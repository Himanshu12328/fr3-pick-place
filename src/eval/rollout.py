"""
Runs a policy in simulation and reports success rate over randomised
trials.

The environment here is deliberately identical to the one used during
collection: same model, same impedance gains, same block distribution,
same success criterion. Any divergence would make the reported number
incomparable to the demonstrations, and a success rate that cannot be
compared to anything is not worth measuring.

A policy is any callable with the signature

    policy(observation) -> action

where observation is a dict of numpy arrays keyed the same way the dataset
is, and action is a length-8 vector of target pose plus gripper. That
interface means a learned policy, a scripted baseline, and a recorded
action replay are all interchangeable, which is what lets the harness be
validated independently of any model.
"""

import os
import time

import mujoco
import numpy as np

from src.config import SCENE_PATH
from src.controllers.impedance import ImpedanceController
from src.data.task import check_success, sample_block_pose, set_block_pose

MODEL_PATH = str(SCENE_PATH)

N_ARM = 7
FINGER_DOFS = [7, 8]
BLOCK_QPOS_ADR = 9
BLOCK_QVEL_ADR = 9

CAMERAS = ["external", "wrist_left", "wrist_right"]
CAM_WIDTH, CAM_HEIGHT = 160, 128

POLICY_HZ = 30
CONTROL_HZ = 1000
STEPS_PER_ACTION = CONTROL_HZ // POLICY_HZ

KP_TRANS, KP_ROT, ZETA = 800.0, 60.0, 0.7
GRIP_KP, GRIP_KD = 300.0, 15.0

WORKSPACE_MIN = np.array([0.25, -0.35, 0.405])
WORKSPACE_MAX = np.array([0.85, 0.35, 0.85])

MAX_POLICY_STEPS = 600  # 20 s at 30 Hz; demos average 343


def setup_model(path=MODEL_PATH):
    """
    Loads the scene configured exactly as it was during collection.

    input:  path (str)
    output: (MjModel, MjData)
    """
    model = mujoco.MjModel.from_xml_path(path)
    model.opt.timestep = 1.0 / CONTROL_HZ
    model.opt.integrator = mujoco.mjtIntegrator.mjINT_IMPLICITFAST
    model.actuator_gainprm[:, :] = 0.0
    model.actuator_biasprm[:, :] = 0.0
    model.dof_frictionloss[:N_ARM] = 0.0

    return model, mujoco.MjData(model)


def gripper_torque(data, target_width):
    """
    Computes PD forces driving both fingers toward a commanded opening.

    input:  data (MjData), target_width (float) metres per finger
    output: numpy array of shape (2,), forces in N
    """
    q = data.qpos[FINGER_DOFS]
    qd = data.qvel[FINGER_DOFS]
    return np.clip(GRIP_KP * (target_width - q) - GRIP_KD * qd, -60.0, 60.0)


def build_state(data):
    """
    Assembles the low-dimensional observation, matching the layout used
    during collection.

    Any mismatch here between collection and evaluation silently feeds the
    policy garbage, and the symptom is a policy that trained well and
    evaluates at zero. Keep this function and collect.py's version
    identical.

    input:  data (MjData)
    output: numpy array of shape (16,) float32
    """
    return np.concatenate(
        [
            data.qpos[:N_ARM],
            data.qvel[:N_ARM],
            data.qpos[FINGER_DOFS],
        ]
    ).astype(np.float32)


def render_all(renderer, data, camera_names=CAMERAS):
    """
    Renders one frame from each camera.

    input:  renderer (mujoco.Renderer), data (MjData),
            camera_names (list of str)
    output: dict mapping camera name to uint8 array
    """
    out = {}
    for name in camera_names:
        renderer.update_scene(data, camera=name)
        out[name] = renderer.render()
    return out


def build_observation(data, renderer, cameras, need_images=True):
    """
    Assembles the observation dict in the same key layout as the dataset.

    input:  data (MjData), renderer (mujoco.Renderer or None),
            cameras (list of str), need_images (bool)
    output: dict with observation.state and observation.images.<camera>
    """
    obs = {"observation.state": build_state(data)}

    if need_images and renderer is not None:
        for name, img in render_all(renderer, data, cameras).items():
            obs[f"observation.images.{name}"] = img

    return obs


def reset_trial(model, data, ctrl, rng):
    """
    Resets the arm to home and places the block at a fresh random pose.

    input:  model (MjModel), data (MjData), ctrl (ImpedanceController),
            rng (numpy Generator)
    output: (block_pos, block_quat)
    """
    kid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "home")
    if kid >= 0:
        mujoco.mj_resetDataKeyframe(model, data, kid)
    else:
        mujoco.mj_resetData(model, data)

    block_pos, block_quat = sample_block_pose(rng)
    set_block_pose(model, data, block_pos, block_quat, BLOCK_QPOS_ADR, BLOCK_QVEL_ADR)
    mujoco.mj_forward(model, data)

    pos, quat = ctrl.current_pose(data)
    ctrl.set_target(pos, quat)

    return block_pos, block_quat


def run_trial(
    model,
    data,
    ctrl,
    policy,
    renderer,
    rng,
    max_steps=MAX_POLICY_STEPS,
    need_images=True,
    record_frames=False,
):
    """
    Runs one episode and returns whether the task was completed.

    The policy is queried at POLICY_HZ and its action held for
    STEPS_PER_ACTION simulation steps. That decimation is what makes the
    rollout match collection, where the operator's target was likewise
    sampled into the dataset at 30 Hz while the controller ran at 1 kHz.

    Actions are clipped to the workspace before being commanded. A policy
    early in training can emit wild values, and without the clip a single
    bad prediction throws the arm into the table and the trial tells you
    nothing.

    input:  model, data, ctrl, policy (callable), renderer, rng,
            max_steps (int) policy steps before timeout,
            need_images (bool) skip rendering for state-only policies,
            record_frames (bool) collect frames for video output
    output: dict with success, steps, distance, block_start, frames
    """
    block_pos, block_quat = reset_trial(model, data, ctrl, rng)
    block_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "block")

    frames = []
    success = False

    for step in range(max_steps):  # noqa: B007 - step is used after the loop
        obs = build_observation(data, renderer, CAMERAS, need_images)
        action = np.asarray(policy(obs), dtype=np.float64)

        target_pos = np.clip(action[:3], WORKSPACE_MIN, WORKSPACE_MAX)
        target_quat = action[3:7] / max(np.linalg.norm(action[3:7]), 1e-9)
        grip = float(np.clip(action[7], 0.0, 0.04))

        ctrl.set_target(target_pos, target_quat)

        for _ in range(STEPS_PER_ACTION):
            data.qfrc_applied[:N_ARM] = ctrl.compute_torque(data)
            data.qfrc_applied[FINGER_DOFS] = gripper_torque(data, grip)
            mujoco.mj_step(model, data)

        if record_frames and renderer is not None:
            renderer.update_scene(data, camera="external")
            frames.append(renderer.render())

        success, dist = check_success(data, block_id)
        if success:
            break

    final_success, final_dist = check_success(data, block_id)

    return {
        "success": final_success,
        "steps": step + 1,
        "distance": final_dist,
        "block_start": block_pos,
        "frames": frames,
    }


def evaluate(
    policy, n_trials=50, seed=0, need_images=True, save_video_dir=None, verbose=True
):
    """
    Runs n_trials randomised episodes and reports the success rate.

    The seed is fixed by default so two policies face the same sequence of
    block placements. Comparing policies on different placements adds
    variance that has nothing to do with the policies.

    input:  policy (callable), n_trials (int), seed (int),
            need_images (bool), save_video_dir (str or None),
            verbose (bool)
    output: dict with success_rate, results list, and timing
    """
    model, data = setup_model()
    ctrl = ImpedanceController(
        model,
        data,
        kp_trans=KP_TRANS,
        kp_rot=KP_ROT,
        zeta=ZETA,
        kp_null=10.0,
        zeta_null=1.0,
        n_arm=N_ARM,
        verbose=False,
    )
    rng = np.random.default_rng(seed)

    renderer = None
    if need_images or save_video_dir:
        renderer = mujoco.Renderer(model, height=CAM_HEIGHT, width=CAM_WIDTH)

    results = []
    start = time.perf_counter()

    for i in range(n_trials):
        if hasattr(policy, "reset"):
            policy.reset()

        r = run_trial(
            model,
            data,
            ctrl,
            policy,
            renderer,
            rng,
            need_images=need_images,
            record_frames=save_video_dir is not None,
        )
        results.append(r)

        if verbose:
            mark = "ok  " if r["success"] else "FAIL"
            print(
                f"  trial {i:3d}  {mark}  {r['steps']:4d} steps  "
                f"dist {r['distance'] * 100:5.1f} cm  "
                f"block {np.round(r['block_start'][:2], 3)}"
            )

        if save_video_dir and r["frames"]:
            save_video(r["frames"], save_video_dir, i, r["success"])

    if renderer is not None:
        renderer.close()

    n_success = sum(r["success"] for r in results)
    rate = n_success / n_trials
    elapsed = time.perf_counter() - start

    if verbose:
        succ_steps = [r["steps"] for r in results if r["success"]]
        print(f"\nsuccess rate: {n_success}/{n_trials} = {rate * 100:.1f}%")
        if succ_steps:
            print(
                f"steps on success: mean {np.mean(succ_steps):.0f}, "
                f"min {min(succ_steps)}, max {max(succ_steps)}"
            )
        print(f"elapsed: {elapsed:.1f} s ({elapsed / n_trials:.1f} s/trial)")

    return {"success_rate": rate, "results": results, "elapsed_s": elapsed}


def save_video(frames, out_dir, index, success):
    """
    Writes a rollout as an mp4 for visual inspection of failures.

    Success rate tells you how often a policy works. It never tells you how
    it fails, and the failure mode is what points at the fix.

    input:  frames (list of arrays), out_dir (str), index (int),
            success (bool)
    output: None
    """
    import imageio

    os.makedirs(out_dir, exist_ok=True)
    tag = "success" if success else "fail"
    path = os.path.join(out_dir, f"trial_{index:03d}_{tag}.mp4")
    imageio.mimsave(path, frames, fps=POLICY_HZ)
