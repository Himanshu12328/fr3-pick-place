"""
Regenerates the image streams of an existing dataset at a different
resolution and camera configuration, by replaying recorded joint states
through the simulator.

Nothing about the demonstrations is re-collected. Each frame's state array
holds all seven arm joint angles plus both finger positions, and each
episode's metadata holds the block's starting pose, so the scene at any
recorded frame can be reconstructed exactly. Only the block's own
trajectory is not directly recorded, so it is recovered by stepping physics
forward under the recorded arm motion rather than teleporting the arm frame
to frame.

The output is a new dataset directory. The original is left untouched, so a
mistake here costs compute rather than data.

Run:
    python src\\scripts\\rerender.py
"""

import json
import os
import shutil
import time

import mujoco
import numpy as np
from PIL import Image

from src.config import RAW_DATA_DIR, SCENE_PATH, SMALL_DATA_DIR
from src.controllers.impedance import ImpedanceController
from src.data.task import set_block_pose

MODEL_PATH = str(SCENE_PATH)
SRC_DIR = RAW_DATA_DIR
DST_DIR = SMALL_DATA_DIR

CAM_WIDTH, CAM_HEIGHT = 160, 128

N_ARM = 7
FINGER_DOFS = [7, 8]
BLOCK_QPOS_ADR = 9
BLOCK_QVEL_ADR = 9

CONTROL_HZ = 1000
POLICY_HZ = 30
STEPS_PER_ACTION = CONTROL_HZ // POLICY_HZ

KP_TRANS, KP_ROT, ZETA = 800.0, 60.0, 0.7
GRIP_KP, GRIP_KD = 300.0, 15.0

WORKSPACE_MIN = np.array([0.25, -0.35, 0.405])
WORKSPACE_MAX = np.array([0.85, 0.35, 0.85])


def setup_model(path=MODEL_PATH):
    """
    Loads the scene configured exactly as it was during collection.

    Any deviation here changes the physics, and the replayed block
    trajectory would then differ from the one the operator actually
    produced.

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


def render_all(renderer, data, camera_names):
    """
    Renders one frame from each camera at the current sim state.

    input:  renderer (mujoco.Renderer), data (MjData),
            camera_names (list of str)
    output: dict mapping camera name to uint8 array
    """
    out = {}
    for name in camera_names:
        renderer.update_scene(data, camera=name)
        out[name] = renderer.render()
    return out


def replay_episode(model, data, ctrl, renderer, ep_dir, cameras):
    """
    Replays one episode's recorded actions and renders every frame.

    Replaying the actions through the controller, rather than writing the
    recorded joint angles directly into qpos, is what makes the block move
    correctly. Teleporting the arm each frame would leave the block's
    contact forces unresolved and it would never be picked up.

    input:  model (MjModel), data (MjData), ctrl (ImpedanceController),
            renderer (mujoco.Renderer), ep_dir (str), cameras (list of str)
    output: (frames, drift_mm) where frames is a list of dicts of images
            and drift_mm measures how far the replayed arm ended up from
            the recorded one
    """
    npz = np.load(os.path.join(ep_dir, "data.npz"))
    with open(os.path.join(ep_dir, "meta.json")) as f:
        meta = json.load(f)

    actions = npz["action"]
    states = npz["state"]

    kid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "home")
    if kid >= 0:
        mujoco.mj_resetDataKeyframe(model, data, kid)
    else:
        mujoco.mj_resetData(model, data)

    set_block_pose(model, data,
                   np.array(meta["block_start_pos"]),
                   np.array(meta["block_start_quat"]),
                   BLOCK_QPOS_ADR, BLOCK_QVEL_ADR)
    mujoco.mj_forward(model, data)

    pos, quat = ctrl.current_pose(data)
    ctrl.set_target(pos, quat)

    frames = []
    for action in actions:
        # Render before stepping, so frame i corresponds to the state the
        # operator saw when issuing action i. Rendering after would shift
        # the whole dataset one step out of alignment with its actions.
        frames.append(render_all(renderer, data, cameras))

        target_pos = np.clip(action[:3], WORKSPACE_MIN, WORKSPACE_MAX)
        target_quat = action[3:7] / max(np.linalg.norm(action[3:7]), 1e-9)
        grip = float(np.clip(action[7], 0.0, 0.04))

        ctrl.set_target(target_pos, target_quat)
        for _ in range(STEPS_PER_ACTION):
            data.qfrc_applied[:N_ARM] = ctrl.compute_torque(data)
            data.qfrc_applied[FINGER_DOFS] = gripper_torque(data, grip)
            mujoco.mj_step(model, data)

    # Compare the replayed final joint configuration against the recorded
    # one. Physics is deterministic here, so any large discrepancy means
    # the replay environment does not match the collection environment.
    drift = np.abs(data.qpos[:N_ARM] - states[-1][:N_ARM]).max()

    return frames, float(drift)


def write_episode(dst_dir, ep_name, src_dir, frames, cameras):
    """
    Writes the re-rendered episode, copying the unchanged arrays and
    metadata across.

    The state, action and timestamp arrays are identical to the original.
    Only the images change, so they are copied rather than regenerated.

    input:  dst_dir (str), ep_name (str), src_dir (str),
            frames (list of dicts), cameras (list of str)
    output: None
    """
    out_dir = os.path.join(dst_dir, ep_name)
    if os.path.exists(out_dir):
        shutil.rmtree(out_dir)
    os.makedirs(out_dir)

    src_ep = os.path.join(src_dir, ep_name)
    shutil.copy(os.path.join(src_ep, "data.npz"), out_dir)
    shutil.copy(os.path.join(src_ep, "meta.json"), out_dir)

    for cam in cameras:
        cam_dir = os.path.join(out_dir, cam)
        os.makedirs(cam_dir)
        for i, frame in enumerate(frames):
            Image.fromarray(frame[cam]).save(
                os.path.join(cam_dir, f"{i:05d}.png"), compress_level=1)


def main():
    """
    Re-renders every episode in the source dataset.

    input:  none
    output: None
    """
    with open(os.path.join(SRC_DIR, "dataset_info.json")) as f:
        info = json.load(f)
    cameras = info["cameras"]

    episodes = sorted(d for d in os.listdir(SRC_DIR) if d.startswith("episode_"))
    print(f"{len(episodes)} episodes, cameras {cameras}, "
          f"target size {CAM_WIDTH}x{CAM_HEIGHT}\n")

    os.makedirs(DST_DIR, exist_ok=True)

    # Carry the metadata forward with the new image size recorded.
    info["image_size"] = [CAM_HEIGHT, CAM_WIDTH]
    with open(os.path.join(DST_DIR, "dataset_info.json"), "w") as f:
        json.dump(info, f, indent=2)

    model, data = setup_model()
    ctrl = ImpedanceController(model, data,
                               kp_trans=KP_TRANS, kp_rot=KP_ROT, zeta=ZETA,
                               kp_null=10.0, zeta_null=1.0,
                               n_arm=N_ARM, verbose=False)
    renderer = mujoco.Renderer(model, height=CAM_HEIGHT, width=CAM_WIDTH)

    start = time.perf_counter()
    drifts = []

    for ep_name in episodes:
        frames, drift = replay_episode(model, data, ctrl, renderer,
                                       os.path.join(SRC_DIR, ep_name), cameras)
        write_episode(DST_DIR, ep_name, SRC_DIR, frames, cameras)
        drifts.append(drift)

        flag = "  <-- check" if drift > 0.05 else ""
        print(f"  {ep_name}: {len(frames)} frames, joint drift "
              f"{drift:.4f} rad{flag}")

    renderer.close()

    elapsed = time.perf_counter() - start
    print(f"\n{len(episodes)} episodes in {elapsed:.0f} s")
    print(f"joint drift: mean {np.mean(drifts):.4f}, max {np.max(drifts):.4f} rad")
    print(f"written to {DST_DIR}")

    if np.max(drifts) > 0.05:
        print("\nSome episodes drifted noticeably from their recorded joint")
        print("angles. The replay physics may not match collection; check the")
        print("gain constants and friction settings before using this dataset.")


if __name__ == "__main__":
    main()