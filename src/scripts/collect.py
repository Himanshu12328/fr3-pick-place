"""
Teleoperated demonstration collection for the pick-and-place task.

Runs the impedance controller at 1 kHz while recording observations and
actions at 30 Hz. Episodes are driven from the pad: start one, perform the
demonstration, save it, and the recorder writes to disk with an automatic
success label.

Pad controls:
    left stick      move in x and y (operator frame)
    L2 / R2         move down / up
    right stick     pitch and yaw
    L1 / R1         roll
    square / cross  close / open gripper

    D-pad UP        start recording a new episode
    D-pad RIGHT     save the current episode and re-randomise
    D-pad DOWN      discard the current episode and re-randomise
    circle          re-anchor the target to the current tip pose

Run:
    python src\\scripts\\collect.py
"""

import time

import mujoco
import mujoco.viewer
import numpy as np

from src.config import RAW_DATA_DIR, SCENE_PATH
from src.controllers.impedance import ImpedanceController
from src.data.recorder import EpisodeRecorder, next_episode_index
from src.data.task import check_success, sample_block_pose, set_block_pose
from src.teleop.dualsense import (
    BTN_CIRCLE,
    BTN_DISCARD_EP,
    BTN_SAVE_EP,
    BTN_START_EP,
    DualSenseTeleop,
)

MODEL_PATH = str(SCENE_PATH)
DATA_DIR = RAW_DATA_DIR

N_ARM = 7
FINGER_DOFS = [7, 8]
BLOCK_QPOS_ADR = 9
BLOCK_QVEL_ADR = 9

CAMERAS = ["external", "wrist_left", "wrist_right"]
CAM_WIDTH, CAM_HEIGHT = 640, 480

RECORD_FPS = 30
CONTROL_HZ = 1000
RECORD_EVERY = CONTROL_HZ // RECORD_FPS  # 33 sim steps per recorded frame

# Transit gain set from the damping sweep: 1.5 Hz bandwidth, 2.9% overshoot.
KP_TRANS, KP_ROT, ZETA = 800.0, 60.0, 0.7
GRIP_KP, GRIP_KD = 300.0, 15.0

WORKSPACE_MIN = np.array([0.25, -0.35, 0.405])
WORKSPACE_MAX = np.array([0.85, 0.35, 0.85])

MAX_EPISODE_S = 60.0  # safety cap so a forgotten recording cannot fill disk


def setup_model(path):
    """
    Loads the scene and configures it for torque control.

    Friction is zeroed only on the arm DOFs. The fingers keep theirs,
    because finger friction is what stops a grasped block from sliding out
    of the jaws under its own weight.

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


def reset_episode(model, data, rng, ctrl, pad):
    """
    Resets the arm to home, places the block at a fresh random pose, and
    reseeds the teleop target so the arm does not lurch.

    input:  model (MjModel), data (MjData), rng (numpy Generator),
            ctrl (ImpedanceController), pad (DualSenseTeleop)
    output: (block_pos, block_quat) the sampled block pose
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
    pad.reseed(pos, quat, gripper=0.04)

    return block_pos, block_quat


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
    Assembles the low-dimensional observation vector.

    Joint velocities are included because a policy predicting target poses
    benefits from knowing whether the arm is already moving. Block pose is
    deliberately excluded: it is available in simulation but not on real
    hardware, and a policy that reads it learns to skip perception
    entirely.

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


def build_action(x_des, quat_des, gripper):
    """
    Assembles the action vector from the commanded target.

    input:  x_des (array (3,)), quat_des (array (4,)), gripper (float)
    output: numpy array of shape (8,) float32
    """
    return np.concatenate([x_des, quat_des, [gripper]]).astype(np.float32)


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


def draw_overlay(viewer, target_pos, recording):
    """
    Draws the commanded target, coloured to indicate recording state.

    A visible recording indicator prevents the two most common collection
    mistakes: performing a demonstration that was never being recorded, and
    leaving the recorder running through a reset.

    input:  viewer (viewer handle), target_pos (array (3,)),
            recording (bool)
    output: None
    """
    rgba = (
        np.array([1.0, 0.2, 0.2, 0.7]) if recording else np.array([0.2, 0.5, 1.0, 0.5])
    )

    viewer.user_scn.ngeom = 0
    mujoco.mjv_initGeom(
        viewer.user_scn.geoms[0],
        type=mujoco.mjtGeom.mjGEOM_SPHERE,
        size=np.array([0.012, 0, 0]),
        pos=np.asarray(target_pos, dtype=np.float64),
        mat=np.eye(3).flatten(),
        rgba=rgba,
    )
    viewer.user_scn.ngeom = 1


def run(model, data, ctrl, pad, recorder, rng):
    """
    Main collection loop.

    Control runs at the simulation rate. Recording is decimated to 30 Hz
    because that is the rate the policy will run at, and because rendering
    three cameras every millisecond would make the loop far slower than
    real time. Rendering happens only on recorded frames and only while
    recording, which keeps free-flying teleop responsive.

    input:  model (MjModel), data (MjData), ctrl (ImpedanceController),
            pad (DualSenseTeleop), recorder (EpisodeRecorder),
            rng (numpy Generator)
    output: None, blocks until the viewer closes
    """
    dt = model.opt.timestep
    block_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "block")

    ep_index = next_episode_index(recorder.out_dir)
    print(f"Next episode index: {ep_index}")

    block_pos, block_quat = reset_episode(model, data, rng, ctrl, pad)

    step_count = 0
    ep_start_time = 0.0
    saved = 0
    successes = 0
    last_report = time.perf_counter()
    sim_start = time.perf_counter()

    renderer = mujoco.Renderer(model, height=CAM_HEIGHT, width=CAM_WIDTH)

    with mujoco.viewer.launch_passive(model, data) as viewer:
        while viewer.is_running():
            twist = pad.read_twist()
            x_des, quat_des, grip = pad.integrate(twist, dt)

            # --- episode control -------------------------------------

            if pad.button_pressed(BTN_START_EP) and not recorder.recording:
                recorder.start()
                ep_start_time = data.time
                print(f"\n[{ep_index}] recording...")

            if pad.button_pressed(BTN_SAVE_EP) and recorder.recording:
                success, dist = check_success(data, block_id)
                recorder.save(
                    ep_index,
                    success,
                    extra_meta={
                        "block_start_pos": block_pos.tolist(),
                        "block_start_quat": block_quat.tolist(),
                        "final_distance_m": dist,
                    },
                )
                saved += 1
                successes += int(success)
                ep_index += 1
                print(f"  session: {saved} saved, {successes} successful")
                block_pos, block_quat = reset_episode(model, data, rng, ctrl, pad)

            if pad.button_pressed(BTN_DISCARD_EP):
                if recorder.recording:
                    n = recorder.discard()
                    print(f"\n  discarded {n} frames")
                block_pos, block_quat = reset_episode(model, data, rng, ctrl, pad)

            if pad.button_pressed(BTN_CIRCLE):
                pos, quat = ctrl.current_pose(data)
                pad.reseed(pos, quat, gripper=pad.gripper)

            if recorder.recording and (data.time - ep_start_time) > MAX_EPISODE_S:
                print(f"\n  episode exceeded {MAX_EPISODE_S:.0f}s, discarding")
                recorder.discard()
                block_pos, block_quat = reset_episode(model, data, rng, ctrl, pad)

            # --- physics ---------------------------------------------

            ctrl.set_target(x_des, quat_des)
            data.qfrc_applied[:N_ARM] = ctrl.compute_torque(data)
            data.qfrc_applied[FINGER_DOFS] = gripper_torque(data, grip)
            mujoco.mj_step(model, data)

            # --- recording, decimated to RECORD_FPS ------------------

            if recorder.recording and step_count % RECORD_EVERY == 0:
                images = render_all(renderer, data, CAMERAS)
                recorder.add(
                    build_state(data),
                    build_action(x_des, quat_des, grip),
                    images,
                    data.time - ep_start_time,
                )

            step_count += 1

            draw_overlay(viewer, x_des, recorder.recording)
            viewer.sync()

            now = time.perf_counter()
            if now - last_report > 1.0:
                pos, _ = ctrl.current_pose(data)
                err = np.linalg.norm(x_des - pos) * 1000.0
                status = (
                    f"REC {recorder.n_frames():4d}f"
                    if recorder.recording
                    else "idle     "
                )
                print(
                    f"\r{status}  err {err:5.1f} mm  grip {grip * 1000:4.1f} mm   ",
                    end="",
                    flush=True,
                )
                last_report = now

            lag = data.time - (now - sim_start)
            if lag > 0:
                time.sleep(lag)

    renderer.close()
    print(f"\n\nSession complete: {saved} episodes saved, {successes} successful")


def main():
    """
    Entry point.

    input:  none
    output: None
    """
    model, data = setup_model(MODEL_PATH)

    ctrl = ImpedanceController(
        model,
        data,
        kp_trans=KP_TRANS,
        kp_rot=KP_ROT,
        zeta=ZETA,
        kp_null=10.0,
        zeta_null=1.0,
        n_arm=N_ARM,
    )

    pos, quat = ctrl.current_pose(data)
    pad = DualSenseTeleop(
        pos,
        quat,
        gripper_init=0.04,
        workspace_min=WORKSPACE_MIN,
        workspace_max=WORKSPACE_MAX,
    )

    recorder = EpisodeRecorder(
        DATA_DIR,
        CAMERAS,
        fps=RECORD_FPS,
        workspace_min=WORKSPACE_MIN,
        workspace_max=WORKSPACE_MAX,
    )
    rng = np.random.default_rng()

    print("\nD-pad UP start   RIGHT save   DOWN discard   circle re-anchor")
    print("Red marker means recording.\n")

    try:
        run(model, data, ctrl, pad, recorder, rng)
    finally:
        pad.close()


if __name__ == "__main__":
    main()
