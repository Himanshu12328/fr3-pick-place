"""
Drives the FR3 with the DualSense through the impedance controller. No
recording; this is for confirming the mapping feels right and finding a
comfortable velocity scale before collecting data.

Pad:
    left stick      move in x and y
    L2 / R2         move down / up
    right stick     pitch and yaw
    L1 / R1         roll
    square / cross  close / open gripper
    triangle        reset the scene
    circle          re-anchor the target to the current tip pose

Run:
    python src\\scripts\\teleop_test.py
"""

import time

import mujoco
import mujoco.viewer
import numpy as np

from src.config import SCENE_PATH
from src.controllers.impedance import ImpedanceController
from src.teleop.dualsense import BTN_CIRCLE, BTN_TRIANGLE, DualSenseTeleop

MODEL_PATH = str(SCENE_PATH)

N_ARM = 7
FINGER_DOFS = [7, 8]

# Transit gain set, chosen from the damping sweep: 1.5 Hz bandwidth at
# 2.9 percent overshoot.
KP_TRANS, KP_ROT, ZETA = 800.0, 60.0, 0.7

# Finger joints are driven by a simple PD, not by the impedance controller.
GRIP_KP, GRIP_KD = 300.0, 15.0

WORKSPACE_MIN = np.array([0.25, -0.35, 0.405])
WORKSPACE_MAX = np.array([0.85, 0.35, 0.85])


def setup_model(path):
    """
    Loads the scene and configures it for torque control.

    input:  path (str)
    output: (MjModel, MjData)
    """
    model = mujoco.MjModel.from_xml_path(path)
    model.opt.timestep = 0.001
    model.opt.integrator = mujoco.mjtIntegrator.mjINT_IMPLICITFAST
    model.actuator_gainprm[:, :] = 0.0
    model.actuator_biasprm[:, :] = 0.0
    model.dof_frictionloss[:N_ARM] = 0.0

    return model, mujoco.MjData(model)


def reset_scene(model, data):
    """
    Resets to the home keyframe.

    input:  model (MjModel), data (MjData)
    output: None
    """
    kid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "home")
    if kid >= 0:
        mujoco.mj_resetDataKeyframe(model, data, kid)
    else:
        mujoco.mj_resetData(model, data)
    mujoco.mj_forward(model, data)


def gripper_torque(data, target_width):
    """
    Computes PD forces driving both fingers toward a commanded opening.

    Each finger slides along its own axis with the same sign convention, so
    both take the same target value. Force is clamped so a closed gripper
    squeezes rather than crushes, which keeps the contact solver stable.

    input:  data (MjData), target_width (float) metres per finger
    output: numpy array of shape (2,), forces in N
    """
    q = data.qpos[FINGER_DOFS]
    qd = data.qvel[FINGER_DOFS]
    force = GRIP_KP * (target_width - q) - GRIP_KD * qd
    return np.clip(force, -60.0, 60.0)


def draw_target(viewer, pos, radius=0.012):
    """
    Draws a translucent sphere at the commanded target position.

    input:  viewer (viewer handle), pos (array (3,)), radius (float)
    output: None
    """
    viewer.user_scn.ngeom = 0
    mujoco.mjv_initGeom(
        viewer.user_scn.geoms[0],
        type=mujoco.mjtGeom.mjGEOM_SPHERE,
        size=np.array([radius, 0, 0]),
        pos=np.asarray(pos, dtype=np.float64),
        mat=np.eye(3).flatten(),
        rgba=np.array([0.2, 0.5, 1.0, 0.6]),
    )
    viewer.user_scn.ngeom = 1


def run(model, data, ctrl, pad):
    """
    Main teleop loop: read the pad, integrate the target, apply torques.

    The pad is polled every simulation step rather than at a lower rate.
    Polling is cheap and integrating at the control rate keeps the target
    trajectory smooth, which matters because that trajectory becomes the
    action stream during collection.

    input:  model (MjModel), data (MjData), ctrl (ImpedanceController),
            pad (DualSenseTeleop)
    output: None, blocks until the viewer closes
    """
    dt = model.opt.timestep
    block_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "block")

    last_report = time.perf_counter()
    sim_start = time.perf_counter()

    with mujoco.viewer.launch_passive(model, data) as viewer:
        while viewer.is_running():
            twist = pad.read_twist()
            x_des, quat_des, grip = pad.integrate(twist, dt)

            if pad.button_pressed(BTN_TRIANGLE):
                reset_scene(model, data)
                pos, quat = ctrl.current_pose(data)
                pad.x_des, pad.quat_des = pos.copy(), quat.copy()
                print("scene reset")

            if pad.button_pressed(BTN_CIRCLE):
                pos, quat = ctrl.current_pose(data)
                pad.x_des, pad.quat_des = pos.copy(), quat.copy()
                print("target re-anchored")

            ctrl.set_target(x_des, quat_des)

            data.qfrc_applied[:N_ARM] = ctrl.compute_torque(data)
            data.qfrc_applied[FINGER_DOFS] = gripper_torque(data, grip)

            mujoco.mj_step(model, data)
            draw_target(viewer, x_des)
            viewer.sync()

            now = time.perf_counter()
            if now - last_report > 1.0:
                pos, _ = ctrl.current_pose(data)
                err = np.linalg.norm(x_des - pos) * 1000.0
                bz = data.xpos[block_id][2]
                print(f"tracking err {err:6.1f} mm   grip {grip*1000:4.1f} mm   "
                      f"block z {bz:.3f}")
                last_report = now

            lag = data.time - (now - sim_start)
            if lag > 0:
                time.sleep(lag)


def main():
    """
    Entry point.

    input:  none
    output: None
    """
    model, data = setup_model(MODEL_PATH)
    reset_scene(model, data)

    ctrl = ImpedanceController(model, data,
                               kp_trans=KP_TRANS, kp_rot=KP_ROT, zeta=ZETA,
                               kp_null=10.0, zeta_null=1.0, n_arm=N_ARM)

    pos, quat = ctrl.current_pose(data)
    pad = DualSenseTeleop(pos, quat, gripper_init=0.04,
                          workspace_min=WORKSPACE_MIN,
                          workspace_max=WORKSPACE_MAX)

    print("\nleft stick: xy   L2/R2: z   right stick: pitch/yaw   L1/R1: roll")
    print("square/cross: close/open gripper   triangle: reset   circle: re-anchor\n")

    try:
        run(model, data, ctrl, pad)
    finally:
        pad.close()


if __name__ == "__main__":
    main()