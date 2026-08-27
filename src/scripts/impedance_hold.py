"""
Interactive feel test for the Cartesian impedance controller.

The arm chases a commanded target. Move the target with the keyboard and
watch the arm follow, or push the arm directly and watch it spring back.
A blue sphere marks the commanded target, so the gap between it and the
gripper is the tracking error made visible.

Keys:
    W / S   move target along +x / -x
    A / D   move target along +y / -y
    Q / E   move target along +z / -z
    [ / ]   halve / double translational stiffness
    r       re-anchor the target to the current tip pose
    n       toggle the nullspace posture term

Run:
    python src\\scripts\\impedance_hold.py
"""

import time

import mujoco
import mujoco.viewer
import numpy as np

from src.config import SCENE_PATH
from src.controllers.impedance import ImpedanceController

MODEL_PATH = str(SCENE_PATH)

N_ARM = 7
NUDGE = 0.05  # metres per keypress


def setup_model(path, timestep=0.001):
    """
    Loads the model and configures it for torque control at a fixed
    timestep.

    Three changes matter. The integrator is switched to implicitfast, since
    the default semi-implicit Euler goes unstable at the stiffnesses worth
    testing. The actuator gains are zeroed so the built-in position servos
    stop fighting the torques written into qfrc_applied. And dry joint
    friction is removed, because it cannot be cancelled by a feedforward
    torque and would otherwise dominate the steady-state error.

    input:  path (str), timestep (float) seconds
    output: (MjModel, MjData)
    """
    model = mujoco.MjModel.from_xml_path(path)
    model.opt.timestep = timestep
    model.opt.integrator = mujoco.mjtIntegrator.mjINT_IMPLICITFAST

    model.actuator_gainprm[:, :] = 0.0
    model.actuator_biasprm[:, :] = 0.0
    model.dof_frictionloss[:] = 0.0

    data = mujoco.MjData(model)
    return model, data


def reset_to_keyframe(model, data, name="home"):
    """
    Resets to a named keyframe, falling back to the default pose.

    Starting from 'home' matters more here than it did for gravity
    compensation. The default zero pose has the arm fully extended, which
    is close to singular, and a badly conditioned Jacobian will make the
    controller behave strangely for reasons that have nothing to do with
    your gains.

    input:  model (MjModel), data (MjData), name (str)
    output: None, mutates data
    """
    kid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, name)
    if kid >= 0:
        mujoco.mj_resetDataKeyframe(model, data, kid)
    else:
        print(f"Keyframe '{name}' not found, using default pose.")
        mujoco.mj_resetData(model, data)
    mujoco.mj_forward(model, data)


def make_key_callback(ctrl, data, state):
    """
    Builds the viewer keypress handler as a closure over the controller.

    launch_passive calls this with a GLFW keycode whenever a key is
    pressed. Keycodes arrive as uppercase regardless of shift state, so the
    handler folds case rather than matching both.

    input:  ctrl (ImpedanceController), data (MjData), state (dict)
    output: function taking an int keycode
    """
    moves = {
        "W": (0, +NUDGE),
        "S": (0, -NUDGE),
        "A": (1, +NUDGE),
        "D": (1, -NUDGE),
        "Q": (2, +NUDGE),
        "E": (2, -NUDGE),
    }

    def callback(keycode):
        key = chr(keycode).upper()

        if key in moves:
            axis, delta = moves[key]
            ctrl.x_des[axis] += delta
            print(f"target -> {np.round(ctrl.x_des, 3)}")

        elif key == "[":
            ctrl.kp_diag[:3] = np.maximum(ctrl.kp_diag[:3] * 0.5, 1.0)
            print(f"Kp translational: {ctrl.kp_diag[0]:.1f} N/m")

        elif key == "]":
            ctrl.kp_diag[:3] = np.minimum(ctrl.kp_diag[:3] * 2.0, 5000.0)
            print(f"Kp translational: {ctrl.kp_diag[0]:.1f} N/m")

        elif key == "R":
            pos, quat = ctrl.current_pose(data)
            ctrl.set_target(pos, quat)
            print("Target re-anchored to current pose.")

        elif key == "N":
            state["nullspace"] = not state["nullspace"]
            ctrl.kp_null = 10.0 if state["nullspace"] else 0.0
            print(f"Nullspace posture: {'on' if state['nullspace'] else 'off'}")

    return callback


def draw_target(viewer, pos, radius=0.012):
    """
    Draws a translucent sphere at the commanded target using the viewer's
    user scene.

    Drawing into user_scn rather than adding a site to the model keeps the
    marker purely visual. A site would appear in camera renders and end up
    baked into the dataset, and moving a site every frame means mutating
    the model, which is the sort of thing that causes confusing bugs later.

    input:  viewer (viewer handle), pos (array (3,)), radius (float) metres
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


def run(model, data, ctrl, key_callback):
    """
    Steps the simulation in real time, applying the controller torque every
    step and reporting tracking error once a second.

    Wall-clock pacing matters for a feel test. Without it MuJoCo runs as
    fast as the CPU allows, so the arm reacts to input in what looks like
    slow motion or fast forward depending on the machine.

    input:  model (MjModel), data (MjData), ctrl (ImpedanceController),
            key_callback (function)
    output: None, blocks until the viewer closes
    """
    print("\nKeys: WASD/QE move target   [ ] stiffness   r re-anchor   n nullspace")
    print("Ctrl + right-drag a link to push the arm directly.\n")

    last_report = time.perf_counter()
    sim_start = time.perf_counter()

    with mujoco.viewer.launch_passive(model, data, key_callback=key_callback) as viewer:
        while viewer.is_running():
            data.qfrc_applied[:N_ARM] = ctrl.compute_torque(data)
            mujoco.mj_step(model, data)

            draw_target(viewer, ctrl.x_des)
            viewer.sync()

            now = time.perf_counter()
            if now - last_report > 1.0:
                pos, _ = ctrl.current_pose(data)
                err_mm = np.linalg.norm(ctrl.x_des - pos) * 1000.0
                print(f"position error: {err_mm:7.3f} mm")
                last_report = now

            # Pace to wall clock so one simulated second takes one real second.
            lag = data.time - (now - sim_start)
            if lag > 0:
                time.sleep(lag)


def main():
    """
    Entry point. Builds the model, controller and viewer loop.

    input:  none
    output: None
    """
    model, data = setup_model(MODEL_PATH)
    reset_to_keyframe(model, data, "home")

    ctrl = ImpedanceController(
        model,
        data,
        kp_trans=300.0,
        kp_rot=30.0,
        zeta=1.0,
        kp_null=10.0,
        zeta_null=1.0,
        n_arm=N_ARM,
    )

    state = {"nullspace": True}
    run(model, data, ctrl, make_key_callback(ctrl, data, state))


if __name__ == "__main__":
    main()
