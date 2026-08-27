"""
Drive the Franka FR3 from Python with gravity and
Coriolis forces cancelled, so the arm floats weightlessly.

This is the foundation the impedance controller sits on. If this works,
the arm stays wherever you drag it. If it sags, something is wrong with
joint indexing or the model's built-in actuators are fighting you.

Run:
    python src\\scripts\\gravity_comp.py
"""

import mujoco
import mujoco.viewer

from src.config import FR3_DIR

MODEL_PATH = str(FR3_DIR / "scene.xml")


def load_model(path):
    """
    Loads a MuJoCo model from an MJCF XML file and creates its state buffer.

    MjModel holds everything constant about the robot: link masses, inertias,
    joint limits, geometry. MjData holds everything that changes as time
    advances: joint positions, velocities, applied forces, contact list.

    input:  path (str) absolute path to the scene XML
    output: (MjModel, MjData) tuple
    """
    model = mujoco.MjModel.from_xml_path(path)
    data = mujoco.MjData(model)
    return model, data


def describe_model(model):
    """
    Prints the joint and actuator layout so you can verify which indices
    belong to the arm versus the gripper.

    Never assume the first 7 DOFs are the arm. Some Menagerie scenes include
    the Franka Hand, some add a free-floating object. Read the output once
    and confirm before writing any controller that slices arrays.

    input:  model (MjModel)
    output: None, prints to stdout
    """
    print(f"nq (position coords):  {model.nq}")
    print(f"nv (velocity coords):  {model.nv}")
    print(f"nu (actuators):        {model.nu}\n")

    print("Joints:")
    for i in range(model.njnt):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, i)
        jnt_type = mujoco.mjtJoint(model.jnt_type[i]).name
        print(
            f"  [{i}] {name:<28} type={jnt_type}  qposadr={model.jnt_qposadr[i]}  dofadr={model.jnt_dofadr[i]}"
        )

    print("\nActuators:")
    for i in range(model.nu):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, i)
        print(f"  [{i}] {name}")
    print()


def disable_actuators(model):
    """
    Neutralises every actuator in the model by zeroing its gain and bias
    parameters, so it produces no force regardless of the ctrl signal.

    The Menagerie FR3 ships with position servos. Left active with ctrl=0
    they will try to drag every joint to zero angle, which fights whatever
    you write into qfrc_applied. Zeroing the gains lets qfrc_applied be the
    only thing acting on the robot, which is what you want for torque
    control.

    input:  model (MjModel) modified in place
    output: None
    """
    model.actuator_gainprm[:, :] = 0.0
    model.actuator_biasprm[:, :] = 0.0


def reset_to_keyframe(model, data, name="home"):
    """
    Resets the simulation state to a named keyframe defined in the XML,
    falling back to the model's default pose if that keyframe is absent.

    The FR3 'home' keyframe is a bent-elbow ready pose. Starting from the
    default zero pose puts the arm fully extended, which is near a
    kinematic singularity and makes the Jacobian badly conditioned later.

    input:  model (MjModel), data (MjData), name (str) keyframe name
    output: None, mutates data in place
    """
    key_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, name)
    if key_id >= 0:
        mujoco.mj_resetDataKeyframe(model, data, key_id)
    else:
        print(f"Keyframe '{name}' not found, using default pose.")
        mujoco.mj_resetData(model, data)
    mujoco.mj_forward(model, data)


def gravity_compensation_torque(data, n_dof):
    """
    Returns the joint torques that exactly cancel gravity and Coriolis
    effects for the first n_dof degrees of freedom.

    MuJoCo's data.qfrc_bias holds C(q, qdot) * qdot + g(q), that is the
    Coriolis/centrifugal term plus the gravity term, evaluated fresh at the
    current state. Applying it as a torque means the equation of motion
    reduces to M(q) * qddot = tau_external, so the arm behaves like a
    frictionless free-floating mass. Anything you add on top of this is
    pure controller behaviour, uncontaminated by the robot's own weight.

    input:  data (MjData) current sim state, n_dof (int) DOFs to compensate
    output: numpy array of shape (n_dof,), torques in Nm
    """
    return data.qfrc_bias[:n_dof].copy()


def run(model, data, n_dof):
    """
    Opens the passive viewer and steps the simulation forward in real time,
    applying gravity compensation every step until the window is closed.

    launch_passive gives you a viewer that does not own the loop, so your
    Python code controls when mj_step is called. That matters because every
    controller you write from here on needs to run between steps.

    input:  model (MjModel), data (MjData), n_dof (int) arm DOF count
    output: None, blocks until the viewer window closes
    """
    print("Viewer open. Ctrl + right-drag a link to push the arm.")
    print("It should move freely and stay put when released.\n")

    with mujoco.viewer.launch_passive(model, data) as viewer:
        while viewer.is_running():
            data.qfrc_applied[:n_dof] = gravity_compensation_torque(data, n_dof)
            mujoco.mj_step(model, data)
            viewer.sync()


def main():
    """
    Entry point. Loads the model, reports its layout, disables the built-in
    position servos, resets to a sane pose, and runs the float test.

    input:  none
    output: None
    """
    model, data = load_model(MODEL_PATH)
    describe_model(model)
    disable_actuators(model)
    reset_to_keyframe(model, data, "home")

    n_dof = 7  # FR3 arm joints; confirm against the printout above
    run(model, data, n_dof)


if __name__ == "__main__":
    main()
