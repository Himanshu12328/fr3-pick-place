"""
Reports the passive joint parameters baked into the model, so you know
what the impedance controller is fighting.

Run:
    python src\\scripts\\inspect_passive.py
"""

import mujoco
import numpy as np

from src.config import FR3_DIR

MODEL_PATH = str(FR3_DIR / "scene.xml")


def report_passive_params(model):
    """
    Prints per-DOF damping, frictionloss and armature.

    damping produces a force proportional to velocity and lands in
    qfrc_passive. frictionloss is dry Coulomb friction, resolved by the
    constraint solver, and is the term that causes a constant steady-state
    offset. armature is rotor inertia added to the diagonal of M, already
    accounted for in the task-space inertia.

    input:  model (MjModel)
    output: None, prints to stdout
    """
    print(f"{'dof':<5}{'damping':>12}{'frictionloss':>15}{'armature':>12}")
    for i in range(model.nv):
        print(
            f"{i:<5}{model.dof_damping[i]:>12.4f}"
            f"{model.dof_frictionloss[i]:>15.4f}"
            f"{model.dof_armature[i]:>12.4f}"
        )

    print(f"\ntotal frictionloss: {np.sum(model.dof_frictionloss):.3f}")
    print(f"total damping:      {np.sum(model.dof_damping):.3f}")


def main():
    """
    Entry point.

    input:  none
    output: None
    """
    model = mujoco.MjModel.from_xml_path(MODEL_PATH)
    report_passive_params(model)


if __name__ == "__main__":
    main()
