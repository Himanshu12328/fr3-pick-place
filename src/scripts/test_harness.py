"""
Validates the evaluation harness itself by replaying recorded actions.

Replaying a successful demonstration's action sequence should reproduce
that demonstration and report success. If it does not, the harness and the
collection environment have diverged somewhere, and every success rate
measured with the harness would be wrong in the same invisible way. Better
to find that now than to spend a week concluding a policy does not work.

Note that replay will not be perfect. The block is randomised to a fresh
pose each trial, whereas the recorded actions were driven against one
specific placement, so replaying against a matching placement is the only
fair test.

Run:
    python src\\scripts\\test_harness.py
"""

import json
import os
import sys

import mujoco
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))
from src.controllers.impedance import ImpedanceController
from src.data.task import check_success, set_block_pose
from src.eval.rollout import (
    BLOCK_QPOS_ADR,
    BLOCK_QVEL_ADR,
    FINGER_DOFS,
    KP_ROT,
    KP_TRANS,
    N_ARM,
    STEPS_PER_ACTION,
    WORKSPACE_MAX,
    WORKSPACE_MIN,
    ZETA,
    gripper_torque,
    run_trial,
    setup_model,
)

RAW_DIR = r"C:\pick_place\data\pick_place_v1"


class ReplayPolicy:
    """
    Emits a recorded action sequence one step at a time, holding the last
    action once exhausted.
    """

    def __init__(self, actions):
        """
        input:  actions (array (N, 8)) recorded action sequence
        output: ReplayPolicy instance
        """
        self.actions = actions
        self.i = 0

    def reset(self):
        """
        Rewinds to the start of the sequence.

        input:  none
        output: None
        """
        self.i = 0

    def __call__(self, obs):
        """
        Returns the next recorded action, ignoring the observation.

        input:  obs (dict) unused
        output: numpy array of shape (8,)
        """
        action = self.actions[min(self.i, len(self.actions) - 1)]
        self.i += 1
        return action


def replay_episode(ep_name, verbose=True):
    """
    Replays one recorded episode against its original block placement.

    input:  ep_name (str), verbose (bool)
    output: dict with success, steps, distance
    """
    ep_dir = os.path.join(RAW_DIR, ep_name)
    actions = np.load(os.path.join(ep_dir, "data.npz"))["action"]
    with open(os.path.join(ep_dir, "meta.json")) as f:
        meta = json.load(f)

    model, data = setup_model()
    ctrl = ImpedanceController(model, data,
                               kp_trans=KP_TRANS, kp_rot=KP_ROT, zeta=ZETA,
                               kp_null=10.0, zeta_null=1.0,
                               n_arm=N_ARM, verbose=False)

    kid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "home")
    mujoco.mj_resetDataKeyframe(model, data, kid)

    set_block_pose(model, data,
                   np.array(meta["block_start_pos"]),
                   np.array(meta["block_start_quat"]),
                   BLOCK_QPOS_ADR, BLOCK_QVEL_ADR)
    mujoco.mj_forward(model, data)

    pos, quat = ctrl.current_pose(data)
    ctrl.set_target(pos, quat)

    block_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "block")

    for action in actions:
        target_pos = np.clip(action[:3], WORKSPACE_MIN, WORKSPACE_MAX)
        target_quat = action[3:7] / max(np.linalg.norm(action[3:7]), 1e-9)
        grip = float(np.clip(action[7], 0.0, 0.04))

        ctrl.set_target(target_pos, target_quat)
        for _ in range(STEPS_PER_ACTION):
            data.qfrc_applied[:N_ARM] = ctrl.compute_torque(data)
            data.qfrc_applied[FINGER_DOFS] = gripper_torque(data, grip)
            mujoco.mj_step(model, data)

    success, dist = check_success(data, block_id)

    if verbose:
        mark = "ok  " if success else "FAIL"
        print(f"  {ep_name}  {mark}  {len(actions):4d} steps  "
              f"dist {dist*100:5.1f} cm  (recorded: {meta['success']})")

    return {"success": success, "steps": len(actions), "distance": dist}


def main():
    """
    Replays a sample of recorded episodes and reports agreement with their
    recorded labels.

    input:  none
    output: None
    """
    names = sorted(d for d in os.listdir(RAW_DIR) if d.startswith("episode_"))
    rng = np.random.default_rng(0)
    sample = rng.choice(names, size=min(10, len(names)), replace=False)

    print(f"Replaying {len(sample)} episodes through the eval harness:\n")

    results = [replay_episode(n) for n in sorted(sample)]
    n_success = sum(r["success"] for r in results)

    print(f"\n{n_success}/{len(results)} replays succeeded")

    if n_success == len(results):
        print("Harness reproduces recorded demonstrations. Ready for policy eval.")
    else:
        print("Replay diverges from collection. Check that build_state, the")
        print("gain constants, and the block address constants match collect.py")
        print("before trusting any policy success rate from this harness.")


if __name__ == "__main__":
    main()