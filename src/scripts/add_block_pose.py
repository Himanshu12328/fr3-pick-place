"""
Recovers the true block pose for every frame of already-collected episodes,
by replaying their recorded actions through the simulator.

Why this is needed. The perception probe established that the block position
is present in the 160x128 images to about 3 mm, which is well inside the
grasp tolerance. So the images are not the problem; extracting the block
position from them is. The intended fix is auxiliary supervision: make the
policy predict where the block is alongside what to do, so the
representation is forced to carry it rather than merely permitted to.

That needs a per-frame block pose as a training target, and the 800 oracle
episodes on disk were collected before the collector recorded one.

Why replay rather than re-collect. These episodes were collected with the
oracle both driving and labelling, so the recorded action is exactly the
action that was executed. Replaying it through the same scene, the same
impedance controller and the same decimation reproduces the same physics.
Nothing is rendered, so this is cheap: no images are touched and none are
rewritten.

The replay is verified rather than assumed. Every episode is checked against
its own recorded joint trajectory, and any episode that drifts is reported
and skipped rather than silently given wrong labels. `rerender.py` already
relies on the same property and measured joint drift under 0.005 rad across
a 400-step episode.

Run:
    python -m src.scripts.add_block_pose --data data/oracle_a data/oracle_b
    python -m src.scripts.add_block_pose --data data/oracle_a --check-only
"""

import argparse
import json
import time
from pathlib import Path

import mujoco
import numpy as np

from src.config import BLOCK_QPOS_ADR, BLOCK_QVEL_ADR
from src.data.task import set_block_pose
from src.eval import rollout as R

# How far the replayed arm may drift from the recorded one before the
# episode is rejected. The recorded state is joint angles in radians, and
# rerender.py measured drift under 0.005 across a 400-step episode, so this
# is an order of magnitude of headroom rather than a guess.
MAX_JOINT_DRIFT_RAD = 0.05


def block_pose_row(data, block_id):
    """
    Returns the block's pose as (x, y, z, yaw).

    Yaw is folded into a quarter turn, because the block is a cube and every
    90 degree rotation presents an identical pair of faces to the fingers.
    Asking a model to predict raw yaw is asking it to distinguish
    orientations that are identical in the image and irrelevant to the
    grasp.

    input:  data (MjData), block_id (int)
    output: numpy array of shape (4,) float32
    """
    pos = np.array(data.xpos[block_id], dtype=np.float64)
    quat = np.array(data.xquat[block_id], dtype=np.float64)
    yaw = 2.0 * np.arctan2(quat[3], quat[0])
    yaw = (yaw + np.pi / 4.0) % (np.pi / 2.0) - np.pi / 4.0
    return np.array([pos[0], pos[1], pos[2], yaw], dtype=np.float32)


def replay(model, data, ctrl, meta, actions):
    """
    Replays one episode's recorded actions and records the block each step.

    input:  model, data, ctrl, meta (dict), actions (array (T,8))
    output: (blocks array (T,4), states array (T,16))
    """
    kid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "home")
    if kid >= 0:
        mujoco.mj_resetDataKeyframe(model, data, kid)
    else:
        mujoco.mj_resetData(model, data)

    set_block_pose(
        model, data,
        np.asarray(meta["block_start_pos"], dtype=np.float64),
        np.asarray(meta["block_start_quat"], dtype=np.float64),
        BLOCK_QPOS_ADR, BLOCK_QVEL_ADR,
    )
    mujoco.mj_forward(model, data)

    pos, quat = ctrl.current_pose(data)
    ctrl.set_target(pos, quat)

    block_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "block")
    blocks, states = [], []

    for a in actions:
        # The block pose is recorded before stepping, so row i is the state
        # the observation at row i was rendered from. Recording it after the
        # step would offset every label by one frame, which on a 30 Hz
        # carry is 3 mm.
        blocks.append(block_pose_row(data, block_id))
        states.append(R.build_state(data))

        target = np.clip(a[:3].astype(np.float64), R.WORKSPACE_MIN, R.WORKSPACE_MAX)
        quat = a[3:7].astype(np.float64)
        quat = quat / max(np.linalg.norm(quat), 1e-9)
        grip = float(np.clip(a[7], 0.0, 0.04))

        ctrl.set_target(target, quat)
        for _ in range(R.STEPS_PER_ACTION):
            data.qfrc_applied[: R.N_ARM] = ctrl.compute_torque(data)
            data.qfrc_applied[R.FINGER_DOFS] = R.gripper_torque(data, grip)
            mujoco.mj_step(model, data)

    return np.asarray(blocks), np.asarray(states)


def main():
    """
    Entry point.

    input:  none
    output: None
    """
    p = argparse.ArgumentParser()
    p.add_argument("--data", nargs="+", required=True)
    p.add_argument("--check-only", action="store_true",
                   help="verify the replay without writing anything")
    p.add_argument("--limit", type=int, default=None)
    args = p.parse_args()

    model, data = R.setup_model()
    ctrl = R.ImpedanceController(
        model, data, kp_trans=R.KP_TRANS, kp_rot=R.KP_ROT, zeta=R.ZETA,
        kp_null=10.0, zeta_null=1.0, n_arm=R.N_ARM, verbose=False,
    )

    total = written = rejected = 0
    drifts = []
    start = time.perf_counter()

    for root in args.data:
        eps = sorted(p for p in Path(root).iterdir()
                     if p.name.startswith("episode_"))
        if args.limit:
            eps = eps[: args.limit]

        for ep in eps:
            npz = dict(np.load(ep / "data.npz"))
            meta = json.loads((ep / "meta.json").read_text())
            total += 1

            blocks, states = replay(model, data, ctrl, meta, npz["action"])

            # The check that makes the labels trustworthy. If the replayed
            # arm did not follow the recorded one, the block positions
            # belong to a different episode than the images do.
            drift = float(np.abs(states[:, :7] - npz["state"][:, :7]).max())
            drifts.append(drift)

            if drift > MAX_JOINT_DRIFT_RAD:
                rejected += 1
                print(f"  {ep.name}: REJECTED, joint drift {drift:.4f} rad")
                continue

            if not args.check_only:
                npz["block"] = blocks
                np.savez(ep / "data.npz", **npz)
                written += 1

            if total % 100 == 0:
                print(f"  {total} episodes, max drift so far "
                      f"{max(drifts):.4f} rad, {time.perf_counter() - start:.0f}s")

    d = np.asarray(drifts)
    print(f"\n{total} episodes in {time.perf_counter() - start:.0f}s")
    print(f"  joint drift: median {np.median(d):.5f}  mean {d.mean():.5f}  "
          f"max {d.max():.5f} rad   (limit {MAX_JOINT_DRIFT_RAD})")
    print(f"  rejected: {rejected}")
    if args.check_only:
        print("  check only, nothing written")
    else:
        print(f"  wrote block poses into {written} episodes")


if __name__ == "__main__":
    main()
