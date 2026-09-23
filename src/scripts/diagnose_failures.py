"""
Replays specific failed trials and records what the fingers and the block
were actually doing.

The strict evaluation says which gate a trial failed. It does not say why.
"The block never rose past 20 mm" is consistent with several different
mechanisms, and they call for different fixes:

  closed on empty air        the policy mislocated the block. A perception
                             or precision problem
  closed on the block and
  it slipped out             the grasp pose was right but the approach or
                             the closing geometry was wrong
  knocked the block away
  before closing             the approach path collided with it
  never attempted a close    the policy stalled or timed out upstream

Telling them apart needs the measured finger width, the tool-to-block
distance at the moment the fingers close, and whether the block moved before
that moment. None of those are in the normal trace, so this script records
them.

Block placements come from the same generator in the same order, so drawing
the sequence forward reproduces any trial index exactly without running the
ones in between.

Run:
    python -m src.scripts.diagnose_failures --ensemble A B --seed 92 --trials 11 21 96
"""

import argparse
import json

import mujoco
import numpy as np

from src.config import BLOCK_QPOS_ADR, BLOCK_QVEL_ADR, LOG_DIR
from src.data.task import BLOCK_Z, sample_block_pose, set_block_pose
from src.eval import rollout as R
from src.eval.strict import evaluate_trace, use_checkpoint_render_size

# How wide the fingers sit when they have closed on the 44 mm block rather
# than on nothing. Each finger travels 0 to 40 mm, so a pair gripping the
# block settles near 22 mm each and a pair closed on air goes to nearly 0.
FINGER_ON_BLOCK_M = 0.012


def build_policy(paths, device="cuda"):
    """
    Loads one policy or an ensemble of several.

    input:  paths (list of str), device (str)
    output: callable
    """
    from src.eval.policy_wrapper import LeRobotPolicyAdapter, load_policy_and_processors

    members = []
    for p in paths:
        use_checkpoint_render_size(p)
        pol, pre, post = load_policy_and_processors(p, policy_type="act", device=device)
        members.append(LeRobotPolicyAdapter(pol, pre, post, device=device))
    if len(members) == 1:
        return members[0]

    from src.eval.ensemble import EnsemblePolicy

    return EnsemblePolicy(members)


def replay(model, data, ctrl, policy, renderer, block_pos, block_quat,
           max_steps=600, frames=None):
    """
    Runs one trial from a given block placement, recording finger state.

    input:  model, data, ctrl, policy, renderer,
            block_pos (3,), block_quat (4,), max_steps (int),
            frames (list or None) to collect video
    output: dict of traces
    """
    kid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "home")
    mujoco.mj_resetDataKeyframe(model, data, kid)
    set_block_pose(model, data, block_pos, block_quat, BLOCK_QPOS_ADR, BLOCK_QVEL_ADR)
    mujoco.mj_forward(model, data)
    pos, quat = ctrl.current_pose(data)
    ctrl.set_target(pos, quat)

    block_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "block")
    if hasattr(policy, "reset"):
        policy.reset()

    t = {k: [] for k in ("block", "ee", "target", "grip", "block_vz",
                         "fingers", "tcp_block_xy", "tcp_block_z", "yaw_err")}
    tcp_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "tcp")
    grip = 0.04

    for _ in range(max_steps):
        obs = R.build_observation(data, renderer, R.CAMERAS, True)
        a = np.asarray(policy(obs), dtype=np.float64)
        target = np.clip(a[:3], R.WORKSPACE_MIN, R.WORKSPACE_MAX)
        quat = a[3:7] / max(np.linalg.norm(a[3:7]), 1e-9)
        grip = float(np.clip(a[7], 0.0, 0.04))
        ctrl.set_target(target, quat)
        for _ in range(R.STEPS_PER_ACTION):
            data.qfrc_applied[: R.N_ARM] = ctrl.compute_torque(data)
            data.qfrc_applied[R.FINGER_DOFS] = R.gripper_torque(data, grip)
            mujoco.mj_step(model, data)

        ee = ctrl.current_pose(data)[0]
        blk = np.array(data.xpos[block_id], dtype=np.float64)
        t["block"].append(blk.astype(np.float32))
        t["ee"].append(ee.astype(np.float32))
        t["target"].append(target.astype(np.float32))
        t["grip"].append(grip)
        t["block_vz"].append(float(data.qvel[BLOCK_QVEL_ADR + 2]))
        t["fingers"].append(float(np.mean(data.qpos[R.FINGER_DOFS])))
        t["tcp_block_xy"].append(float(np.linalg.norm(ee[:2] - blk[:2])))
        t["tcp_block_z"].append(float(ee[2] - blk[2]))

        # A cube is symmetric every 90 degrees, so the fingers only have to
        # line up with a pair of faces. Fold both yaws into a quarter turn
        # and take the smallest remaining angle: that is the misalignment
        # the grasp actually sees.
        bq = np.array(data.xquat[block_id])
        byaw = 2.0 * np.arctan2(bq[3], bq[0])
        q = np.zeros(4)
        mujoco.mju_mat2Quat(q, np.array(data.site_xmat[tcp_id]).ravel())
        gyaw = 2.0 * np.arctan2(q[2], q[1])
        err = (gyaw - byaw) % (np.pi / 2.0)
        t["yaw_err"].append(float(np.degrees(min(err, np.pi / 2.0 - err))))

        if frames is not None and renderer is not None:
            renderer.update_scene(data, camera="external")
            frames.append(renderer.render())

    return {k: np.asarray(v) for k, v in t.items()}


def diagnose(t):
    """
    Works out which mechanism produced the failure.

    input:  t (dict of arrays) from replay
    output: dict
    """
    grip = t["grip"]
    fingers = t["fingers"]
    block = t["block"]
    closes = np.where(np.diff((grip < 0.02).astype(int)) == 1)[0] + 1

    out = {
        "close_attempts": int(len(closes)),
        "max_lift_mm": float((block[:, 2].max() - BLOCK_Z) * 1000),
        "block_moved_before_first_close_mm": None,
        "attempts": [],
    }

    start_xy = block[0, :2]
    if len(closes):
        out["block_moved_before_first_close_mm"] = float(
            np.linalg.norm(block[closes[0], :2] - start_xy) * 1000)

    for c in closes[:6]:
        # Fingers need time to travel; look at where they settle rather than
        # at the instant the command changed.
        settle = min(c + 25, len(fingers) - 1)
        out["attempts"].append({
            "step": int(c),
            "tcp_block_xy_mm": float(t["tcp_block_xy"][c] * 1000),
            "tcp_above_block_mm": float(t["tcp_block_z"][c] * 1000),
            "finger_width_after_close_mm": float(fingers[settle] * 1000),
            "gripped_something": bool(fingers[settle] > FINGER_ON_BLOCK_M),
            "block_shifted_mm": float(
                np.linalg.norm(block[settle, :2] - block[c, :2]) * 1000),
            "yaw_err_deg": float(t["yaw_err"][c]),
        })
    return out


def main():
    """
    Entry point.

    input:  none
    output: None
    """
    p = argparse.ArgumentParser()
    p.add_argument("--ensemble", nargs="+", required=True)
    p.add_argument("--seed", type=int, default=92)
    p.add_argument("--trials", type=int, nargs="+", required=True)
    p.add_argument("--total", type=int, default=100)
    p.add_argument("--video-dir", default=None)
    p.add_argument("--device", default="cuda")
    args = p.parse_args()

    # Draw the whole placement sequence so any index can be reproduced.
    rng = np.random.default_rng(args.seed)
    placements = [sample_block_pose(rng) for _ in range(args.total)]

    policy = build_policy(args.ensemble, args.device)
    model, data = R.setup_model()
    ctrl = R.ImpedanceController(
        model, data, kp_trans=R.KP_TRANS, kp_rot=R.KP_ROT, zeta=R.ZETA,
        kp_null=10.0, zeta_null=1.0, n_arm=R.N_ARM, verbose=False)
    if hasattr(policy, "bind"):
        policy.bind(model, data)
    renderer = mujoco.Renderer(model, height=R.CAM_HEIGHT, width=R.CAM_WIDTH)

    report = []
    for idx in args.trials:
        pos, quat = placements[idx]
        frames = [] if args.video_dir else None
        t = replay(model, data, ctrl, policy, renderer, pos, quat, frames=frames)
        d = diagnose(t)
        d["trial"] = idx
        d["block_start"] = [float(v) for v in pos]

        row = evaluate_trace(t, len(t["grip"]), False, float(
            np.linalg.norm(t["block"][-1, :2] - np.array([0.55, 0.20]))))
        d["failed_gates"] = [g for g, ok in row["gates"].items() if not ok]
        report.append(d)

        print(f"\n=== trial {idx}  block start "
              f"x={pos[0]:.3f} y={pos[1]:.3f} ===")
        print(f"  max lift            {d['max_lift_mm']:.1f} mm")
        print(f"  gripper close attempts {d['close_attempts']}")
        if d["block_moved_before_first_close_mm"] is not None:
            print(f"  block moved before first close "
                  f"{d['block_moved_before_first_close_mm']:.1f} mm")
        for a in d["attempts"]:
            print(f"   step {a['step']:3d}: tool {a['tcp_block_xy_mm']:5.1f} mm "
                  f"lateral, {a['tcp_above_block_mm']:+6.1f} mm vertical | "
                  f"fingers settle {a['finger_width_after_close_mm']:5.1f} mm "
                  f"-> {'HOLDING' if a['gripped_something'] else 'closed on air'}"
                  f" | yaw off {a['yaw_err_deg']:4.1f} deg"
                  f" | block shifted {a['block_shifted_mm']:.1f} mm")

        if args.video_dir and frames:
            import os

            import imageio
            os.makedirs(args.video_dir, exist_ok=True)
            out = os.path.join(args.video_dir, f"failure_trial_{idx:03d}.mp4")
            imageio.mimsave(out, frames, fps=30)
            print(f"  wrote {out}")

    renderer.close()
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    (LOG_DIR / f"diagnose_seed{args.seed}.json").write_text(
        json.dumps(report, indent=2, default=float), encoding="utf-8")
    print(f"\nwrote {LOG_DIR / f'diagnose_seed{args.seed}.json'}")


if __name__ == "__main__":
    main()
