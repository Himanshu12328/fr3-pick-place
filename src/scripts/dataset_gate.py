"""
Checks a raw dataset against the recorded demonstrations before it is
converted or trained on.

This exists because of one specific failure. The first DAgger dataset had
the gripper commanded open in 96% of its frames against 55.7% in the
demonstrations, and 29.4% of its frames came from episodes where the
demonstrator itself failed. Nobody looked. It converted cleanly, trained for
eighty minutes, and produced a policy that scored 11%.

Every number below is one line of arithmetic that would have caught it. The
rule is that this runs *before* conversion, not after a disappointing
evaluation.

The reference column is measured from the 221 recorded teleoperation
episodes, not asserted. Run with no `--reference` and it recomputes them.

Run:
    python -m src.scripts.dataset_gate --data data/oracle_a data/oracle_b
    python -m src.scripts.dataset_gate --data data/dagger_v2 --strict
"""

import argparse
import json
from pathlib import Path

import numpy as np

from src.config import LOG_DIR
from src.rl import reference as REF

DEMO_DIR = "data/pick_place_v1"

# Tolerances on the comparison against the demonstrations. These are wide,
# because the point is to catch a dataset that is wrong by a factor, not to
# insist on a match. The DAgger dataset that failed was out by 0.40 on the
# gripper fraction and by 26 points on failed-episode frames.
TOL = {
    "gripper_open_fraction": 0.12,
    "mean_episode_len": 90.0,
    "failed_frame_share": 0.10,
    "min_trajectory_score": 0.10,
}


def episode_stats(ep):
    """
    Reduces one raw episode to the quantities the gate compares.

    input:  ep (Path) an episode directory
    output: dict, or None if the episode has no usable action trace
    """
    npz = np.load(ep / "data.npz")
    action = npz["action"]
    if len(action) < 5:
        return None

    meta = {}
    meta_path = ep / "meta.json"
    if meta_path.exists():
        meta = json.loads(meta_path.read_text())

    described = REF.describe(action[:, :3], action[:, 7])
    score, _ = REF.score(described) if described else (0.0, {})

    return {
        "frames": int(len(action)),
        "gripper_open_fraction": float(np.mean(action[:, 7] > 0.02)),
        "grasped": described is not None,
        "released": bool(described["released"]) if described else False,
        "trajectory_score": float(score),
        "succeeded": bool(meta.get("student_succeeded", meta.get("success", True))),
        "block_start_x": float(meta.get("block_start_pos", [np.nan])[0]),
        "has_block_pose": "block" in npz,
    }


def summarise(rows):
    """
    Aggregates per-episode statistics into the gate's comparison row.

    input:  rows (list of dict)
    output: dict
    """
    frames = np.array([r["frames"] for r in rows])
    failed = np.array([not r["succeeded"] for r in rows])
    scores = np.array([r["trajectory_score"] for r in rows])
    # Weighted by frames, because that is what the loss sees. A handful of
    # failed episodes that run to the full step cap contribute far more
    # frames than their episode count suggests, which is exactly how the
    # first DAgger dataset went wrong.
    gof = np.array([r["gripper_open_fraction"] for r in rows])

    return {
        "episodes": len(rows),
        "frames": int(frames.sum()),
        "mean_episode_len": float(frames.mean()),
        "sd_episode_len": float(frames.std()),
        "gripper_open_fraction": float(np.average(gof, weights=frames)),
        "failed_episode_share": float(failed.mean()),
        "failed_frame_share": float(frames[failed].sum() / frames.sum())
        if failed.any() else 0.0,
        "never_grasped": float(np.mean([not r["grasped"] for r in rows])),
        "never_released": float(np.mean([not r["released"] for r in rows])),
        "mean_trajectory_score": float(scores.mean()),
        "min_trajectory_score": float(scores.min()),
        "p05_trajectory_score": float(np.percentile(scores, 5)),
        "with_block_pose": float(np.mean([r["has_block_pose"] for r in rows])),
    }


def load(dirs):
    """
    Reads every episode under the given directories.

    input:  dirs (list of str)
    output: list of dict
    """
    rows = []
    for d in dirs:
        for ep in sorted(Path(d).iterdir()):
            if not ep.name.startswith("episode_"):
                continue
            s = episode_stats(ep)
            if s:
                rows.append(s)
    return rows


def main():
    """
    Entry point.

    input:  none
    output: None
    """
    p = argparse.ArgumentParser()
    p.add_argument("--data", nargs="+", required=True)
    p.add_argument("--reference", default=DEMO_DIR)
    p.add_argument("--strict", action="store_true",
                   help="exit non-zero if any check fails")
    p.add_argument("--tag", default="dataset_gate")
    args = p.parse_args()

    print(f"reference: {args.reference}")
    ref = summarise(load([args.reference]))
    print(f"dataset:   {args.data}")
    got = summarise(load(args.data))

    print(f"\n{'':<26}{'dataset':>12}{'demos':>12}{'delta':>10}")
    fails = []
    for key in [
        "episodes", "frames", "mean_episode_len", "sd_episode_len",
        "gripper_open_fraction", "failed_episode_share", "failed_frame_share",
        "never_grasped", "never_released", "mean_trajectory_score",
        "min_trajectory_score", "p05_trajectory_score", "with_block_pose",
    ]:
        a, b = got[key], ref[key]
        delta = a - b
        flag = ""
        if key in TOL and abs(delta) > TOL[key]:
            flag = "  <-- OUT OF TOLERANCE"
            fails.append(key)
        print(f"{key:<26}{a:>12.3f}{b:>12.3f}{delta:>10.3f}{flag}")

    print("\nthe two that have actually broken a dataset:")
    print(f"  gripper open fraction   {got['gripper_open_fraction']:.3f}   "
          f"demos {ref['gripper_open_fraction']:.3f}   "
          f"the failed DAgger set was 0.960")
    print(f"  frames from failures    {got['failed_frame_share']:.3f}   "
          f"demos {ref['failed_frame_share']:.3f}   "
          f"the failed DAgger set was 0.294")

    LOG_DIR.mkdir(parents=True, exist_ok=True)
    out = LOG_DIR / f"{args.tag}.json"
    out.write_text(json.dumps({"dataset": got, "reference": ref,
                               "data": args.data, "failed_checks": fails},
                              indent=2), encoding="utf-8")
    print(f"\nwrote {out}")

    if fails:
        print(f"\nFAILED: {', '.join(fails)}")
        if args.strict:
            raise SystemExit(1)
    else:
        print("\nall checks inside tolerance")


if __name__ == "__main__":
    main()
