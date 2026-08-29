"""
Reports success rate split by where the block started, for one checkpoint.

An aggregate success rate hides blind spots. If a policy fails
systematically in one part of the workspace, that is actionable — it says
where to collect more demonstrations — while a single number does not.

Run:
    python -m src.scripts.region_analysis --run act_v2 --checkpoint 005000 --type act --trials 100
"""

import argparse

import numpy as np

from src.config import OUTPUT_ROOT, TASK_STRING
from src.eval.policy_wrapper import LeRobotPolicyAdapter, load_policy_and_processors
from src.eval.rollout import evaluate


def report_regions(results, split_x=0.53, split_y=-0.04):
    """
    Prints success rate for each half of the workspace along both axes.

    input:  results (list of trial dicts), split_x (float), split_y (float)
    output: None
    """
    regions = {
        f"x <  {split_x}": [r for r in results if r["block_start"][0] < split_x],
        f"x >= {split_x}": [r for r in results if r["block_start"][0] >= split_x],
        f"y <  {split_y}": [r for r in results if r["block_start"][1] < split_y],
        f"y >= {split_y}": [r for r in results if r["block_start"][1] >= split_y],
    }

    print(f"\n{'region':<14}{'success':>10}{'n':>6}{'se':>8}")
    for name, rs in regions.items():
        if not rs:
            continue
        rate = float(np.mean([r["success"] for r in rs]))
        se = np.sqrt(rate * (1 - rate) / len(rs))
        print(f"{name:<14}{rate * 100:>9.1f}%{len(rs):>6}{se * 100:>7.1f}%")


def main():
    """
    Entry point.

    input:  none
    output: None
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", default="act_v2")
    parser.add_argument("--checkpoint", default="005000")
    parser.add_argument("--type", default="act")
    parser.add_argument("--trials", type=int, default=100)
    parser.add_argument("--seed", type=int, default=10)
    args = parser.parse_args()

    ckpt = OUTPUT_ROOT / args.run / "checkpoints" / args.checkpoint / "pretrained_model"
    policy, pre, post = load_policy_and_processors(str(ckpt), args.type)
    adapter = LeRobotPolicyAdapter(policy, pre, post, task=TASK_STRING)

    result = evaluate(adapter, n_trials=args.trials, seed=args.seed,
                      need_images=True, verbose=False)

    print(f"\n{args.run} @ {args.checkpoint}: "
          f"{result['success_rate'] * 100:.1f}% over {args.trials} trials")
    report_regions(result["results"])


if __name__ == "__main__":
    main()