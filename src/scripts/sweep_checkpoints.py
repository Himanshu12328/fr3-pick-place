"""
Evaluates every checkpoint in a training run and reports success rate
against training step.

Denoising loss decreases monotonically while task success does not, so the
only way to find the best checkpoint is to evaluate them. Every trial uses
the same seed, so all checkpoints face the identical sequence of block
placements and the comparison between them is not confounded by luck.

Run:
    python src\\scripts\\sweep_checkpoints.py
    python src\\scripts\\sweep_checkpoints.py --run diffusion_v2 --trials 20
"""

import argparse
import json
import os
import time

import matplotlib.pyplot as plt
import numpy as np

from src.config import LOG_DIR, OUTPUT_ROOT
from src.eval.policy_wrapper import LeRobotPolicyAdapter, load_policy_and_processors
from src.eval.rollout import evaluate

OUTPUT_ROOT = OUTPUT_ROOT
LOG_DIR = LOG_DIR
TASK = "pick up the red block and place it on the green target"


def list_checkpoints(run_dir):
    """
    Returns every checkpoint step directory in a run, sorted numerically.

    The 'last' entry is skipped because it is a symlink to one of the
    numbered checkpoints and would otherwise be evaluated twice.

    input:  run_dir (str) path to the run's output directory
    output: list of str, the checkpoint directory names
    """
    ckpt_root = os.path.join(run_dir, "checkpoints")
    names = [
        d
        for d in os.listdir(ckpt_root)
        if d.isdigit() and os.path.isdir(os.path.join(ckpt_root, d))
    ]
    return sorted(names, key=int)


def evaluate_checkpoint(run_dir, name, policy_type, trials, seed):
    """
    Loads one checkpoint and runs the rollout harness against it.

    input:  run_dir (str), name (str) checkpoint directory name,
            policy_type (str), trials (int), seed (int)
    output: dict with step, success_rate, and per-trial results
    """
    ckpt = os.path.join(run_dir, "checkpoints", name, "pretrained_model")
    policy, pre, post = load_policy_and_processors(ckpt, policy_type)
    adapter = LeRobotPolicyAdapter(policy, pre, post, task=TASK)

    result = evaluate(
        adapter, n_trials=trials, seed=seed, need_images=True, verbose=False
    )

    return {
        "step": int(name),
        "success_rate": result["success_rate"],
        "results": result["results"],
    }


def summarise_by_region(results, split_x=0.53):
    """
    Splits the success rate by where the block started along x.

    A policy trained on an uneven distribution of object positions often
    works well in the well-represented region and fails elsewhere, and an
    aggregate success rate hides that entirely. Splitting makes it visible.

    input:  results (list of trial dicts), split_x (float) metres
    output: (near_rate, far_rate, near_n, far_n)
    """
    near = [r for r in results if r["block_start"][0] < split_x]
    far = [r for r in results if r["block_start"][0] >= split_x]

    near_rate = np.mean([r["success"] for r in near]) if near else float("nan")
    far_rate = np.mean([r["success"] for r in far]) if far else float("nan")

    return near_rate, far_rate, len(near), len(far)


def plot_curve(records, path):
    """
    Plots success rate against training step, split by block region.

    input:  records (list of checkpoint dicts), path (str)
    output: None
    """
    steps = [r["step"] for r in records]
    overall = [r["success_rate"] * 100 for r in records]
    near = [r["near_rate"] * 100 for r in records]
    far = [r["far_rate"] * 100 for r in records]

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(steps, overall, "o-", lw=2, label="overall")
    ax.plot(steps, near, "s--", alpha=0.7, label="block x < 0.53")
    ax.plot(steps, far, "^--", alpha=0.7, label="block x >= 0.53")

    ax.set_xlabel("training step")
    ax.set_ylabel("success rate (%)")
    ax.set_title("Task success vs training duration")
    ax.set_ylim(-5, 105)
    ax.grid(alpha=0.3)
    ax.legend()

    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"saved: {path}")


def main():
    """
    Entry point.

    input:  none
    output: None
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", default="diffusion_v2")
    parser.add_argument("--type", default="diffusion")
    parser.add_argument("--trials", type=int, default=20)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--split-x", type=float, default=0.53)
    args = parser.parse_args()

    run_dir = os.path.join(OUTPUT_ROOT, args.run)
    names = list_checkpoints(run_dir)
    print(f"{len(names)} checkpoints in {args.run}: {', '.join(names)}")
    print(f"{args.trials} trials each, seed {args.seed}\n")

    records = []
    start = time.perf_counter()

    for name in names:
        r = evaluate_checkpoint(run_dir, name, args.type, args.trials, args.seed)
        near, far, n_near, n_far = summarise_by_region(r["results"], args.split_x)
        r.update({"near_rate": near, "far_rate": far})
        records.append(r)

        print(
            f"  step {r['step']:>7}: {r['success_rate'] * 100:5.1f}%   "
            f"near {near * 100:5.1f}% (n={n_near})   "
            f"far {far * 100:5.1f}% (n={n_far})"
        )

    best = max(records, key=lambda r: r["success_rate"])
    elapsed = time.perf_counter() - start

    print(f"\nbest: step {best['step']} at {best['success_rate'] * 100:.1f}%")
    print(f"elapsed: {elapsed / 60:.1f} min")

    out = {
        "run": args.run,
        "trials": args.trials,
        "seed": args.seed,
        "checkpoints": [
            {k: v for k, v in r.items() if k != "results"} for r in records
        ],
    }
    json_path = os.path.join(LOG_DIR, f"sweep_{args.run}.json")
    with open(json_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"saved: {json_path}")

    plot_curve(records, os.path.join(LOG_DIR, f"sweep_{args.run}.png"))


if __name__ == "__main__":
    main()
