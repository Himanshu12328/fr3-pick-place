"""
Sweeps training checkpoints under the strict gates.

Every checkpoint decision in this project so far was made on
`check_success`. That criterion cannot see a shove, a drop, a missing
retreat or a rushed trajectory, so the checkpoint it selects is the one best
at putting the block near the target by any means, which is not the same as
the one best at performing a pick and place. Whether those two are the same
checkpoint is an open question, and it decides how every later run gets
selected.

Two protocol rules this project already learned the hard way, both enforced
here:

  Sweep on a screening seed and report on different ones. The 86.8% result
  was contaminated because seed 51 drove the checkpoint choice and was then
  included in the headline. Screening seeds here are 61 to 63 and reporting
  seeds are 71 to 76, with no overlap.

  Never select on a single checkpoint. Adjacent checkpoints 1,250 steps
  apart have differed by 27 points on this task while the loss fell
  smoothly and monotonically the whole way.

Run:
    python -m src.scripts.strict_sweep --run act_oracle_v1 --trials 40 --seed 61
"""

import argparse
import json
import re

from src.config import LOG_DIR, OUTPUT_ROOT
from src.eval.strict import (
    evaluate_strict,
    use_checkpoint_render_size,
)


def checkpoint_steps(run_dir):
    """
    Lists the numeric checkpoints in a run, in training order.

    input:  run_dir (Path)
    output: list of str
    """
    ckpts = run_dir / "checkpoints"
    if not ckpts.exists():
        raise SystemExit(f"no checkpoints under {ckpts}")
    names = [p.name for p in ckpts.iterdir() if re.fullmatch(r"\d+", p.name)]
    return sorted(names, key=int)


def main():
    """
    Entry point.

    input:  none
    output: None
    """
    p = argparse.ArgumentParser()
    p.add_argument("--run", default="act_oracle_v1")
    p.add_argument("--policy-type", default="act")
    p.add_argument("--steps", nargs="+", default=None,
                   help="checkpoint names, default every one in the run")
    p.add_argument("--trials", type=int, default=40)
    p.add_argument("--seed", type=int, default=61,
                   help="screening seed, must not be a reporting seed")
    p.add_argument("--device", default="cuda")
    p.add_argument("--task", default=None)
    p.add_argument("--tag", default=None)
    args = p.parse_args()

    run_dir = OUTPUT_ROOT / args.run
    steps = args.steps or checkpoint_steps(run_dir)
    tag = args.tag or f"strict_sweep_{args.run}"

    print(f"{args.run}: {len(steps)} checkpoints, {args.trials} trials, "
          f"screening seed {args.seed}\n")

    from src.eval.policy_wrapper import LeRobotPolicyAdapter, load_policy_and_processors

    table = []
    for name in steps:
        ckpt = run_dir / "checkpoints" / name / "pretrained_model"
        if not ckpt.exists():
            print(f"  {name}: missing, skipped")
            continue

        use_checkpoint_render_size(ckpt)
        policy, pre, post = load_policy_and_processors(
            str(ckpt), policy_type=args.policy_type, device=args.device
        )
        adapter = LeRobotPolicyAdapter(policy, pre, post,
                                       device=args.device, task=args.task)

        summary, rows = evaluate_strict(
            adapter, n_trials=args.trials, seed=args.seed,
            need_images=True, verbose=False,
        )
        summary["step"] = int(name)
        table.append(summary)

        print(f"  {name:>7}  strict {summary['strict'] * 100:5.1f}%   "
              f"loose {summary['loose'] * 100:5.1f}%   "
              f"steps {summary['mean_steps']:5.0f}   "
              f"traj {summary['mean_trajectory_score']:.3f}")

        del policy, adapter
        import torch
        torch.cuda.empty_cache()

    if not table:
        raise SystemExit("nothing evaluated")

    print(f"\n{'step':>8}{'strict':>9}{'loose':>9}{'gap':>7}   worst gates")
    for r in table:
        worst = sorted(r["gates"].items(), key=lambda kv: kv[1])[:3]
        w = ", ".join(f"{k} {v * 100:.0f}%" for k, v in worst)
        print(f"{r['step']:>8}{r['strict'] * 100:>8.1f}%{r['loose'] * 100:>8.1f}%"
              f"{(r['loose'] - r['strict']) * 100:>6.1f}   {w}")

    best_strict = max(table, key=lambda r: r["strict"])
    best_loose = max(table, key=lambda r: r["loose"])
    print(f"\n  best under STRICT: step {best_strict['step']} "
          f"at {best_strict['strict'] * 100:.1f}%")
    print(f"  best under loose:  step {best_loose['step']} "
          f"at {best_loose['loose'] * 100:.1f}%")
    if best_strict["step"] != best_loose["step"]:
        print("  THESE DIFFER. Selecting on the loose criterion picks a "
              "different checkpoint than selecting on the task.")
    else:
        print("  Same checkpoint. The loose criterion happened to agree here.")

    LOG_DIR.mkdir(parents=True, exist_ok=True)
    out = LOG_DIR / f"{tag}.json"
    out.write_text(json.dumps({
        "run": args.run, "trials": args.trials, "screening_seed": args.seed,
        "table": table,
    }, indent=2, default=float), encoding="utf-8")
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
