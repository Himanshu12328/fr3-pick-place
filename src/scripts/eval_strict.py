"""
Scores a policy under the strict pick-and-place gates.

Every success rate in this project before this script was measured with
`check_success`, which is satisfied by a shove, a drop and a hover. This
one requires the episode to be the sequence the demonstrations perform:
approach, grasp, lift, carry, descend, release, lift off, home.

Run:
    python -m src.scripts.eval_strict --oracle --trials 20
    python -m src.scripts.eval_strict --checkpoint outputs/act_oracle_v1/checkpoints/020000/pretrained_model --trials 100 --seeds 51
"""

import argparse
import json

import numpy as np

from src.config import LOG_DIR
from src.eval.strict import GATES, evaluate_strict, format_report


def build_policy(args):
    """
    Constructs the policy under test and says whether it needs images.

    input:  args (Namespace)
    output: (policy callable, need_images bool, name str)
    """
    if args.oracle:
        from src.eval.oracle import ScriptedOracle

        return (
            ScriptedOracle(jitter=args.jitter, seed=args.seed),
            False,
            f"oracle(jitter={args.jitter})",
        )

    from src.eval.policy_wrapper import LeRobotPolicyAdapter, load_policy_and_processors
    from src.eval.strict import use_checkpoint_render_size

    # Render at whatever the checkpoint was trained at, not at whatever
    # config.py currently says. See strict.render_size_for_checkpoint.
    use_checkpoint_render_size(args.checkpoint)

    policy, pre, post = load_policy_and_processors(
        args.checkpoint,
        policy_type=args.policy_type,
        device=args.device,
        n_action_steps=args.n_action_steps,
        temporal_ensemble_coeff=args.temporal_ensemble_coeff,
    )
    return (
        LeRobotPolicyAdapter(policy, pre, post, device=args.device, task=args.task),
        True,
        args.checkpoint,
    )


def main():
    """
    Entry point.

    input:  none
    output: None
    """
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint")
    p.add_argument("--policy-type", default="act")
    p.add_argument("--oracle", action="store_true")
    p.add_argument("--jitter", type=float, default=0.0)
    p.add_argument("--trials", type=int, default=100)
    p.add_argument("--seeds", type=int, nargs="+", default=[51])
    p.add_argument("--seed", type=int, default=0, help="oracle jitter seed")
    p.add_argument("--device", default="cuda")
    p.add_argument("--task", default=None)
    p.add_argument("--n-action-steps", type=int, default=None)
    p.add_argument("--temporal-ensemble-coeff", type=float, default=None)
    p.add_argument("--video-dir", default=None)
    p.add_argument("--video-n", type=int, default=0)
    p.add_argument("--quiet", action="store_true")
    p.add_argument("--tag", default="strict")
    args = p.parse_args()

    if not args.oracle and not args.checkpoint:
        raise SystemExit("give --checkpoint or --oracle")

    policy, need_images, name = build_policy(args)
    print(f"{name}: {args.trials} trials x {len(args.seeds)} seed(s)\n")

    all_rows, per_seed = [], []
    for seed in args.seeds:
        summary, rows = evaluate_strict(
            policy,
            n_trials=args.trials,
            seed=seed,
            need_images=need_images,
            verbose=not args.quiet,
            video_dir=args.video_dir,
            video_n=args.video_n,
        )
        per_seed.append(summary["strict"])
        all_rows += rows
        print(f"\n  seed {seed}: strict {summary['strict'] * 100:.1f}%  "
              f"loose {summary['loose'] * 100:.1f}%\n")

    from src.eval.strict import summarise

    total = summarise(all_rows)
    print(format_report(total, f"\n=== {name} ==="))
    if len(per_seed) > 1:
        print(f"\n  per seed: {' / '.join(f'{s * 100:.0f}' for s in per_seed)}")
        print(f"  strict {total['strict'] * 100:.1f}% +- "
              f"{np.std(per_seed) * 100:.1f}%")

    near = [r for r in all_rows if r["block_start"][0] < 0.53]
    far = [r for r in all_rows if r["block_start"][0] >= 0.53]
    print(f"\n  {'region':<12}{'strict':>9}{'loose':>9}{'n':>6}")
    for label, rows in (("x <  0.53", near), ("x >= 0.53", far)):
        if rows:
            print(f"  {label:<12}"
                  f"{np.mean([r['strict_success'] for r in rows]) * 100:>8.1f}%"
                  f"{np.mean([r['loose_success'] for r in rows]) * 100:>8.1f}%"
                  f"{len(rows):>6}")

    LOG_DIR.mkdir(parents=True, exist_ok=True)
    out = LOG_DIR / f"{args.tag}.json"
    out.write_text(json.dumps({
        "name": name, "trials": args.trials, "seeds": args.seeds,
        "per_seed": per_seed, "summary": total,
        "rows": [
            {k: v for k, v in r.items() if k != "trajectory_per_metric"}
            for r in all_rows
        ],
    }, indent=2, default=float), encoding="utf-8")
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
