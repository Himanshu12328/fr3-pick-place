"""
Runs a trained policy and reports what its failures actually were.

Run:
    python -m src.scripts.classify_failures --run act_v4 --checkpoint 020000
    python -m src.scripts.classify_failures --oracle --trials 100
"""

import argparse
import json

from src.config import LOG_DIR, OUTPUT_ROOT, TASK_STRING
from src.eval.failure_analysis import classify, report, summarise
from src.eval.oracle import ScriptedOracle
from src.eval.policy_wrapper import LeRobotPolicyAdapter, load_policy_and_processors
from src.eval.rollout import evaluate


def main():
    """
    Entry point.

    input:  none
    output: None
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", default="act_v4")
    parser.add_argument("--checkpoint", default="020000")
    parser.add_argument("--type", default="act")
    parser.add_argument("--trials", type=int, default=100)
    parser.add_argument("--seeds", type=int, nargs="+", default=[51, 52, 53])
    parser.add_argument("--oracle", action="store_true",
                        help="classify the privileged oracle instead of a policy")
    args = parser.parse_args()

    if args.oracle:
        policy, tag, need_images = ScriptedOracle(), "oracle", False
    else:
        ckpt = (OUTPUT_ROOT / args.run / "checkpoints" / args.checkpoint
                / "pretrained_model")
        p, pre, post = load_policy_and_processors(str(ckpt), args.type)
        policy = LeRobotPolicyAdapter(p, pre, post, task=TASK_STRING)
        tag, need_images = f"{args.run}_{args.checkpoint}", True

    results = []
    for seed in args.seeds:
        r = evaluate(policy, n_trials=args.trials, seed=seed,
                     need_images=need_images, verbose=False, record_trace=True)
        results += r["results"]
        print(f"  seed {seed}: {r['success_rate'] * 100:5.1f}%")

    summary = summarise(results)
    print(f"\n{tag}: {summary['counts']['success'] / summary['n'] * 100:.1f}% "
          f"over {summary['n']} trials")
    report(summary, results)

    # Where in the workspace each failure mode lives. The near/far split is
    # the one this project already tracks, so keep reporting on it.
    print(f"\n{'failure mode':<20}{'near x<0.53':>13}{'far x>=0.53':>13}")
    print("-" * 46)
    for name in sorted({classify(r) for r in results} - {"success"}):
        near = sum(1 for r in results
                   if classify(r) == name and r["block_start"][0] < 0.53)
        far = sum(1 for r in results
                  if classify(r) == name and r["block_start"][0] >= 0.53)
        print(f"{name:<20}{near:>13}{far:>13}")

    LOG_DIR.mkdir(parents=True, exist_ok=True)
    out = LOG_DIR / f"failures_{tag}.json"
    out.write_text(json.dumps({
        "tag": tag,
        "trials": args.trials,
        "seeds": args.seeds,
        "summary": {"counts": summary["counts"], "timeouts": summary["timeouts"],
                    "n": summary["n"]},
        "failures": [
            {"mode": classify(r), "steps": int(r["steps"]),
             "distance": float(r["distance"]),
             "block_start": [float(v) for v in r["block_start"]]}
            for r in results if not r["success"]
        ],
    }, indent=2), encoding="utf-8")
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
