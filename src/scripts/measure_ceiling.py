"""
Measures the environment's ceiling by running the privileged scripted
oracle over the evaluation distribution.

Whatever this reports is an upper bound on any vision policy trained in
this environment, because the oracle has the one thing no vision policy can
have: the block's exact pose, for free, every step. It runs through the
same harness, the same impedance controller and the same success criterion
as every other number in this project, so the comparison is direct.

The result decides what the remaining work is worth. An oracle at 100%
means the gap to a trained policy is entirely learnable. An oracle in the
mid nineties means part of the gap belongs to the contact model, the
gripper geometry or the step budget, and no amount of policy work moves it.

Run:
    python -m src.scripts.measure_ceiling
    python -m src.scripts.measure_ceiling --trials 100 --seeds 51 52 53 54 55 56
"""

import argparse
import json

import numpy as np

from src.config import LOG_DIR
from src.eval.oracle import ScriptedOracle
from src.eval.rollout import evaluate
from src.scripts.region_analysis import report_regions


def main():
    """
    Entry point.

    input:  none
    output: None
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--trials", type=int, default=100)
    parser.add_argument("--seeds", type=int, nargs="+", default=[51, 52, 53, 54, 55, 56])
    parser.add_argument("--step-mm", type=float, default=4.0)
    args = parser.parse_args()

    oracle = ScriptedOracle(step_m=args.step_mm / 1000.0)

    per_seed, results = [], []
    for seed in args.seeds:
        r = evaluate(oracle, n_trials=args.trials, seed=seed,
                     need_images=False, verbose=False)
        per_seed.append(r["success_rate"])
        results += r["results"]
        print(f"  seed {seed}: {r['success_rate'] * 100:5.1f}%")

    rate = float(np.mean([x["success"] for x in results]))
    n = len(results)
    sd = float(np.std(per_seed))
    se = float(np.sqrt(rate * (1 - rate) / n))

    print(f"\nCEILING: {rate * 100:.1f}% over {n} trials "
          f"({len(args.seeds)} seeds x {args.trials}), sd {sd * 100:.1f}, se {se * 100:.1f}")
    report_regions(results)

    fails = [x for x in results if not x["success"]]
    if fails:
        print(f"\n{len(fails)} failure(s):")
        for f in fails[:20]:
            print(f"  {f['steps']:4d} steps  dist {f['distance'] * 100:5.1f} cm  "
                  f"block {np.round(f['block_start'][:2], 3)}")

    LOG_DIR.mkdir(parents=True, exist_ok=True)
    out = LOG_DIR / "ceiling.json"
    out.write_text(json.dumps({
        "trials": args.trials,
        "seeds": args.seeds,
        "step_mm": args.step_mm,
        "per_seed": per_seed,
        "rate": rate,
        "n": n,
        "failures": [
            {"steps": int(f["steps"]), "distance": float(f["distance"]),
             "block_start": [float(v) for v in f["block_start"]]}
            for f in fails
        ],
    }, indent=2), encoding="utf-8")
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
