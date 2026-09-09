"""
Compares inference-time action settings for a single trained ACT
checkpoint: temporal ensembling at several coefficients against plain
chunk execution at several chunk lengths.

Nothing here retrains anything. Both settings are inference-time only, so
every configuration runs against the same weights already on disk. That
makes this the cheapest experiment in the project and the right one to run
before any of the expensive ones.

The sweep includes one control that carries most of the scientific weight.
Temporal ensembling changes two things at once: the policy re-plans every
step instead of every 32, and it averages every overlapping prediction.
Running n_action_steps=1 with no ensembling isolates the first from the
second. Without that control a gain from ensembling cannot be told apart
from a gain from simply re-planning more often, and the two have very
different implications for what to build next.

Every configuration sees identical block placements on each seed, because
the harness seeds its own generator and draws placements in the same order.
Comparisons are therefore paired, and the report uses a paired test rather
than comparing two independent rates.

Run:
    python -m src.scripts.ensemble_sweep --run act_v4 --checkpoint 020000
    python -m src.scripts.ensemble_sweep --run act_v4 --checkpoint 020000 \
        --seeds 51 52 53 54 55 56 --configs baseline te_0.01
"""

import argparse
import json
import time

import numpy as np

from src.config import LOG_DIR, OUTPUT_ROOT, TASK_STRING
from src.eval.policy_wrapper import LeRobotPolicyAdapter, load_policy_and_processors
from src.eval.rollout import evaluate

# name -> (n_action_steps, temporal_ensemble_coeff)
#
# baseline is how the policy was trained and how every number in the README
# was measured. te_* vary the ensembling coefficient at a fixed chunk. n*
# vary the chunk length with no ensembling, and n1 is the control described
# in the module docstring.
CONFIGS = {
    "baseline": (None, None),
    "te_0.01": (None, 0.01),
    "te_0.00": (None, 0.0),
    "te_0.05": (None, 0.05),
    "n16": (16, None),
    "n8": (8, None),
    "n4": (4, None),
    "n1": (1, None),
}


def run_config(ckpt, policy_type, n_action_steps, coeff, seeds, trials):
    """
    Evaluates one configuration across every seed.

    The policy is reloaded per configuration rather than mutated in place.
    Reloading costs a few seconds and removes any chance that a setting
    from a previous configuration survives into the next one.

    input:  ckpt (str), policy_type (str), n_action_steps (int or None),
            coeff (float or None), seeds (list of int), trials (int)
    output: dict with per_seed rates, per-trial success and block x, timing
    """
    policy, pre, post = load_policy_and_processors(
        ckpt,
        policy_type,
        n_action_steps=n_action_steps,
        temporal_ensemble_coeff=coeff,
    )
    adapter = LeRobotPolicyAdapter(policy, pre, post, task=TASK_STRING)

    per_seed, success, block_x = [], [], []
    start = time.perf_counter()

    for seed in seeds:
        result = evaluate(
            adapter, n_trials=trials, seed=seed, need_images=True, verbose=False
        )
        per_seed.append(result["success_rate"])
        success += [bool(r["success"]) for r in result["results"]]
        block_x += [float(r["block_start"][0]) for r in result["results"]]
        print(f"    seed {seed}: {result['success_rate'] * 100:5.1f}%")

    elapsed = time.perf_counter() - start

    return {
        "per_seed": per_seed,
        "success": success,
        "block_x": block_x,
        "elapsed_s": elapsed,
        "s_per_trial": elapsed / max(len(success), 1),
    }


def paired_delta(baseline_success, variant_success):
    """
    Compares two configurations on the trials they both ran.

    Trial i faced the same block placement under both configurations, so
    the two outcome sequences are paired. Only the trials where they
    disagree carry information about which is better, which is McNemar's
    observation. Treating the two rates as independent samples throws that
    pairing away and inflates the standard error.

    input:  baseline_success (list of bool), variant_success (list of bool)
    output: (delta, se, n_flips) as fractions, or (None, None, 0)
    """
    n = min(len(baseline_success), len(variant_success))
    if n == 0:
        return None, None, 0

    b = sum(1 for i in range(n) if baseline_success[i] and not variant_success[i])
    c = sum(1 for i in range(n) if variant_success[i] and not baseline_success[i])

    delta = (c - b) / n
    se = np.sqrt(b + c) / n if (b + c) else 0.0
    return delta, se, b + c


def report(results, trials, seeds):
    """
    Prints the comparison table.

    input:  results (dict name -> run_config output), trials (int),
            seeds (list of int)
    output: None
    """
    n = trials * len(seeds)
    base = results.get("baseline")

    print(f"\n{'config':<12}{'success':>10}{'sd':>8}{'vs base':>10}"
          f"{'se':>8}{'flips':>7}{'s/trial':>9}")
    print("-" * 64)

    for name, r in results.items():
        rate = float(np.mean(r["success"]))
        sd = float(np.std(r["per_seed"])) if len(r["per_seed"]) > 1 else float("nan")

        if base is None or name == "baseline":
            delta_s = se_s = flips_s = ""
        else:
            delta, se, flips = paired_delta(base["success"], r["success"])
            delta_s = f"{delta * 100:+.1f}"
            se_s = f"{se * 100:.1f}"
            flips_s = str(flips)

        sd_s = "" if np.isnan(sd) else f"{sd * 100:.1f}"
        print(f"{name:<12}{rate * 100:>9.1f}%{sd_s:>8}{delta_s:>10}"
              f"{se_s:>8}{flips_s:>7}{r['s_per_trial']:>9.1f}")

    print(f"\nn = {n} per config ({len(seeds)} seed(s) x {trials} trials), paired.")
    print("vs base is a paired difference. Only trials where the two configs")
    print("disagreed contribute to it, so se shrinks with the flip count, not n.")


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
    parser.add_argument("--seeds", type=int, nargs="+", default=[41])
    parser.add_argument("--configs", nargs="+", default=list(CONFIGS))
    args = parser.parse_args()

    unknown = [c for c in args.configs if c not in CONFIGS]
    if unknown:
        raise SystemExit(f"unknown configs {unknown}, choose from {list(CONFIGS)}")

    ckpt = OUTPUT_ROOT / args.run / "checkpoints" / args.checkpoint / "pretrained_model"
    if not ckpt.exists():
        raise SystemExit(f"no checkpoint at {ckpt}")

    print(f"{args.run} @ {args.checkpoint}, {args.trials} trials x "
          f"{len(args.seeds)} seed(s), configs: {', '.join(args.configs)}")

    results = {}
    for name in args.configs:
        n_action_steps, coeff = CONFIGS[name]
        print(f"\n[{name}]")
        results[name] = run_config(
            str(ckpt), args.type, n_action_steps, coeff, args.seeds, args.trials
        )

    report(results, args.trials, args.seeds)

    LOG_DIR.mkdir(parents=True, exist_ok=True)
    out = LOG_DIR / f"ensemble_sweep_{args.run}_{args.checkpoint}.json"
    out.write_text(
        json.dumps(
            {
                "run": args.run,
                "checkpoint": args.checkpoint,
                "trials": args.trials,
                "seeds": args.seeds,
                "configs": {k: CONFIGS[k] for k in args.configs},
                "results": results,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
