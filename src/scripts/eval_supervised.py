"""
Scores a policy, or an ensemble of policies, under the strict gates with
the runtime supervisor in one of four configurations.

Four configurations, because the question is not whether the combination
works but which half of it does what:

    monitor   the supervisor observes and changes nothing. Byte for byte
              the inner policy's actions, so this reproduces the baseline
              and simultaneously records the distributions the thresholds
              are set from
    veto      closes away from grasp height are refused and the action
              chunk is flushed
    retry     failed grasps are detected, the fingers reopen and the policy
              replans
    both      veto and retry together

Reporting one number for "supervisor on" would leave it unknown whether the
prevention or the recovery earned it, and this project has already paid
twice for a result whose cause was not isolated.

Parallelism, and the one rule it has to respect. The harness draws block
placements from a single generator advanced trial by trial, so trial 47 on a
seed depends on the 46 before it. Splitting one seed across processes would
change the placements and silently produce a different experiment. Work is
therefore split **by seed**, one process per seed, which leaves every trial
bit-identical to the serial run and matches the six-seeds-of-a-hundred
protocol exactly.

Run:
    python -m src.scripts.eval_supervised --mode monitor --seeds 81 --trials 100
    python -m src.scripts.eval_supervised --mode both \
        --seeds 101 102 103 104 105 106 --trials 100 --workers 6 --tag headline
"""

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

from src.config import LOG_DIR, OUTPUT_ROOT

# The two members of the 97.0% ensemble. Complementary regional profiles:
# the ResNet18 is 6.8 points better near the base, the ResNet34 5.4 points
# better away from it, which is the only situation where averaging two
# policies can beat either.
DEFAULT_MEMBERS = [
    str(OUTPUT_ROOT / "act_oracle_v2" / "checkpoints" / "030000" / "pretrained_model"),
    str(OUTPUT_ROOT / "act_r34" / "checkpoints" / "025000" / "pretrained_model"),
]

MODES = {
    "monitor": dict(veto=False, retry=False),
    "veto": dict(veto=True, retry=False),
    "retry": dict(veto=False, retry=True),
    "both": dict(veto=True, retry=True),
}


def build_policy(members, mode, device="cuda", oracle=False, jitter=1.0,
                 **kwargs):
    """
    Loads the members, ensembles them if there is more than one, and wraps
    the result in the supervisor.

    The oracle is wrapped by the same supervisor for one reason: it scores
    99.7% strict against the student's 96.4%, and the instrumentation that
    measured the student's clearance budget can measure the demonstrator's
    on the same terms. If the teacher's clearances are much larger, the
    budget is the whole account of the gap. If they are similar, something
    else separates them and the budget is not where to look.

    input:  members (list of str) checkpoint paths, mode (str) one of MODES,
            device (str), oracle (bool) score the demonstrator instead,
            jitter (float) oracle jitter, kwargs passed to SupervisedPolicy
    output: (policy callable, name str)
    """
    from src.eval.ensemble import EnsemblePolicy
    from src.eval.policy_wrapper import LeRobotPolicyAdapter, load_policy_and_processors
    from src.eval.strict import use_checkpoint_render_size
    from src.eval.supervisor import SupervisedPolicy

    flags = MODES[mode]

    if oracle:
        from src.eval.oracle import ScriptedOracle

        inner = ScriptedOracle(jitter=jitter, seed=0)
        policy = SupervisedPolicy(inner, veto=flags["veto"],
                                  retry=flags["retry"], **kwargs)
        return policy, f"oracle(jitter={jitter})+{mode}"

    adapters = []
    for path in members:
        # Render at the resolution the checkpoint was trained at, not at
        # whatever config.py currently says. A policy trained at 160x128 and
        # evaluated at 640x480 does not raise, it goes blind.
        use_checkpoint_render_size(path)
        pol, pre, post = load_policy_and_processors(
            path, policy_type="act", device=device
        )
        adapters.append(LeRobotPolicyAdapter(pol, pre, post, device=device))

    inner = adapters[0] if len(adapters) == 1 else EnsemblePolicy(adapters)
    policy = SupervisedPolicy(inner, veto=flags["veto"], retry=flags["retry"],
                              **kwargs)
    name = f"{'ensemble' if len(adapters) > 1 else Path(members[0]).parts[-3]}+{mode}"
    return policy, name


def run_seed(members, mode, seed, trials, max_steps, device, video_dir,
             video_n, thresholds, oracle=False, jitter=1.0):
    """
    Evaluates one seed in this process, in trial order.

    input:  members (list of str), mode (str), seed (int), trials (int),
            max_steps (int), device (str), video_dir (str or None),
            video_n (int), thresholds (dict) SupervisedPolicy overrides
    output: list of row dicts
    """
    from src.eval.strict import evaluate_strict

    policy, _ = build_policy(members, mode, device=device, oracle=oracle,
                             jitter=jitter, **thresholds)
    _, rows = evaluate_strict(
        policy,
        n_trials=trials,
        seed=seed,
        need_images=not oracle,
        verbose=False,
        video_dir=video_dir,
        video_n=video_n,
        max_steps=max_steps,
    )
    return rows


def spawn_workers(args, thresholds):
    """
    Runs one subprocess per seed and collects their rows.

    input:  args (Namespace), thresholds (dict)
    output: list of row dicts, ordered by seed then trial
    """
    tmp = LOG_DIR / f"_parts_{args.tag}"
    tmp.mkdir(parents=True, exist_ok=True)

    env = dict(os.environ)
    env["PYTHONIOENCODING"] = "utf-8"
    env["PYTHONUNBUFFERED"] = "1"

    pending, running, done = list(args.seeds), [], []

    while pending or running:
        while pending and len(running) < args.workers:
            seed = pending.pop(0)
            out = tmp / f"seed_{seed}.json"
            cmd = [
                sys.executable, "-u", "-m", "src.scripts.eval_supervised",
                "--worker-seed", str(seed), "--worker-out", str(out),
                "--mode", args.mode, "--trials", str(args.trials),
                "--max-steps", str(args.max_steps), "--device", args.device,
                "--members", *args.members,
                "--grasp-max-above-mm", str(args.grasp_max_above_mm),
                "--finger-hold-min-mm", str(args.finger_hold_min_mm),
                "--settle-wait", str(args.settle_wait),
                "--max-recoveries", str(args.max_recoveries),
            ]
            log = open(tmp / f"seed_{seed}.log", "w", encoding="utf-8")
            proc = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT,
                                    env=env, cwd=str(Path.cwd()))
            running.append((seed, proc, out, log))
            print(f"  seed {seed} started (pid {proc.pid})", flush=True)

        time.sleep(2.0)

        for entry in list(running):
            seed, proc, out, log = entry
            if proc.poll() is None:
                continue
            running.remove(entry)
            log.close()
            if proc.returncode != 0 or not out.exists():
                raise SystemExit(
                    f"seed {seed} failed with code {proc.returncode}; "
                    f"see {tmp / f'seed_{seed}.log'}"
                )
            rows = json.loads(out.read_text(encoding="utf-8"))
            done.append((seed, rows))
            print(f"  seed {seed} done: "
                  f"{sum(r['strict_success'] for r in rows)}/{len(rows)} strict",
                  flush=True)

    done.sort(key=lambda kv: args.seeds.index(kv[0]))
    return [r for _, rows in done for r in rows]


def report(rows, name, seeds):
    """
    Prints the full result: gates, supervisor activity, regional split.

    input:  rows (list of dict), name (str), seeds (list of int)
    output: dict summary
    """
    from src.eval.strict import GATES, TIME_BAND, summarise

    n = len(rows)
    total = summarise(rows)
    strict = total["strict"]
    unwidened = float(np.mean([r["strict_success_unwidened"] for r in rows]))

    per_seed = []
    for s in seeds:
        sub = [r for r in rows if r["seed"] == s]
        if sub:
            per_seed.append(float(np.mean([r["strict_success"] for r in sub])))

    vetoes = [r.get("supervisor", {}).get("vetoes", 0) for r in rows]
    recoveries = [r.get("recovery_events", 0) for r in rows]

    lines = [
        "",
        f"=== {name}, {n} trials ===",
        "",
        f"  STRICT pick and place:  {strict * 100:7.2f}%   ({n} trials)",
        f"  strict, original band:  {unwidened * 100:7.2f}%   "
        f"(no recovery allowance)",
        f"  loose check_success:    {total['loose'] * 100:7.2f}%",
        "",
        f"  {'gate':<16}{'pass':>9}",
    ]
    for g in GATES:
        lines.append(f"  {g:<16}{total['gates'][g] * 100:8.2f}%")

    failures = [r for r in rows if not r["strict_success"]]
    lines += [
        "",
        f"  failures: {len(failures)} of {n}",
        f"  trials using a recovery: {sum(1 for c in recoveries if c)} "
        f"({sum(1 for c in recoveries if c) / max(n, 1) * 100:.2f}%)",
        f"  recoveries total {sum(recoveries)}, max in one trial "
        f"{max(recoveries) if recoveries else 0}",
        f"  closes refused total {sum(vetoes)}, in "
        f"{sum(1 for v in vetoes if v)} trials, max in one trial "
        f"{max(vetoes) if vetoes else 0}",
        "",
        f"  steps  mean {np.mean([r['total_steps'] for r in rows]):.0f}  "
        f"max {max(r['total_steps'] for r in rows)}  "
        f"(demo band {TIME_BAND[0]}-{TIME_BAND[1]})",
        f"  lift   mean {np.mean([r['lift_m'] for r in rows]) * 1000:.0f} mm "
        f"(demos 80)",
        f"  trajectory score {np.mean([r['trajectory_score'] for r in rows]):.3f} "
        f"(oracle 0.991)",
    ]

    if per_seed:
        lines.append("")
        lines.append("  per seed: " + " / ".join(f"{p * 100:.0f}" for p in per_seed))
        lines.append(f"  strict {strict * 100:.2f}% +- {np.std(per_seed) * 100:.2f}%")

    near = [r for r in rows if r["block_start"][0] < 0.53]
    far = [r for r in rows if r["block_start"][0] >= 0.53]
    lines.append("")
    for label, sub in (("x <  0.53", near), ("x >= 0.53", far)):
        if sub:
            lines.append(
                f"  {label}  strict "
                f"{np.mean([r['strict_success'] for r in sub]) * 100:6.2f}%  "
                f"n={len(sub)}"
            )

    if failures:
        lines.append("")
        lines.append("  failing trials:")
        for r in failures[:40]:
            miss = ",".join(g for g, ok in r["gates"].items() if not ok)
            lines.append(
                f"    seed {r['seed']} trial {r['trial']:3d}  "
                f"steps {r['total_steps']:4d}  lift "
                f"{r['lift_m'] * 1000:5.1f} mm  rec {r.get('recovery_events', 0)}  "
                f"missing: {miss}"
            )

    # A zero-failure run bounds the rate from below rather than pinning it.
    # Reporting a point estimate of 100% from n trials without that bound
    # is the kind of claim this project exists to avoid.
    if not failures:
        lines += [
            "",
            f"  no failures in {n} trials. One-sided 95% lower bound on the "
            f"success rate: {(1 - 3.0 / n) * 100:.3f}%",
            f"  (a 99.99% claim needs 30,000 clean trials; this is {n})",
        ]

    print("\n".join(lines))
    return {
        "name": name,
        "n": n,
        "strict": strict,
        "strict_unwidened": unwidened,
        "loose": total["loose"],
        "gates": total["gates"],
        "per_seed": per_seed,
        "failures": len(failures),
        "recovery_trials": sum(1 for c in recoveries if c),
        "recoveries_total": sum(recoveries),
        "vetoes_total": sum(vetoes),
    }


def main():
    """
    Entry point.

    input:  none
    output: None
    """
    p = argparse.ArgumentParser()
    p.add_argument("--members", nargs="+", default=DEFAULT_MEMBERS)
    p.add_argument("--mode", default="monitor", choices=sorted(MODES))
    p.add_argument("--seeds", type=int, nargs="+", default=[81])
    p.add_argument("--trials", type=int, default=100)
    p.add_argument("--workers", type=int, default=1)
    p.add_argument("--max-steps", type=int, default=600)
    p.add_argument("--device", default="cuda")
    p.add_argument("--tag", default="supervised")
    p.add_argument("--video-dir", default=None)
    p.add_argument("--video-n", type=int, default=0)
    p.add_argument("--grasp-max-above-mm", type=float, default=12.0)
    p.add_argument("--finger-hold-min-mm", type=float, default=24.0)
    p.add_argument("--settle-wait", type=int, default=12)
    p.add_argument("--max-recoveries", type=int, default=3)
    p.add_argument("--oracle", action="store_true",
                   help="score the scripted demonstrator instead of a "
                        "checkpoint, with the same instrumentation")
    p.add_argument("--jitter", type=float, default=1.0)
    p.add_argument("--worker-seed", type=int, default=None)
    p.add_argument("--worker-out", default=None)
    args = p.parse_args()

    thresholds = dict(
        grasp_max_above_m=args.grasp_max_above_mm / 1000.0,
        finger_hold_min_m=args.finger_hold_min_mm / 1000.0,
        settle_wait_steps=args.settle_wait,
        max_recoveries=args.max_recoveries,
    )

    # Worker branch: one seed, write rows, say nothing else.
    if args.worker_seed is not None:
        rows = run_seed(
            args.members, args.mode, args.worker_seed, args.trials,
            args.max_steps, args.device, args.video_dir, args.video_n,
            thresholds, oracle=args.oracle, jitter=args.jitter,
        )
        Path(args.worker_out).write_text(
            json.dumps(rows, default=float), encoding="utf-8"
        )
        return

    t0 = time.time()
    print(f"{args.mode}: {args.trials} trials x {len(args.seeds)} seed(s), "
          f"{args.workers} worker(s)\n", flush=True)

    if args.workers > 1 and len(args.seeds) > 1:
        rows = spawn_workers(args, thresholds)
        name = f"{'oracle' if args.oracle else 'ensemble'}+{args.mode}"
    else:
        rows = []
        for seed in args.seeds:
            rows += run_seed(
                args.members, args.mode, seed, args.trials, args.max_steps,
                args.device, args.video_dir, args.video_n, thresholds,
                oracle=args.oracle, jitter=args.jitter,
            )
            print(f"  seed {seed} done: "
                  f"{sum(r['strict_success'] for r in rows[-args.trials:])}"
                  f"/{args.trials} strict", flush=True)
        name = f"{'oracle' if args.oracle else 'ensemble'}+{args.mode}"

    summary = report(rows, name, args.seeds)
    summary["elapsed_s"] = time.time() - t0
    summary["mode"] = args.mode
    summary["members"] = args.members
    summary["thresholds"] = {k: float(v) for k, v in thresholds.items()}
    summary["seeds"] = args.seeds

    out = LOG_DIR / f"strict_{args.tag}.json"
    out.write_text(
        json.dumps({"summary": summary, "rows": rows}, default=float),
        encoding="utf-8",
    )
    print(f"\n  {len(rows)} trials in {(time.time() - t0) / 60:.1f} min")
    print(f"  wrote {out}")


if __name__ == "__main__":
    main()
