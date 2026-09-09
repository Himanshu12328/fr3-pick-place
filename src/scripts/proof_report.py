"""
Renders visual evidence that a policy performs the task, not just that it
scores well.

This project's own history is the argument for it. Four separate policies
reported success rates above 99% while shoving the block across the table,
dropping it from 95 mm, or ending with it gripped in mid-air, and the
success rate saw none of it. What caught each one was plotting the block's
height against time and looking: a shove never leaves the table, a drop
falls, a hover never comes down.

So the headline panel here is that same plot. A correct episode reads flat
on the table, one clean rise, a plateau through the carry, one clean
descent, and flat afterwards because the arm retreats without disturbing
what it put down.

The other two panels give the numbers behind it: which of the nine gates
each trial passed, and how long each episode took against the band the
recorded demonstrations span.

Run:
    python -m src.scripts.proof_report --checkpoint <dir> --trials 20
    python -m src.scripts.proof_report --ensemble <dir_a> <dir_b> --trials 20
"""

import argparse
import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mujoco
import numpy as np

from src.config import DOCS_DIR, LOG_DIR
from src.data.task import BLOCK_Z
from src.eval import rollout as R
from src.eval.strict import (
    GATES,
    TIME_BAND,
    run_strict_trial,
    summarise,
    use_checkpoint_render_size,
)
from src.rl import reference as REF

PASS_COLOUR = "#1b7f5a"
FAIL_COLOUR = "#b3402f"
BAND_COLOUR = "#c8d6e5"


def build_policy(args):
    """
    Constructs the policy under test.

    input:  args (Namespace)
    output: (policy callable, need_images bool, label str)
    """
    if args.oracle:
        from src.eval.oracle import ScriptedOracle

        return ScriptedOracle(jitter=args.jitter, seed=0), False, "scripted oracle"

    from src.eval.policy_wrapper import LeRobotPolicyAdapter, load_policy_and_processors

    paths = args.ensemble if args.ensemble else [args.checkpoint]
    members = []
    for p in paths:
        use_checkpoint_render_size(p)
        pol, pre, post = load_policy_and_processors(
            p, policy_type=args.policy_type, device=args.device
        )
        members.append(LeRobotPolicyAdapter(pol, pre, post, device=args.device))

    if len(members) == 1:
        return members[0], True, args.checkpoint

    from src.eval.ensemble import EnsemblePolicy

    return EnsemblePolicy(members), True, f"ensemble of {len(members)} policies"


def collect(policy, need_images, trials, seed):
    """
    Runs the trials, keeping the full trace of each.

    input:  policy (callable), need_images (bool), trials (int), seed (int)
    output: list of dict
    """
    model, data = R.setup_model()
    ctrl = R.ImpedanceController(
        model, data, kp_trans=R.KP_TRANS, kp_rot=R.KP_ROT, zeta=R.ZETA,
        kp_null=10.0, zeta_null=1.0, n_arm=R.N_ARM, verbose=False,
    )
    rng = np.random.default_rng(seed)
    if hasattr(policy, "bind"):
        policy.bind(model, data)

    renderer = None
    if need_images:
        renderer = mujoco.Renderer(model, height=R.CAM_HEIGHT, width=R.CAM_WIDTH)

    rows = []
    for i in range(trials):
        if hasattr(policy, "reset"):
            policy.reset()
        row = run_strict_trial(model, data, ctrl, policy, renderer, rng,
                               need_images=need_images, record_frames=False)
        rows.append(row)
        mark = "pass" if row["strict_success"] else "FAIL"
        print(f"  trial {i:2d}  {mark}  {row['total_steps']:4d} steps  "
              f"lift {row['lift_m'] * 1000:5.1f} mm", flush=True)

    if renderer is not None:
        renderer.close()
    return rows


def render(rows, label, out_path, note=None):
    """
    Draws the three-panel figure.

    input:  rows (list of dict), label (str), out_path (Path)
    output: dict summary
    """
    summary = summarise(rows)
    n = len(rows)
    n_pass = sum(r["strict_success"] for r in rows)

    fig = plt.figure(figsize=(13, 9))
    grid = fig.add_gridspec(2, 2, height_ratios=[1.25, 1.0], hspace=0.32, wspace=0.22)

    # ---- block height traces, the panel that catches what rates cannot ----
    ax = fig.add_subplot(grid[0, :])
    for r in rows:
        block = np.asarray(r["trace"]["block"])
        z_mm = (block[:, 2] - BLOCK_Z) * 1000.0
        ax.plot(np.arange(len(z_mm)), z_mm,
                color=PASS_COLOUR if r["strict_success"] else FAIL_COLOUR,
                alpha=0.55, linewidth=1.1)

    ax.axhline(80.0, color="#444", linestyle="--", linewidth=1.0)
    ax.text(4, 83, "demonstrations lift 80 mm", fontsize=9, color="#444")
    ax.axhline(0.0, color="#444", linewidth=0.8)
    ax.text(4, 4, "resting on the table", fontsize=9, color="#444")
    ax.set_xlabel("policy step")
    ax.set_ylabel("block height above resting (mm)")
    ax.set_title(
        f"Block height through each episode  —  {label}\n"
        f"flat, one clean rise, a plateau through the carry, one clean descent, "
        f"then flat because the arm retreats without disturbing it",
        fontsize=11, loc="left")
    ax.grid(alpha=0.25)
    ax.margins(x=0.01)

    # ---- gate pass rates ----
    ax = fig.add_subplot(grid[1, 0])
    names = list(reversed(GATES))
    vals = [summary["gates"][g] * 100 for g in names]
    bars = ax.barh(names, vals, color=PASS_COLOUR, alpha=0.85)
    for b, v in zip(bars, vals):
        ax.text(min(v + 1.5, 101), b.get_y() + b.get_height() / 2,
                f"{v:.0f}%", va="center", fontsize=9)
    ax.set_xlim(0, 112)
    ax.set_xlabel("percent of trials passing")
    ax.set_title("Each of the nine gates", fontsize=11, loc="left")
    ax.grid(axis="x", alpha=0.25)

    # ---- per-trial episode length against the demonstration band ----
    ax = fig.add_subplot(grid[1, 1])
    steps = [r["total_steps"] for r in rows]
    colours = [PASS_COLOUR if r["strict_success"] else FAIL_COLOUR for r in rows]
    ax.bar(range(n), steps, color=colours, alpha=0.85)
    ax.axhspan(TIME_BAND[0], TIME_BAND[1], color=BAND_COLOUR, alpha=0.55, zorder=0)
    ax.axhline(REF.TOTAL_STEPS[0], color="#444", linestyle="--", linewidth=1.0)
    ax.text(0.2, REF.TOTAL_STEPS[0] + 6, "demonstration mean, 295 steps",
            fontsize=9, color="#444")
    ax.set_xlabel("trial")
    ax.set_ylabel("episode length (policy steps)")
    ax.set_title("Episode length, shaded band is the demonstrations' range",
                 fontsize=11, loc="left")
    ax.grid(axis="y", alpha=0.25)

    fig.suptitle(
        f"STRICT pick-and-place: {n_pass} of {n} trials passed all nine gates "
        f"({n_pass / n * 100:.0f}%)   |   "
        f"mean lift {summary['mean_lift_mm']:.0f} mm   |   "
        f"block moved {summary['mean_drift_mm']:.2f} mm after release   |   "
        f"trajectory score {summary['mean_trajectory_score']:.3f}",
        fontsize=12.5, y=0.985)

    if note:
        # A single sample is not the claim. Say so on the figure itself, so a
        # panel showing 20 of 20 cannot be read as the headline number.
        fig.text(0.5, 0.005, note, ha="center", fontsize=10, color="#444",
                 style="italic")

    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    print(f"\nwrote {out_path}")
    return summary


def main():
    """
    Entry point.

    input:  none
    output: None
    """
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint")
    p.add_argument("--ensemble", nargs="+")
    p.add_argument("--oracle", action="store_true")
    p.add_argument("--jitter", type=float, default=1.0)
    p.add_argument("--policy-type", default="act")
    p.add_argument("--trials", type=int, default=20)
    p.add_argument("--seed", type=int, default=91)
    p.add_argument("--device", default="cuda")
    p.add_argument("--out", default=None)
    p.add_argument("--tag", default="proof")
    p.add_argument("--note", default=None,
                   help="caption placed under the figure, for stating the "
                        "headline result a single sample should not be "
                        "mistaken for")
    args = p.parse_args()

    if not (args.checkpoint or args.ensemble or args.oracle):
        raise SystemExit("give --checkpoint, --ensemble or --oracle")

    policy, need_images, label = build_policy(args)
    print(f"{label}: {args.trials} trials, seed {args.seed}\n")
    rows = collect(policy, need_images, args.trials, args.seed)

    out = args.out or (DOCS_DIR / f"{args.tag}.png")
    summary = render(rows, label, out, note=args.note)

    LOG_DIR.mkdir(parents=True, exist_ok=True)
    (LOG_DIR / f"{args.tag}.json").write_text(json.dumps({
        "label": label, "trials": args.trials, "seed": args.seed,
        "summary": summary,
    }, indent=2, default=float), encoding="utf-8")

    print(f"\nSTRICT {summary['strict'] * 100:.1f}%  "
          f"loose {summary['loose'] * 100:.1f}%  "
          f"lift {summary['mean_lift_mm']:.0f} mm  "
          f"trajectory {summary['mean_trajectory_score']:.3f}")


if __name__ == "__main__":
    main()
