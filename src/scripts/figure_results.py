"""
Draws the two charts the README leads with: where the success rate went,
and the quantity that explains it.

Both panels are hand-entered rather than computed, because they pool
numbers from measurements made months apart under different sample sizes,
and every one of them is sourced in `docs/PATH_TO_97.md` or
`docs/PATH_TO_99.md`. The sample size is printed next to each bar for the
same reason: this project has twice drawn a conclusion from forty trials
and had to withdraw it, so an unlabelled rate is not worth plotting.

Run:
    python -m src.scripts.figure_results
"""

import argparse

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from src.config import DOCS_DIR  # noqa: E402

TEACHER = "#8e6c1f"
STUDENT = "#1f6f4a"
OLD = "#9aa5ab"

# (label, strict %, n, colour). Ordered worst to best so the bars read
# upward. Every figure is on the strict nine-gate criterion; the loose
# numbers this project reported before Stage 3c are not comparable and are
# deliberately absent.
MILESTONES = [
    ("act_oracle_v1\nprevious headline", 72.5, 40, OLD),
    ("act_oracle_v2\nlabels corrected", 92.3, 600, OLD),
    ("+ act_r34\nensembled", 96.38, 2400, OLD),
    ("act_aux_xy\nalone", 96.50, 200, STUDENT),
    ("act_oracle_v2 + act_aux_xy\nthis result", 98.83, 1200, STUDENT),
    ("scripted demonstrator\nthe ceiling", 99.70, 1000, TEACHER),
]

# (label, p5 clearance mm, strict %, colour, n)
BUDGET = [
    ("act_oracle_v2 alone", 1.83, 92.00, STUDENT, 200),
    ("+ act_r34", 5.08, 97.00, STUDENT, 200),
    ("+ act_aux_xy", 7.20, 98.83, STUDENT, 1200),
    ("demonstrator", 11.94, 99.70, TEACHER, 400),
]


def main():
    """
    Entry point.

    input:  none
    output: None
    """
    p = argparse.ArgumentParser()
    p.add_argument("--out", default=str(DOCS_DIR / "results.png"))
    args = p.parse_args()

    fig, axes = plt.subplots(1, 2, figsize=(14.5, 5.6),
                             gridspec_kw={"width_ratios": [1.35, 1]})

    # ---- panel 1: where the rate went -----------------------------------
    ax = axes[0]
    labels = [m[0] for m in MILESTONES]
    vals = [m[1] for m in MILESTONES]
    ns = [m[2] for m in MILESTONES]
    colours = [m[3] for m in MILESTONES]
    y = np.arange(len(labels))

    ax.barh(y, vals, color=colours, alpha=0.9, height=0.62)
    for i, (v, n) in enumerate(zip(vals, ns, strict=True)):
        ax.text(v + 0.4, i, f"{v:.2f}%   n={n:,}", va="center", fontsize=9)

    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=9)
    ax.set_xlim(65, 108)
    ax.set_xlabel("strict success rate (all nine gates)")
    ax.set_title("Every number on the strict criterion, with its sample size",
                 fontsize=11)
    ax.grid(axis="x", alpha=0.25)
    ax.axvline(99.70, color=TEACHER, ls=":", lw=1.4)

    # ---- panel 2: the quantity that explains it -------------------------
    ax = axes[1]
    xs = np.array([b[1] for b in BUDGET])
    ys = np.array([b[2] for b in BUDGET])

    ax.plot(xs, ys, color="#2c3e50", lw=1.2, alpha=0.5, zorder=1)
    for label, x, v, colour, n in BUDGET:
        ax.scatter([x], [v], s=130, c=colour, zorder=3, edgecolors="white",
                   linewidths=1.5)
        ax.annotate(f"{label}\nn={n:,}", (x, v), textcoords="offset points",
                    xytext=(6, -22), fontsize=8.5, color="#333")

    ax.axvspan(-1, 6.0, color="#c0392b", alpha=0.08)
    ax.annotate("every jam measured\nsits below 6 mm", (0.4, 93.4),
                fontsize=8.5, color="#c0392b")

    ax.set_xlim(-0.5, 15)
    ax.set_ylim(90.5, 100.6)
    ax.set_xlabel("5th-percentile clearance at descent onset (mm)\n"
                  r"$40 - (d + 22(\cos e + \sin e))$")
    ax.set_ylabel("strict success rate")
    ax.set_title("The clearance budget predicts the rate", fontsize=11)
    ax.grid(alpha=0.25)

    fig.suptitle(
        "A pick-and-place policy scored on whether it performed the task, "
        "not on where the block ended up",
        fontsize=12.5, y=0.985,
    )
    fig.text(
        0.5, 0.015,
        "Left: the strict criterion requires all nine gates — a shove, a drop "
        "and a hover all fail it. Right: the gripper's open fingers must clear "
        "the block to descend around it; d is the lateral offset and e the "
        "commanded wrist yaw error.",
        ha="center", fontsize=8.5, color="#555",
    )
    fig.tight_layout(rect=(0, 0.055, 1, 0.94))
    fig.savefig(args.out, dpi=150)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
