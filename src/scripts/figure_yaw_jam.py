"""
Draws the three measurements that identify the residual failure.

Panel 1, the jam. Commanded tool height against achieved tool height, both
relative to the block's centre, for one failing and one passing trial. The
failing trial's policy asks for the descent and the arm does not go: it
stops at the block's top face and stays there while the command sits 25 mm
below it. The passing trial tracks the command to under a millimetre. This
is what rules out a timing explanation, and with it the whole class of
fixes that refuse the mistimed close.

Panel 2, one of the two terms. Commanded wrist yaw error against the
block's yaw offset from the nearest face alignment, every trial of a
200-trial run. The relationship is proportional — error = 0.115 x offset —
which is shrinkage toward the mean rather than a failure at one particular
angle, and it is why failures concentrate where the wrist has furthest to
turn.

Panel 3, the budget both terms are spent from. A finger face sits 40 mm
from the tool centre when open, and a 44 mm cube at yaw error e presents
22(cos e + sin e) of half-extent, so a gripper offset laterally by d has
40 - (d + extent) millimetres of room to pass the block's top face. Every
jam in 200 trials sits under 6 mm of it against a median of 12. Neither
term separates alone, which is why four single-variable explanations each
looked right on a handful of trials and failed at scale, and why correcting
the yaw with the true block orientation moves the rate from 97.0% to
96.0%: it buys down one term of two.

Run:
    python -m src.scripts.figure_yaw_jam
"""

import argparse
import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import mujoco  # noqa: E402
import numpy as np  # noqa: E402

from src.config import DOCS_DIR, LOG_DIR  # noqa: E402
from src.data.task import sample_block_pose  # noqa: E402
from src.eval import rollout as R  # noqa: E402


def run_pair(seed, trials, members, device="cuda"):
    """
    Replays the named trials and returns their height traces.

    input:  seed (int), trials (list of int), members (list of str),
            device (str)
    output: dict trial -> dict of arrays
    """
    from src.scripts.eval_supervised import build_policy
    from src.scripts.probe_stuck_grasp import replay

    rng = np.random.default_rng(seed)
    placements = [sample_block_pose(rng) for _ in range(100)]

    model, data = R.setup_model()
    ctrl = R.ImpedanceController(
        model, data, kp_trans=R.KP_TRANS, kp_rot=R.KP_ROT, zeta=R.ZETA,
        kp_null=10.0, zeta_null=1.0, n_arm=R.N_ARM, verbose=False,
    )
    renderer = mujoco.Renderer(model, height=R.CAM_HEIGHT, width=R.CAM_WIDTH)
    policy, _ = build_policy(members, "monitor", device=device)
    policy.bind(model, data)
    inner = policy.policy

    out = {}
    for idx in trials:
        pos, quat = placements[idx]
        t, row, steps = replay(model, data, ctrl, policy, renderer, pos, quat,
                               max_steps=900, inner_ensemble=inner)
        out[idx] = {
            "target": t["target_above_block"] * 1000.0,
            "ee": t["above_block"] * 1000.0,
            "grip": t["grip"],
            "yaw": t["yaw_err"],
            "strict": bool(row["strict_success"]),
            "steps": int(steps),
        }
        print(f"  seed {seed} trial {idx}: strict={row['strict_success']}, "
              f"{steps} steps", flush=True)
    renderer.close()
    return out


def main():
    """
    Entry point.

    input:  none
    output: None
    """
    from src.scripts.eval_supervised import DEFAULT_MEMBERS

    p = argparse.ArgumentParser()
    p.add_argument("--members", nargs="+", default=DEFAULT_MEMBERS)
    p.add_argument("--seed", type=int, default=92)
    p.add_argument("--fail-trial", type=int, default=11)
    p.add_argument("--pass-trial", type=int, default=0)
    p.add_argument("--scatter", default=str(LOG_DIR / "strict_yawscatter.json"))
    p.add_argument("--out", default=str(DOCS_DIR / "yaw_jam.png"))
    p.add_argument("--device", default="cuda")
    args = p.parse_args()

    traces = run_pair(args.seed, [args.fail_trial, args.pass_trial],
                      args.members, args.device)

    rows = json.load(open(args.scatter, encoding="utf-8"))["rows"]
    off, err, ok = [], [], []
    for r in rows:
        ev = r.get("supervisor", {}).get("close_events", [])
        if not ev or "block_yaw_offset_deg" not in ev[0]:
            continue
        off.append(ev[0]["block_yaw_offset_deg"])
        err.append(ev[0]["cmd_yaw_err_deg"])
        ok.append(bool(r["strict_success"]))
    off, err, ok = np.array(off), np.array(err), np.array(ok)

    fig, axes = plt.subplots(1, 3, figsize=(18.5, 5.4))

    # ---- panel 1: the jam ------------------------------------------------
    ax = axes[0]
    styles = {
        args.fail_trial: ("#c0392b", "failed"),
        args.pass_trial: ("#1f6f4a", "passed"),
    }
    for idx, (colour, label) in styles.items():
        t = traces[idx]
        n = len(t["ee"])
        x = np.arange(n)
        ax.plot(x, t["target"], color=colour, ls="--", lw=1.2, alpha=0.75,
                label=f"trial {idx} ({label}) — commanded")
        ax.plot(x, t["ee"], color=colour, lw=2.0,
                label=f"trial {idx} ({label}) — achieved")

    ax.axhline(0.0, color="#444", lw=0.9)
    ax.axhline(22.0, color="#888", lw=0.9, ls=":")
    ax.annotate("block top face, +22 mm", xy=(0.99, 22.0),
                xycoords=("axes fraction", "data"), ha="right", va="bottom",
                fontsize=8, color="#555")
    ax.annotate("block centre, a grasp closes 1-2 mm below this", xy=(0.99, 0.0),
                xycoords=("axes fraction", "data"), ha="right", va="bottom",
                fontsize=8, color="#555")
    ax.set_xlim(0, 350)
    ax.set_ylim(-40, 200)
    ax.set_xlabel("policy step")
    ax.set_ylabel("height above block centre (mm)")
    ax.set_title("The policy commands the descent; the arm does not go",
                 fontsize=11)
    ax.legend(fontsize=7.5, loc="upper right", framealpha=0.95)
    ax.grid(alpha=0.25)

    # ---- panel 2: the shrinkage -----------------------------------------
    ax = axes[1]
    ax.scatter(off[ok], err[ok], s=16, c="#1f6f4a", alpha=0.55,
               label=f"passed strict (n={int(ok.sum())})", edgecolors="none")
    ax.scatter(off[~ok], err[~ok], s=58, c="#c0392b", marker="X",
               label=f"failed strict (n={int((~ok).sum())})", zorder=5)

    k = float((off * err).sum() / (off * off).sum())
    xs = np.linspace(0, 45, 50)
    ax.plot(xs, k * xs, color="#2c3e50", lw=1.8,
            label=f"least squares through origin: {k:.3f} x offset")

    ax.set_xlabel("block yaw offset from nearest face alignment (deg)")
    ax.set_ylabel("commanded wrist yaw error at the grasp (deg)")
    ax.set_title("The wrist is under-rotated in proportion, at every angle",
                 fontsize=11)
    ax.legend(fontsize=8, loc="upper left", framealpha=0.95)
    ax.grid(alpha=0.25)
    ax.set_xlim(0, 46)

    # ---- panel 3: the clearance budget ----------------------------------
    ax = axes[2]
    lat, eps, jam = [], [], []
    for r in rows:
        d = r.get("supervisor", {}).get("descent_onset")
        if not d or "cmd_yaw_err_deg" not in d:
            continue
        ce = r.get("supervisor", {}).get("close_events", [])
        is_jam = (not r["strict_success"] and ce
                  and not any(e.get("gripped") for e in ce))
        lat.append(d["tool_block_xy_mm"])
        eps.append(d["cmd_yaw_err_deg"])
        jam.append(bool(is_jam))
    lat = np.array(lat)
    extent = 22.0 * (np.cos(np.radians(eps)) + np.sin(np.radians(eps)))
    clear = 40.0 - (lat + extent)
    jam = np.array(jam)

    bins = np.linspace(-10, 18, 29)
    ax.hist(clear[~jam], bins=bins, color="#1f6f4a", alpha=0.55,
            label=f"did not jam (n={int((~jam).sum())})")
    ax.hist(clear[jam], bins=bins, color="#c0392b",
            label=f"jammed (n={int(jam.sum())})")
    ax.axvline(6.0, color="#2c3e50", ls="--", lw=1.6)
    ax.annotate("every jam below 6 mm\n(4 of 4, against 7 of 194\n"
                "trials that did not jam)",
                xy=(8.5, ax.get_ylim()[1] * 0.62),
                fontsize=8, color="#2c3e50", va="center")
    ax.set_xlabel("clearance at descent onset (mm)\n"
                  r"$40 - (d + 22(\cos e + \sin e))$")
    ax.set_ylabel("trials")
    ax.set_title("Two terms, one budget: neither alone separates",
                 fontsize=11)
    ax.legend(fontsize=8, loc="upper left", framealpha=0.95)
    ax.grid(alpha=0.25)

    fig.suptitle(
        "The residual failure is a clearance event, not a mistimed grasp: "
        "a finger lands on the block and the descent stops",
        fontsize=12.5, y=0.99,
    )
    fig.text(
        0.5, 0.055,
        "Left: two trials from held-out seed 92. Middle and right: every trial of a "
        "200-trial run on seeds 61, 62, 91, 92 in monitor mode, which leaves the "
        "policy's actions untouched.",
        ha="center", fontsize=8, color="#555",
    )
    fig.text(
        0.5, 0.017,
        "Right: d is the lateral offset, e the commanded wrist yaw error. Correcting "
        "e alone, using the block's true orientation, moves the rate from 97.0% to "
        "96.0% — it buys down one term of two.",
        ha="center", fontsize=8, color="#555",
    )
    fig.tight_layout(rect=(0, 0.085, 1, 0.94))
    fig.savefig(args.out, dpi=150)
    print(f"\n  wrote {args.out}")


if __name__ == "__main__":
    main()
