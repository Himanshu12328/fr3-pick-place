"""
Reports the clearance budget of one or more evaluation runs.

The budget is the quantity that accounts for the residual failures, and
docs/PATH_TO_99.md S25 and S30 derive it. The gripper descends with its
fingers open; each finger face sits 40 mm from the tool centre; a 44 mm cube
at wrist yaw error `e` presents 22*(cos e + sin e) of half-extent along the
closing axis. So a tool offset laterally by `d` has

    clearance = 40 - (d + 22 * (cos e + sin e))

millimetres of room to pass the block's top face. Negative clearance means a
finger is over the block and the descent cannot complete; small positive
clearance means it completes only if nothing else goes slightly wrong.

Measured at descent onset, over 200 trials of the 97%-class ensemble, every
jam sits under 6 mm of clearance against a median of 12, and 11 trials in
200 are in that band of which 4 jam — 36% against a 2% base rate. The
scripted demonstrator, which scores 99.75% over 400 trials, never goes
below 6.40 mm and never jams.

**Why this is the metric to tune against rather than the success rate.**
Two clean 1,200-trial measurements of one unchanged policy landed 1.25
points apart (S27), so a success rate cannot resolve a point of improvement
without thousands of trials. The fifth percentile of the clearance
distribution is a continuous quantity measurable on 200, and the target is
explicit: the ensemble is at 5.08 mm and the demonstrator at 11.94.

Reads the JSON written by `eval_supervised.py`, which records the descent
onset in monitor mode.

Run:
    python -m src.scripts.clearance_report logs/strict_yawscatter.json
    python -m src.scripts.clearance_report logs/strict_*.json --labels ...
"""

import argparse
import json
from pathlib import Path

import numpy as np

# Finger face distance from the tool centre with the gripper open, and the
# block's half width. Both from the scene.
FINGER_OPEN_MM = 40.0
BLOCK_HALF_MM = 22.0

# The band every measured jam falls inside. Derived from 4 jams against 194
# passes and then independently respected by 400 demonstrator trials, which
# is why it is quoted as a threshold rather than as a fitted boundary.
DANGER_MM = 6.0


def classify(row):
    """
    Says what happened in one trial, distinguishing the jam from the other
    failure modes.

    Lumping them together is what made the residual look like one thing for
    two sessions. A trial that never commands a close and a trial that
    grasps cleanly and fails the retreat are not jams and are not addressed
    by anything the clearance budget describes.

    input:  row (dict) one evaluation row
    output: str, one of pass, jam, never closed, grasped then failed
    """
    sup = row.get("supervisor", {})
    closes = sup.get("close_events", [])
    if row["strict_success"]:
        return "pass"
    if not closes:
        return "never closed"
    if any(e.get("gripped") for e in closes):
        return "grasped then failed"
    return "jam"


def clearance_of(row):
    """
    The clearance at descent onset, or None if the trial never began one.

    input:  row (dict)
    output: (clearance mm, lateral mm, yaw error deg) or None
    """
    d = row.get("supervisor", {}).get("descent_onset")
    if not d or "cmd_yaw_err_deg" not in d:
        return None
    lat = float(d["tool_block_xy_mm"])
    eps = np.radians(float(d["cmd_yaw_err_deg"]))
    extent = BLOCK_HALF_MM * (np.cos(eps) + np.sin(eps))
    return FINGER_OPEN_MM - (lat + extent), lat, float(np.degrees(eps))


def report(path, label):
    """
    Prints the budget for one run.

    input:  path (str or Path), label (str)
    output: dict summary, or None if the file carries no descent onsets
    """
    blob = json.loads(Path(path).read_text(encoding="utf-8"))
    rows = blob["rows"] if "rows" in blob else blob

    recs, kinds = [], []
    for r in rows:
        c = clearance_of(r)
        if c is None:
            continue
        recs.append(c)
        kinds.append(classify(r))

    if not recs:
        print(f"{label}: no descent onsets recorded "
              f"(was this run made in monitor mode?)")
        return None

    a = np.asarray(recs, dtype=float)
    kinds = np.asarray(kinds)
    clear, lat, yaw = a[:, 0], a[:, 1], a[:, 2]
    jam = kinds == "jam"

    n_band = int((clear < DANGER_MM).sum())
    n_jam_band = int((jam & (clear < DANGER_MM)).sum())

    print(f"\n=== {label} ===")
    print(f"  n={len(a)}   " + "  ".join(
        f"{k}={int((kinds == k).sum())}" for k in
        ("pass", "jam", "never closed", "grasped then failed")
        if (kinds == k).any()))
    print(f"  clearance mm     min {clear.min():7.2f}   p1 "
          f"{np.percentile(clear, 1):7.2f}   **p5 "
          f"{np.percentile(clear, 5):7.2f}**   median "
          f"{np.median(clear):7.2f}   max {clear.max():7.2f}")
    print(f"  lateral mm       median {np.median(lat):7.2f}   p95 "
          f"{np.percentile(lat, 95):7.2f}   max {lat.max():7.2f}")
    print(f"  cmd yaw err deg  median {np.median(yaw):7.2f}   p95 "
          f"{np.percentile(yaw, 95):7.2f}   max {yaw.max():7.2f}")
    print(f"  under {DANGER_MM:.0f} mm: {n_band} trials "
          f"({n_band / len(a) * 100:5.2f}%), of which {n_jam_band} jammed")
    if jam.any():
        print(f"  every jam's clearance: "
              f"{[round(v, 2) for v in sorted(clear[jam])]}")

    return {
        "label": label,
        "n": int(len(a)),
        "p5_mm": float(np.percentile(clear, 5)),
        "median_mm": float(np.median(clear)),
        "min_mm": float(clear.min()),
        "under_band": n_band,
        "jams": int(jam.sum()),
    }


def main():
    """
    Entry point.

    input:  none
    output: None
    """
    p = argparse.ArgumentParser()
    p.add_argument("runs", nargs="+", help="strict_*.json files")
    p.add_argument("--labels", nargs="*", default=None)
    args = p.parse_args()

    labels = args.labels or [Path(r).stem.replace("strict_", "")
                             for r in args.runs]
    if len(labels) != len(args.runs):
        raise SystemExit("--labels must match the number of runs")

    summaries = [s for s, in
                 ((report(r, label),) for r, label in
                  zip(args.runs, labels, strict=True)) if s]

    if len(summaries) > 1:
        print(f"\n  {'run':<34}{'p5 mm':>9}{'median':>9}{'min':>9}"
              f"{'<6mm':>7}{'jams':>6}")
        for s in summaries:
            print(f"  {s['label'][:34]:<34}{s['p5_mm']:>9.2f}"
                  f"{s['median_mm']:>9.2f}{s['min_mm']:>9.2f}"
                  f"{s['under_band']:>7}{s['jams']:>6}")
        print("\n  the demonstrator, for reference: p5 11.94, median 14.91, "
              "min 6.40, 0 under 6 mm, 0 jams")


if __name__ == "__main__":
    main()
