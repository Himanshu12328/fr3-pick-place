"""
Sorts failed rollouts into named failure modes.

A success rate says how often a policy works. It never says how it fails,
and the failure mode is what points at the fix. A policy losing 16 points
to missed grasps needs better perception. One losing 16 points to dropping
the block halfway needs a firmer hold. One losing them to stopping 6 cm
short needs a small precision correction and nothing else. Those are three
different projects, and an aggregate number cannot tell them apart.

The classes below are deliberately mechanical. Each is decided from the
recorded block and gripper traces by a threshold that means something
physical, not by eyeballing videos. Reading the same rollout twice gives
the same label.

Timeout is reported alongside the class rather than as one of the classes,
because it is not a separate way to fail. A trial that never grasped and a
trial that dropped the block can both then run out of steps, and collapsing
those into one bucket would hide which of them happened.
"""

import numpy as np

from src.data.task import BLOCK_Z, PLACE_TARGET, SUCCESS_RADIUS

# The block is a 44 mm cube resting with its centre at BLOCK_Z. Counting it
# as lifted needs a margin comfortably above settling jitter but well below
# the transit height the demonstrations used, which is about 78 mm up.
LIFT_M = 0.020

# A block that never got picked up but ended this far from where it started
# was hit rather than missed. Those are different errors: one is a grasp
# that never happened, the other is a grasp attempt that fouled the block.
NUDGE_M = 0.030

# Ending inside this radius but outside SUCCESS_RADIUS is a near miss. The
# distinction matters because a near miss is a precision problem worth a
# bounded correction, while a larger error is a navigation problem.
NEAR_MISS_M = 0.10

CLASSES = [
    "success",
    "near_miss",
    "dropped_in_transit",
    "knocked_away",
    "never_grasped",
    "unclassified",
]


def classify(result):
    """
    Assigns one failure mode to a single trial.

    input:  result (dict) a run_trial return with a recorded trace
    output: str, one of CLASSES
    """
    if result["success"]:
        return "success"

    trace = result.get("trace")
    if trace is None or len(trace["block"]) == 0:
        return "unclassified"

    block = np.asarray(trace["block"])
    start = np.asarray(result["block_start"])
    final = block[-1]

    lifted = bool(np.max(block[:, 2]) > BLOCK_Z + LIFT_M)
    moved = float(np.linalg.norm(final[:2] - start[:2]))
    dist = float(np.linalg.norm(final[:2] - PLACE_TARGET[:2]))

    if not lifted:
        return "knocked_away" if moved > NUDGE_M else "never_grasped"

    # It was picked up. The only question left is where it ended up.
    if dist < NEAR_MISS_M:
        return "near_miss"
    return "dropped_in_transit"


def summarise(results):
    """
    Counts failure modes across a set of trials.

    input:  results (list of run_trial dicts)
    output: dict with counts, timeout count and total
    """
    counts = dict.fromkeys(CLASSES, 0)
    timeouts = 0

    for r in results:
        counts[classify(r)] += 1
        if r.get("timed_out"):
            timeouts += 1

    return {"counts": counts, "timeouts": timeouts, "n": len(results)}


def report(summary, results=None):
    """
    Prints the taxonomy table.

    input:  summary (dict) from summarise(), results (list or None) used
            for the distance detail on near misses
    output: None
    """
    n = summary["n"]
    counts = summary["counts"]
    failures = n - counts["success"]

    print(f"\n{'failure mode':<20}{'n':>6}{'of all':>9}{'of failures':>13}")
    print("-" * 48)
    for name in CLASSES:
        c = counts[name]
        if c == 0 and name != "success":
            continue
        share = f"{c / failures * 100:.1f}%" if failures and name != "success" else ""
        print(f"{name:<20}{c:>6}{c / n * 100:>8.1f}%{share:>13}")

    print(f"\n{failures} failures in {n} trials. "
          f"{summary['timeouts']} hit the step limit.")

    if results:
        near = [r for r in results if classify(r) == "near_miss"]
        if near:
            d = np.array([r["distance"] for r in near]) * 100
            print(f"near misses landed {d.min():.1f} to {d.max():.1f} cm out "
                  f"(median {np.median(d):.1f}), against a "
                  f"{SUCCESS_RADIUS * 100:.0f} cm success radius.")
