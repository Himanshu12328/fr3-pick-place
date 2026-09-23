"""
The demonstration trajectory profile, and how to score a policy against it.

Stage 2 produced a teacher that solves the task 98.8% of the time and looks
nothing like the demonstrations doing it: 73 steps against their 295, a
31 mm carry against their 80 mm, one sustained fast blur against their
deliberate slow-fast-slow-fast profile, and no retreat at all.

Success rate cannot see any of that, which is the same lesson Stage 2 kept
teaching. So the target is written down here as numbers, measured from all
221 recorded episodes, and every policy gets scored against them.

Everything below was measured by reading the recorded actions. The action
is the absolute target pose, so target speed is the step to step change in
its position, and phase boundaries are the frames where the binary gripper
changes state.

    approach   start to gripper close    105 steps, p10 75,  p90 145
    carry      close to gripper open     131 steps, p10 96,  p90 166
    retreat    open to end                59 steps, p10 50,  p90 71
    total                                295 steps, p10 227, p90 367

    speed      approach                 1.85 mm/step, p10 1.33, p90 2.49
               carry                    3.27 mm/step, p10 2.67, p90 3.81
               retreat                  4.78 mm/step, p10 4.17, p90 5.17
               overall                  3.06 mm/step, p10 2.46, p90 3.64

    heights    grasp                    0.4190
               max while carrying       0.4993   (80 mm above the grasp)
               max after release        0.5181   (99 mm above the grasp)

    jerk       mean step to step change in speed, 0.98 mm/step^2

The profile across normalised episode time reads 1.9, 1.8, 1.9, 2.8, 4.7,
4.1, 2.4, 1.6, 4.7, 4.8. Slow to position, quick to lift and transport,
slow again to descend and release, quick to leave. That shape is the thing
to reproduce, not just the average.
"""

import numpy as np

# Phase durations in policy steps, as (mean, p10, p90).
APPROACH_STEPS = (105.0, 75.0, 145.0)
CARRY_STEPS = (131.0, 96.0, 166.0)
RETREAT_STEPS = (59.0, 50.0, 71.0)
TOTAL_STEPS = (295.0, 227.0, 367.0)

# Target speed in mm per policy step, as (mean, p10, p90).
APPROACH_SPEED = (1.85, 1.33, 2.49)
CARRY_SPEED = (3.27, 2.67, 3.81)
RETREAT_SPEED = (4.78, 4.17, 5.17)
OVERALL_SPEED = (3.06, 2.46, 3.64)

GRASP_Z = 0.4190
CARRY_Z_MAX = 0.4993          # 80 mm above the grasp
RETREAT_Z_MAX = 0.5181        # 99 mm above the grasp
JERK = 0.98                   # mm per step squared

# What the reward aims the policy at, one per phase.
SPEED_TARGET = {
    "approach": APPROACH_SPEED[0],
    "carry": CARRY_SPEED[0],
    "retreat": RETREAT_SPEED[0],
}

# Heights the environment requires, chosen a little inside the measured
# demonstration values so they are reachable rather than marginal.
TRANSIT_Z = 0.480             # lift to here before carrying, 61 mm up
RETREAT_Z = 0.500             # rise to here after releasing, 81 mm up

# The scene's home keyframe. The demonstrations do not stop once the block
# is down, they bring the arm back out here, 282 mm of travel over 59 steps.
HOME_XY = (0.5545, 0.0)
RETREAT_HOME_TOL = 0.06       # how near home counts as retreated

# A place is only a place if the gripper opened low and the block was
# already at rest. Releasing from 95 mm up and letting the block fall
# satisfies a height-and-open test, and that is what the first version of
# this check permitted.
PLACE_EE_Z = 0.45             # tool must have descended this far to release
PLACE_SETTLED_M = 0.005       # block within 5 mm of resting, not 10
PLACE_MAX_VZ = 0.05           # block moving slower than 50 mm/s


def phases(gripper):
    """
    Finds the grasp and release frames from the binary gripper trace.

    input:  gripper (array (T,)) commanded finger width per step
    output: (grasp int or None, release int or None)
    """
    g = np.asarray(gripper)
    closed = np.where(g < 0.02)[0]
    if len(closed) == 0:
        return None, None

    grasp = int(closed[0])
    opened = np.where(g[grasp:] > 0.02)[0]
    release = grasp + int(opened[0]) if len(opened) else None
    return grasp, release


def describe(targets, gripper, grasp=None, release=None):
    """
    Reduces one episode to the quantities the demonstration profile is
    written in.

    `phases` takes the first close and the first open after it, which is
    right for a demonstration and wrong for an episode that closed on
    nothing, reopened and grasped properly on the second attempt. On such an
    episode the first close is the failed attempt, so `carry` measures the
    handful of steps before the fingers reopened and `retreat` measures
    everything from there to the end. The shape score then collapses, and it
    collapses for the wrong reason: it reports the recovery as a malformed
    trajectory rather than scoring the trajectory that actually carried the
    block.

    `strict.evaluate_trace` already computes the phase frames the robust
    way, from the last moment the block was genuinely in a closed gripper,
    precisely because seven of its nine gates had the same problem. Passing
    them in lets one definition serve both. Left at None the behaviour is
    unchanged, so every number this module has ever reported still holds.

    input:  targets (array (T,3)) commanded target position per step,
            gripper (array (T,)) commanded finger width per step,
            grasp (int or None) grasp frame, defaults to the first close,
            release (int or None) release frame, defaults to the first open
    output: dict of measurements, or None if the episode never grasped
    """
    targets = np.asarray(targets, dtype=float)
    if grasp is None:
        grasp, release = phases(gripper)
    if grasp is None:
        return None

    n = len(targets)
    speed = np.linalg.norm(np.diff(targets, axis=0), axis=1) * 1000.0
    end = release if release is not None else n - 1

    def seg(a, b):
        s = speed[a:b]
        return float(s.mean()) if len(s) else 0.0

    return {
        "total": n,
        "approach": grasp,
        "carry": end - grasp,
        "retreat": n - end,
        "speed_overall": float(speed.mean()) if len(speed) else 0.0,
        "speed_approach": seg(0, grasp),
        "speed_carry": seg(grasp, end),
        "speed_retreat": seg(end, n - 1),
        "jerk": float(np.abs(np.diff(speed)).mean()) if len(speed) > 1 else 0.0,
        "carry_z_max": float(targets[grasp:end, 2].max()) if end > grasp else GRASP_Z,
        "retreat_z_max": float(targets[end:, 2].max()) if end < n else GRASP_Z,
        "released": release is not None,
    }


def band_score(value, low, high):
    """
    Scores a measurement as 1.0 inside the demonstration band and decaying
    outside it.

    A band rather than a point, because the demonstrations themselves span
    one. Matching the mean exactly is not the goal; landing where a human
    plausibly landed is.

    input:  value (float), low (float), high (float)
    output: float in 0 to 1
    """
    if low <= value <= high:
        return 1.0
    width = max(high - low, 1e-9)
    off = (low - value) if value < low else (value - high)
    return float(np.exp(-((off / width) ** 2)))


METRICS = [
    ("total", TOTAL_STEPS, "episode length"),
    ("approach", APPROACH_STEPS, "approach steps"),
    ("carry", CARRY_STEPS, "carry steps"),
    ("retreat", RETREAT_STEPS, "retreat steps"),
    ("speed_approach", APPROACH_SPEED, "approach speed"),
    ("speed_carry", CARRY_SPEED, "carry speed"),
    ("speed_retreat", RETREAT_SPEED, "retreat speed"),
]


def score(measurements):
    """
    Scores one episode against the demonstration profile.

    input:  measurements (dict) from describe()
    output: (total float in 0 to 1, per_metric dict)
    """
    if measurements is None:
        return 0.0, {}

    per = {k: band_score(measurements[k], lo, hi)
           for k, (_, lo, hi), _ in METRICS}

    # Heights are one sided: lifting higher than a human is not a fault,
    # skimming the table is. Same for the retreat.
    per["carry_z"] = band_score(measurements["carry_z_max"], CARRY_Z_MAX - 0.02, 10.0)
    per["retreat_z"] = band_score(measurements["retreat_z_max"], RETREAT_Z_MAX - 0.03, 10.0)
    per["jerk"] = band_score(measurements["jerk"], 0.0, JERK * 2.0)

    return float(np.mean(list(per.values()))), per
