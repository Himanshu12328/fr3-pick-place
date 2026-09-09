"""
Scores a policy on whether it performed a pick and place, not on where the
block ended up.

`check_success` asks two things: the block is within 5 cm of the target and
it is resting. Stage 2 established, five separate times, that this is not a
description of the task. A block shoved across the table satisfies it. A
block dropped from 95 mm satisfies it. A block hovering inside the 20 mm
resting tolerance while still gripped satisfies it. Teachers scoring 99.0%,
99.8% and 100% were each rejected on video after passing it.

Those lessons were written into `eval_teacher.py` and never applied to the
vision student, so every ACT number in this project, 83.5% and 86.8%
included, was measured with the criterion that certifies the exact
behaviours the project already knows to reject.

This module applies them. The sequence being scored is the one the
demonstrations perform and the one specified for this task:

    approach -> grasp -> lift -> carry -> descend -> release -> lift off -> home

Nine gates, every one of which exists because something failed it:

    delivered      the block ends within 5 cm of the target and resting
    lifted         the block came off the table, so this was not a shove
    carried        the fingers were shut around it while it was up
    released_low   the tool had descended and the block was already at rest
                   and nearly motionless when the fingers opened, so this
                   was a place and not a drop
    lifted_off     the tool rose clear of the block before translating
    home           the arm returned to where it started
    undisturbed    the block did not move after being let go
    in_time        the episode ran as long as a demonstration ran
    demo_like      the trajectory has the shape of a demonstration, scored
                   per phase against the measured profile

A trial passes only if all nine hold. That is the number this project now
reports.

The bands are not invented here. `src/rl/reference.py` holds them as
measured constants from all 221 recorded episodes, and the same constants
already score the oracle at 0.991.
"""

import mujoco
import numpy as np

from src.config import BLOCK_QVEL_ADR
from src.data.task import BLOCK_Z, PLACE_TARGET, check_success
from src.rl import reference as REF

# Policy rate, needed to turn a per step height change into a speed. The
# release criterion is written in metres per second because that is how it
# was measured when the dropping exploit was found.
REF_HZ = 30.0

# Block geometry, from the scene. Half edge 22 mm, so the top face of a
# resting block sits 22 mm above its centre.
BLOCK_HALF = 0.022

# A genuine lift. The demonstrations raise the block about 80 mm and the
# oracle 78. A shove never leaves the table and the worst reward exploit in
# this project averaged 8 mm, so 40 mm separates the two cases with room to
# spare on both sides.
LIFT_MIN_M = 0.040

# How near the tool the block has to be to count as carried rather than
# coincidentally airborne. The block is 44 mm across and the fingers close
# on its centre.
CARRY_NEAR_M = 0.060

# The tool must rise at least the height of the block before it translates,
# or the open fingers drag the block it has just set down. Measured at
# 17.4 mm of disturbance, up to 38.4 mm, before the oracle gained its
# two-stage retreat.
LIFT_OFF_M = 0.030
LATERAL_EPS_M = 0.020

# How far the block may move between release and the end of the settle.
# The oracle moves it 0.06 mm. Anything approaching a centimetre means the
# arm knocked it on the way out.
DISTURB_MAX_M = 0.005

# Simulation steps of settling before the final check. Two successes in a
# hundred did not survive this when it was added to the teacher evaluation,
# because the block was released near the edge of the radius and rolled out
# of it afterwards.
SETTLE_STEPS = 60

# Episode length band, and the trajectory-shape threshold.
#
# Both are calibrated so that every one of the 221 recorded demonstrations
# passes them. That is the property a gate has to have: it must require
# behaviour the data actually contains, not behaviour tighter than any
# human ever produced.
#
# The first version of this file used the p10 to p90 band, 227 to 367 steps.
# That was wrong in a way worth recording. By construction 20% of the
# demonstrations fall outside their own p10-p90, so the gate rejected
# behaviour the operator actually performed. Measured on ACT it threw out
# eight episodes, two of which scored 0.998 against the demonstration
# profile at 218 and 221 steps. The gate was rejecting near-perfect
# trajectories for being nine steps quicker than a percentile.
#
# Measured over all 221 episodes:
#   episode length   min 195, max 452
#   trajectory score min 0.856, and 100% score at or above 0.85
TIME_BAND = (195, 452)
DEMO_LIKE_MIN = 0.85

# Widening the length band alone would have let the wrong thing through: an
# oracle driven at three times demonstration speed finishes in 205 steps,
# which is inside 195 to 452. What separates it is the shape of the
# trajectory rather than its duration, and `reference.score` already
# measures that per phase. It rates the rushed oracle 0.594 against the real
# oracle's 0.988. The two gates together reject it; neither does alone.
GATES = [
    "delivered",
    "lifted",
    "carried",
    "released_low",
    "lifted_off",
    "home",
    "undisturbed",
    "in_time",
    "demo_like",
]


def evaluate_trace(trace, total_steps, settled_ok, settled_dist):
    """
    Applies the nine gates to one recorded episode.

    input:  trace (dict) with block, ee, target and grip arrays of length T,
            total_steps (int) steps before the episode was declared over,
            settled_ok (bool) check_success after the settle,
            settled_dist (float) horizontal distance after the settle
    output: dict of gate booleans, the measurements behind them, and pass
    """
    block = np.asarray(trace["block"], dtype=float)
    ee = np.asarray(trace["ee"], dtype=float)
    target = np.asarray(trace["target"], dtype=float)
    grip = np.asarray(trace["grip"], dtype=float)

    # Where the carry actually ended, rather than the first time the fingers
    # happened to open.
    #
    # `reference.phases` takes the first close and the first open after it,
    # which is right for a clean demonstration and wrong for a policy that
    # grasps, misses, opens, and grasps again. One RL teacher dropped and
    # re-grasped in 72 of 100 episodes. On such an episode the first open is
    # the failed grasp, so `released_low`, `lifted_off`, `home` and
    # `undisturbed` would all be measured at a frame in the middle of the
    # approach, tens of centimetres and hundreds of steps from the actual
    # place.
    #
    # The release is the frame after the last one where the block was
    # genuinely in a closed gripper. That is robust to chatter and to
    # re-grasping, because it takes the final time the block was held.
    closed = grip < 0.02
    in_hand = np.linalg.norm(block - ee, axis=1) < CARRY_NEAR_M
    held = closed & in_hand
    held_idx = np.where(held)[0]

    if len(held_idx) == 0:
        grasp = release = None
    else:
        grasp = int(held_idx[0])
        last_held = int(held_idx[-1])
        # If the block was still held on the final frame, nothing was ever
        # let go of, and there is no release to score.
        release = last_held + 1 if last_held + 1 < len(grip) else None

    m = {
        "grasp_frame": grasp,
        "release_frame": release,
        "total_steps": int(total_steps),
        "settled_distance": float(settled_dist),
    }

    # 1. Delivered. The original criterion, applied after the block has been
    # given time to land and stop rolling rather than at the instant it
    # first entered the radius.
    delivered = bool(settled_ok)

    # 2. Lifted. Peak block height over the whole episode, against its
    # resting height. This is the shove test.
    lift = float(block[:, 2].max() - BLOCK_Z)
    m["lift_m"] = lift
    lifted = lift >= LIFT_MIN_M

    # 3. Carried. There has to be a frame where the fingers were commanded
    # shut, the block was off the table, and the block was in the hand. A
    # block that is airborne because it was flicked fails the third test.
    airborne = block[:, 2] > BLOCK_Z + 0.020
    carried = bool(np.any(airborne & held))
    m["carry_frames"] = int(np.sum(airborne & held))

    # Everything below needs a release to talk about. A policy that never
    # opened its fingers fails the remaining gates by construction, which is
    # correct: it never placed anything.
    released_low = lifted_off = home = undisturbed = False

    if release is not None and grasp is not None:
        # 4. Released low, and gently. Three conditions, because the first
        # two alone were satisfied by a teacher that opened its fingers
        # 95 mm up and let the block fall 303 mm/s into the target.
        ee_z = float(ee[release, 2])
        block_rest_err = float(abs(block[release, 2] - BLOCK_Z))
        # The simulator's own block velocity, not a finite difference of the
        # recorded height. At 30 Hz one step of numerical settling is about
        # 2 mm, which the difference reports as 60 mm/s against a 50 mm/s
        # limit, so the first version of this gate failed four episodes for
        # solver noise while their tools sat 25 mm below the release ceiling.
        vz = (
            float(abs(np.asarray(trace["block_vz"])[release]))
            if "block_vz" in trace
            else (
                float(abs(block[release, 2] - block[release - 1, 2])) * REF_HZ
                if release > 0
                else 0.0
            )
        )

        m["release_ee_z"] = ee_z
        m["release_block_rest_err"] = block_rest_err
        m["release_block_vz"] = vz

        released_low = bool(
            ee_z <= REF.PLACE_EE_Z
            and block_rest_err <= REF.PLACE_SETTLED_M
            and vz <= REF.PLACE_MAX_VZ
        )

        # 5. Lifted off before translating. Walk forward from the release
        # frame to the first moment the tool has moved more than 2 cm
        # sideways, and require that it had already risen a block height by
        # then. This is the fix that took block disturbance from 17.4 mm to
        # 0.06 mm on the oracle.
        rel_xy = ee[release, :2]
        rel_z = ee[release, 2]
        after = ee[release:]
        lateral = np.linalg.norm(after[:, :2] - rel_xy, axis=1)
        moved = np.where(lateral > LATERAL_EPS_M)[0]

        if len(moved) == 0:
            # Never translated at all. It cannot have dragged the block,
            # but it also never went home, which gate 6 will catch.
            rise_before_move = float(after[:, 2].max() - rel_z)
        else:
            rise_before_move = float(after[: moved[0] + 1, 2].max() - rel_z)

        m["rise_before_move_m"] = rise_before_move
        lifted_off = rise_before_move >= LIFT_OFF_M

        # 6. Went home. The demonstrations do not stop once the block is
        # down, they bring the arm back out: 282 mm of travel over 59 steps.
        end_xy = float(np.linalg.norm(ee[-1, :2] - np.array(REF.HOME_XY)))
        peak_z_after = float(ee[release:, 2].max())
        m["end_home_dist_m"] = end_xy
        m["retreat_z_max"] = peak_z_after
        home = bool(end_xy <= REF.RETREAT_HOME_TOL and peak_z_after >= REF.RETREAT_Z)

        # 7. Undisturbed. Where the block was when it was let go against
        # where it finished, horizontally. The vertical component is left
        # out because a block released 2 mm up and settling is not a
        # disturbance.
        drift = float(np.linalg.norm(block[-1, :2] - block[release, :2]))
        m["post_release_drift_m"] = drift
        undisturbed = drift <= DISTURB_MAX_M

    # 8. In time. The demonstrations run 227 to 367 steps. A policy that
    # blurs through in 73 is not doing what they did, and one that runs the
    # clock out to 600 has not finished.
    in_time = bool(TIME_BAND[0] <= total_steps <= TIME_BAND[1])

    # The trajectory profile score, reported rather than gated. The gates
    # above test the structure of the episode; this scores its shape
    # against the measured demonstration bands, and the oracle sits at
    # 0.991 on it.
    described = REF.describe(target, grip)
    traj_score, traj_per = REF.score(described)
    m["trajectory_score"] = float(traj_score)
    m["trajectory_per_metric"] = {k: float(v) for k, v in traj_per.items()}

    gates = {
        "delivered": delivered,
        "lifted": bool(lifted),
        "carried": carried,
        "released_low": bool(released_low),
        "lifted_off": bool(lifted_off),
        "home": bool(home),
        "undisturbed": bool(undisturbed),
        "in_time": in_time,
        "demo_like": bool(traj_score >= DEMO_LIKE_MIN),
    }

    out = dict(m)
    out["gates"] = gates
    out["strict_success"] = all(gates.values())
    out["loose_success"] = delivered
    return out


def episode_complete(ee, block, grip):
    """
    Says whether the full sequence has finished, so the episode can stop.

    Where the episode ends decides what its length is, and length is one of
    the eight gates, so this cannot be "the first frame the block is near
    the target". That is the early break the old harness used, and it makes
    every episode look shorter than it was and hides everything that
    happens afterwards.

    The task is over when the block is delivered and resting, the fingers
    are open, and the arm has climbed and returned to where it started.

    input:  ee (array (3,)), block (array (3,)), grip (float)
    output: bool
    """
    horizontal = float(np.linalg.norm(block[:2] - PLACE_TARGET[:2]))
    resting = abs(float(block[2]) - BLOCK_Z) < 0.010
    open_fingers = float(grip) > 0.02
    # Exactly the height the `home` gate requires, not 10 mm under it.
    # With the slack, an episode could be declared complete at 0.492 and
    # then fail `home` for never having reached 0.500 — a failure the
    # harness created by stopping the policy one step early rather than one
    # the policy earned. The termination condition must never end an episode
    # in a state the gates then reject.
    high = float(ee[2]) >= REF.RETREAT_Z
    at_home = float(np.linalg.norm(ee[:2] - np.array(REF.HOME_XY))) <= REF.RETREAT_HOME_TOL

    return bool(
        horizontal < 0.05 and resting and open_fingers and high and at_home
    )


def summarise(rows):
    """
    Aggregates per-episode results into the numbers that get reported.

    The per-gate pass rates are the diagnosis. A strict rate well under the
    loose rate is expected; which gate is eating the difference is what says
    what to fix next.

    input:  rows (list of dict) from evaluate_trace
    output: dict
    """
    n = len(rows)
    if n == 0:
        return {}

    out = {
        "n": n,
        "strict": float(np.mean([r["strict_success"] for r in rows])),
        "loose": float(np.mean([r["loose_success"] for r in rows])),
        "gates": {g: float(np.mean([r["gates"][g] for r in rows])) for g in GATES},
        "mean_lift_mm": float(np.mean([r["lift_m"] for r in rows]) * 1000),
        "mean_steps": float(np.mean([r["total_steps"] for r in rows])),
        "mean_trajectory_score": float(np.mean([r["trajectory_score"] for r in rows])),
    }

    drift = [r["post_release_drift_m"] for r in rows if "post_release_drift_m" in r]
    out["mean_drift_mm"] = float(np.mean(drift) * 1000) if drift else None
    out["max_drift_mm"] = float(np.max(drift) * 1000) if drift else None

    return out


def format_report(summary, title=""):
    """
    Renders a summary as the block of text this project prints everywhere.

    input:  summary (dict) from summarise, title (str)
    output: str
    """
    if not summary:
        return "no trials"

    g = summary["gates"]
    lines = []
    if title:
        lines.append(title)
    lines.append(
        f"  STRICT pick and place:  {summary['strict'] * 100:5.1f}%   "
        f"({summary['n']} trials)"
    )
    lines.append(
        f"  loose check_success:    {summary['loose'] * 100:5.1f}%   "
        f"the old number, for comparison"
    )
    lines.append("")
    lines.append("  gate                     pass    fails because")
    reasons = {
        "delivered": "block not at the target, or not resting",
        "lifted": "shoved rather than picked",
        "carried": "block airborne but not in the hand",
        "released_low": "dropped from height, not placed",
        "lifted_off": "dragged the block on the way out",
        "home": "never retreated to the start pose",
        "undisturbed": "knocked the block after letting go",
        "in_time": "longer or shorter than any demonstration",
        "demo_like": "trajectory shape unlike the demonstrations",
    }
    for name in GATES:
        lines.append(f"  {name:<22} {g[name] * 100:6.1f}%   {reasons[name]}")

    lines.append("")
    lines.append(
        f"  lift {summary['mean_lift_mm']:.0f} mm (demos 80)   "
        f"steps {summary['mean_steps']:.0f} (demos 295, band "
        f"{TIME_BAND[0]:.0f}-{TIME_BAND[1]:.0f})   "
        f"trajectory {summary['mean_trajectory_score']:.3f} (oracle 0.991)"
    )
    if summary.get("mean_drift_mm") is not None:
        lines.append(
            f"  block moved after release: mean "
            f"{summary['mean_drift_mm']:.2f} mm, max "
            f"{summary['max_drift_mm']:.2f} mm (oracle 0.06)"
        )
    return "\n".join(lines)


def run_strict_trial(model, data, ctrl, policy, renderer, rng,
                     max_steps=600, need_images=True, record_frames=False):
    """
    Runs one episode to completion and records everything the gates need.

    Three things differ from `rollout.run_trial`, and each of them is the
    reason a number measured there could not be trusted as a pick and place.

    It does not stop when `check_success` first fires. That break made the
    recorded episode length meaningless and hid the entire retreat, which is
    half of what the task was specified to be.

    It stops when the whole sequence is done, block placed and arm home, so
    episode length means what it means in the demonstrations.

    It records the commanded target and gripper alongside the tool and block
    positions, because the demonstration profile in `reference.py` is
    written in terms of the commanded target and cannot be scored without
    it.

    input:  model, data, ctrl, policy (callable), renderer, rng,
            max_steps (int), need_images (bool), record_frames (bool)
    output: dict with the trace, the settle result and the frames
    """
    from src.eval import rollout as R

    block_pos, _ = R.reset_trial(model, data, ctrl, rng)
    block_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "block")

    trace = {"block": [], "ee": [], "target": [], "grip": [], "block_vz": []}
    frames = []
    grip = 0.04
    steps = 0

    for step in range(max_steps):
        obs = R.build_observation(data, renderer, R.CAMERAS, need_images)
        action = np.asarray(policy(obs), dtype=np.float64)

        target_pos = np.clip(action[:3], R.WORKSPACE_MIN, R.WORKSPACE_MAX)
        target_quat = action[3:7] / max(np.linalg.norm(action[3:7]), 1e-9)
        grip = float(np.clip(action[7], 0.0, 0.04))

        ctrl.set_target(target_pos, target_quat)

        for _ in range(R.STEPS_PER_ACTION):
            data.qfrc_applied[: R.N_ARM] = ctrl.compute_torque(data)
            data.qfrc_applied[R.FINGER_DOFS] = R.gripper_torque(data, grip)
            mujoco.mj_step(model, data)

        ee = ctrl.current_pose(data)[0]
        block = np.array(data.xpos[block_id], dtype=np.float64)

        trace["block"].append(block.astype(np.float32))
        trace["ee"].append(ee.astype(np.float32))
        trace["target"].append(target_pos.astype(np.float32))
        trace["grip"].append(grip)
        trace["block_vz"].append(float(data.qvel[BLOCK_QVEL_ADR + 2]))

        if record_frames and renderer is not None:
            renderer.update_scene(data, camera="external")
            frames.append(renderer.render())

        steps = step + 1
        if episode_complete(ee, block, grip):
            break

    # Let the block land and stop rolling before the last check. The last
    # commanded gripper is held rather than forced open, so a policy that
    # ended still gripping the block keeps gripping it and fails honestly
    # instead of having the harness release it.
    for _ in range(SETTLE_STEPS):
        data.qfrc_applied[: R.N_ARM] = ctrl.compute_torque(data)
        data.qfrc_applied[R.FINGER_DOFS] = R.gripper_torque(data, grip)
        mujoco.mj_step(model, data)

        if record_frames and renderer is not None:
            renderer.update_scene(data, camera="external")
            frames.append(renderer.render())

    settled_ok, settled_dist = check_success(data, block_id)

    # The settle moves the block, and the disturbance gate is about where it
    # came to rest, so the trace has to carry the settled position as its
    # last entry.
    trace["block"].append(np.array(data.xpos[block_id], dtype=np.float32))
    trace["ee"].append(ctrl.current_pose(data)[0].astype(np.float32))
    trace["target"].append(trace["target"][-1])
    trace["grip"].append(grip)
    trace["block_vz"].append(float(data.qvel[BLOCK_QVEL_ADR + 2]))

    row = evaluate_trace(trace, steps, settled_ok, settled_dist)
    row["block_start"] = [float(v) for v in block_pos]
    row["timed_out"] = steps >= max_steps
    row["frames"] = frames
    row["trace"] = {k: np.asarray(v) for k, v in trace.items()}
    return row


def evaluate_strict(policy, n_trials=100, seed=51, need_images=True,
                    verbose=True, video_dir=None, video_n=0):
    """
    Runs n_trials episodes under the strict gates and reports the result.

    input:  policy (callable), n_trials (int), seed (int),
            need_images (bool), verbose (bool),
            video_dir (str or None), video_n (int) episodes to record
    output: (summary dict, rows list)
    """
    from src.eval import rollout as R

    model, data = R.setup_model()
    ctrl = R.ImpedanceController(
        model, data, kp_trans=R.KP_TRANS, kp_rot=R.KP_ROT, zeta=R.ZETA,
        kp_null=10.0, zeta_null=1.0, n_arm=R.N_ARM, verbose=False,
    )
    rng = np.random.default_rng(seed)

    if hasattr(policy, "bind"):
        policy.bind(model, data)

    renderer = None
    if need_images or video_dir:
        renderer = mujoco.Renderer(model, height=R.CAM_HEIGHT, width=R.CAM_WIDTH)

    rows = []
    for i in range(n_trials):
        if hasattr(policy, "reset"):
            policy.reset()

        row = run_strict_trial(
            model, data, ctrl, policy, renderer, rng,
            need_images=need_images,
            record_frames=bool(video_dir) and i < video_n,
        )

        if video_dir and row["frames"]:
            _save_video(row["frames"], video_dir, i, row["strict_success"])
        row.pop("frames", None)
        row.pop("trace", None)
        rows.append(row)

        if verbose:
            failed = [g for g, ok in row["gates"].items() if not ok]
            mark = "PASS" if row["strict_success"] else "fail"
            note = "" if not failed else "  missing: " + ",".join(failed)
            print(f"  trial {i:3d}  {mark}  {row['total_steps']:4d} steps{note}")

    if renderer is not None:
        renderer.close()

    summary = summarise(rows)
    if verbose:
        print()
        print(format_report(summary))
    return summary, rows


def _save_video(frames, out_dir, index, ok):
    """
    Writes an episode as mp4, including the retreat and the settle.

    The old harness stopped recording the moment the block touched down,
    which is precisely where the interesting failures live.

    input:  frames (list), out_dir (str), index (int), ok (bool)
    output: None
    """
    import os

    import imageio

    os.makedirs(out_dir, exist_ok=True)
    tag = "pass" if ok else "fail"
    imageio.mimsave(
        os.path.join(out_dir, f"strict_{index:03d}_{tag}.mp4"), frames, fps=30
    )


def render_size_for_checkpoint(checkpoint_dir):
    """
    Reads the image resolution the checkpoint was trained at.

    The vision backbone is fully convolutional and accepts any input size
    without complaint, so a policy trained at 160x128 and evaluated at
    640x480 does not raise, it just goes blind. This project measured that
    once already: 60% success became 0.4%.

    Two copies of a constant that must agree is a latent version of the same
    bug. Rather than trusting config.py to have been set correctly for
    whichever checkpoint is being run, ask the checkpoint.

    input:  checkpoint_dir (str or Path) the pretrained_model folder
    output: (width int, height int)
    """
    import json
    from pathlib import Path

    cfg = json.loads(
        (Path(checkpoint_dir) / "config.json").read_text(encoding="utf-8")
    )
    shapes = [
        v["shape"]
        for k, v in cfg.get("input_features", {}).items()
        if k.startswith("observation.images.")
    ]
    if not shapes:
        raise SystemExit(f"{checkpoint_dir} declares no image inputs")
    if len({tuple(s) for s in shapes}) != 1:
        raise SystemExit(f"{checkpoint_dir} mixes image resolutions: {shapes}")

    _, height, width = shapes[0]
    return int(width), int(height)


def use_checkpoint_render_size(checkpoint_dir):
    """
    Points the rollout harness at the checkpoint's own resolution.

    input:  checkpoint_dir (str or Path)
    output: (width int, height int)
    """
    from src.eval import rollout as R

    width, height = render_size_for_checkpoint(checkpoint_dir)
    if (width, height) != (R.CAM_WIDTH, R.CAM_HEIGHT):
        print(
            f"  render size {R.CAM_WIDTH}x{R.CAM_HEIGHT} -> "
            f"{width}x{height}, taken from the checkpoint"
        )
    R.CAM_WIDTH, R.CAM_HEIGHT = width, height
    return width, height
