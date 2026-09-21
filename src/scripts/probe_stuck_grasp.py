"""
Asks one question about the trials the close veto could not rescue: when the
policy commands a close at hover height, is it *asking* for a descent that
the arm fails to deliver, or is it not asking?

The two answers call for completely different work, and nothing measured so
far distinguishes them.

    the commanded target descends and the tool does not follow
        -> a control failure. The impedance controller cannot place the
           tool where the policy asked, and the policy is blameless. No
           amount of replanning, vetoing or retraining fixes it; the gains,
           the nullspace posture or a joint limit does.

    the commanded target never descends
        -> a policy failure. The policy genuinely believes, from this
           observation, that the grasp happens here. Vetoing the close and
           forcing it to look again cannot help, because looking again
           returns the same decision — which is exactly what 29 refused
           closes in one episode look like.

The veto run settled that refusing the close is not sufficient: it fired on
all four air-close trials and all four still failed. So this replays those
trials and records, per step, the commanded target height against the
achieved tool height against the block, under both the untouched policy and
the veto.

Placements come from the same generator in the same order, so any trial
index is reproducible without running the ones before it.

Run:
    python -m src.scripts.probe_stuck_grasp --seed 61 --trials 18 23
    python -m src.scripts.probe_stuck_grasp --seed 92 --trials 11 21 --mode veto
"""

import argparse
import json

import mujoco
import numpy as np

from src.config import BLOCK_QPOS_ADR, BLOCK_QVEL_ADR, LOG_DIR
from src.data.task import BLOCK_Z, sample_block_pose, set_block_pose
from src.eval import rollout as R
from src.eval.strict import evaluate_trace


def replay(model, data, ctrl, policy, renderer, block_pos, block_quat,
           max_steps=900, inner_ensemble=None):
    """
    Runs one trial from a fixed placement, recording the commanded target
    and the achieved pose separately.

    input:  model, data, ctrl, policy (callable), renderer,
            block_pos (3,), block_quat (4,), max_steps (int),
            inner_ensemble (EnsemblePolicy or None) for member disagreement
    output: dict of per-step arrays plus the strict row
    """
    kid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "home")
    mujoco.mj_resetDataKeyframe(model, data, kid)
    set_block_pose(model, data, block_pos, block_quat,
                   BLOCK_QPOS_ADR, BLOCK_QVEL_ADR)
    mujoco.mj_forward(model, data)
    pos, quat = ctrl.current_pose(data)
    ctrl.set_target(pos, quat)

    if hasattr(policy, "reset"):
        policy.reset()

    block_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "block")
    keys = ("target_z", "ee_z", "block_z", "block_xy", "grip", "fingers",
            "tool_block_xy", "above_block", "target_above_block",
            "yaw_err", "cmd_yaw_err", "member_yaw_spread", "member_pos_spread",
            "q6", "q7", "rot_err_deg")
    t = {k: [] for k in keys}
    trace = {"block": [], "ee": [], "target": [], "grip": [], "block_vz": []}

    steps = 0
    for step in range(max_steps):
        obs = R.build_observation(data, renderer, R.CAMERAS, True)
        action = np.asarray(policy(obs), dtype=np.float64)

        target_pos = np.clip(action[:3], R.WORKSPACE_MIN, R.WORKSPACE_MAX)
        target_quat = action[3:7] / max(np.linalg.norm(action[3:7]), 1e-9)
        grip = float(np.clip(action[7], 0.0, 0.04))
        ctrl.set_target(target_pos, target_quat)

        for _ in range(R.STEPS_PER_ACTION):
            data.qfrc_applied[: R.N_ARM] = ctrl.compute_torque(data)
            data.qfrc_applied[R.FINGER_DOFS] = R.gripper_torque(data, grip)
            mujoco.mj_step(model, data)

        ee, ee_quat = ctrl.current_pose(data)
        block = np.array(data.xpos[block_id], dtype=np.float64)

        # Yaw misalignment between the gripper and the block, folded into a
        # quarter turn because a cube is symmetric every 90 degrees and the
        # fingers only have to line up with a pair of faces. Same
        # convention as diagnose_failures.py so the numbers are comparable.
        bq = np.array(data.xquat[block_id], dtype=np.float64)
        byaw = 2.0 * np.arctan2(bq[3], bq[0])
        gyaw = 2.0 * np.arctan2(ee_quat[2], ee_quat[1])
        err = (gyaw - byaw) % (np.pi / 2.0)
        t["yaw_err"].append(
            float(np.degrees(min(err, np.pi / 2.0 - err)))
        )

        # The same misalignment measured on the *commanded* quaternion.
        # If the command is aligned and the achieved pose is not, the arm
        # cannot rotate to where it was asked and the descent is blocked by
        # a wrist limit. If the command itself is misaligned, the policy is
        # asking for a grasp that cannot work.
        cerr = (2.0 * np.arctan2(target_quat[2], target_quat[1]) - byaw)
        cerr = cerr % (np.pi / 2.0)
        t["cmd_yaw_err"].append(
            float(np.degrees(min(cerr, np.pi / 2.0 - cerr)))
        )

        # The two wrist joints, to see whether either is against a stop.
        t["q6"].append(float(data.qpos[5]))
        t["q7"].append(float(data.qpos[6]))

        # Total orientation error the controller is fighting.
        dq = np.abs(np.dot(ee_quat, target_quat))
        t["rot_err_deg"].append(
            float(np.degrees(2.0 * np.arccos(min(dq, 1.0))))
        )

        # What the ensemble members disagreed about this step. A cube is
        # 90-degree symmetric, so two policies can choose equally valid
        # orientations a quarter turn apart; averaging those quaternions
        # lands the wrist 45 degrees from both.
        members = getattr(inner_ensemble, "last_member_actions", None)
        if members is not None and len(members) > 1:
            yaws = []
            for a in members:
                q = a[3:7] / max(np.linalg.norm(a[3:7]), 1e-9)
                yaws.append(2.0 * np.arctan2(q[2], q[1]))
            d = (yaws[0] - yaws[1]) % (np.pi / 2.0)
            t["member_yaw_spread"].append(
                float(np.degrees(min(d, np.pi / 2.0 - d)))
            )
            t["member_pos_spread"].append(
                float(np.linalg.norm(members[0][:3] - members[1][:3]) * 1000)
            )
        else:
            t["member_yaw_spread"].append(float("nan"))
            t["member_pos_spread"].append(float("nan"))

        t["target_z"].append(float(target_pos[2]))
        t["ee_z"].append(float(ee[2]))
        t["block_z"].append(float(block[2]))
        t["block_xy"].append([float(block[0]), float(block[1])])
        t["grip"].append(grip)
        t["fingers"].append(float(np.mean(data.qpos[R.FINGER_DOFS])))
        t["tool_block_xy"].append(float(np.linalg.norm(ee[:2] - block[:2])))
        t["above_block"].append(float(ee[2] - block[2]))
        t["target_above_block"].append(float(target_pos[2] - block[2]))

        trace["block"].append(block.astype(np.float32))
        trace["ee"].append(ee.astype(np.float32))
        trace["target"].append(target_pos.astype(np.float32))
        trace["grip"].append(grip)
        trace["block_vz"].append(float(data.qvel[BLOCK_QVEL_ADR + 2]))

        steps = step + 1
        from src.eval.strict import episode_complete
        if episode_complete(ee, block, grip):
            break

    t = {k: np.asarray(v) for k, v in t.items()}
    row = evaluate_trace(trace, steps, False, 0.0,
                         recovery_events=int(getattr(policy, "recovery_events", 0)))
    return t, row, steps


def summarise(t, steps, label):
    """
    Prints the answer to the question in the module docstring.

    input:  t (dict of arrays), steps (int), label (str)
    output: dict
    """
    # Only the pre-grasp part of the episode is relevant: once the block is
    # in the hand, height above it means nothing.
    closed = t["grip"] < 0.02
    lift = (t["block_z"] - BLOCK_Z) * 1000.0

    # The deepest the policy ever *asked* the tool to go, and the deepest it
    # actually got, both relative to the block's centre.
    tgt_min = float(t["target_above_block"].min() * 1000)
    ee_min = float(t["above_block"].min() * 1000)

    # How well the arm tracked the command, over the steps where the policy
    # was asking for something near the block.
    near = t["target_above_block"] < 0.02
    track = (t["above_block"] - t["target_above_block"])[near]
    track_mm = float(np.median(track) * 1000) if len(track) else float("nan")

    block_moved = float(
        np.linalg.norm(t["block_xy"][-1] - t["block_xy"][0]) * 1000
    )

    # Yaw misalignment at the deepest point of the descent, where it
    # decides whether the open fingers can straddle the block at all.
    deepest = int(np.argmin(t["above_block"]))
    stall = t["above_block"] > 0.010
    yaw_at_stall = (
        float(np.median(t["yaw_err"][stall & near])) if np.any(stall & near)
        else float("nan")
    )

    out = {
        "label": label,
        "steps": int(steps),
        "cmd_yaw_err_at_deepest_deg": float(t["cmd_yaw_err"][deepest]),
        "rot_err_at_deepest_deg": float(t["rot_err_deg"][deepest]),
        "q7_at_deepest": float(t["q7"][deepest]),
        "q6_at_deepest": float(t["q6"][deepest]),
        "q7_range": [float(t["q7"].min()), float(t["q7"].max())],
        "yaw_err_at_deepest_deg": float(t["yaw_err"][deepest]),
        "yaw_err_while_stalled_deg": yaw_at_stall,
        "member_yaw_spread_median_deg": float(
            np.nanmedian(t["member_yaw_spread"])),
        "member_pos_spread_median_mm": float(
            np.nanmedian(t["member_pos_spread"])),
        "closes": int(np.sum(np.diff(closed.astype(int)) == 1)),
        "target_min_above_block_mm": tgt_min,
        "ee_min_above_block_mm": ee_min,
        "tracking_error_mm": track_mm,
        "steps_target_below_5mm": int(np.sum(t["target_above_block"] < 0.005)),
        "steps_ee_below_5mm": int(np.sum(t["above_block"] < 0.005)),
        "min_tool_block_xy_mm": float(t["tool_block_xy"].min() * 1000),
        "max_lift_mm": float(lift.max()),
        "block_moved_mm": block_moved,
        "final_fingers_mm": float(t["fingers"][-1] * 1000),
    }

    print(f"  {label}")
    print(f"    steps {out['steps']}, closes commanded {out['closes']}")
    print(f"    deepest COMMANDED target, above block centre: "
          f"{tgt_min:+8.2f} mm   ({out['steps_target_below_5mm']} steps under +5)")
    print(f"    deepest ACHIEVED tool,    above block centre: "
          f"{ee_min:+8.2f} mm   ({out['steps_ee_below_5mm']} steps under +5)")
    print(f"    tracking error while commanding near the block: "
          f"{track_mm:+8.2f} mm")
    print(f"    closest lateral approach {out['min_tool_block_xy_mm']:6.2f} mm, "
          f"block pushed {block_moved:6.2f} mm, max lift {out['max_lift_mm']:6.2f} mm")
    print(f"    YAW error at deepest descent {out['yaw_err_at_deepest_deg']:5.2f} deg"
          f"   while stalled {out['yaw_err_while_stalled_deg']:5.2f} deg")
    print(f"    COMMANDED yaw error at deepest {out['cmd_yaw_err_at_deepest_deg']:5.2f} deg"
          f"   |  total orientation error {out['rot_err_at_deepest_deg']:5.2f} deg")
    print(f"    wrist q7 {out['q7_at_deepest']:+6.3f} rad (range "
          f"{out['q7_range'][0]:+.3f} to {out['q7_range'][1]:+.3f}), "
          f"q6 {out['q6_at_deepest']:+6.3f} rad")
    print(f"    members disagree: yaw {out['member_yaw_spread_median_deg']:5.2f} deg, "
          f"position {out['member_pos_spread_median_mm']:5.2f} mm")
    return out


def main():
    """
    Entry point.

    input:  none
    output: None
    """
    from src.scripts.eval_supervised import DEFAULT_MEMBERS, build_policy

    p = argparse.ArgumentParser()
    p.add_argument("--members", nargs="+", default=DEFAULT_MEMBERS)
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--trials", type=int, nargs="+", required=True)
    p.add_argument("--total", type=int, default=100)
    p.add_argument("--modes", nargs="+", default=["monitor", "veto"])
    p.add_argument("--max-steps", type=int, default=900)
    p.add_argument("--device", default="cuda")
    p.add_argument("--tag", default="stuck")
    args = p.parse_args()

    rng = np.random.default_rng(args.seed)
    placements = [sample_block_pose(rng) for _ in range(args.total)]

    model, data = R.setup_model()
    ctrl = R.ImpedanceController(
        model, data, kp_trans=R.KP_TRANS, kp_rot=R.KP_ROT, zeta=R.ZETA,
        kp_null=10.0, zeta_null=1.0, n_arm=R.N_ARM, verbose=False,
    )
    renderer = mujoco.Renderer(model, height=R.CAM_HEIGHT, width=R.CAM_WIDTH)

    report = []
    for mode in args.modes:
        policy, _ = build_policy(args.members, mode, device=args.device)
        if hasattr(policy, "bind"):
            policy.bind(model, data)
        inner = getattr(policy, "policy", None)

        print(f"\n=== mode {mode}, seed {args.seed} ===")
        for idx in args.trials:
            pos, quat = placements[idx]
            t, row, steps = replay(model, data, ctrl, policy, renderer,
                                   pos, quat, max_steps=args.max_steps,
                                   inner_ensemble=inner)
            out = summarise(t, steps, f"trial {idx}")
            out["mode"] = mode
            out["trial"] = int(idx)
            out["seed"] = int(args.seed)
            out["block_start"] = [float(v) for v in pos]
            out["strict"] = bool(row["strict_success"])
            out["missing_gates"] = [g for g, ok in row["gates"].items() if not ok]
            if hasattr(policy, "diagnostics"):
                d = policy.diagnostics()
                out["vetoes"] = d["vetoes"]
                out["recoveries"] = d["recoveries"]
            report.append(out)

    renderer.close()
    path = LOG_DIR / f"probe_{args.tag}_s{args.seed}.json"
    path.write_text(json.dumps(report, indent=1, default=float), encoding="utf-8")
    print(f"\nwrote {path}")


if __name__ == "__main__":
    main()
