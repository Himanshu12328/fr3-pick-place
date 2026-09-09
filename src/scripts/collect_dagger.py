"""
Collects DAgger episodes: the vision student drives, the oracle says what
it should have done.

This is the direct antidote to the failure Stage 0 measured. 62% of ACT's
failures are grasps that never happened and 18% are blocks knocked aside,
which together are 80% of everything that goes wrong, and every one of them
happens in a state no demonstration contains. A human operator who rarely
botches an approach cannot show a policy how to recover from one. The
student can reach those states on its own, and the oracle can label them,
because it reads the true block pose and recomputes its waypoints from
wherever the arm actually is.

So the student controls the arm and the oracle is asked, at every single
step, what the correct action would have been. The student's observations
are paired with the oracle's answers. Nothing about the resulting episode is
a demonstration of the student's behaviour; it is a demonstration of the
correct behaviour on the student's own state distribution, which is exactly
what behaviour cloning never gets to see.

Episodes are written in the same raw directory format as teleoperated data,
so `rerender.py` and `to_lerobot.py` work on them unchanged, and a mixed
dataset of human and relabelled episodes needs no special handling.

Two things make this safe to mix with the recorded demonstrations. The
oracle's grasp pose, wrist yaw, hover height, transit height and release
pose were all extracted from those same 221 episodes, and its per-phase
speeds are the measured demonstration speeds. It scores 0.991 against the
demonstration trajectory profile, so its labels look like the data they will
sit beside rather than like a different operator.

Run:
    python -m src.scripts.collect_dagger --run act_v4 --checkpoint 020000 --episodes 50
    python -m src.scripts.collect_dagger --episodes 20 --out data/dagger_v1
"""

import argparse
import json
import shutil
import time

import imageio
import mujoco
import numpy as np

from src.config import (
    DATA_ROOT,
    OUTPUT_ROOT,
    RECORD_FPS,
    TASK_STRING,
    TRAIN_HEIGHT,
    TRAIN_WIDTH,
)
from src.data.task import check_success
from src.eval import rollout as R
from src.eval.oracle import HOME_XY as ORACLE_HOME_XY
from src.eval.oracle import RETREAT_Z as ORACLE_RETREAT_Z
from src.eval.oracle import ScriptedOracle
from src.eval.policy_wrapper import LeRobotPolicyAdapter, load_policy_and_processors
from src.rl.env import MAX_DELTA_M


def oracle_label(env_target, oracle):
    """
    Asks the oracle for the correct absolute target pose from here.

    Returned in the dataset's own 8-element layout, so the label is directly
    comparable to a recorded teleoperation action with no conversion.

    input:  env_target (array (3,)) unused, kept for symmetry,
            oracle (ScriptedOracle)
    output: numpy array of shape (8,) float32
    """
    return np.asarray(oracle(None), dtype=np.float32)


def block_pose(data, block_id):
    """
    Returns the block's true pose as (x, y, z, yaw).

    Yaw is folded into a quarter turn because the block is a cube and every
    90 degree rotation presents an identical pair of faces to the fingers.
    Predicting the raw yaw would ask a model to distinguish orientations
    that are indistinguishable in the image and irrelevant to the grasp.

    input:  data (MjData), block_id (int)
    output: numpy array of shape (4,) float32
    """
    pos = np.array(data.xpos[block_id], dtype=np.float32)
    quat = np.array(data.xquat[block_id])
    yaw = 2.0 * np.arctan2(quat[3], quat[0])
    yaw = (yaw + np.pi / 4.0) % (np.pi / 2.0) - np.pi / 4.0
    return np.array([pos[0], pos[1], pos[2], yaw], dtype=np.float32)


def write_episode(out_dir, index, states, actions, images, meta, blocks=None):
    """
    Writes one episode in the raw teleoperation format.

    input:  out_dir (Path), index (int), states (list), actions (list),
            images (dict of camera name to list of frames), meta (dict)
    output: None
    """
    ep = out_dir / f"episode_{index:04d}"
    ep.mkdir(parents=True, exist_ok=True)

    arrays = {
        "state": np.asarray(states, dtype=np.float32),
        "action": np.asarray(actions, dtype=np.float32),
        "timestamp": np.arange(len(states), dtype=np.float32) / RECORD_FPS,
    }
    # The true block pose per frame. Never an input: the student stays vision
    # only, and excluding block pose from its observation is what stops it
    # skipping perception. It is recorded as a possible auxiliary training
    # target, because Stage 0 says 62% of failures are grasps that never
    # happened with zero near misses, which is what failing to locate the
    # block looks like rather than failing to control the arm.
    if blocks is not None:
        arrays["block"] = np.asarray(blocks, dtype=np.float32)

    np.savez(ep / "data.npz", **arrays)
    for cam, frames in images.items():
        cam_dir = ep / cam
        cam_dir.mkdir(exist_ok=True)
        for i, frame in enumerate(frames):
            imageio.imwrite(cam_dir / f"{i:05d}.png", frame)

    (ep / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")


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
    parser.add_argument("--episodes", type=int, default=50)
    parser.add_argument("--seed", type=int, default=201)
    parser.add_argument("--out", default=None)
    parser.add_argument("--max-steps", type=int, default=R.MAX_POLICY_STEPS)
    parser.add_argument("--failures-only", action="store_true",
                        help="keep only episodes the student failed, which is "
                             "where the labels it has never seen live")
    parser.add_argument("--driver", choices=["student", "oracle"], default="student",
                        help="student drives and the oracle labels (DAgger), or "
                             "the oracle both drives and labels (plain "
                             "demonstrations at scale)")
    parser.add_argument("--jitter", type=float, default=1.0,
                        help="per-episode variation in the oracle's hover and "
                             "transit heights, grasp offset, speed and target "
                             "lead. 0 is the deterministic oracle, which "
                             "produces degenerate data when driving")
    args = parser.parse_args()

    out_dir = DATA_ROOT / (args.out or f"dagger_{args.run}_{args.checkpoint}")
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True)

    student = None
    if args.driver == "student":
        ckpt = (OUTPUT_ROOT / args.run / "checkpoints" / args.checkpoint
                / "pretrained_model")
        policy, pre, post = load_policy_and_processors(str(ckpt), args.type)
        student = LeRobotPolicyAdapter(policy, pre, post, task=TASK_STRING)

    model, data = R.setup_model()
    ctrl = R.ImpedanceController(
        model, data, kp_trans=R.KP_TRANS, kp_rot=R.KP_ROT, zeta=R.ZETA,
        kp_null=10.0, zeta_null=1.0, n_arm=R.N_ARM, verbose=False,
    )
    oracle = ScriptedOracle(jitter=args.jitter, seed=args.seed)
    oracle.bind(model, data)
    # TRAIN resolution, not COLLECT. Two independent reasons, and the
    # first one is the documented trap in this repo: the student's vision
    # backbone accepts any input size silently, but crop_shape is applied
    # in pixels, so feeding it 640x480 when it trained on 160x128 turns a
    # whole-scene crop into a small centre patch and the policy goes
    # effectively blind. Collecting at 640 here scored the student 0 out
    # of 3 against its true 83.5%.
    #
    # The second reason is that these episodes must never be re-rendered.
    # rerender.py reconstructs frames by replaying the recorded actions,
    # and the recorded actions here are the ORACLE's labels, not what the
    # student executed. Replaying them would visit entirely different
    # states from the ones the images show. So the images are written at
    # training resolution once and used as-is.
    renderer = mujoco.Renderer(model, height=TRAIN_HEIGHT, width=TRAIN_WIDTH)
    block_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "block")
    rng = np.random.default_rng(args.seed)

    who = (f"{args.run} @ {args.checkpoint}" if student is not None
           else f"oracle (jitter {args.jitter})")
    mode = ("reactive label()" if student is not None
            else "phase machine, demonstration speeds")
    print(f"{who} driving, oracle labelling via {mode}, "
          f"{args.episodes} episodes -> {out_dir}\n")

    kept, student_ok, start = 0, 0, time.perf_counter()
    for i in range(args.episodes):
        block_pos, block_quat = R.reset_trial(model, data, ctrl, rng)
        if student is not None:
            student.reset()
        oracle.reset()

        states, actions, images = [], [], {c: [] for c in R.CAMERAS}
        blocks = []
        success = False

        for _ in range(args.max_steps):
            obs = R.build_observation(data, renderer, R.CAMERAS, True)

            # Label first: what the oracle would do from this exact state.
            # It has to be asked before the student moves the arm, or the
            # label describes a state the student never saw.
            #
            # Which oracle depends on who is driving, and getting this wrong
            # cost this project a whole dataset.
            #
            # `label()` is reactive: it recomputes from the current world
            # state with no memory, which is what DAgger needs, because the
            # student reaches states the oracle's own trajectory never
            # visits. But it places the target a fixed 12 mm ahead of the
            # tool, so the tool moves at one speed for the whole episode.
            # Measured over the 800 episodes collected that way: approach
            # 2.28, carry 2.86, retreat 2.74 mm per step, all converging on
            # the same number, against demonstrations of 1.85, 3.27 and
            # 4.78. The demonstrations are deliberately slow to position,
            # quicker to carry and quickest to leave, and that shape was
            # absent from every label. The dataset scored 0.832 against the
            # demonstration profile with a 5th percentile of 0.628, under
            # the 0.85 that all 221 demonstrations clear. ACT imitated it
            # faithfully, which is why `demo_like` is its worst gate.
            #
            # `__call__` is the phase machine, rate limited to the measured
            # per-phase demonstration speed, and it scores 0.987. When the
            # oracle drives there is no student to be reactive for, so it is
            # simply the better demonstrator.
            label = (oracle.label() if student is not None
                     else np.asarray(oracle(None), dtype=np.float32))

            states.append(R.build_state(data))
            actions.append(label)
            blocks.append(block_pose(data, block_id))
            for cam in R.CAMERAS:
                images[cam].append(obs[f"observation.images.{cam}"])

            # Then drive. With the student at the wheel the states visited
            # are its own, which is the point of DAgger. With the oracle
            # driving this is plain demonstration collection, and the
            # label is simply the action that was executed.
            act = (np.asarray(student(obs), dtype=np.float64)
                   if student is not None else label.astype(np.float64))
            target = np.clip(act[:3], R.WORKSPACE_MIN, R.WORKSPACE_MAX)
            quat = act[3:7] / max(np.linalg.norm(act[3:7]), 1e-9)
            grip = float(np.clip(act[7], 0.0, 0.04))
            ctrl.set_target(target, quat)

            for _ in range(R.STEPS_PER_ACTION):
                data.qfrc_applied[: R.N_ARM] = ctrl.compute_torque(data)
                data.qfrc_applied[R.FINGER_DOFS] = R.gripper_torque(data, grip)
                mujoco.mj_step(model, data)

            success, _ = check_success(data, block_id)
            if success:
                # Both drivers keep going until the arm is clear and back at
                # home. Stopping at check_success would cut every collected
                # episode off before the retreat, which is a third of what
                # the recorded episodes contain and the phase the dataset is
                # least able to spare.
                #
                # The student path used to break here immediately, and the
                # composition gate caught what that did. The label is asked
                # for *before* the driver moves, so on the step where the
                # student's release makes the block land, the recorded label
                # was computed while the block was still held and therefore
                # says "gripper closed". Breaking right after meant **100%
                # of DAgger episodes contained no open-gripper label at
                # all**: `never_released` was 1.000 against 0.000 in the
                # demonstrations. A policy trained on that learns to carry
                # the block and never let go, which is precisely the failure
                # RL teacher v2 was rejected for.
                ee = np.array(data.site_xpos[oracle.tcp_id])
                home = float(np.linalg.norm(ee[:2] - np.array(ORACLE_HOME_XY)))
                if ee[2] >= ORACLE_RETREAT_Z - 0.01 and home <= 0.06:
                    break

        student_ok += int(success)
        if args.failures_only and success:
            print(f"  episode {i:3d}  student ok, discarded "
                  f"({len(states)} steps)")
            continue

        meta = {
            "episode_index": kept,
            "task": TASK_STRING,
            "n_frames": len(states),
            "fps": RECORD_FPS,
            "success": True,
            "source": args.driver,
            "labelled_by": "scripted_oracle",
            "student": (f"{args.run}/{args.checkpoint}"
                        if student is not None else None),
            "student_succeeded": bool(success),
            "block_start_pos": [float(v) for v in block_pos],
            "block_start_quat": [float(v) for v in block_quat],
        }
        write_episode(out_dir, kept, states, actions, images, meta, blocks)
        kept += 1
        who_ok = ("ok  " if success else "FAIL") if student is None else (
            "student ok  " if success else "student FAIL")
        print(f"  episode {i:3d}  {who_ok}  "
              f"{len(states)} steps  kept as {kept - 1:04d}")

    renderer.close()
    elapsed = time.perf_counter() - start

    info = {
        "task": TASK_STRING,
        "fps": RECORD_FPS,
        "cameras": R.CAMERAS,
        "image_size": [TRAIN_WIDTH, TRAIN_HEIGHT],
        "rerender": False,  # actions are oracle labels, not executed actions
        "source": args.driver,
        "jitter": args.jitter,
        "student": (f"{args.run}/{args.checkpoint}"
                    if args.driver == "student" else None),
        "episodes": kept,
        "student_success_rate": student_ok / max(args.episodes, 1),
        "max_delta_m": MAX_DELTA_M,
    }
    (out_dir / "dataset_info.json").write_text(json.dumps(info, indent=2),
                                               encoding="utf-8")

    print(f"\nstudent succeeded on {student_ok}/{args.episodes} "
          f"({student_ok / max(args.episodes, 1) * 100:.1f}%)")
    print(f"kept {kept} relabelled episodes in {elapsed / 60:.1f} min")
    print(f"wrote {out_dir}")


if __name__ == "__main__":
    main()
