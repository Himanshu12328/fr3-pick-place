"""
Evaluates the Stage 2 RL teacher on the same protocol as every other number
in this project, and checks that it is solving the task rather than
exploiting the reward.

The success rate here is directly comparable to ACT's 83.5% and the
oracle's 100%, because the environment draws block placements from the same
generator in the same order given the same seed. Seed 51 in this script and
seed 51 in `region_analysis.py` are the same hundred placements.

The diagnostics matter as much as the rate. The training callback reported
100% while earning a return of 119 against the oracle's 104, and a policy
scoring above the reference is exactly the signal this project wrote down
as meaning the reward has been found rather than the task. Three checks
separate those cases:

  lift height      a block slid across the table never rises. A genuine
                   pick shows the block well above resting height
  grasp fraction   the share of episodes where the block was ever held,
                   which a shoving solution would fail
  final height     `check_success` already requires the block to be resting
                   rather than held, but reporting it makes a policy that
                   drops the block from height visible

Run:
    python -m src.scripts.eval_teacher --model best
    python -m src.scripts.eval_teacher --model latest --trials 100 --video 3
"""

import argparse
import json

import mujoco
import numpy as np
from stable_baselines3 import SAC

from src.config import LOG_DIR, OUTPUT_ROOT
from src.data.task import BLOCK_Z, check_success
from src.eval import rollout as R
from src.rl.env import FR3PickPlaceEnv

REPORT_SEEDS = [51, 52, 53, 54, 55, 56]

# Simulation steps to let a released block land and come to rest before the
# success criterion is applied for the last time.
SETTLE_STEPS = 60


def run_episode(env, model, renderer=None, frames=None):
    """
    Runs one episode and records what actually happened to the block.

    input:  env (FR3PickPlaceEnv), model (SAC),
            renderer (mujoco.Renderer or None), frames (list or None)
    output: dict of per-episode diagnostics
    """
    obs, _ = env.reset()
    max_z, held, steps, ok = BLOCK_Z, False, 0, False

    while True:
        action, _ = model.predict(obs, deterministic=True)
        obs, _, terminated, truncated, info = env.step(action)
        steps += 1
        ok = info["success"]
        held = held or info["holding"]
        max_z = max(max_z, float(env.data.xpos[env.block_id][2]))

        if renderer is not None and frames is not None:
            renderer.update_scene(env.data, camera="external")
            frames.append(renderer.render())

        if terminated or truncated:
            break

    # The episode ends the instant the place criterion is met, which can be
    # with the block still 9 mm up and falling. Keep simulating with the
    # gripper open to let it land and settle, then re-check. Two successes
    # in a hundred did not survive this, because the block was released near
    # the edge of the radius and rolled out of it after we stopped watching.
    for _ in range(SETTLE_STEPS):
        env.data.qfrc_applied[: R.N_ARM] = env.ctrl.compute_torque(env.data)
        env.data.qfrc_applied[R.FINGER_DOFS] = R.gripper_torque(env.data, 0.04)
        mujoco.mj_step(env.model, env.data)
    settled_ok, settled_dist = check_success(env.data, env.block_id)

    block = np.array(env.data.xpos[env.block_id])
    return {
        "settled_success": bool(settled_ok),
        "settled_distance": float(settled_dist),
        "success": bool(ok),
        "steps": steps,
        "distance": float(info["distance"]),
        "block_start": [float(v) for v in env.block_start],
        "lift_m": float(max_z - BLOCK_Z),
        "held": bool(held),
        "final_z": float(block[2]),
        # Whether the gripper actually opened and the block came to rest,
        # rather than ending held in the air inside check_success's 20 mm
        # tolerance. A teacher scored 99.8% doing the latter.
        "placed": bool(info.get("placed", False)),
        "released": bool(env.grip > 0.02),
    }


def main():
    """
    Entry point.

    input:  none
    output: None
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", default="rl_teacher_v1")
    parser.add_argument("--model", default="best")
    parser.add_argument("--trials", type=int, default=100)
    parser.add_argument("--seeds", type=int, nargs="+", default=REPORT_SEEDS)
    parser.add_argument("--video", type=int, default=0,
                        help="record this many rollouts to logs/ as mp4")
    args = parser.parse_args()

    path = OUTPUT_ROOT / args.run / f"{args.model}.zip"
    if not path.exists():
        raise SystemExit(f"no model at {path}")

    model = SAC.load(str(path), device="cuda")
    env = FR3PickPlaceEnv()
    print(f"{args.run}/{args.model}, {args.trials} trials x {len(args.seeds)} seeds\n")

    results, per_seed = [], []
    for seed in args.seeds:
        env.rng = np.random.default_rng(seed)
        rows = [run_episode(env, model) for _ in range(args.trials)]
        rate = float(np.mean([r["success"] for r in rows]))
        per_seed.append(rate)
        results += rows
        print(f"  seed {seed}: {rate * 100:5.1f}%")

    n = len(results)
    rate = float(np.mean([r["success"] for r in results]))
    sd = float(np.std(per_seed))
    steps = np.array([r["steps"] for r in results])
    lifts = np.array([r["lift_m"] for r in results])
    held = float(np.mean([r["held"] for r in results]))

    picked = float(np.mean([r["success"] and r["held"] for r in results]))

    print(f"\nTEACHER: {rate * 100:.1f}% +- {sd * 100:.1f} over {n} trials")
    print(f"  per seed: {' / '.join(f'{p * 100:.0f}' for p in per_seed)}")

    # check_success is satisfied by a block slid across the table, so the
    # raw rate alone cannot distinguish a pick from a shove. An earlier
    # teacher scored 99.0% here while never holding the block in two thirds
    # of its successes. Report both rates always; the gap between them is
    # the thing to watch.
    placed = float(np.mean([r["placed"] for r in results]))
    released = float(np.mean([r["released"] for r in results]))

    print(f"  picked (success with a genuine lift):        {picked * 100:5.1f}%  "
          f"a gap here means shoving")
    print(f"  PLACED (picked, released and settled):       {placed * 100:5.1f}%  "
          f"a gap here means it never let go")
    print(f"  gripper open at the end:                     {released * 100:5.1f}%")

    settled = float(np.mean([r["settled_success"] for r in results]))
    print(f"  STILL PLACED after a {SETTLE_STEPS}-step settle:        {settled * 100:5.1f}%  "
          f"this is the honest number")

    print("\nsanity checks against a reward-exploiting solution")
    print(f"  episodes where the block was held: {held * 100:.1f}%")
    print(f"  block lift above resting height:   mean {lifts.mean() * 1000:.0f} mm, "
          f"min {lifts.min() * 1000:.0f} mm, max {lifts.max() * 1000:.0f} mm")
    print(f"  steps to finish:                   mean {steps.mean():.0f}, "
          f"min {steps.min()}, max {steps.max()}")
    print("  oracle for comparison:             100%, 152 steps, "
          "lifts the block about 78 mm")

    near = [r for r in results if r["block_start"][0] < 0.53]
    far = [r for r in results if r["block_start"][0] >= 0.53]
    print(f"\n{'region':<14}{'success':>10}{'n':>6}")
    for name, rows in (("x <  0.53", near), ("x >= 0.53", far)):
        if rows:
            print(f"{name:<14}{np.mean([r['success'] for r in rows]) * 100:>9.1f}%"
                  f"{len(rows):>6}")

    fails = [r for r in results if not r["success"]]
    if fails:
        print(f"\n{len(fails)} failure(s):")
        for f in fails[:10]:
            print(f"  {f['steps']:4d} steps  dist {f['distance'] * 100:5.1f} cm  "
                  f"lift {f['lift_m'] * 1000:4.0f} mm  held {f['held']}")

    if args.video:
        import imageio
        import mujoco
        renderer = mujoco.Renderer(env.model, height=480, width=640)
        env.rng = np.random.default_rng(args.seeds[0])
        for i in range(args.video):
            frames = []
            r = run_episode(env, model, renderer, frames)
            out = LOG_DIR / f"teacher_{args.model}_{i:02d}_{'ok' if r['success'] else 'fail'}.mp4"
            imageio.mimsave(str(out), frames, fps=30)
            print(f"  wrote {out}  ({r['steps']} steps, lift {r['lift_m'] * 1000:.0f} mm)")
        renderer.close()

    LOG_DIR.mkdir(parents=True, exist_ok=True)
    out = LOG_DIR / f"teacher_eval_{args.run}_{args.model}.json"
    out.write_text(json.dumps({
        "run": args.run, "model": args.model, "trials": args.trials,
        "seeds": args.seeds, "per_seed": per_seed, "rate": rate, "n": n,
        "held_fraction": held, "picked": picked, "placed": placed, "released": released,
        "settled_success": settled, "mean_lift_m": float(lifts.mean()),
        "mean_steps": float(steps.mean()),
        "failures": fails,
    }, indent=2), encoding="utf-8")
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
