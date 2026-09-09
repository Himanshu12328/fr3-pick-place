"""
Validates the RL environment by driving it with the scripted oracle.

This is the same argument `test_harness.py` makes. The environment is a
second implementation of the collection dynamics: it has its own action
integration, its own stepping loop and its own episode termination. If it
has drifted from the harness, then a teacher trained inside it learns to
solve a task that is not the one being measured, and every number that
follows is wrong in a way nothing raises.

The oracle solves the harness 600 times out of 600. Driving the environment
with the same oracle should give the same result. Anything less means the
environment diverged, and the difference has to be found before any
training starts rather than after.

It also reports the reward the oracle earns. That number is the reference
scale for the whole of Stage 2: a trained policy scoring well below it has
not solved the task, and one scoring far above it has found something in
the reward that the task did not intend.

Run:
    python -m src.scripts.test_rl_env --trials 20
"""

import argparse

import numpy as np

from src.eval.oracle import ScriptedOracle, quat_yaw, wrap
from src.rl.env import MAX_DELTA_M, MAX_DYAW_RAD, FR3PickPlaceEnv


def oracle_action(env, oracle):
    """
    Converts the oracle's absolute target into the environment's delta
    action.

    input:  env (FR3PickPlaceEnv), oracle (ScriptedOracle)
    output: numpy array of shape (5,)
    """
    want = oracle(None)

    dpos = (want[:3] - env.target_pos) / MAX_DELTA_M
    dyaw = wrap(quat_yaw(want[3:7]) - env.target_yaw) / MAX_DYAW_RAD
    grip = -1.0 if want[7] < 0.02 else 1.0

    return np.clip(np.concatenate([dpos, [dyaw], [grip]]), -1.0, 1.0)


def main():
    """
    Entry point.

    input:  none
    output: None
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--trials", type=int, default=20)
    parser.add_argument("--seed", type=int, default=41)
    args = parser.parse_args()

    env = FR3PickPlaceEnv(seed=args.seed)
    oracle = ScriptedOracle()
    oracle.bind(env.model, env.data)

    successes, returns, lengths = 0, [], []

    print(f"Driving the RL environment with the oracle, {args.trials} episodes:\n")
    for i in range(args.trials):
        env.reset()
        oracle.reset()
        total, steps, ok = 0.0, 0, False

        while True:
            _, r, terminated, truncated, info = env.step(oracle_action(env, oracle))
            total += r
            steps += 1
            ok = info["success"]
            if terminated or truncated:
                break

        successes += int(ok)
        returns.append(total)
        lengths.append(steps)
        print(f"  episode {i:3d}  {'ok  ' if ok else 'FAIL'}  {steps:4d} steps  "
              f"return {total:9.1f}  dist {info['distance'] * 100:5.1f} cm")

    rate = successes / args.trials
    print(f"\n{successes}/{args.trials} succeeded = {rate * 100:.1f}%")
    print(f"return: mean {np.mean(returns):.1f}, min {np.min(returns):.1f}, "
          f"max {np.max(returns):.1f}")
    print(f"length: mean {np.mean(lengths):.0f}, max {np.max(lengths)}")

    if rate < 1.0:
        print("\nENVIRONMENT DIVERGED. The oracle solves the harness 600/600 but "
              "not this environment. Find the difference before training.")
    else:
        print("\nEnvironment reproduces the harness under the oracle. "
              f"Reference return is {np.mean(returns):.0f}.")


if __name__ == "__main__":
    main()
