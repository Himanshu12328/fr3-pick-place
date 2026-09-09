"""
Builds the frozen oracle replay buffer once and writes it to disk.

Seeding used to happen inside the training process, which meant 20
SubprocVecEnv workers and 20 more local environments alive at the same time
before a single gradient step had run. Two consecutive training launches
died at exactly the same point in that phase, 500 of 2500 lockstep steps,
about 45 seconds in, with no error reported. Whatever the cause, the fix is
the same: do it once, in its own short-lived process, and let training load
the result.

That also makes the seed data inspectable and reusable. A buffer whose
contents were verified once is worth more than one rebuilt on every launch
and reported on by a progress message that describes what was sent rather
than what was kept.

Run:
    python -m src.scripts.build_demo_buffer --steps 2500 --envs 20
"""

import argparse

import gymnasium as gym
import numpy as np
from stable_baselines3.common.save_util import save_to_pkl

from src.config import DATA_ROOT
from src.rl import seed as SEED
from src.rl.buffer import make_demo_buffer
from src.rl.env import OBS_DIM


def main():
    """
    Entry point.

    input:  none
    output: None
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=2500,
                        help="lockstep steps; each writes one transition per env")
    parser.add_argument("--envs", type=int, default=20)
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    obs_space = gym.spaces.Box(-np.inf, np.inf, (OBS_DIM,), np.float32)
    act_space = gym.spaces.Box(-1.0, 1.0, (5,), np.float32)
    transitions = args.steps * args.envs

    demo = make_demo_buffer(obs_space, act_space, args.envs, transitions, "cpu")
    holder = type("H", (), {"replay_buffer": demo})()

    print(f"collecting {args.steps} lockstep steps across {args.envs} envs "
          f"({transitions:,} transitions)")
    stats = SEED.fill(holder, args.envs, args.steps)

    # Report what the buffer kept, not what the collector sent. The
    # difference between those two is exactly how a demo buffer sized a
    # factor of twenty too small went unnoticed for a whole run.
    kept = demo.size() * demo.n_envs
    print(f"\ncollected {stats['transitions']:,} transitions from "
          f"{stats['episodes']} oracle episodes, "
          f"{stats['retreated']} completed the retreat")
    print(f"buffer kept {kept:,} transitions in {demo.size()} positions")
    if kept < stats["transitions"]:
        print(f"WARNING: buffer wrapped, {stats['transitions'] - kept:,} "
              f"transitions were discarded")

    out = args.out or (DATA_ROOT / "oracle_demo_buffer.pkl")
    save_to_pkl(str(out), demo)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
