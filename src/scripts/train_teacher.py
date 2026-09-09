"""
Trains the Stage 2 privileged RL teacher with SAC.

The teacher sees the block pose and no images. That is what makes it
affordable: with nothing to render, 20 worker processes reach 4,753 policy
steps per second on this machine, so a five million step run is about
18 minutes of wall clock. The expensive half of the simulator is simply not
running.

What this policy is for is Stage 3. Behaviour cloning fails here by
compounding error, which the Stage 0 taxonomy localised precisely: 62% of
ACT's failures are grasps that never happened, and not one failure in 300
trials was a near miss. A teacher that can recover from a botched approach
is the thing a student needs to be shown, and a demonstration set collected
by a human who rarely botches approaches cannot show it.

Evaluation here deliberately ignores reward. Success is `check_success`,
unmodified, on held-out seeds, reported as a rate. This project has
measured loss and reward curves diverging from success repeatedly, most
starkly when diffusion's loss fell smoothly while its success rate swung
between 10 and 80 percent. Reward is for the optimiser. The success rate is
for us.

Run:
    python -m src.scripts.train_teacher --steps 2000000 --workers 16
    python -m src.scripts.train_teacher --steps 50000 --workers 8 --eval-every 10000
"""

import argparse
import json
import time

import numpy as np
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import BaseCallback, CallbackList
from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor

from src.config import OUTPUT_ROOT
from src.rl import seed as SEED
from src.rl.buffer import MixedReplayBuffer, make_demo_buffer
from src.rl.env import FR3PickPlaceEnv, make_env
from src.rl.pretrain import evaluate_actor, pretrain_actor

# Seeds held apart from the 51 to 56 used for headline reporting, so the
# teacher is never selected on the placements it will be judged on. Same
# rule the checkpoint sweeps follow.
EVAL_SEEDS = [71, 72]


def evaluate_success(model, n_episodes=40, seeds=EVAL_SEEDS, deterministic=True):
    """
    Runs the policy and reports the real success rate.

    input:  model (SAC), n_episodes (int) per seed, seeds (list of int),
            deterministic (bool)
    output: (rate float, mean_return float, mean_steps float)
    """
    env = FR3PickPlaceEnv()
    successes, returns, lengths = 0, [], []

    for seed in seeds:
        env.rng = np.random.default_rng(seed)
        for _ in range(n_episodes):
            obs, _ = env.reset()
            total, steps, ok = 0.0, 0, False
            while True:
                action, _ = model.predict(obs, deterministic=deterministic)
                obs, r, terminated, truncated, info = env.step(action)
                total += r
                steps += 1
                ok = info["success"]
                if terminated or truncated:
                    break
            successes += int(ok)
            returns.append(total)
            lengths.append(steps)

    n = n_episodes * len(seeds)
    return successes / n, float(np.mean(returns)), float(np.mean(lengths))


class SuccessEval(BaseCallback):
    """
    Periodically measures the success rate and keeps the best checkpoint.

    Selecting on reward would be a mistake here for the same reason
    selecting on loss was a mistake earlier in this project. The reward is a
    shaped proxy; the success rate is the thing.
    """

    def __init__(self, every, episodes, out_dir, verbose=0):
        """
        input:  every (int) timesteps between evaluations,
                episodes (int) per seed, out_dir (Path), verbose (int)
        output: SuccessEval instance
        """
        super().__init__(verbose)
        self.every = every
        self.episodes = episodes
        self.out_dir = out_dir
        self.best = -1.0
        self.last = 0
        self.history = []

    def _on_step(self):
        """
        input:  none
        output: bool, always True to continue training
        """
        if self.num_timesteps - self.last < self.every:
            return True
        self.last = self.num_timesteps

        rate, ret, steps = evaluate_success(self.model, self.episodes)
        self.history.append((self.num_timesteps, rate))
        flag = ""

        if rate > self.best:
            self.best = rate
            self.model.save(str(self.out_dir / "best"))
            flag = "  <- best, saved"

        print(f"  {self.num_timesteps:>9,} steps   success {rate * 100:5.1f}%   "
              f"return {ret:8.1f}   len {steps:5.0f}{flag}", flush=True)
        return True


class ResumableCheckpoint(BaseCallback):
    """
    Periodically saves the policy together with its replay buffer.

    Saving the policy alone is close to useless for resuming an off-policy
    run. SAC's competence lives as much in the buffer as in the weights, so
    reloading the network against an empty buffer throws away most of the
    compute that produced it and re-enters the warmup phase. The buffer has
    to travel with the checkpoint.

    A single pair of files is overwritten rather than one per interval. The
    buffer is a few hundred megabytes, so keeping every checkpoint would
    fill the disk long before the run finished.
    """

    def __init__(self, every, out_dir, verbose=0):
        """
        input:  every (int) timesteps between checkpoints, out_dir (Path),
                verbose (int)
        output: ResumableCheckpoint instance
        """
        super().__init__(verbose)
        self.every = every
        self.out_dir = out_dir
        self.last = 0

    def _save(self):
        """
        Writes the policy, the replay buffer and the step counter.

        input:  none
        output: None
        """
        self.model.save(str(self.out_dir / "latest"))
        self.model.save_replay_buffer(str(self.out_dir / "latest_buffer"))
        (self.out_dir / "progress.json").write_text(
            json.dumps({"num_timesteps": int(self.num_timesteps)}), encoding="utf-8"
        )

    def _on_step(self):
        """
        input:  none
        output: bool, always True to continue training
        """
        if self.num_timesteps - self.last >= self.every:
            self.last = self.num_timesteps
            self._save()
            print(f"  {self.num_timesteps:>9,} steps   checkpoint saved", flush=True)
        return True

    def _on_training_end(self):
        """
        input:  none
        output: None
        """
        self._save()


def main():
    """
    Entry point.

    input:  none
    output: None
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=2_000_000)
    parser.add_argument("--workers", type=int, default=20)
    parser.add_argument("--run", default="rl_teacher_v1")
    parser.add_argument("--eval-every", type=int, default=50_000)
    parser.add_argument("--eval-episodes", type=int, default=20)
    parser.add_argument("--gradient-steps", type=int, default=10)
    parser.add_argument("--learning-starts", type=int, default=10_000)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--buffer-size", type=int, default=1_000_000)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--gamma", type=float, default=0.997)
    parser.add_argument("--checkpoint-every", type=int, default=100_000)
    parser.add_argument("--seed-oracle", type=int, default=0,
                        help="lockstep steps of scripted oracle data to preload "
                             "into the replay buffer before training")
    parser.add_argument("--demo-buffer", default=None,
                        help="path to a demo buffer built by "
                             "src.scripts.build_demo_buffer")
    parser.add_argument("--bc-only", action="store_true",
                        help="behaviour-clone the actor and stop, skipping RL. "
                             "RL fine-tuning destroyed the cloned policy in "
                             "every attempt: it fell to 0%% within 50k steps "
                             "and never recovered across 600k")
    parser.add_argument("--bc-epochs", type=int, default=0,
                        help="behaviour-clone the actor on the oracle data "
                             "for this many epochs before training")
    parser.add_argument("--demo-ratio", type=float, default=0.5,
                        help="share of each gradient batch drawn from the "
                             "frozen oracle buffer")
    parser.add_argument("--resume", action="store_true",
                        help="continue from latest.zip and its replay buffer")
    args = parser.parse_args()

    out_dir = OUTPUT_ROOT / args.run
    out_dir.mkdir(parents=True, exist_ok=True)

    venv = VecMonitor(
        SubprocVecEnv([make_env(seed=1000 + i) for i in range(args.workers)])
    )

    latest = out_dir / "latest.zip"
    buffer = out_dir / "latest_buffer.pkl"
    resuming = args.resume and latest.exists()

    if resuming:
        model = SAC.load(str(latest), env=venv, device="cuda")
        done = 0
        if buffer.exists():
            model.load_replay_buffer(str(buffer))
            # size() counts buffer positions, and each position holds one
            # transition per parallel environment. Reporting positions here
            # reads as though most of the buffer were lost on resume.
            filled = model.replay_buffer.size() * model.n_envs
            print(f"resumed with {filled:,} transitions in buffer")
        else:
            print("WARNING: no replay buffer found, resuming with an empty one. "
                  "SAC will re-enter its warmup phase.")
        progress = out_dir / "progress.json"
        if progress.exists():
            done = json.loads(progress.read_text(encoding="utf-8"))["num_timesteps"]
        print(f"resuming {args.run} from {done:,} steps")
    else:
        model = SAC(
            "MlpPolicy",
            venv,
            learning_rate=args.lr,
            buffer_size=args.buffer_size,
            batch_size=args.batch_size,
            learning_starts=args.learning_starts,
            tau=0.005,
            gamma=args.gamma,
            # One gradient step per several environment steps. The environment
            # produces transitions far faster than the GPU consumes them here,
            # so matching them one to one would leave the workers idle.
            train_freq=(1, "step"),
            gradient_steps=args.gradient_steps,
            policy_kwargs={"net_arch": [512, 512]},
            replay_buffer_class=MixedReplayBuffer,
            device="cuda",
            verbose=0,
        )

    if args.demo_buffer and not resuming:
        # Loaded from disk rather than collected here. Seeding in this
        # process meant 20 SubprocVecEnv workers and 20 more local
        # environments alive simultaneously, and two launches died at
        # exactly the same point in that phase with no error reported.
        from stable_baselines3.common.save_util import load_from_pkl
        demo = load_from_pkl(args.demo_buffer)
        demo.device = model.device
        kept = demo.size() * demo.n_envs
        model.replay_buffer.attach_demo(demo, ratio=args.demo_ratio)
        print(f"loaded {kept:,} oracle transitions from {args.demo_buffer}; "
              f"{args.demo_ratio:.0%} of every batch will come from them")

        if args.bc_epochs:
            # The actor is what fails on this task, not the critic. RLPD
            # already teaches the critic what a finished episode is worth;
            # nothing teaches the actor the 280 step sequence that earns it.
            print(f"behaviour cloning the actor for {args.bc_epochs} epochs")
            stats = pretrain_actor(model, demo, epochs=args.bc_epochs)
            print(f"  final mse {stats['loss']:.4f} over "
                  f"{stats['samples']:,} oracle transitions")
            before = evaluate_actor(model, FR3PickPlaceEnv(), episodes=10)
            print(f"  actor before any RL: success {before['success']:.0%}, "
                  f"placed {before['placed']:.0%}, "
                  f"retreated {before['retreated']:.0%}, "
                  f"{before['mean_steps']:.0f} steps")

            if args.bc_only:
                model.save(str(out_dir / "best"))
                model.save(str(out_dir / "latest"))
                venv.close()
                print(f"bc-only, saved to {out_dir / 'best.zip'}")
                return
    elif args.seed_oracle and not resuming:
        print(f"filling a frozen oracle buffer with {args.seed_oracle} lockstep "
              f"steps ({args.seed_oracle * args.workers:,} transitions)")
        demo = make_demo_buffer(
            model.observation_space, model.action_space,
            args.workers, args.seed_oracle * args.workers, model.device,
        )
        holder = type("H", (), {"replay_buffer": demo})()
        stats = SEED.fill(holder, args.workers, args.seed_oracle)
        model.replay_buffer.attach_demo(demo, ratio=args.demo_ratio)
        print(f"  {stats['transitions']:,} sent, "
              f"{demo.size() * demo.n_envs:,} kept from "
              f"{stats['episodes']} oracle episodes")
    print()

    print(f"{args.run}: SAC, {args.workers} workers, {args.steps:,} steps")
    print(f"evaluating every {args.eval_every:,} steps on seeds {EVAL_SEEDS}, "
          f"{args.eval_episodes} episodes each")
    print("oracle reference: 100% success, return about 131\n")

    start = time.perf_counter()
    cb = SuccessEval(args.eval_every, args.eval_episodes, out_dir)
    ckpt = ResumableCheckpoint(args.checkpoint_every, out_dir)
    model.learn(
        total_timesteps=args.steps,
        callback=CallbackList([cb, ckpt]),
        reset_num_timesteps=not resuming,
        progress_bar=False,
    )
    elapsed = time.perf_counter() - start

    model.save(str(out_dir / "final"))
    venv.close()

    print(f"\ndone in {elapsed / 60:.1f} min "
          f"({args.steps / max(elapsed, 1e-9):.0f} env steps/s)")
    if cb.best < 0:
        print("no evaluation ran, so no best checkpoint was selected")
    else:
        print(f"best success {cb.best * 100:.1f}%, saved to {out_dir / 'best.zip'}")


if __name__ == "__main__":
    main()
