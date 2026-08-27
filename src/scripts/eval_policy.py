"""
Evaluates a trained checkpoint with the rollout harness.

Before running a full evaluation, use --check to compare the policy's
prediction against a recorded action on a frame taken straight from the
training set. If those disagree wildly, the observation or action plumbing
is broken and a low success rate would tell you nothing about the policy.

Run:
    python src\\scripts\\eval_policy.py --run act_v1 --type act --checkpoint 020000 --check
    python src\\scripts\\eval_policy.py --run diffusion_v2 --checkpoint 020000 --trials 20 --video
"""

import argparse
import os

import numpy as np
from PIL import Image

from src.config import LOG_DIR, OUTPUT_ROOT, RAW_DATA_DIR
from src.eval.policy_wrapper import LeRobotPolicyAdapter, load_policy_and_processors
from src.eval.rollout import evaluate

OUTPUT_ROOT = OUTPUT_ROOT
RAW_DIR = RAW_DATA_DIR
VIDEO_ROOT = LOG_DIR / "rollout"

CAMERAS = ["external", "wrist_left", "wrist_right"]
TASK = "pick up the red block and place it on the green target"


def sanity_check(adapter, episode="episode_0000", n_frames=3):
    """
    Feeds recorded observations through the adapter and prints the
    predicted action beside the recorded one.

    The frames come from the training set, so a correctly wired policy
    should predict something close to what was recorded. Large
    disagreement, or predictions confined to the range minus one to plus
    one, indicate the action is still in normalised space and the
    postprocessor is not being applied. Predictions that barely change
    between frames indicate the policy is ignoring its observations
    entirely and emitting the dataset mean.

    input:  adapter (LeRobotPolicyAdapter), episode (str), n_frames (int)
    output: None, prints to stdout
    """
    ep_dir = os.path.join(RAW_DIR, episode)
    npz = np.load(os.path.join(ep_dir, "data.npz"))
    state, action = npz["state"], npz["action"]

    print(f"\nSanity check against {episode}\n")
    adapter.reset()

    for i in range(n_frames):
        obs = {"observation.state": state[i]}
        for cam in CAMERAS:
            path = os.path.join(ep_dir, cam, f"{i:05d}.png")
            obs[f"observation.images.{cam}"] = np.array(Image.open(path))

        pred = adapter(obs)
        rec = action[i]
        err = np.linalg.norm(pred[:3] - rec[:3]) * 1000

        print(f"  frame {i}")
        print(f"    predicted {np.round(pred, 4)}")
        print(f"    recorded  {np.round(rec, 4)}")
        print(f"    position error {err:.1f} mm\n")


def main():
    """
    Entry point.

    input:  none
    output: None
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--run", default="diffusion_v2", help="training run directory under outputs/"
    )
    parser.add_argument("--checkpoint", default="020000")
    parser.add_argument(
        "--type", default="diffusion", help="diffusion or act; must match the run"
    )
    parser.add_argument("--trials", type=int, default=20)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--video", action="store_true")
    parser.add_argument(
        "--check", action="store_true", help="run the sanity check and exit"
    )
    parser.add_argument(
        "--n-action-steps",
        type=int,
        default=None,
        help="override how many steps of each predicted chunk "
        "are executed before re-planning",
    )
    args = parser.parse_args()

    ckpt = os.path.join(
        OUTPUT_ROOT, args.run, "checkpoints", args.checkpoint, "pretrained_model"
    )
    if not os.path.isdir(ckpt):
        raise SystemExit(f"No checkpoint at {ckpt}")

    policy, pre, post = load_policy_and_processors(
        ckpt, args.type, n_action_steps=args.n_action_steps
    )
    adapter = LeRobotPolicyAdapter(policy, pre, post, task=TASK)

    if args.check:
        sanity_check(adapter)
        return

    video_dir = None
    if args.video:
        video_dir = os.path.join(VIDEO_ROOT, f"{args.run}_{args.checkpoint}")

    print(
        f"\nEvaluating {args.run} ({args.type}) @ step {args.checkpoint}, "
        f"{args.trials} trials\n"
    )

    evaluate(
        adapter,
        n_trials=args.trials,
        seed=args.seed,
        need_images=True,
        save_video_dir=video_dir,
    )


if __name__ == "__main__":
    main()
