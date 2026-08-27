"""
Feeds a recorded episode's observations through a trained policy and
compares the predicted action trajectory against the recorded one.

This is teacher forcing: the policy sees observations that its own actions
did not produce, so the result measures how well it fits the
demonstrations, not how it behaves in closed loop. A policy that tracks
well here can still fail in rollout through compounding error. But a
policy that fails here has a problem that closed-loop evaluation would
only obscure.

The per-dimension and per-phase breakdown is the point. An aggregate
position error hides whether the policy is fine during transit and wrong
at the grasp, and hides gripper timing entirely.

Run:
    python src\\scripts\\replay_predict.py --run act_v2 --checkpoint 035000 --type act
    python src\\scripts\\replay_predict.py --run act_v2 --checkpoint 035000 --type act --episodes 5
"""

import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from src.config import LOG_DIR, OUTPUT_ROOT, SMALL_DATA_DIR
from src.eval.policy_wrapper import LeRobotPolicyAdapter, load_policy_and_processors

OUTPUT_ROOT = OUTPUT_ROOT
RAW_DIR = SMALL_DATA_DIR
LOG_DIR = LOG_DIR

CAMERAS = ["external", "wrist_left", "wrist_right"]
TASK = "pick up the red block and place it on the green target"
AXES = ["x", "y", "z"]


def load_episode(ep_dir):
    """
    Loads one episode's arrays and image paths.

    Images are not decoded here. An episode holds hundreds of frames per
    camera and decoding them all up front wastes memory when they are
    consumed one at a time.

    input:  ep_dir (str)
    output: (state, action) arrays
    """
    npz = np.load(os.path.join(ep_dir, "data.npz"))
    return npz["state"], npz["action"]


def predict_trajectory(adapter, ep_dir, n_frames):
    """
    Runs the policy over every frame of a recorded episode in order.

    The adapter is reset once at the start, so chunked policies build and
    consume their action queues exactly as they would in a rollout. That
    matters: querying frames independently would hide any error introduced
    by executing a stale chunk.

    input:  adapter (LeRobotPolicyAdapter), ep_dir (str), n_frames (int)
    output: numpy array of shape (n_frames, 8), predicted actions
    """
    state, _ = load_episode(ep_dir)
    adapter.reset()

    preds = []
    for i in range(n_frames):
        obs = {"observation.state": state[i]}
        for cam in CAMERAS:
            path = os.path.join(ep_dir, cam, f"{i:05d}.png")
            obs[f"observation.images.{cam}"] = np.array(Image.open(path))
        preds.append(adapter(obs))

    return np.array(preds)


def phase_bounds(action):
    """
    Locates the grasp and release frames from the recorded gripper signal.

    input:  action (array (N, 8)) recorded actions
    output: (grasp, release) frame indices, or (None, None) if the gripper
            never closed
    """
    closed = action[:, 7] < 0.01
    if not closed.any():
        return None, None
    grasp = int(np.argmax(closed))
    release = int(len(closed) - np.argmax(closed[::-1]) - 1)
    return grasp, release


def report_errors(pred, rec, ep_name):
    """
    Prints per-dimension and per-phase error statistics.

    Splitting by phase matters because the phases have very different
    tolerances. Being 2 cm off during transit is harmless; being 2 cm off
    at the grasp means missing the block entirely.

    input:  pred (array (N, 8)), rec (array (N, 8)), ep_name (str)
    output: dict of summary statistics
    """
    pos_err = np.linalg.norm(pred[:, :3] - rec[:, :3], axis=1) * 1000
    grip_err = np.abs(pred[:, 7] - rec[:, 7])

    grasp, release = phase_bounds(rec)

    print(f"\n{ep_name}  ({len(rec)} frames)")
    print(
        f"  position error   mean {pos_err.mean():6.1f} mm   "
        f"median {np.median(pos_err):6.1f}   max {pos_err.max():6.1f}"
    )

    for i, ax in enumerate(AXES):
        e = (pred[:, i] - rec[:, i]) * 1000
        print(f"    {ax}: bias {e.mean():+7.1f} mm   spread {e.std():6.1f} mm")

    if grasp is not None:
        phases = {
            "approach": (0, grasp),
            "grasp": (max(0, grasp - 10), min(len(rec), grasp + 10)),
            "carry": (grasp, release),
            "release": (max(0, release - 10), min(len(rec), release + 10)),
        }
        print("  by phase:")
        for name, (a, b) in phases.items():
            if b > a:
                print(f"    {name:<10} {pos_err[a:b].mean():6.1f} mm")

        # Gripper timing: when does the prediction first close, versus
        # when the demonstration did.
        pred_closed = pred[:, 7] < 0.02
        if pred_closed.any():
            pred_grasp = int(np.argmax(pred_closed))
            lag = pred_grasp - grasp
            print(
                f"  gripper closes at frame {pred_grasp}, "
                f"recorded {grasp}, lag {lag:+d} frames "
                f"({lag / 30:+.2f} s)"
            )
        else:
            print("  gripper never closes in the prediction")

    print(f"  gripper error    mean {grip_err.mean() * 1000:6.2f} mm")

    return {
        "pos_mean": float(pos_err.mean()),
        "pos_max": float(pos_err.max()),
        "grasp": grasp,
    }


def plot_comparison(pred, rec, ep_name, path):
    """
    Plots predicted against recorded trajectories for each position axis
    and the gripper.

    input:  pred (array (N, 8)), rec (array (N, 8)), ep_name (str),
            path (str)
    output: None
    """
    fig, axes = plt.subplots(4, 1, figsize=(11, 11), sharex=True)
    t = np.arange(len(rec))
    grasp, release = phase_bounds(rec)

    for i, ax_name in enumerate(AXES):
        ax = axes[i]
        ax.plot(t, rec[:, i] * 1000, lw=2, label="recorded")
        ax.plot(t, pred[:, i] * 1000, lw=1.5, alpha=0.8, label="predicted")
        ax.set_ylabel(f"{ax_name} (mm)")
        ax.grid(alpha=0.3)
        if grasp is not None:
            ax.axvline(grasp, color="k", ls=":", alpha=0.5)
            ax.axvline(release, color="k", ls=":", alpha=0.5)
        if i == 0:
            ax.legend()

    ax = axes[3]
    ax.plot(t, rec[:, 7] * 1000, lw=2, label="recorded")
    ax.plot(t, pred[:, 7] * 1000, lw=1.5, alpha=0.8, label="predicted")
    ax.set_ylabel("gripper (mm)")
    ax.set_xlabel("frame")
    ax.grid(alpha=0.3)

    fig.suptitle(
        f"{ep_name}: predicted vs recorded actions "
        f"(dotted lines mark grasp and release)"
    )

    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved: {path}")


def main():
    """
    Entry point.

    input:  none
    output: None
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", default="act_v2")
    parser.add_argument("--checkpoint", default="035000")
    parser.add_argument("--type", default="act")
    parser.add_argument("--episodes", type=int, default=3)
    parser.add_argument("--start", type=int, default=0)
    args = parser.parse_args()

    ckpt = os.path.join(
        OUTPUT_ROOT, args.run, "checkpoints", args.checkpoint, "pretrained_model"
    )
    policy, pre, post = load_policy_and_processors(ckpt, args.type)
    adapter = LeRobotPolicyAdapter(policy, pre, post, task=TASK)

    names = sorted(d for d in os.listdir(RAW_DIR) if d.startswith("episode_"))
    chosen = names[args.start : args.start + args.episodes]

    for ep_name in chosen:
        ep_dir = os.path.join(RAW_DIR, ep_name)
        state, rec = load_episode(ep_dir)
        pred = predict_trajectory(adapter, ep_dir, len(rec))

        report_errors(pred, rec, ep_name)
        plot_comparison(
            pred,
            rec,
            ep_name,
            os.path.join(LOG_DIR, f"replay_{args.run}_{args.checkpoint}_{ep_name}.png"),
        )


if __name__ == "__main__":
    main()
