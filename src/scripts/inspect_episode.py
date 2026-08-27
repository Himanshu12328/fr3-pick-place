"""
Renders a filmstrip of frames from one or more episodes so the image
streams can be checked by eye.

The numeric validation confirms the arrays are well formed. It cannot tell
you whether the block is visible during the approach, whether the wrist
views are occluded at the moment of grasp, or whether the lighting changed
partway through a session. Those are the failure modes that only show up
by looking.

Run:
    python src\\scripts\\inspect_episode.py            # random episode
    python src\\scripts\\inspect_episode.py 42         # specific episode
    python src\\scripts\\inspect_episode.py --batch 6  # six random episodes
"""

import argparse
import json
import os

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from src.config import LOG_DIR, RAW_DATA_DIR

DATA_DIR = RAW_DATA_DIR
LOG_DIR = LOG_DIR


def list_episodes(data_dir):
    """
    Returns every episode directory name, sorted.

    input:  data_dir (str)
    output: list of str
    """
    return sorted(d for d in os.listdir(data_dir) if d.startswith("episode_"))


def key_frame_indices(action, n_extra=3):
    """
    Picks frames that span the task's phases rather than sampling evenly.

    Even sampling wastes most of the filmstrip on transit, where nothing
    interesting happens, and often misses the grasp entirely. The moments
    that matter are the approach, the instant the gripper closes, the lift,
    and the release.

    input:  action (array (N, 8)), n_extra (int) filler frames
    output: list of (index, label) tuples
    """
    n = len(action)
    grip = action[:, 7]
    closed = grip < 0.01

    frames = [(0, "start")]

    if closed.any():
        grasp = int(np.argmax(closed))
        release = int(len(closed) - np.argmax(closed[::-1]) - 1)

        frames.append((max(0, grasp - 15), "pre-grasp"))
        frames.append((grasp, "grasp"))
        frames.append((min(n - 1, grasp + 30), "lift"))
        frames.append(((grasp + release) // 2, "transit"))
        frames.append((release, "release"))
    else:
        step = max(1, n // (n_extra + 1))
        frames += [(i * step, f"t={i * step}") for i in range(1, n_extra + 1)]

    frames.append((n - 1, "end"))
    return sorted(set(frames), key=lambda f: f[0])


def load_frame(ep_dir, camera, index):
    """
    Loads one PNG frame.

    input:  ep_dir (str), camera (str), index (int)
    output: numpy array (H, W, 3) uint8, or None if missing
    """
    path = os.path.join(ep_dir, camera, f"{index:05d}.png")
    if not os.path.exists(path):
        return None
    return np.array(Image.open(path))


def filmstrip(data_dir, ep_name, cameras, out_path):
    """
    Writes a grid with one row per camera and one column per key frame.

    input:  data_dir (str), ep_name (str), cameras (list of str),
            out_path (str)
    output: None
    """
    ep_dir = os.path.join(data_dir, ep_name)
    action = np.load(os.path.join(ep_dir, "data.npz"))["action"]
    with open(os.path.join(ep_dir, "meta.json")) as f:
        meta = json.load(f)

    frames = key_frame_indices(action)

    fig, axes = plt.subplots(
        len(cameras), len(frames), figsize=(2.6 * len(frames), 2.2 * len(cameras))
    )
    if len(cameras) == 1:
        axes = axes[None, :]

    for r, cam in enumerate(cameras):
        for c, (idx, label) in enumerate(frames):
            ax = axes[r, c]
            img = load_frame(ep_dir, cam, idx)
            if img is not None:
                ax.imshow(img)
            ax.axis("off")
            if r == 0:
                ax.set_title(f"{label}\n{idx}", fontsize=8)
            if c == 0:
                ax.text(
                    -0.08,
                    0.5,
                    cam,
                    rotation=90,
                    va="center",
                    ha="right",
                    transform=ax.transAxes,
                    fontsize=9,
                )

    fig.suptitle(
        f"{ep_name}   {meta['n_frames']} frames   "
        f"{meta['duration_s']:.1f}s   success={meta['success']}",
        fontsize=11,
    )

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=100, bbox_inches="tight")
    plt.close(fig)
    print(f"saved: {out_path}")


def brightness_audit(data_dir, ep_names, camera="external"):
    """
    Reports mean image brightness per episode to catch rendering drift.

    A session where lighting or exposure changed partway through gives the
    policy a spurious cue it can latch onto, and the drift is invisible
    unless you compare episodes directly. Sampling one frame per episode is
    enough to spot it.

    input:  data_dir (str), ep_names (list of str), camera (str)
    output: None, prints outliers
    """
    means = []
    for name in ep_names:
        img = load_frame(os.path.join(data_dir, name), camera, 0)
        means.append(np.nan if img is None else float(img.mean()))

    arr = np.array(means)
    valid = arr[~np.isnan(arr)]
    if len(valid) < 2:
        return

    mu, sd = valid.mean(), valid.std()
    print(f"\n{camera} frame-0 brightness: mean {mu:.1f}, std {sd:.2f}")

    outliers = [
        (n, m)
        for n, m in zip(ep_names, arr, strict=True)
        if not np.isnan(m) and abs(m - mu) > 3 * sd
    ]
    if outliers:
        print("  outliers (>3 sd):")
        for n, m in outliers:
            print(f"    {n}: {m:.1f}")
    else:
        print("  no outliers")


def main():
    """
    Entry point.

    input:  none
    output: None
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("episode", nargs="?", type=int, default=None)
    parser.add_argument("--batch", type=int, default=0)
    args = parser.parse_args()

    with open(os.path.join(DATA_DIR, "dataset_info.json")) as f:
        cameras = json.load(f)["cameras"]

    names = list_episodes(DATA_DIR)
    rng = np.random.default_rng()

    if args.batch:
        chosen = rng.choice(names, size=min(args.batch, len(names)), replace=False)
    elif args.episode is not None:
        chosen = [f"episode_{args.episode:04d}"]
    else:
        chosen = [rng.choice(names)]

    for name in chosen:
        filmstrip(
            DATA_DIR, name, cameras, os.path.join(LOG_DIR, f"filmstrip_{name}.png")
        )

    brightness_audit(DATA_DIR, names)


if __name__ == "__main__":
    main()
