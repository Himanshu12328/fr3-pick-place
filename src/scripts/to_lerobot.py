"""
Converts the raw episode directories into a LeRobotDataset that the
LeRobot training scripts can consume directly.

The raw format stays on disk untouched. Conversion is deliberately a
separate offline step rather than something the collection script does
inline: LeRobot's dataset API has changed shape across versions, and a
collection session should never be lost to a library upgrade.

Feature naming follows LeRobot convention:
    observation.state          low-dimensional proprioception
    observation.images.<name>  one entry per camera
    action                     what the policy predicts

Run:
    python src\\scripts\\to_lerobot.py
"""

import json
import os
import shutil
import sys

import numpy as np
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from PIL import Image

# RAW_DIR = r"C:\pick_place\data\pick_place_v1"
# REPO_ID = "local/fr3_pick_place"
OUT_ROOT = r"C:\pick_place\data\lerobot"
RAW_DIR = r"C:\pick_place\data\pick_place_v1_small"
REPO_ID = "local/fr3_pick_place_small"

# PNG frames rather than encoded video. Video is smaller and loads faster,
# but the encode/decode path depends on ffmpeg bindings that are the least
# reliable part of this stack on native Windows. Start with the option that
# cannot fail; revisit only if dataloading becomes the training bottleneck.
USE_VIDEO = False


def load_raw_info(raw_dir):
    """
    Reads the dataset-level metadata written during collection.

    input:  raw_dir (str)
    output: dict
    """
    with open(os.path.join(raw_dir, "dataset_info.json")) as f:
        return json.load(f)


def list_episodes(raw_dir):
    """
    Returns sorted episode directory names.

    input:  raw_dir (str)
    output: list of str
    """
    return sorted(d for d in os.listdir(raw_dir) if d.startswith("episode_"))


def build_features(info, image_shape):
    """
    Describes every stream in the dataset so LeRobot can allocate storage
    and report shapes to the policy.

    Naming matters more than it looks. LeRobot's policy configs select
    inputs by these exact keys, so a camera named observation.images.wrist
    here must be referenced by that name in the training config.

    input:  info (dict) raw dataset_info, image_shape (tuple) H, W, C
    output: dict mapping feature name to spec
    """
    features = {
        "observation.state": {
            "dtype": "float32",
            "shape": (info["state_dim"],),
            "names": [f"state_{i}" for i in range(info["state_dim"])],
        },
        "action": {
            "dtype": "float32",
            "shape": (info["action_dim"],),
            "names": ["x", "y", "z", "qw", "qx", "qy", "qz", "gripper"],
        },
    }

    for cam in info["cameras"]:
        features[f"observation.images.{cam}"] = {
            "dtype": "video" if USE_VIDEO else "image",
            "shape": image_shape,
            "names": ["height", "width", "channel"],
        }

    return features


def probe_image_shape(raw_dir, ep_name, camera):
    """
    Reads one frame to determine image dimensions.

    Hardcoding the resolution would silently produce a broken dataset if
    the collection resolution ever changed, so it is read from the data
    instead.

    input:  raw_dir (str), ep_name (str), camera (str)
    output: tuple (height, width, channels)
    """
    path = os.path.join(raw_dir, ep_name, camera, "00000.png")
    return np.array(Image.open(path)).shape


def convert(raw_dir, repo_id, out_root):
    """
    Writes every raw episode into a new LeRobotDataset.

    Frames are added one at a time and committed per episode. LeRobot
    buffers within an episode and flushes on save_episode, so a crash
    midway loses only the episode in progress.

    input:  raw_dir (str), repo_id (str), out_root (str)
    output: LeRobotDataset
    """
    info = load_raw_info(raw_dir)
    episodes = list_episodes(raw_dir)
    if not episodes:
        raise RuntimeError(f"No episodes found in {raw_dir}")

    cameras = info["cameras"]
    image_shape = probe_image_shape(raw_dir, episodes[0], cameras[0])
    print(f"{len(episodes)} episodes, cameras {cameras}, image shape {image_shape}")

    root = os.path.join(out_root, repo_id.replace("/", "_"))
    if os.path.exists(root):
        print(f"Removing existing {root}")
        shutil.rmtree(root)

    dataset = LeRobotDataset.create(
        repo_id=repo_id,
        fps=info["fps"],
        root=root,
        features=build_features(info, image_shape),
        use_videos=USE_VIDEO,
    )

    total = 0
    for ep_name in episodes:
        ep_dir = os.path.join(raw_dir, ep_name)
        npz = np.load(os.path.join(ep_dir, "data.npz"))
        with open(os.path.join(ep_dir, "meta.json")) as f:
            meta = json.load(f)

        state = npz["state"]
        action = npz["action"]
        n = len(state)

        for i in range(n):
            frame = {
                "observation.state": state[i].astype(np.float32),
                "action": action[i].astype(np.float32),
                "task": meta.get("task", info["task"]),
            }
            for cam in cameras:
                img_path = os.path.join(ep_dir, cam, f"{i:05d}.png")
                frame[f"observation.images.{cam}"] = np.array(Image.open(img_path))

            dataset.add_frame(frame)

        dataset.save_episode()
        total += n
        print(f"  {ep_name}: {n} frames")

    print(f"\nWrote {len(episodes)} episodes, {total} frames to {root}")
    return dataset


def verify(root, repo_id):
    """
    Reloads the converted dataset and reports its structure.

    Reloading rather than inspecting the object still in memory is the
    point: it exercises the same read path training will use, so a
    metadata problem surfaces here rather than on the first training step.

    input:  root (str), repo_id (str)
    output: None
    """
    ds = LeRobotDataset(repo_id, root=root)

    print(f"\nepisodes: {ds.num_episodes}")
    print(f"frames:   {ds.num_frames}")
    print(f"fps:      {ds.fps}")

    print("\nfeatures:")
    for key, spec in ds.features.items():
        print(f"  {key:<38} {spec.get('dtype'):<8} {spec.get('shape')}")

    sample = ds[0]
    print("\nsample [0]:")
    for key, value in sample.items():
        if hasattr(value, "shape"):
            print(f"  {key:<38} {tuple(value.shape)}  {value.dtype}")
        else:
            print(f"  {key:<38} {value}")


def main():
    """
    Entry point.

    input:  none
    output: None
    """
    root = os.path.join(OUT_ROOT, REPO_ID.replace("/", "_"))
    convert(RAW_DIR, REPO_ID, OUT_ROOT)
    verify(root, REPO_ID)


if __name__ == "__main__":
    main()