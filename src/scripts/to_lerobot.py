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

import numpy as np
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from PIL import Image

from src.config import LEROBOT_DIR, SMALL_DATA_DIR

OUT_ROOT = LEROBOT_DIR
RAW_DIR = SMALL_DATA_DIR
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
        info = json.load(f)

    # The oracle collector never wrote these, and the mixed conversion only
    # worked because a human directory was listed first and supplied them.
    # Converting the oracle episodes on their own raised a KeyError three
    # frames into a run. Read them off the data instead of trusting the
    # metadata to be complete, which is right regardless of collector.
    if "state_dim" not in info or "action_dim" not in info:
        first = sorted(d for d in os.listdir(raw_dir)
                       if d.startswith("episode_"))[0]
        npz = np.load(os.path.join(raw_dir, first, "data.npz"))
        info.setdefault("state_dim", int(npz["state"].shape[1]))
        info.setdefault("action_dim", int(npz["action"].shape[1]))

    return info


def list_episodes(raw_dir):
    """
    Returns sorted episode directory names.

    input:  raw_dir (str)
    output: list of str
    """
    return sorted(d for d in os.listdir(raw_dir) if d.startswith("episode_"))


# How the block pose is added as an auxiliary target.
#
# The perception probe showed the block position is legible from these
# images to about 3 mm, so the policy is not blind; it simply is not
# required to represent where the block is. Auxiliary supervision makes it
# required.
#
# It is appended to the *action* vector rather than added as a new head.
# ACT already regresses a chunk of actions, so three extra action dimensions
# are three extra regression targets on the same chunk, and the whole thing
# needs no change to LeRobot's ACT implementation at all. At rollout the
# harness reads action[:3], action[3:7] and action[7], so the extra columns
# are ignored without any code knowing they exist.
#
# It is a training target and never an input. The student's observation is
# unchanged and still carries no privileged state, so the rule that a policy
# must not skip perception still holds.
AUX_NAMES = ["block_x", "block_y", "block_yaw"]
AUX_COLUMNS = [0, 1, 3]  # x, y and yaw from the recorded (x, y, z, yaw)


def build_features(info, image_shape, aux_dim=0):
    """
    Describes every stream in the dataset so LeRobot can allocate storage
    and report shapes to the policy.

    Naming matters more than it looks. LeRobot's policy configs select
    inputs by these exact keys, so a camera named observation.images.wrist
    here must be referenced by that name in the training config.

    input:  info (dict) raw dataset_info, image_shape (tuple) H, W, C
    output: dict mapping feature name to spec
    """
    action_names = ["x", "y", "z", "qw", "qx", "qy", "qz", "gripper"]
    if aux_dim:
        action_names = action_names + AUX_NAMES[:aux_dim]

    features = {
        "observation.state": {
            "dtype": "float32",
            "shape": (info["state_dim"],),
            "names": [f"state_{i}" for i in range(info["state_dim"])],
        },
        "action": {
            "dtype": "float32",
            "shape": (info["action_dim"] + aux_dim,),
            "names": action_names,
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


def convert(raw_dirs, repo_id, out_root, exclude_failed=False,
            with_block_pose=False):
    """
    Writes every raw episode into a new LeRobotDataset.

    Frames are added one at a time and committed per episode. LeRobot
    buffers within an episode and flushes on save_episode, so a crash
    midway loses only the episode in progress.

    Several raw directories can be merged into one dataset, which is how
    relabelled DAgger episodes are mixed with recorded teleoperation.
    They are only safe to mix because the oracle that produced the labels
    was built from those same demonstrations and scores 0.991 against
    their trajectory profile, so both halves look like one operator.

    input:  raw_dirs (str or list of str), repo_id (str), out_root (str)
    output: LeRobotDataset
    """
    if isinstance(raw_dirs, (str, os.PathLike)):
        raw_dirs = [raw_dirs]
    raw_dirs = [str(d) for d in raw_dirs]

    info = load_raw_info(raw_dirs[0])
    sources = [(d, list_episodes(d)) for d in raw_dirs]

    if exclude_failed:
        # The recorded teleoperation contains only successes, because
        # the operator discarded the rest. Machine-collected episodes
        # should match that, and an episode where the demonstrator
        # itself failed runs to the full step cap, so a handful of them
        # contribute a disproportionate share of frames.
        kept_sources, dropped = [], 0
        for d, eps in sources:
            keep = []
            for e in eps:
                meta_path = os.path.join(d, e, 'meta.json')
                if os.path.exists(meta_path):
                    with open(meta_path) as fh:
                        m = json.load(fh)
                    if m.get('student_succeeded') is False:
                        dropped += 1
                        continue
                keep.append(e)
            kept_sources.append((d, keep))
        sources = kept_sources
        print(f'excluded {dropped} episodes whose demonstrator failed')
    for d, eps in sources:
        if not eps:
            raise RuntimeError(f"No episodes found in {d}")

    cameras = info["cameras"]
    image_shape = probe_image_shape(sources[0][0], sources[0][1][0], cameras[0])
    for d, eps in sources:
        shape = probe_image_shape(d, eps[0], cameras[0])
        if shape != image_shape:
            raise RuntimeError(
                f"{d} has image shape {shape}, expected {image_shape}. "
                "Mixing resolutions silently blinds the policy, because "
                "crop_shape is applied in pixels."
            )
    n_eps = sum(len(eps) for _, eps in sources)
    print(f"{n_eps} episodes from {len(sources)} source(s), "
          f"cameras {cameras}, image shape {image_shape}")

    aux_dim = len(AUX_COLUMNS) if with_block_pose else 0
    if with_block_pose:
        # Fail here rather than three hours into training. An episode
        # without a recorded block pose cannot be given one later, and a
        # dataset silently mixing 8- and 11-dimensional actions is exactly
        # the class of bug this project keeps paying for.
        for d, eps in sources:
            missing = [e for e in eps
                       if "block" not in np.load(os.path.join(d, e, "data.npz"))]
            if missing:
                raise RuntimeError(
                    f"{len(missing)} episodes in {d} have no recorded block "
                    f"pose, first is {missing[0]}. Run "
                    "`python -m src.scripts.add_block_pose` over it first."
                )
        print(f"auxiliary block pose on: action widened by {aux_dim} to "
              f"{info['action_dim'] + aux_dim} dimensions")

    root = os.path.join(out_root, repo_id.replace("/", "_"))
    if os.path.exists(root):
        print(f"Removing existing {root}")
        shutil.rmtree(root)

    dataset = LeRobotDataset.create(
        repo_id=repo_id,
        fps=info["fps"],
        root=root,
        features=build_features(info, image_shape, aux_dim=aux_dim),
        use_videos=USE_VIDEO,
    )

    total = 0
    for src_dir, episodes in sources:
      for ep_name in episodes:
        ep_dir = os.path.join(src_dir, ep_name)
        npz = np.load(os.path.join(ep_dir, "data.npz"))
        with open(os.path.join(ep_dir, "meta.json")) as f:
            meta = json.load(f)

        state = npz["state"]
        action = npz["action"]
        n = len(state)

        if aux_dim:
            block = npz["block"][:, AUX_COLUMNS].astype(np.float32)
            if len(block) != n:
                raise RuntimeError(
                    f"{ep_dir}: {len(block)} block rows against {n} frames"
                )
            action = np.concatenate([action, block], axis=1)

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
        print(f"  {os.path.basename(src_dir)}/{ep_name}: {n} frames")

    print(f"\nWrote {n_eps} episodes, {total} frames to {root}")
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
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--raw", nargs="+", default=[str(RAW_DIR)],
                        help="one or more raw episode directories to merge")
    parser.add_argument("--repo-id", default=REPO_ID)
    parser.add_argument("--with-block-pose", action="store_true",
                        help="append the true block x, y and yaw to the "
                             "action vector as auxiliary training targets")
    parser.add_argument("--exclude-failed", action="store_true",
                        help="skip episodes whose demonstrator did not "
                             "solve the task")
    args = parser.parse_args()

    root = os.path.join(OUT_ROOT, args.repo_id.replace("/", "_"))
    convert(args.raw, args.repo_id, OUT_ROOT,
            exclude_failed=args.exclude_failed,
            with_block_pose=args.with_block_pose)
    verify(root, args.repo_id)


if __name__ == "__main__":
    main()
