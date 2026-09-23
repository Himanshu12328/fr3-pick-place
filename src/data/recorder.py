"""
Buffers observations and actions during teleoperation and writes completed
episodes to disk.

Storage format is one directory per episode containing a compressed npz of
the low-dimensional streams plus PNG frames per camera. This is
deliberately not LeRobotDataset yet: writing raw first means a collection
session cannot be lost to a LeRobot version mismatch, and conversion is a
separate offline step that can be re-run.

Observation and action design:

    observation.state    (16,) 7 joint pos, 7 joint vel, 2 finger pos
    observation.images   three 640x480 RGB streams
    action               (8,)  target position 3, target quaternion 4,
                               gripper opening 1

The action is the commanded target pose, not the achieved pose. This is
the quantity the impedance controller consumes, so a policy trained to
predict it plugs into exactly the same controller used during collection.
Recording achieved poses instead would train the policy to predict where
the arm ended up, which is not something it can command.
"""

import json
import os
import shutil
import time

import numpy as np
from PIL import Image


class EpisodeRecorder:
    """
    Accumulates frames for one episode at a time and writes them on demand.
    """

    def __init__(self, out_dir, camera_names, fps=30, 
                 task="pick up the red block and place it on the green target",
                 workspace_min=None, workspace_max=None):
        """
        Prepares the output directory and records the dataset-level
        metadata that every episode shares.

        input:  out_dir (str) dataset root,
                camera_names (list of str),
                fps (int) recording rate
        output: EpisodeRecorder instance
        """
        self.out_dir = out_dir
        self.camera_names = list(camera_names)
        self.fps = fps

        os.makedirs(out_dir, exist_ok=True)
        self._reset_buffer()

        meta_path = os.path.join(out_dir, "dataset_info.json")
        if not os.path.exists(meta_path):
            with open(meta_path, "w") as f:
                json.dump({
                    "task": task,
                    "fps": fps,
                    "cameras": self.camera_names,
                    "state_dim": 16,
                    "action_dim": 8,
                    "state_layout": "qpos[0:7], qvel[0:7], finger_qpos[0:2]",
                    "action_layout": "target_pos[0:3], target_quat[3:7], gripper[7]",
                    "gripper_open": 0.04,
                    "gripper_closed": 0.0,
                    "workspace_min": list(workspace_min) if workspace_min is not None else None,
                    "workspace_max": list(workspace_max) if workspace_max is not None else None,

                }, f, indent=2)

    def _reset_buffer(self):
        """
        Clears the in-memory episode buffer.

        input:  none
        output: None
        """
        self.states = []
        self.actions = []
        self.frames = {name: [] for name in self.camera_names}
        self.timestamps = []
        self.recording = False
        self.start_wall = None

    def start(self):
        """
        Begins buffering a new episode, discarding anything already held.

        input:  none
        output: None
        """
        self._reset_buffer()
        self.recording = True
        self.start_wall = time.perf_counter()

    def add(self, state, action, images, sim_time):
        """
        Appends one timestep to the buffer.

        Arrays are copied because the caller reuses its own buffers between
        frames. Storing references would leave every entry pointing at the
        same final values.

        input:  state (array (16,)), action (array (8,)),
                images (dict of camera name to uint8 array),
                sim_time (float) seconds since episode start
        output: None
        """
        if not self.recording:
            return

        self.states.append(np.asarray(state, dtype=np.float32).copy())
        self.actions.append(np.asarray(action, dtype=np.float32).copy())
        self.timestamps.append(float(sim_time))

        for name in self.camera_names:
            self.frames[name].append(images[name].copy())

    def n_frames(self):
        """
        Returns how many timesteps are buffered.

        input:  none
        output: int
        """
        return len(self.states)

    def discard(self):
        """
        Throws away the current episode without writing anything.

        Used when a demonstration goes wrong. A fumbled episode is worse
        than no episode: imitation learning reproduces what it is shown, so
        a recovery from a dropped block teaches the policy to drop blocks.

        input:  none
        output: None
        """
        n = self.n_frames()
        self._reset_buffer()
        return n

    def save(self, episode_index, success, extra_meta=None):
        """
        Writes the buffered episode to disk and clears the buffer.

        Low-dimensional streams go into a single compressed npz, which
        loads in one call. Images are written as individual PNGs, which
        keeps the format inspectable and avoids any video encoding
        dependency.

        input:  episode_index (int), success (bool),
                extra_meta (dict or None) additional per-episode metadata
        output: str path to the episode directory
        """
        if self.n_frames() == 0:
            raise RuntimeError("Nothing buffered.")

        ep_dir = os.path.join(self.out_dir, f"episode_{episode_index:04d}")
        if os.path.exists(ep_dir):
            shutil.rmtree(ep_dir)
        os.makedirs(ep_dir)

        np.savez_compressed(
            os.path.join(ep_dir, "data.npz"),
            state=np.stack(self.states),
            action=np.stack(self.actions),
            timestamp=np.array(self.timestamps, dtype=np.float32),
        )

        for name in self.camera_names:
            cam_dir = os.path.join(ep_dir, name)
            os.makedirs(cam_dir)
            for i, img in enumerate(self.frames[name]):
                Image.fromarray(img).save(
                    os.path.join(cam_dir, f"{i:05d}.png"),
                    compress_level=1,   # fast; disk is cheaper than time
                )

        meta = {
            "episode_index": episode_index,
            "n_frames": self.n_frames(),
            "fps": self.fps,
            "success": bool(success),
            "duration_s": float(self.timestamps[-1]),
            "wall_time_s": time.perf_counter() - self.start_wall,
        }
        if extra_meta:
            meta.update(extra_meta)

        with open(os.path.join(ep_dir, "meta.json"), "w") as f:
            json.dump(meta, f, indent=2)

        n = self.n_frames()
        self._reset_buffer()
        print(f"  saved {ep_dir}  ({n} frames, success={success})")
        return ep_dir


def next_episode_index(out_dir):
    """
    Returns the next unused episode index by scanning the output directory.

    Scanning rather than counting means a session can be interrupted and
    resumed without overwriting earlier episodes.

    input:  out_dir (str)
    output: int
    """
    if not os.path.isdir(out_dir):
        return 0

    used = [int(d.split("_")[1]) for d in os.listdir(out_dir)
            if d.startswith("episode_") and d.split("_")[1].isdigit()]
    return max(used) + 1 if used else 0