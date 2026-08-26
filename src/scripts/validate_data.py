"""
Audits a collected dataset before more time is spent extending it.

Checks for the failure modes that are cheap to fix at 25 episodes and
expensive at 150: action sequences the arm cannot track, episodes with no
grasp, inconsistent lengths, missing frames, and demonstrations that are
outliers relative to the rest of the set.

Run:
    python src\\scripts\\validate_data.py
"""

import json
import os
import sys

import matplotlib.pyplot as plt
import numpy as np

DATA_DIR = r"C:\pick_place\data\pick_place_v1"
LOG_DIR = r"C:\pick_place\logs"


def load_episodes(data_dir):
    """
    Loads every episode's arrays and metadata into memory.

    Images are not loaded. They dominate disk size but none of the checks
    here need them, and loading tens of thousands of PNGs would make this
    slow enough that you would stop running it.

    input:  data_dir (str)
    output: list of dicts with keys state, action, timestamp, meta, name
    """
    episodes = []
    names = sorted(d for d in os.listdir(data_dir) if d.startswith("episode_"))

    for name in names:
        ep_dir = os.path.join(data_dir, name)
        npz = np.load(os.path.join(ep_dir, "data.npz"))
        with open(os.path.join(ep_dir, "meta.json")) as f:
            meta = json.load(f)

        episodes.append({
            "name": name,
            "state": npz["state"],
            "action": npz["action"],
            "timestamp": npz["timestamp"],
            "meta": meta,
            "dir": ep_dir,
        })

    return episodes


def check_frame_counts(episodes, camera_names):
    """
    Confirms every camera directory holds exactly as many PNGs as there
    are state rows.

    A mismatch means the recorder dropped frames, which would silently
    misalign images against actions during training. That misalignment is
    nearly impossible to diagnose from a training curve.

    input:  episodes (list), camera_names (list of str)
    output: list of str describing problems, empty if clean
    """
    problems = []
    for ep in episodes:
        n = len(ep["state"])
        for cam in camera_names:
            cam_dir = os.path.join(ep["dir"], cam)
            if not os.path.isdir(cam_dir):
                problems.append(f"{ep['name']}: missing camera dir {cam}")
                continue
            n_png = len([f for f in os.listdir(cam_dir) if f.endswith(".png")])
            if n_png != n:
                problems.append(f"{ep['name']}: {cam} has {n_png} frames, expected {n}")
    return problems


def check_grasp(episodes):
    """
    Confirms each episode contains a closed-gripper phase of plausible
    duration.

    An episode where the gripper never closed is not a demonstration of
    this task. One where it closed for only a handful of frames usually
    means a failed grasp that was not noticed.

    input:  episodes (list)
    output: list of str describing problems
    """
    problems = []
    for ep in episodes:
        grip = ep["action"][:, 7]
        closed = grip < 0.01
        n_closed = int(closed.sum())

        if n_closed == 0:
            problems.append(f"{ep['name']}: gripper never closed")
        elif n_closed < 30:
            problems.append(f"{ep['name']}: gripper closed only {n_closed} frames")

        # Gripper should toggle a small number of times. Many transitions
        # means repeated grab attempts, which teaches retrying.
        transitions = int(np.sum(np.abs(np.diff(closed.astype(int)))))
        if transitions > 2:
            problems.append(f"{ep['name']}: gripper toggled {transitions} times")

    return problems


def check_action_reachability(episodes, workspace_min, workspace_max, tol=1e-4):
    """
    Reports how much of each episode has a target pinned against a
    workspace bound.

    A clamped target is one the arm cannot reach, so the recorded action
    and the achieved pose diverge for as long as the clamp holds. A policy
    trained on this learns to command unreachable poses, and the resulting
    steady offset is baked into every rollout.

    input:  episodes (list), workspace_min (array (3,)),
            workspace_max (array (3,)), tol (float) metres
    output: dict mapping episode name to fraction of frames clamped
    """
    clamped = {}
    for ep in episodes:
        pos = ep["action"][:, :3]
        at_min = np.any(np.abs(pos - workspace_min) < tol, axis=1)
        at_max = np.any(np.abs(pos - workspace_max) < tol, axis=1)
        clamped[ep["name"]] = float(np.mean(at_min | at_max))
    return clamped


def check_action_smoothness(episodes, fps=30):
    """
    Measures the largest per-step position jump in each episode.

    Large jumps mean the target moved faster than the impedance controller
    can follow. At a 1.5 Hz bandwidth the arm tracks smooth motion well and
    steps poorly, so a jumpy action stream produces a policy whose own
    output it cannot execute.

    input:  episodes (list), fps (int)
    output: dict mapping episode name to (max_jump_mm, mean_speed_mm_s)
    """
    out = {}
    for ep in episodes:
        pos = ep["action"][:, :3]
        deltas = np.linalg.norm(np.diff(pos, axis=0), axis=1)
        out[ep["name"]] = (float(deltas.max() * 1000),
                           float(deltas.mean() * 1000 * fps))
    return out


def summarise(episodes, clamped, smoothness):
    """
    Prints a per-episode table and dataset-level statistics.

    input:  episodes (list), clamped (dict), smoothness (dict)
    output: None
    """
    print(f"\n{'episode':<16}{'frames':>8}{'dur(s)':>8}{'success':>9}"
          f"{'clamped%':>10}{'maxjump(mm)':>13}{'speed(mm/s)':>13}")

    for ep in episodes:
        name = ep["name"]
        n = len(ep["state"])
        dur = float(ep["timestamp"][-1])
        succ = ep["meta"].get("success", False)
        jump, speed = smoothness[name]
        print(f"{name:<16}{n:>8}{dur:>8.1f}{str(succ):>9}"
              f"{clamped[name]*100:>10.1f}{jump:>13.2f}{speed:>13.1f}")

    lengths = np.array([len(ep["state"]) for ep in episodes])
    successes = sum(ep["meta"].get("success", False) for ep in episodes)

    print(f"\nepisodes:        {len(episodes)}")
    print(f"successful:      {successes} ({successes/len(episodes)*100:.0f}%)")
    print(f"total frames:    {lengths.sum()}")
    print(f"length mean/std: {lengths.mean():.0f} / {lengths.std():.0f} frames")
    print(f"length min/max:  {lengths.min()} / {lengths.max()} frames")


def plot_trajectories(episodes, path):
    """
    Overlays every episode's end-effector target path in the xy plane and
    the z profile over normalised time.

    Consistency is the thing to look for. Tightly bundled trajectories mean
    the demonstrations agree on how the task is done, which is what lets a
    policy learn from relatively few of them. A scattered bundle means the
    policy has to average across genuinely different strategies.

    input:  episodes (list), path (str)
    output: None
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

    for ep in episodes:
        pos = ep["action"][:, :3]
        grip = ep["action"][:, 7]
        ax1.plot(pos[:, 0], pos[:, 1], alpha=0.4, lw=1)

        # Mark where the grasp happens
        closed = np.nonzero(grip < 0.01)[0]
        if len(closed):
            ax1.plot(pos[closed[0], 0], pos[closed[0], 1], "k.", ms=6)

        t_norm = np.linspace(0, 1, len(pos))
        ax2.plot(t_norm, pos[:, 2], alpha=0.4, lw=1)

    ax1.set_xlabel("x (m)")
    ax1.set_ylabel("y (m)")
    ax1.set_title("target path, top view (dots = grasp)")
    ax1.grid(alpha=0.3)
    ax1.axis("equal")

    ax2.set_xlabel("normalised episode time")
    ax2.set_ylabel("target z (m)")
    ax2.set_title("height profile")
    ax2.grid(alpha=0.3)

    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"\nsaved: {path}")


def main():
    """
    Runs every check and reports.

    input:  none
    output: None
    """
    with open(os.path.join(DATA_DIR, "dataset_info.json")) as f:
        info = json.load(f)

    episodes = load_episodes(DATA_DIR)
    if not episodes:
        print("No episodes found.")
        return

    ws_min = np.array(info["workspace_min"])
    ws_max = np.array(info["workspace_max"])

    clamped = check_action_reachability(episodes, ws_min, ws_max)
    smoothness = check_action_smoothness(episodes, info["fps"])
    summarise(episodes, clamped, smoothness)

    problems = []
    problems += check_frame_counts(episodes, info["cameras"])
    problems += check_grasp(episodes)

    heavy_clamp = [n for n, f in clamped.items() if f > 0.15]
    if heavy_clamp:
        problems.append(f"targets clamped >15% of frames: {', '.join(heavy_clamp)}")

    print("\n" + ("-" * 60))
    if problems:
        print(f"{len(problems)} issue(s):")
        for p in problems:
            print(f"  {p}")
    else:
        print("No issues found.")

    plot_trajectories(episodes, os.path.join(LOG_DIR, "dataset_trajectories.png"))


if __name__ == "__main__":
    main()