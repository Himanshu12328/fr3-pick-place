"""
Records rollouts of the trained teacher for visual inspection.

Statistics said the previous teacher was working. It scored 99.0% over 600
trials while shoving the block across the table in two thirds of them, and
no aggregate number in this project would have shown that. What shows it is
watching the block.

So this writes three things:

  mp4 per episode      the rollout as it actually looks
  contact sheet        six key frames per episode, all episodes on one page,
                       for judging at a glance whether the motion is sane
  block height trace   the block's height against time, which separates a
                       pick from a shove with no interpretation needed. A
                       pick is a clear arc well above the table. A shove is
                       a flat line at resting height

The height trace is the one that matters. A grid of small images can be
squinted at and read as a success; a flat height curve cannot.

Run:
    python -m src.scripts.teacher_demos --episodes 10
    python -m src.scripts.teacher_demos --run rl_teacher_v1 --model best
"""

import argparse

import imageio
import matplotlib
import mujoco
import numpy as np
from stable_baselines3 import SAC

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from src.config import DOCS_DIR, LOG_DIR, OUTPUT_ROOT  # noqa: E402
from src.data.task import BLOCK_Z  # noqa: E402
from src.eval import rollout as R  # noqa: E402
from src.rl.env import FR3PickPlaceEnv  # noqa: E402
from src.rl.seed import oracle_action  # noqa: E402

KEY_FRAMES = 6


def rollout(env, model, renderer, oracle=None, tail=45):
    """
    Runs one episode, keeping every frame and the block height trace.

    `tail` keeps recording after the episode ends. The episode terminates
    the moment the arm has retreated, which is the right place to stop
    counting but the wrong place to stop watching: the last thing a viewer
    needs to see is the block sitting still on the target with the arm
    clear of it.

    input:  env (FR3PickPlaceEnv), model (SAC or None),
            renderer (mujoco.Renderer), oracle (ScriptedOracle or None),
            tail (int) extra frames to record after termination
    output: dict with frames, heights and episode outcome
    """
    obs, _ = env.reset()
    if oracle is not None:
        oracle.reset()
    frames, heights = [], []
    held, ok, steps = False, False, 0

    while True:
        if oracle is not None:
            action = oracle_action(env, oracle)
        else:
            action, _ = model.predict(obs, deterministic=True)
        obs, _, terminated, truncated, info = env.step(action)
        steps += 1
        ok = info["success"]
        held = held or info["holding"]

        renderer.update_scene(env.data, camera="external")
        frames.append(renderer.render())
        heights.append(float(env.data.xpos[env.block_id][2]))

        if terminated or truncated:
            break

    # Hold position and keep filming so the settled result is visible.
    for _ in range(tail):
        env.data.qfrc_applied[: R.N_ARM] = env.ctrl.compute_torque(env.data)
        env.data.qfrc_applied[R.FINGER_DOFS] = R.gripper_torque(env.data, env.grip)
        mujoco.mj_step(env.model, env.data)
        renderer.update_scene(env.data, camera="external")
        frames.append(renderer.render())
        heights.append(float(env.data.xpos[env.block_id][2]))

    return {
        "frames": frames,
        "heights": np.array(heights),
        "success": bool(ok),
        "held": bool(held),
        "steps": steps,
        "lift_mm": float((np.max(heights) - BLOCK_Z) * 1000),
        "block_start": np.array(env.block_start),
    }


def contact_sheet(episodes, path, title):
    """
    Lays six evenly spaced frames from every episode onto one page.

    input:  episodes (list of rollout dicts), path (Path), title (str)
    output: None
    """
    n = len(episodes)
    fig, axes = plt.subplots(n, KEY_FRAMES, figsize=(KEY_FRAMES * 2.1, n * 1.75))
    if n == 1:
        axes = axes[None, :]

    for row, ep in enumerate(episodes):
        idx = np.linspace(0, len(ep["frames"]) - 1, KEY_FRAMES).astype(int)
        for col, k in enumerate(idx):
            ax = axes[row, col]
            ax.imshow(ep["frames"][k])
            ax.set_xticks([])
            ax.set_yticks([])
            for s in ax.spines.values():
                s.set_visible(False)
            if row == 0:
                ax.set_title(f"{int(100 * col / (KEY_FRAMES - 1))}%", fontsize=9)
        mark = "ok" if ep["success"] else "FAIL"
        axes[row, 0].set_ylabel(
            f"{mark}\n{ep['steps']} steps\nlift {ep['lift_mm']:.0f} mm",
            fontsize=8, rotation=0, ha="right", va="center", labelpad=34,
        )

    fig.suptitle(title, fontsize=12)
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.98))
    fig.savefig(path, dpi=110)
    plt.close(fig)


def height_plot(episodes, path, title):
    """
    Plots block height against time for every episode.

    This is the plot that distinguishes a pick from a shove. The resting
    height and the holding threshold are drawn in, so a trace that never
    clears the threshold is visibly not a pick.

    input:  episodes (list of rollout dicts), path (Path), title (str)
    output: None
    """
    fig, ax = plt.subplots(figsize=(9, 5))

    for i, ep in enumerate(episodes):
        ax.plot((ep["heights"] - BLOCK_Z) * 1000, lw=1.4,
                color="#c0392b" if not ep["success"] else "#0b5566",
                alpha=0.85, label="_nolegend_" if i else None)

    ax.axhline(0, color="#444", lw=1.2, ls="-")
    ax.axhline(10, color="#b8781c", lw=1.4, ls="--")
    ax.axhline(78, color="#5a8f3c", lw=1.4, ls=":")

    ax.text(2, 2.5, "resting on table", fontsize=9, color="#444")
    ax.text(2, 12, "holding threshold, 10 mm", fontsize=9, color="#b8781c")
    ax.text(2, 80, "oracle transit height, 78 mm", fontsize=9, color="#5a8f3c")

    ax.set_xlabel("policy step")
    ax.set_ylabel("block height above resting (mm)")
    ax.set_title(title)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def main():
    """
    Entry point.

    input:  none
    output: None
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", default="rl_teacher_v2")
    parser.add_argument("--model", default="best")
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--seed", type=int, default=51)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--tail", type=int, default=45,
                        help="frames to keep recording after the episode ends")
    parser.add_argument("--oracle", action="store_true",
                        help="record the scripted oracle instead of a model")
    args = parser.parse_args()

    env = FR3PickPlaceEnv()
    model, oracle = None, None

    if args.oracle:
        from src.eval.oracle import ScriptedOracle
        oracle = ScriptedOracle()
        oracle.bind(env.model, env.data)
        args.run, args.model = "oracle", "scripted"
    else:
        path = OUTPUT_ROOT / args.run / f"{args.model}.zip"
        if not path.exists():
            raise SystemExit(f"no model at {path}")
        model = SAC.load(str(path), device="cuda")
    env.rng = np.random.default_rng(args.seed)
    renderer = mujoco.Renderer(env.model, height=args.height, width=args.width)

    out_dir = LOG_DIR / f"demos_{args.run}_{args.model}"
    out_dir.mkdir(parents=True, exist_ok=True)

    episodes = []
    print(f"{args.run}/{args.model}, {args.episodes} episodes from seed {args.seed}\n")
    for i in range(args.episodes):
        ep = rollout(env, model, renderer, oracle=oracle, tail=args.tail)
        episodes.append(ep)
        tag = "ok" if ep["success"] else "FAIL"
        mp4 = out_dir / f"episode_{i:02d}_{tag}.mp4"
        imageio.mimsave(str(mp4), ep["frames"], fps=30)
        print(f"  episode {i:2d}  {tag:4s}  {ep['steps']:4d} steps  "
              f"lift {ep['lift_mm']:5.0f} mm  held {str(ep['held']):5s}  -> {mp4.name}")

    renderer.close()

    picked = sum(e["success"] and e["held"] for e in episodes)
    lifts = np.array([e["lift_mm"] for e in episodes])
    title = (f"{args.run}/{args.model}  |  "
             f"{sum(e['success'] for e in episodes)}/{len(episodes)} success, "
             f"{picked}/{len(episodes)} picked and placed, "
             f"lift {lifts.mean():.0f} mm mean")

    sheet = out_dir / "contact_sheet.png"
    trace = out_dir / "block_height.png"
    contact_sheet(episodes, sheet, title)
    height_plot(episodes, trace, f"Block height, {args.run}/{args.model}")

    DOCS_DIR.mkdir(parents=True, exist_ok=True)
    for src in (sheet, trace):
        (DOCS_DIR / f"teacher_{src.name}").write_bytes(src.read_bytes())

    print(f"\n{sum(e['success'] for e in episodes)}/{len(episodes)} succeeded, "
          f"{picked}/{len(episodes)} with a genuine lift")
    print(f"lift: mean {lifts.mean():.0f} mm, min {lifts.min():.0f}, "
          f"max {lifts.max():.0f}  (a shove would be near 0)")
    print(f"\nvideos      {out_dir}")
    print(f"contact     {sheet}")
    print(f"height plot {trace}")


if __name__ == "__main__":
    main()
