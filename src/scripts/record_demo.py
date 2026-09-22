"""
Records the policy at presentation resolution without letting it see that
resolution.

The distinction is the whole reason this script exists separately from the
`--video-dir` flag on `eval_supervised.py`. That flag records the same
frames the policy is given, which are 160x128 — correct for diagnosis, and
far too small to show anyone. Rendering the *policy's* cameras larger is
not an option: `crop_shape` is applied in pixels, so a policy trained at
160x128 and fed 640x480 receives a small centre patch instead of the scene
and goes blind. This project measured that once, at 60% success becoming
0.4%.

So there are two renderers. The policy's observation is built exactly as
training built it, and a second renderer watches the same simulation from a
free orbit camera at whatever size is asked for. The recording cannot
affect the behaviour it records.

Produces an mp4, and optionally a GIF, because a README on GitHub will not
play a local mp4 but will always animate a GIF.

Run:
    python -m src.scripts.record_demo --trials 4 --seed 131
    python -m src.scripts.record_demo --trials 1 --seed 131 --gif --size 720
"""

import argparse
from pathlib import Path

import imageio.v2 as imageio
import mujoco

from src.config import OUTPUT_ROOT

# Where the orbit camera sits. Chosen to frame the table, the block's
# starting region and the place target together, from an angle where the
# gripper's approach and the lift are both legible.
CAM_LOOKAT = (0.50, 0.08, 0.46)
CAM_DISTANCE = 1.05
CAM_AZIMUTH = 200.0
CAM_ELEVATION = -32.0


class Recorder:
    """
    Wraps a policy and renders one high-resolution frame per policy step.

    Presents the same callable interface the harness expects, so it is
    interchangeable with the policy it wraps and needs no harness changes.
    It returns the inner policy's action untouched.
    """

    def __init__(self, inner, size=1080):
        """
        input:  inner (callable) the policy to record, size (int) pixels
        output: Recorder instance
        """
        self.inner = inner
        self.size = int(size)
        self.frames = []
        self.renderer = None
        self.model = None
        self.data = None

        self.cam = mujoco.MjvCamera()
        self.cam.type = mujoco.mjtCamera.mjCAMERA_FREE
        self.cam.lookat[:] = CAM_LOOKAT
        self.cam.distance = CAM_DISTANCE
        self.cam.azimuth = CAM_AZIMUTH
        self.cam.elevation = CAM_ELEVATION

    def bind(self, model, data):
        """
        input:  model (MjModel), data (MjData)
        output: None
        """
        self.model, self.data = model, data
        # The scene's declared offscreen buffer is sized for the policy's
        # cameras, so it has to be raised before a larger renderer works.
        model.vis.global_.offwidth = max(model.vis.global_.offwidth, self.size)
        model.vis.global_.offheight = max(model.vis.global_.offheight, self.size)
        self.renderer = mujoco.Renderer(model, height=self.size, width=self.size)
        if hasattr(self.inner, "bind"):
            self.inner.bind(model, data)

    def reset(self):
        """
        input:  none
        output: None
        """
        self.frames = []
        if hasattr(self.inner, "reset"):
            self.inner.reset()

    def __call__(self, obs):
        """
        input:  obs (dict) harness observation, built at training resolution
        output: numpy array, the inner policy's action unmodified
        """
        self.renderer.update_scene(self.data, camera=self.cam)
        self.frames.append(self.renderer.render().copy())
        return self.inner(obs)


def write_gif(frames, path, size, fps, max_seconds):
    """
    Writes a GIF small enough for a README to load.

    Downsamples spatially by slicing and temporally by taking every nth
    frame, because a 1080-pixel GIF of a 300-step episode is tens of
    megabytes and nobody waits for it.

    input:  frames (list of arrays), path (Path), size (int) target pixels,
            fps (int) playback rate, max_seconds (float) trim length
    output: None
    """
    step = max(1, len(frames) // int(max_seconds * fps))
    picked = frames[::step]

    h, w = picked[0].shape[:2]
    factor = max(1, h // size)
    small = [f[::factor, ::factor] for f in picked]

    # Hold the final frame so the placed block is visible before it loops.
    small = small + [small[-1]] * max(1, fps // 2)
    imageio.mimwrite(path, small, duration=1000.0 / fps, loop=0)
    mb = path.stat().st_size / 1e6
    print(f"  wrote {path}  {len(small)} frames  {small[0].shape[1]}px  {mb:.1f} MB")


def main():
    """
    Entry point.

    input:  none
    output: None
    """
    from src.scripts.eval_supervised import build_policy
    from src.scripts.proof_report import collect

    p = argparse.ArgumentParser()
    p.add_argument("--members", nargs="+", default=[
        str(OUTPUT_ROOT / "act_oracle_v2" / "checkpoints" / "030000" / "pretrained_model"),
        str(OUTPUT_ROOT / "act_aux_xy" / "checkpoints" / "025000" / "pretrained_model"),
    ])
    p.add_argument("--trials", type=int, default=4)
    p.add_argument("--seed", type=int, default=131)
    p.add_argument("--size", type=int, default=1080)
    p.add_argument("--out", default=str(OUTPUT_ROOT / "demo"))
    p.add_argument("--device", default="cuda")
    p.add_argument("--gif", action="store_true", help="also write a GIF")
    p.add_argument("--gif-size", type=int, default=480)
    p.add_argument("--gif-fps", type=int, default=12)
    p.add_argument("--gif-seconds", type=float, default=9.0)
    args = p.parse_args()

    inner, name = build_policy(args.members, "monitor", device=args.device)
    rec = Recorder(inner, size=args.size)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    # `collect` resets the policy between trials, so each trial's frames are
    # captured by intercepting the reset rather than by asking afterwards.
    captured = []
    inner_reset = rec.reset

    def reset_and_flush():
        if rec.frames:
            captured.append(rec.frames)
        inner_reset()

    rec.reset = reset_and_flush
    rows = collect(rec, True, args.trials, args.seed)
    captured.append(rec.frames)

    print(f"\n{name}, seed {args.seed}:")
    for i, (frames, row) in enumerate(zip(captured, rows, strict=True)):
        tag = "pass" if row["strict_success"] else "fail"
        mp4 = out / f"demo_seed{args.seed}_{i:02d}_{tag}.mp4"
        hold = [frames[-1]] * 30
        imageio.mimwrite(mp4, frames + hold, fps=30, quality=8,
                         macro_block_size=1)
        print(f"  wrote {mp4}  {len(frames)} frames  "
              f"lift {row['lift_m'] * 1000:.0f} mm  {tag}")

        if args.gif and tag == "pass":
            write_gif(frames, out / f"demo_seed{args.seed}_{i:02d}.gif",
                      args.gif_size, args.gif_fps, args.gif_seconds)


if __name__ == "__main__":
    main()
