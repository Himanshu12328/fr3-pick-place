"""
Asks whether the block's yaw is in the images at all, and at what
resolution it becomes readable.

This reopens a question `PATH_TO_97.md` S3 closed. That section is titled
"The resolution hypothesis is dead. Measured, not argued." — and it is
right about what it measured. `perception_probe.py` established that the
block's **position** is legible to 3.1 mm at 160x128, well inside the grasp
tolerance, so resolution was dropped as an explanation and never revisited.

Nobody measured the block's **yaw**.

It matters now because the residual failures are a yaw error, and because
the yaw error has the specific form of *shrinkage*: measured over 200
trials, the commanded wrist yaw is under-rotated by a constant 11.5% of the
rotation the block requires, err = 0.115 x offset with r = 0.476. A
regressor shrinks toward the mean when its input does not determine its
target — hedging is the loss-minimising response to an unreadable input —
so the shape of the error is itself evidence about observability.

A first attempt at a camera-only yaw estimator came back at a median of
6.9 degrees and a p90 of 36.2, against 3.1 mm for position. That is either
a fact about the pixels or a fact about that estimator, and the difference
decides what to do next: re-render at a higher resolution and retrain, or
stop blaming the images.

### Why this renders rather than reading the dataset

The recorded episodes exist only at 160x128, so the comparison has to be
generated. It is generated from the one frame that cannot leak the answer.

Every episode begins with the arm at the same home keyframe, so at frame 0
the only thing that differs between two episodes' images is the block.
Sampling any later frame leaks: the oracle rotates the wrist into alignment
with the block, so a probe can read the block's yaw off the arm's own pose
without ever seeing the block, and would report the leak as legibility.
That is the same control `perception_probe.py` uses and the same reason.

So this resets the simulator to each episode's recorded block pose, renders
frame 0 at several resolutions, and trains the same estimator on each.

Run:
    python -m src.scripts.yaw_legibility --episodes 1000 --sizes 160x128 320x256 640x480
"""

import argparse
import json
import time
from pathlib import Path

import mujoco
import numpy as np
import torch
import torch.nn as nn

from src.config import BLOCK_QPOS_ADR, BLOCK_QVEL_ADR, CAMERAS, LOG_DIR
from src.data.task import set_block_pose
from src.scripts.train_yaw_head import SYMMETRY, YawHead, angle_error_deg


def render_synthetic(sizes, n, cameras, seed=7):
    """
    Renders the home-pose view of randomly posed blocks.

    Why synthetic rather than the recorded episodes. Frame 0 is the only
    frame of a recorded episode that cannot leak the answer, so 1,000
    episodes yield exactly 1,000 non-leaky images, and 800 of them after a
    holdout. A convolutional network trained from scratch on 800 images
    reaches chance on this target, which says nothing about whether the
    information is there — the first attempt at this comparison was
    confounded exactly that way, reporting 21.8, 17.2 and 21.8 degrees at
    three resolutions against a chance level of 22.5.

    Nothing about the question needs recorded episodes. The arm is at the
    same home keyframe at frame 0 of every episode, so the only thing that
    varies is the block, and the block's pose can simply be drawn. That
    makes the sample size a free parameter and removes the confound.

    input:  sizes (list of (w,h)), n (int) samples, cameras (list of str),
            seed (int)
    output: (dict of (w,h) -> dict of camera -> uint8 array, targets (N,2))
    """
    from src.data.task import sample_block_pose
    from src.eval import rollout as R

    model, data = R.setup_model()
    home = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "home")

    rng = np.random.default_rng(seed)
    poses = [sample_block_pose(rng) for _ in range(n)]

    out, targets = {}, None
    for (w, h) in sizes:
        renderer = mujoco.Renderer(model, height=h, width=w)
        imgs = {c: [] for c in cameras}
        tgt = []
        t0 = time.perf_counter()

        for pos, quat in poses:
            mujoco.mj_resetDataKeyframe(model, data, home)
            set_block_pose(model, data, np.asarray(pos, dtype=np.float64),
                           np.asarray(quat, dtype=np.float64),
                           BLOCK_QPOS_ADR, BLOCK_QVEL_ADR)
            mujoco.mj_forward(model, data)
            for c in cameras:
                renderer.update_scene(data, camera=c)
                imgs[c].append(renderer.render().copy())
            theta = 2.0 * np.arctan2(quat[3], quat[0])
            tgt.append([np.sin(SYMMETRY * theta), np.cos(SYMMETRY * theta)])

        renderer.close()
        out[(w, h)] = {c: np.stack(v) for c, v in imgs.items()}
        targets = np.asarray(tgt, dtype=np.float32)
        print(f"  rendered {n} synthetic frames at {w}x{h} in "
              f"{time.perf_counter() - t0:.0f}s", flush=True)

    return out, targets


def render_frame_zero(sizes, episodes, cameras, data_dir):
    """
    Renders the home-pose view of each episode's block at each resolution.

    input:  sizes (list of (w,h)), episodes (int) cap, cameras (list of str),
            data_dir (Path) episodes to take block poses from
    output: (dict of (w,h) -> dict of camera -> uint8 array, targets (N,2))
    """
    from src.eval import rollout as R

    model, data = R.setup_model()
    home = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_KEY, "home")

    eps = sorted(p for p in Path(data_dir).iterdir()
                 if p.name.startswith("episode_"))[:episodes]

    poses = []
    for ep in eps:
        meta = json.loads((ep / "meta.json").read_text())
        poses.append((np.asarray(meta["block_start_pos"], dtype=np.float64),
                      np.asarray(meta["block_start_quat"], dtype=np.float64)))

    out = {}
    targets = None
    for (w, h) in sizes:
        renderer = mujoco.Renderer(model, height=h, width=w)
        imgs = {c: [] for c in cameras}
        tgt = []
        t0 = time.perf_counter()

        for pos, quat in poses:
            mujoco.mj_resetDataKeyframe(model, data, home)
            set_block_pose(model, data, pos, quat,
                           BLOCK_QPOS_ADR, BLOCK_QVEL_ADR)
            mujoco.mj_forward(model, data)

            for c in cameras:
                renderer.update_scene(data, camera=c)
                imgs[c].append(renderer.render().copy())

            theta = 2.0 * np.arctan2(quat[3], quat[0])
            tgt.append([np.sin(SYMMETRY * theta), np.cos(SYMMETRY * theta)])

        renderer.close()
        out[(w, h)] = {c: np.stack(v) for c, v in imgs.items()}
        targets = np.asarray(tgt, dtype=np.float32)
        print(f"  rendered {len(poses)} frames at {w}x{h} in "
              f"{time.perf_counter() - t0:.0f}s", flush=True)

    return out, targets


def train_and_score(imgs, tgt, cameras, holdout, epochs, batch, lr, device):
    """
    Trains the yaw estimator on one resolution and returns its held-out
    error.

    input:  imgs (dict camera -> array), tgt (N,2), cameras (list of str),
            holdout (int) episodes held out, epochs, batch, lr, device
    output: dict of best-epoch metrics
    """
    n = len(tgt)
    cut = n - holdout
    tr = np.arange(0, cut)
    te = np.arange(cut, n)

    net = YawHead(len(cameras)).to(device)
    opt = torch.optim.AdamW(net.parameters(), lr=lr)

    def batches(idx, shuffle):
        order = idx.copy()
        if shuffle:
            np.random.shuffle(order)
        for i in range(0, len(order), batch):
            j = order[i:i + batch]
            x = [torch.from_numpy(imgs[c][j]).permute(0, 3, 1, 2)
                 .float().div_(255).to(device) for c in cameras]
            y = torch.from_numpy(tgt[j]).to(device)
            yield x, y

    best = None
    for epoch in range(epochs):
        net.train()
        for x, y in batches(tr, True):
            loss = nn.functional.mse_loss(net(x), y)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()

        net.eval()
        preds = np.zeros((len(te), 2), dtype=np.float32)
        pos = 0
        with torch.no_grad():
            for x, _ in batches(te, False):
                o = net(x).cpu().numpy()
                preds[pos:pos + len(o)] = o
                pos += len(o)

        err = angle_error_deg(preds, tgt[te])
        row = {
            "epoch": epoch,
            "median_deg": float(np.median(err)),
            "mean_deg": float(err.mean()),
            "p90_deg": float(np.percentile(err, 90)),
            "over_5deg": float(np.mean(err > 5.0)),
        }
        if best is None or row["median_deg"] < best["median_deg"]:
            best = row
    return best


def main():
    """
    Entry point.

    input:  none
    output: None
    """
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="data/oracle_v2")
    p.add_argument("--episodes", type=int, default=1000)
    p.add_argument("--sizes", nargs="+", default=["160x128", "320x256", "640x480"])
    p.add_argument("--cameras", nargs="+", default=CAMERAS)
    p.add_argument("--holdout", type=int, default=200)
    p.add_argument("--epochs", type=int, default=25)
    p.add_argument("--batch", type=int, default=64)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--device", default="cuda")
    p.add_argument("--synthetic", type=int, default=0,
                   help="draw this many random block poses instead of "
                        "reading recorded episodes")
    p.add_argument("--tag", default="yaw_legibility")
    args = p.parse_args()

    sizes = []
    for s in args.sizes:
        w, h = s.lower().split("x")
        sizes.append((int(w), int(h)))

    if args.synthetic:
        print(f"rendering {args.synthetic} random block poses at the home "
              f"pose, cameras {args.cameras}")
        rendered, tgt = render_synthetic(sizes, args.synthetic, args.cameras)
    else:
        print(f"rendering frame 0 for up to {args.episodes} episodes, "
              f"cameras {args.cameras}")
        rendered, tgt = render_frame_zero(sizes, args.episodes, args.cameras,
                                          args.data)
    print(f"  {len(tgt)} episodes, {args.holdout} held out\n")

    results = {}
    for (w, h) in sizes:
        print(f"training at {w}x{h}", flush=True)
        best = train_and_score(rendered[(w, h)], tgt, args.cameras,
                               args.holdout, args.epochs, args.batch,
                               args.lr, args.device)
        results[f"{w}x{h}"] = best
        print(f"  best: median {best['median_deg']:5.2f} deg  "
              f"mean {best['mean_deg']:5.2f}  p90 {best['p90_deg']:5.2f}  "
              f"over 5 deg {best['over_5deg'] * 100:5.1f}%\n", flush=True)

    print(f"  {'resolution':<12}{'median':>9}{'mean':>9}{'p90':>9}{'>5 deg':>9}")
    for k, v in results.items():
        print(f"  {k:<12}{v['median_deg']:>9.2f}{v['mean_deg']:>9.2f}"
              f"{v['p90_deg']:>9.2f}{v['over_5deg'] * 100:>8.1f}%")

    # What the answer means. A chance-level estimator on a target folded
    # into a quarter turn averages 22.5 degrees of error, so anything near
    # that has learned nothing.
    print("\n  chance level is 22.5 deg median; the policy's own commanded")
    print("  yaw error is 0.115 x the required rotation, so 5 deg at 45 deg")
    print("  of offset is the scale that matters.")

    (LOG_DIR / f"{args.tag}.json").write_text(
        json.dumps({"results": results, "episodes": int(len(tgt)),
                    "cameras": args.cameras, "holdout": args.holdout},
                   indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
