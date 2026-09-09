"""
Asks one question: can the block's position be read out of the images the
policy is actually given?

Stage 0's failure taxonomy is the reason this exists. 62% of ACT's failures
are grasps that never happened, 18% are blocks knocked aside, and **zero of
300 failures were near misses**. That is not the signature of a policy with
imprecise control. A policy with imprecise control produces near misses. A
policy that closes its fingers somewhere the block is not has failed to
locate the block.

Two explanations remain, and they call for completely different work:

  the images do not contain the block position accurately enough
      → resolution, camera placement or backbone capacity. More
        demonstrations cannot fix it, which would also explain why the
        1,006-episode dataset peaked at 4.3 epochs and gained 3 points.

  the images contain it and ACT does not extract it
      → an auxiliary supervision or capacity problem, not a sensing one.

This script separates them by training a small convolutional regressor on
exactly the observations the policy sees, with nothing else to do but
predict where the block is. If a purpose-built probe cannot read the block
position off these pixels, no policy trained on them can either, and that
is a fact about the dataset rather than about ACT.

The labels are free. Before the gripper first closes, the block has not
moved, so every pre-grasp frame of every episode is labelled by that
episode's recorded `block_start_pos`. No new collection is needed.

The number that matters is the error against the grasp tolerance. The
fingers open to 40 mm each side of a 44 mm block, and the oracle positions
itself to 6 mm before closing. An error much past 10 mm means the grasp is
a coin flip.

Run:
    python -m src.scripts.perception_probe --data data/oracle_a --epochs 12
    python -m src.scripts.perception_probe --data data/oracle_a --cameras external
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from PIL import Image

from src.config import CAMERAS, LOG_DIR

# How many pre-grasp frames to take from each episode. The arm is at home
# for the first few and on top of the block by the end, so sampling across
# the phase covers both the far view and the close one.
FRAMES_PER_EPISODE = 6

# The grasp is decided by lateral position. Height is fixed by the table
# and yaw is recovered separately, so the probe predicts x and y.
TARGET_DIM = 2

# Block start x bins, matching the bins the strict evaluation is
# reported over so the two can be read against each other.
BINS = [0.46, 0.49, 0.52, 0.55, 0.58, 0.61, 0.64]


class BlockRegressor(nn.Module):
    """
    A small convolutional stack per camera, concatenated into a linear head.

    Deliberately not a ResNet. The question is whether the information is
    present in the pixels, so the probe should be as unconstrained as it can
    be while still training in minutes. If a network with one job and no
    distractions cannot read the block position, ACT reading it as a side
    effect of predicting 32 actions certainly cannot.
    """

    def __init__(self, n_cameras, height, width):
        """
        input:  n_cameras (int), height (int), width (int)
        output: BlockRegressor
        """
        super().__init__()

        def trunk():
            return nn.Sequential(
                nn.Conv2d(3, 32, 5, stride=2, padding=2), nn.ReLU(),
                nn.Conv2d(32, 64, 3, stride=2, padding=1), nn.ReLU(),
                nn.Conv2d(64, 128, 3, stride=2, padding=1), nn.ReLU(),
                nn.Conv2d(128, 128, 3, stride=2, padding=1), nn.ReLU(),
                nn.AdaptiveAvgPool2d((2, 2)),
                nn.Flatten(),
            )

        self.trunks = nn.ModuleList([trunk() for _ in range(n_cameras)])
        self.head = nn.Sequential(
            nn.Linear(128 * 4 * n_cameras, 256), nn.ReLU(),
            nn.Linear(256, TARGET_DIM),
        )

    def forward(self, images):
        """
        input:  images (list of tensors, each B,3,H,W)
        output: tensor B,2
        """
        feats = [t(x) for t, x in zip(self.trunks, images)]
        return self.head(torch.cat(feats, dim=1))


def grasp_frame(action):
    """
    Finds the first frame where the gripper was commanded shut.

    input:  action (array (T,8))
    output: int, or the episode length if it never closed
    """
    closed = np.where(action[:, 7] < 0.02)[0]
    return int(closed[0]) if len(closed) else len(action)


def load_split(data_dirs, cameras, limit=None, per_episode=FRAMES_PER_EPISODE,
               frac_lo=0.0, frac_hi=1.0):
    """
    Loads pre-grasp frames and their block positions.

    input:  data_dirs (list of Path), cameras (list of str),
            limit (int or None) episodes per directory
    output: (images dict of camera to uint8 array, targets array (N,2))
    """
    images = {c: [] for c in cameras}
    targets = []

    for d in data_dirs:
        eps = sorted(p for p in Path(d).iterdir() if p.name.startswith("episode_"))
        if limit:
            eps = eps[:limit]

        for ep in eps:
            meta = json.loads((ep / "meta.json").read_text())
            npz = np.load(ep / "data.npz")
            g = grasp_frame(npz["action"])
            if g < max(per_episode, 4):
                continue

            # Which part of the approach to sample from, as a fraction.
            #
            # This is the control that makes the experiment mean anything.
            # Sampling the whole approach leaks the answer: the oracle
            # drives the arm to the block, so by the end of the approach the
            # tool is directly above it and a probe can read the block
            # position off the arm's own pose without ever seeing the block.
            # Restricting to the start of the approach removes that path,
            # because at frame 0 the arm is at the same home keyframe in
            # every single episode and the only thing that differs between
            # images is the block itself.
            lo = int(round(frac_lo * (g - 1)))
            hi = int(round(frac_hi * (g - 1)))
            if hi <= lo:
                idx = np.array([lo])
            else:
                idx = np.unique(np.linspace(lo, hi, per_episode).astype(int))
            xy = np.asarray(meta["block_start_pos"][:2], dtype=np.float32)

            for i in idx:
                for c in cameras:
                    images[c].append(
                        np.asarray(Image.open(ep / c / f"{i:05d}.png"), dtype=np.uint8)
                    )
                targets.append(xy)

    return (
        {c: np.stack(v) for c, v in images.items()},
        np.stack(targets),
    )


def main():
    """
    Entry point.

    input:  none
    output: None
    """
    p = argparse.ArgumentParser()
    p.add_argument("--data", nargs="+", default=["data/oracle_a"])
    p.add_argument("--cameras", nargs="+", default=CAMERAS)
    p.add_argument("--episodes", type=int, default=250)
    p.add_argument("--holdout", type=int, default=50)
    p.add_argument("--frames", type=int, default=FRAMES_PER_EPISODE,
                   help="pre-grasp frames sampled per episode")
    p.add_argument("--frac-lo", type=float, default=0.0,
                   help="sample frames from this fraction of the approach on")
    p.add_argument("--frac-hi", type=float, default=1.0,
                   help="up to this fraction; use a low value to stop the "
                        "probe reading the block position off the arm pose")
    p.add_argument("--epochs", type=int, default=12)
    p.add_argument("--batch", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--device", default="cuda")
    p.add_argument("--tag", default="probe")
    args = p.parse_args()

    print(f"loading up to {args.episodes} episodes from {args.data}, "
          f"cameras {args.cameras}")
    t0 = time.perf_counter()
    imgs, tgt = load_split([Path(d) for d in args.data], args.cameras,
                           limit=args.episodes, per_episode=args.frames,
                           frac_lo=args.frac_lo, frac_hi=args.frac_hi)
    print(f"  {len(tgt)} frames in {time.perf_counter() - t0:.0f}s, "
          f"image shape {imgs[args.cameras[0]].shape[1:]}")

    # Split by episode block rather than at random. Frames from one episode
    # are near duplicates of each other, so a random split leaks the answer
    # across it and the held-out error comes back meaninglessly small.
    per_ep = max(1, len(tgt) // max(1, args.episodes * len(args.data)))
    n_hold = args.holdout * per_ep
    tr = slice(0, len(tgt) - n_hold)
    te = slice(len(tgt) - n_hold, len(tgt))
    print(f"  train {tr.stop - tr.start} frames, held out {n_hold}")

    dev = args.device
    height, width = imgs[args.cameras[0]].shape[1:3]
    net = BlockRegressor(len(args.cameras), height, width).to(dev)
    opt = torch.optim.AdamW(net.parameters(), lr=args.lr)

    # Normalise the target, otherwise the loss is dominated by nothing at
    # all: x and y live in an 18 cm box around 0.55 and -0.03.
    mu = tgt[tr].mean(0)
    sd = tgt[tr].std(0) + 1e-8

    def batches(sl, shuffle):
        n = sl.stop - sl.start
        order = np.arange(sl.start, sl.stop)
        if shuffle:
            np.random.shuffle(order)
        for i in range(0, n, args.batch):
            j = order[i:i + args.batch]
            x = [
                torch.from_numpy(imgs[c][j]).permute(0, 3, 1, 2).float().div_(255).to(dev)
                for c in args.cameras
            ]
            y = torch.from_numpy((tgt[j] - mu) / sd).to(dev)
            yield x, y

    history = []
    for epoch in range(args.epochs):
        net.train()
        losses = []
        for x, y in batches(tr, True):
            opt.zero_grad()
            loss = nn.functional.mse_loss(net(x), y)
            loss.backward()
            opt.step()
            losses.append(loss.item())

        net.eval()
        errs = []
        with torch.no_grad():
            for x, y in batches(te, False):
                pred = net(x).cpu().numpy() * sd + mu
                true = y.cpu().numpy() * sd + mu
                errs.append(np.linalg.norm(pred - true, axis=1))
        err = np.concatenate(errs) * 1000.0

        # Where in the workspace the error sits matters more than its mean.
        # ACT delivers 100% for blocks in the middle of the range and 66.7%
        # in the far bin, so the question is whether the probe degrades in
        # the same place. If it does, the far-edge failures are perceptual.
        # If it does not, they are geometry or control and no amount of
        # auxiliary supervision will touch them.
        by_bin = {}
        xs = tgt[te][:, 0]
        for lo, hi in zip(BINS[:-1], BINS[1:]):
            mask = (xs >= lo) & (xs < hi)
            if mask.sum():
                by_bin[f"{lo:.3f}-{hi:.3f}"] = {
                    "n": int(mask.sum()),
                    "median_mm": float(np.median(err[mask])),
                    "p90_mm": float(np.percentile(err[mask], 90)),
                }

        row = {
            "epoch": epoch,
            "by_x_bin": by_bin,
            "train_mse": float(np.mean(losses)),
            "median_mm": float(np.median(err)),
            "mean_mm": float(err.mean()),
            "p90_mm": float(np.percentile(err, 90)),
            "over_10mm": float(np.mean(err > 10.0)),
        }
        history.append(row)
        print(f"  epoch {epoch:2d}  loss {row['train_mse']:.4f}  "
              f"held-out error median {row['median_mm']:5.1f} mm  "
              f"mean {row['mean_mm']:5.1f}  p90 {row['p90_mm']:5.1f}  "
              f"over 10 mm {row['over_10mm'] * 100:4.1f}%")

    best = min(history, key=lambda r: r["median_mm"])
    print("\n--- can the block be located from these pixels? ---")
    print(f"  cameras          {args.cameras}")
    print(f"  resolution       {width}x{height}")
    print(f"  best median      {best['median_mm']:.1f} mm")
    print(f"  best mean        {best['mean_mm']:.1f} mm")
    print(f"  p90              {best['p90_mm']:.1f} mm")
    print(f"  frames over 10mm {best['over_10mm'] * 100:.1f}%")
    print("")
    print("  error by block start x, at the best epoch:")
    for k, v in best.get("by_x_bin", {}).items():
        print(f"    x [{k}]  median {v['median_mm']:5.1f} mm   "
              f"p90 {v['p90_mm']:5.1f} mm   n={v['n']}")
    print("  the oracle positions itself to 6 mm before closing; a lateral")
    print("  error much past 10 mm makes the grasp a coin flip")

    LOG_DIR.mkdir(parents=True, exist_ok=True)
    out = LOG_DIR / f"{args.tag}.json"
    out.write_text(json.dumps({
        "data": args.data, "cameras": args.cameras,
        "resolution": [width, height], "episodes": args.episodes,
        "history": history, "best": best,
    }, indent=2), encoding="utf-8")
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
