"""
Trains a camera-only estimator of the block's yaw, in the one
representation that is single-valued for a cube.

Why this exists. S23 established that the ensemble's residual 3% is a
discontinuous regression target: the policy outputs an absolute gripper
quaternion, a cube is symmetric every 90 degrees, and the aligned label is
therefore a sawtooth in the block's yaw. Two blocks at 44 and 46 degrees
look nearly identical and their labels sit 88 degrees apart, so an
L1-regressed policy smears across the jump and the fingers arrive 12 to 18
degrees off. Failures climb monotonically from 0% below a 30 degree yaw
offset to 12.5% between 40 and 45.

The fix that does not need the policy retrained is to measure the block's
yaw and snap the commanded wrist yaw to the nearest exact alignment.
`probe_yaw_ceiling.py` measures what that is worth using the simulator's
own block orientation; this trains the part that would make it deployable,
from the same three camera images the policy sees and nothing else.

**The representation is the whole point.** A raw angle cannot be regressed
here: theta and theta+90 are the same cube in the same pose, so identical
images would carry labels 90 degrees apart, which is the defect being
fixed rather than a fix for it. The target is instead

    (sin 4*theta, cos 4*theta)

which is continuous, and identical for all four orientations that are
physically identical. Recovering theta from it gives the alignment modulo
90 degrees, which is exactly and only what a grasp needs.

Written as a separate script rather than as a flag on
`perception_probe.py` on purpose. That script's reported result — the block
position is legible to 3.1 mm at 160x128 — is quoted in three documents,
and widening its target would quietly change what the number refers to.

Run:
    python -m src.scripts.train_yaw_head --data data/oracle_v2 --epochs 14
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from PIL import Image

from src.config import CAMERAS, LOG_DIR, MODELS_DIR

# The cube's symmetry order about the vertical axis. Four faces, so the
# grasp only cares about yaw modulo 90 degrees.
SYMMETRY = 4

# How many pre-grasp frames per episode. The arm is at the shared home
# keyframe at frame 0 and above the block by the last pre-grasp frame, so
# sampling across the approach covers both the far view and the near one.
FRAMES_PER_EPISODE = 6

# Restrict sampling to the first part of the approach.
#
# This is the control that keeps the experiment honest, and it is inherited
# from perception_probe.py for the same reason. Sampling the whole approach
# leaks the answer: the oracle drives the wrist into alignment with the
# block, so by the end of the approach the block's yaw can be read off the
# arm's own joint angles without the block ever being seen. At the start of
# the approach every episode's arm is in the identical pose and the only
# thing that differs between images is the block.
FRAC_LO, FRAC_HI = 0.0, 0.35


class YawHead(nn.Module):
    """
    A small convolutional stack per camera into a two-number head.

    The same architecture as the position probe, deliberately: the question
    is whether the information is in the pixels, so the network should be
    unconstrained enough to answer that and small enough to train in
    minutes.
    """

    def __init__(self, n_cameras):
        """
        input:  n_cameras (int)
        output: YawHead
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
            nn.Linear(256, 2),
        )

    def forward(self, images):
        """
        input:  images (list of tensors, each B,3,H,W)
        output: tensor B,2 holding (sin 4*theta, cos 4*theta), unnormalised
        """
        feats = [t(x) for t, x in zip(self.trunks, images, strict=True)]
        out = self.head(torch.cat(feats, dim=1))
        # Project onto the unit circle. The target lives there by
        # construction, so constraining the output there removes a degree of
        # freedom the network would otherwise have to learn to ignore.
        return out / out.norm(dim=1, keepdim=True).clamp_min(1e-6)


def block_yaw(meta):
    """
    Reads the block's yaw from an episode's recorded start pose.

    input:  meta (dict) episode metadata
    output: float, yaw in radians
    """
    q = np.asarray(meta["block_start_quat"], dtype=np.float64)
    return float(2.0 * np.arctan2(q[3], q[0]))


def grasp_frame(action):
    """
    First frame where the gripper was commanded shut.

    input:  action (array (T,8))
    output: int
    """
    closed = np.where(action[:, 7] < 0.02)[0]
    return int(closed[0]) if len(closed) else len(action)


def load_split(data_dirs, cameras, limit=None):
    """
    Loads pre-grasp frames and their symmetry-folded yaw targets.

    input:  data_dirs (list of Path), cameras (list of str),
            limit (int or None) episodes per directory
    output: (images dict camera -> uint8 array, targets (N,2), yaws (N,),
             episode index per frame (N,))
    """
    images = {c: [] for c in cameras}
    targets, yaws, ep_ids = [], [], []
    ep_counter = 0

    for d in data_dirs:
        eps = sorted(p for p in Path(d).iterdir()
                     if p.name.startswith("episode_"))
        if limit:
            eps = eps[:limit]

        for ep in eps:
            meta = json.loads((ep / "meta.json").read_text())
            npz = np.load(ep / "data.npz")
            g = grasp_frame(npz["action"])
            if g < max(FRAMES_PER_EPISODE, 4):
                continue

            lo = int(round(FRAC_LO * (g - 1)))
            hi = int(round(FRAC_HI * (g - 1)))
            idx = (np.array([lo]) if hi <= lo else
                   np.unique(np.linspace(lo, hi, FRAMES_PER_EPISODE).astype(int)))

            theta = block_yaw(meta)
            target = np.array([np.sin(SYMMETRY * theta),
                               np.cos(SYMMETRY * theta)], dtype=np.float32)

            for i in idx:
                for c in cameras:
                    images[c].append(np.asarray(
                        Image.open(ep / c / f"{i:05d}.png"), dtype=np.uint8))
                targets.append(target)
                yaws.append(theta)
                ep_ids.append(ep_counter)
            ep_counter += 1

    return ({c: np.stack(v) for c, v in images.items()},
            np.stack(targets), np.asarray(yaws), np.asarray(ep_ids))


def angle_error_deg(pred, target):
    """
    Angular error in the folded frame, in degrees.

    Both vectors encode 4*theta, so the angle between them is four times
    the yaw error. Dividing by the symmetry order gives the error in the
    only quantity a grasp cares about: the alignment modulo 90 degrees.

    input:  pred (N,2), target (N,2)
    output: array (N,) of degrees
    """
    a = np.arctan2(pred[:, 0], pred[:, 1])
    b = np.arctan2(target[:, 0], target[:, 1])
    d = np.abs(np.mod(a - b + np.pi, 2 * np.pi) - np.pi)
    return np.degrees(d) / SYMMETRY


def main():
    """
    Entry point.

    input:  none
    output: None
    """
    p = argparse.ArgumentParser()
    p.add_argument("--data", nargs="+", default=["data/oracle_v2"])
    p.add_argument("--cameras", nargs="+", default=CAMERAS)
    p.add_argument("--episodes", type=int, default=600)
    p.add_argument("--holdout", type=int, default=120,
                   help="episodes held out, taken from the end")
    p.add_argument("--epochs", type=int, default=14)
    p.add_argument("--batch", type=int, default=64)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--device", default="cuda")
    p.add_argument("--out", default=str(MODELS_DIR / "yaw_head.pt"))
    p.add_argument("--tag", default="yaw_head")
    args = p.parse_args()

    print(f"loading up to {args.episodes} episodes from {args.data}")
    t0 = time.perf_counter()
    imgs, tgt, yaws, ep_ids = load_split(
        [Path(d) for d in args.data], args.cameras, limit=args.episodes)
    print(f"  {len(tgt)} frames from {ep_ids.max() + 1} episodes in "
          f"{time.perf_counter() - t0:.0f}s")

    # Split by episode, never by frame. Frames from one episode are near
    # duplicates, so a random split leaks across it and the held-out error
    # comes back meaninglessly small.
    n_eps = int(ep_ids.max()) + 1
    cut = n_eps - args.holdout
    tr_mask = ep_ids < cut
    te_mask = ~tr_mask
    tr_idx = np.where(tr_mask)[0]
    te_idx = np.where(te_mask)[0]
    print(f"  train {len(tr_idx)} frames from {cut} episodes, "
          f"held out {len(te_idx)} frames from {n_eps - cut}")

    dev = args.device
    net = YawHead(len(args.cameras)).to(dev)
    opt = torch.optim.AdamW(net.parameters(), lr=args.lr)

    def batches(idx, shuffle):
        order = idx.copy()
        if shuffle:
            np.random.shuffle(order)
        for i in range(0, len(order), args.batch):
            j = order[i:i + args.batch]
            x = [torch.from_numpy(imgs[c][j]).permute(0, 3, 1, 2)
                 .float().div_(255).to(dev) for c in args.cameras]
            y = torch.from_numpy(tgt[j]).to(dev)
            yield j, x, y

    best = None
    history = []
    for epoch in range(args.epochs):
        net.train()
        losses = []
        for _, x, y in batches(tr_idx, True):
            loss = nn.functional.mse_loss(net(x), y)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
            losses.append(float(loss))

        net.eval()
        preds = np.zeros((len(te_idx), 2), dtype=np.float32)
        pos = 0
        with torch.no_grad():
            for _, x, _y in batches(te_idx, False):
                out = net(x).cpu().numpy()
                preds[pos:pos + len(out)] = out
                pos += len(out)

        err = angle_error_deg(preds, tgt[te_idx])
        row = {
            "epoch": epoch,
            "train_loss": float(np.mean(losses)),
            "median_deg": float(np.median(err)),
            "mean_deg": float(err.mean()),
            "p90_deg": float(np.percentile(err, 90)),
            "over_5deg": float(np.mean(err > 5.0)),
        }
        history.append(row)
        print(f"  epoch {epoch:2d}  loss {row['train_loss']:.4f}  "
              f"held-out yaw error median {row['median_deg']:5.2f} deg  "
              f"mean {row['mean_deg']:5.2f}  p90 {row['p90_deg']:5.2f}  "
              f"over 5 deg {row['over_5deg'] * 100:5.1f}%")

        if best is None or row["median_deg"] < best["median_deg"]:
            best = row
            torch.save({
                "state_dict": net.state_dict(),
                "cameras": args.cameras,
                "symmetry": SYMMETRY,
                "epoch": epoch,
                "median_deg": row["median_deg"],
            }, args.out)

    print(f"\n  best epoch {best['epoch']}: median {best['median_deg']:.2f} deg, "
          f"mean {best['mean_deg']:.2f}, p90 {best['p90_deg']:.2f}")
    print(f"  saved {args.out}")

    # The number that decides whether snapping is deployable. The failures
    # being fixed are 12 to 18 degrees off, and a grasp tolerates a few
    # degrees, so an estimator much past 5 would be replacing one error
    # with another.
    verdict = ("usable: snapping on this estimate would put the wrist "
               "inside the tolerance every time"
               if best["p90_deg"] < 5.0 else
               "NOT usable as-is: p90 exceeds 5 degrees")
    print(f"  {verdict}")

    (LOG_DIR / f"{args.tag}.json").write_text(
        json.dumps({"history": history, "best": best,
                    "data": args.data, "cameras": args.cameras},
                   indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
