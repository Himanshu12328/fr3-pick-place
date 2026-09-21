"""
Measures how much of the residual failure rate is the gripper yaw, by
fixing the yaw with privileged information and scoring what is left.

**This is a diagnostic and it is not a deliverable.** It reads the block's
true orientation out of the simulator and it overwrites a motion command,
so it violates both of the constraints the supervised result is held to. It
exists to answer one question before four hours of GPU time are spent on
the answer: if the commanded gripper yaw were exactly right, how many of
the remaining failures would go away?

The reason to ask. The residual failures are a yaw error, and the yaw error
has a specific form. Measured over 200 trials, the commanded wrist yaw is
under-rotated by a constant fraction of the rotation the block asks for:

    commanded yaw error = 0.115 x block yaw offset,  r = 0.476

The ratio holds across every bin from 0 to 45 degrees of offset, which is
**shrinkage toward the mean** rather than the averaging-at-a-symmetry-
boundary story an earlier draft of this file told. Failures concentrate at
large offsets because a proportional error grows with the rotation
required — mean 5.5 degrees and worst 18.3 at 40 to 45 degrees of offset —
and its upper tail crosses what a grasp tolerates. Beyond that tolerance
the open fingers cannot straddle the cube, one lands on its top face, and
the descent jams 21 mm up.

If snapping the yaw recovers the teacher's 99.7%, the residual is the yaw
and nothing else, and a retraining run aimed at it is justified. If it
recovers little, the yaw is a symptom of something else again and the four
hours would be wasted.

Run:
    python -m src.scripts.probe_yaw_ceiling --seeds 61 62 91 92 --trials 50
"""

import argparse
import json

import numpy as np

from src.config import LOG_DIR
from src.data.task import BLOCK_Z

# How near grasp height the commanded target has to be before the yaw is
# corrected. Forty millimetres is below the hover the policy approaches at,
# so by then it has already rotated the wrist most of the way and what is
# left is the shrinkage residual.
CORRECT_BELOW_M = 0.040


class YawSnappedPolicy:
    """
    Wraps a policy and replaces the commanded yaw with the nearest exact
    alignment to the block.

    Position, height and the gripper channel are the policy's own. Only the
    yaw of the commanded orientation is overwritten, and it is overwritten
    with the answer, read from the simulator.
    """

    def __init__(self, policy, enabled=True):
        """
        input:  policy (callable), enabled (bool) snap or pass through
        output: YawSnappedPolicy instance
        """
        self.policy = policy
        self.enabled = bool(enabled)
        self.model = None
        self.data = None
        self.snaps = 0
        self.max_correction_deg = 0.0
        self.target_yaw = None
        self.done = False

    def bind(self, model, data):
        """
        input:  model (MjModel), data (MjData)
        output: None
        """
        import mujoco

        self.model = model
        self.data = data
        self._block = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "block")
        from src.config import FINGER_DOFS
        self._fingers = list(FINGER_DOFS)
        if hasattr(self.policy, "bind"):
            self.policy.bind(model, data)

    def reset(self):
        """
        input:  none
        output: None
        """
        if hasattr(self.policy, "reset"):
            self.policy.reset()
        self.target_yaw = None
        self.done = False

    def __call__(self, obs):
        """
        input:  obs (dict)
        output: numpy array of shape (8,)
        """
        action = np.asarray(self.policy(obs), dtype=np.float64).copy()
        if not self.enabled or self.model is None:
            return action

        # Stop once the grasp has happened. After the fingers are on the
        # block, the block turns with the hand and its yaw no longer says
        # anything about where the wrist should be; the place and the
        # retreat are left entirely to the policy.
        if self.done:
            return action
        fingers_shut = float(np.mean(self.data.qpos[self._fingers])) < 0.024
        if fingers_shut and action[7] < 0.02:
            self.done = True
            return action

        q = action[3:7] / max(np.linalg.norm(action[3:7]), 1e-9)

        # Yaw of the commanded gripper frame and of the block, in the same
        # convention diagnose_failures.py uses.
        gyaw = 2.0 * np.arctan2(q[2], q[1])
        bq = np.asarray(self.data.xquat[self._block], dtype=np.float64)
        byaw = 2.0 * np.arctan2(bq[3], bq[0])

        quarter = np.pi / 2.0

        # Correct the residual error, late in the approach, and nothing else.
        #
        # Two earlier versions of this were wrong and both were wrong in the
        # same direction — too large an intervention — so they are recorded
        # here rather than deleted.
        #
        # The first recomputed the nearest alignment every step. Near the
        # boundary between two alignments the rounding flips branch step to
        # step, the commanded yaw jumps 90 degrees each time and the arm
        # chatters: 1,461 corrections over three trials, largest 44.98
        # degrees, 33% strict against a 97% baseline. It reproduced the very
        # defect it was built to diagnose.
        #
        # The second latched one absolute target yaw at step 0 and held it.
        # That removes the chatter and introduces a worse problem: at step 0
        # the wrist is at the home pose, so the latched target can be 45
        # degrees away and the correction yanks the wrist to its final
        # orientation immediately. The approach no longer looks like
        # anything in the training data, the policy is fed a pose it has
        # never seen, and the result was 70% with corrections up to 88.74
        # degrees.
        #
        # What the measurement actually calls for is small. The policy's own
        # yaw is under-rotated by 0.115 of the required rotation, so the
        # residual is a few degrees and at worst 18. Correcting *that*,
        # only once the policy has committed to the descent and has already
        # rotated the wrist most of the way, leaves the trajectory shape
        # untouched and tests exactly the quantity in question.
        if float(action[2] - BLOCK_Z) > CORRECT_BELOW_M:
            return action

        signed = float(np.mod(gyaw - byaw + quarter / 2.0, quarter)
                       - quarter / 2.0)
        correction = -signed

        if abs(correction) > 1e-9:
            self.snaps += 1
            self.max_correction_deg = max(
                self.max_correction_deg, abs(np.degrees(correction))
            )

        # Rotate the commanded quaternion about the world z axis by the
        # correction. The gripper is down at all times, so a z rotation is
        # the yaw and leaves the approach direction untouched.
        h = correction / 2.0
        rz = np.array([np.cos(h), 0.0, 0.0, np.sin(h)])
        w0, x0, y0, z0 = rz
        w1, x1, y1, z1 = q
        action[3:7] = np.array([
            w0 * w1 - x0 * x1 - y0 * y1 - z0 * z1,
            w0 * x1 + x0 * w1 + y0 * z1 - z0 * y1,
            w0 * y1 - x0 * z1 + y0 * w1 + z0 * x1,
            w0 * z1 + x0 * y1 - y0 * x1 + z0 * w1,
        ])
        return action


def main():
    """
    Entry point.

    input:  none
    output: None
    """
    from src.eval.strict import evaluate_strict, summarise
    from src.scripts.eval_supervised import DEFAULT_MEMBERS, build_policy

    p = argparse.ArgumentParser()
    p.add_argument("--members", nargs="+", default=DEFAULT_MEMBERS)
    p.add_argument("--seeds", type=int, nargs="+", default=[61, 62, 91, 92])
    p.add_argument("--trials", type=int, default=50)
    p.add_argument("--device", default="cuda")
    p.add_argument("--off", action="store_true", help="control: do not snap")
    p.add_argument("--tag", default="yaw_ceiling")
    args = p.parse_args()

    inner, _ = build_policy(args.members, "monitor", device=args.device)
    policy = YawSnappedPolicy(inner, enabled=not args.off)

    print(f"yaw snapping {'OFF (control)' if args.off else 'ON'}: "
          f"{args.trials} trials x {len(args.seeds)} seeds\n", flush=True)

    rows = []
    for seed in args.seeds:
        _, r = evaluate_strict(policy, n_trials=args.trials, seed=seed,
                               need_images=True, verbose=False)
        rows += r
        print(f"  seed {seed}: {sum(x['strict_success'] for x in r)}/{len(r)}",
              flush=True)

    total = summarise(rows)
    fails = [r for r in rows if not r["strict_success"]]
    print(f"\n  STRICT {total['strict'] * 100:.2f}%  ({len(rows)} trials, "
          f"{len(fails)} failures)")
    print(f"  loose  {total['loose'] * 100:.2f}%")
    print(f"  yaw corrections applied: {policy.snaps}, largest "
          f"{policy.max_correction_deg:.2f} deg")
    for g, v in total["gates"].items():
        print(f"    {g:<16}{v * 100:7.2f}%")
    if fails:
        print("\n  failing trials:")
        for r in fails:
            miss = ",".join(g for g, ok in r["gates"].items() if not ok)
            print(f"    seed {r['seed']} trial {r['trial']:3d}  "
                  f"lift {r['lift_m'] * 1000:5.1f} mm  missing: {miss}")

    out = LOG_DIR / f"probe_{args.tag}.json"
    out.write_text(json.dumps(
        {"strict": total["strict"], "n": len(rows), "snaps": policy.snaps,
         "max_correction_deg": policy.max_correction_deg,
         "enabled": not args.off, "rows": rows}, default=float),
        encoding="utf-8")
    print(f"\n  wrote {out}")


if __name__ == "__main__":
    main()
