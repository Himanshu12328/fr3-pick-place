# The residual 3% diagnosed, the runtime fix for it ruled out, and two reported numbers corrected

Nine measurements. **None of them improves the policy.** Two correct a
previously reported number downward and the rest close off directions. The
value of the branch is the diagnosis and the ruling-out, not a better rate.

Full running log in [`docs/PATH_TO_99.md`](docs/PATH_TO_99.md), in the form
`PATH_TO_97.md` uses: predictions written before the runs that test them,
and wrong entries kept rather than deleted. Four of this branch's own
sections propose a cause that a later section refutes.

---

## 1. The teacher was never 97.5%. It is 99.7%.

`PATH_TO_97.md` S16 closed the previous session with:

> **The student has effectively caught its teacher.** … further gains need a
> better demonstrator rather than a better student — collect at lower
> jitter, or filter the training set to strict-clean episodes.

That rested on the scripted oracle scoring 97.5% strict, which is **39/40**.
Forty trials, against a protocol this repository states elsewhere as "20
trials is ±11% and 100 is ±4%".

| | strict | n |
|---|---|---|
| oracle, jitter 1.0, as reported | 97.5% | **40** |
| **oracle, jitter 1.0, remeasured** | **99.7%** | **1,000** |

The student is **2.7 points behind its teacher**, not level with it. Both
data interventions that conclusion recommended are aimed at a problem that
does not exist: the demonstrator fails three times in a thousand.

## 2. The headline remeasures at 95.75% on seeds that have never been used

| seed set | strict | n |
|---|---|---|
| 71-76 and 81-86, as reported | 97.00% | 1,200 |
| **101-106 and 111-116, never used for anything** | **95.75%** | 1,200 |
| **all 24 seeds pooled** | **96.38%** (CI 95.63-97.12) | **2,400** |

The fresh set is 1.25 points lower. The standard error of that difference
is 0.76 points, **z = 1.64, not significant at the 5% level** — nothing was
fitted to the earlier seeds and the earlier measurement was not wrong. What
it demonstrates is that **1,200 trials cannot quote this rate to the tenth
of a point**: two clean 1,200-trial measurements of one unchanged policy
landed 1.25 points apart.

## 3. The residual failure is a clearance event, not a mistimed grasp

`PATH_TO_97.md` S17 read it as a timing failure: ACT commits to a gripper
close inside its open-loop chunk while still at hover height. Measured
per-step, the policy **commands the descent correctly** — target at −3.5 mm
relative to the block, held 15 consecutive steps — while the tool sits at
+21.9 mm moving 0.01 to 0.08 mm per step against 1.2 to 1.5 in a free
descent, fingers wide open, commanded orientation tracked to 0.4° and both
wrist joints at a fifth of their range.

The arm is **in contact**. A finger is resting on the block's top face:

```
clearance = 40 mm − ( lateral offset + 22·(cos e + sin e) )
```

a finger face sits 40 mm from the tool centre when open; a 44 mm cube at
yaw error `e` presents `22(cos e + sin e)` of half-extent.

| clearance at descent onset | jams caught | non-jams caught |
|---|---|---|
| < 0 mm | 1 of 4 | 3 of 194 |
| < 4 mm | 3 of 4 | 3 of 194 |
| **< 6 mm** | **4 of 4** | **7 of 194** |

Eleven trials in 200 sit under 6 mm and four of them jam: **36% against a
2% base rate**, against a median clearance of 12 mm.

![the residual failure](docs/yaw_jam.png)

**Two terms that trade off**, which is why four separate single-variable
explanations each looked right on a handful of trials and each failed at
scale. One jam happens with the wrist 0.8° from perfect alignment and
13.78 mm of lateral offset; another with 5.26 mm of offset and 29.69 mm of
extent.

## 4. What was ruled out

| intervention | outcome |
|---|---|
| close veto at 5 / 8 / 12 mm | **97.00%**, no change at any threshold |
| grasp-failure detector and replan | **97.00%** |
| both together | **97.00%** |
| stall flush | no change |
| third ensemble member (`act_chunk64`) | **97.0% → 92.0%**, a regression |
| correcting the wrist yaw with the **true** block orientation | **97.0% → 96.0%**, worse |
| camera-only yaw estimator, 6,500 samples | **at chance** (22.3° at 160×128, 21.4° at 320×256, chance 22.5°) |

**All six supervisor configurations fail the same six trials.** The veto
fired on exactly the right four — 36 refused closes, up to 29 in one
episode — and refusing a close does not unjam an arm. The retry fired on
exactly the right four and none recovered.

Correcting the yaw with privileged ground truth rescues the two jams whose
*extent* term is large, buys **0.3 mm** for the one at 0.8°, and breaks four
previously-passing trials by walking the policy off its training
distribution. Fixing one term of a two-term budget cannot clear a budget
short on the other.

And `PATH_TO_97.md` S3 — "The resolution hypothesis is dead" — is correct
about **position**, which is legible to 3.1 mm. Nobody had measured **yaw**,
and yaw is at chance at both resolutions tested.

---

## Code

* `src/eval/supervisor.py` — the runtime layer. Three rules, all
  proprioception only. Its docstring leads with the fact that it is worth
  zero points, because the obvious extensions are already ruled out.
  **Monitor mode is the part that earned its place**: it returns the inner
  policy's action byte for byte, reproduces the baseline at 97.00% over 200
  trials, and records every close, the height and offset it happened at,
  the finger width 12 steps later, where a descent began, whether it
  jammed, the commanded yaw error, and the ensemble members' disagreement.
  Every threshold in the file came from those distributions over hundreds
  of trials.
* `src/scripts/eval_supervised.py` — runs the four configurations and
  **parallelises by seed**. The harness draws placements from one generator
  advanced trial by trial, so splitting a seed across processes would
  change the placements and silently produce a different experiment;
  splitting by seed keeps every trial bit-identical. 200 trials in 3.6 min
  against ~15 serial.
* `src/eval/strict.py`, `src/rl/reference.py` — `in_time` widened by one
  measured approach phase (105 steps) per recovery, with
  `strict_success_unwidened` reported alongside every result; `demo_like`
  now scores from the robust phase frames the gates already compute.
  **Proven inert**: all 221 demonstrations close the gripper exactly once
  and their trajectory scores are bit-identical under both definitions
  (max difference 0.00e+00).
* Diagnostics: `probe_stuck_grasp.py`, `probe_yaw_ceiling.py`,
  `train_yaw_head.py`, `yaw_legibility.py`, `figure_yaw_jam.py`.
* `tests/test_invariants.py` — **repairs a guard that had been silently
  failing.** `test_eval_resolution_matches_training_resolution` protects
  against the most expensive defect in this project's history (60% → 0.4%
  from a resolution mismatch), and `module_constant` could not read
  `CAM_WIDTH, CAM_HEIGHT = TRAIN_WIDTH, TRAIN_HEIGHT` because
  `ast.literal_eval` raises on a bare Name. It returned `None`, the
  assertion compared `None` against 160, and with CI disabled nothing
  reported it. Plus three new invariants, including one that parses
  `supervisor.py` and fails if anything but `action[7]` is ever assigned —
  the constraint that every target pose is the policy's own output is the
  basis of the result and is now enforced mechanically rather than by
  docstring.

## Two constants in the existing diagnostic did not survive being used at scale

* `FINGER_ON_BLOCK_M = 0.012` reasons from geometry that a pair of fingers
  closed on air "goes to nearly zero". They stop at **15.35 to 19.11 mm**,
  catching the block's top edge. That constant calls all four air closes
  successful grasps.
* The same script reads finger width 25 steps after a close. A real grasp
  resolves in **one** step; an air close is still travelling at 8. The two
  populations first separate cleanly at 12.

## What the evidence points at next

The **lateral** term is the one with room in it — position is legible to
3.1 mm against a p5 clearance of 7 mm. The auxiliary block-position target
built and validated in `PATH_TO_97.md` S5 and never used aims exactly
there, and `to_lerobot.py` gains an `--aux xy` option that supervises
position only, omitting the yaw that the images do not contain.

The **yaw** term cannot be improved from these three views. This is the
first evidence in the project pointing at camera placement rather than at
the policy.
