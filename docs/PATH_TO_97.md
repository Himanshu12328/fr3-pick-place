# The path to 97%, under a criterion that means something

A running log. Every step, every prediction written before the run, and
every failure. Nothing is removed when it turns out to be wrong; wrong
entries are the point of the document.

The target: **97% strict pick-and-place success**, six seeds of 100 trials
minimum, reporting seeds disjoint from selection seeds.

---

## S0. The instrument, and why every earlier number needed replacing

### The problem

Every ACT number in this project, 58.7% through 86.8%, was measured with
`check_success`:

```python
horizontal < 0.05 and abs(pos[2] - BLOCK_Z) < 0.02
```

The block ends near the target and is roughly at table height. That is all.

Stage 2 already established, five separate times and each time on video,
that this is not a description of the task. A block **shoved** across the
table satisfies it. A block **dropped** from 95 mm satisfies it the instant
it lands. A block still **gripped** and hovering inside the 20 mm tolerance
satisfies it. Teachers scoring 99.0%, 99.8% and 100% were each rejected
after passing it.

Those lessons were written into `eval_teacher.py`, which scores the RL
teacher on five gates. **They were never applied to the vision student.**
So the entire ACT results table was measured with the criterion this project
had already proven certifies the behaviours it rejects.

The task was specified as a sequence:

> approach → grasp → move up → go to the green spot → descend → release the
> block → move the arm up, retracting in z by about the block height before
> translating home, so the block is not disturbed.

`check_success` tests one instant of that and nothing else. Worse,
`rollout.run_trial` **breaks the loop the moment it fires**, so the recorded
episode length is meaningless and every frame after the block touches down,
the entire retreat and a third of what a demonstration contains, was never
simulated, never scored, and never seen.

### What was built

`src/eval/strict.py`. Eight gates, each one traceable to something that
actually failed:

| gate | requirement | exists because |
|---|---|---|
| `delivered` | block within 5 cm and resting, **after a 60-step settle** | 2 successes in 100 rolled out of the radius after the harness stopped watching |
| `lifted` | block reached 40 mm above resting | a teacher scored 99.0% shoving, mean lift 8 mm |
| `carried` | fingers commanded shut, block airborne **and** within 60 mm of the tool | separates a carry from a flick |
| `released_low` | at the release frame: tool below 0.45, block within 5 mm of resting, block under 50 mm/s | a teacher opened its fingers 95 mm up at 303 mm/s and the criterion fired as it landed |
| `lifted_off` | tool rose ≥ 30 mm before translating > 20 mm sideways | open fingers dragged the placed block 17.4 mm on average, up to 38.4 mm |
| `home` | tool ends within 60 mm of the home xy, having reached z ≥ 0.50 | the demonstrations retreat 282 mm over 59 steps; nothing required it |
| `undisturbed` | block moved ≤ 5 mm between release and the end of the settle | the oracle moves it 0.06 mm |
| `in_time` | episode length within the demonstrations' p10 to p90, 227 to 367 steps | a teacher did it in 73 steps and looked nothing like a demonstration |

A trial passes only if **all eight** hold.

Two changes to how a trial is run, both required for the gates to be
measurable at all:

* The episode **no longer breaks on `check_success`**. It ends when the
  whole sequence is complete: block delivered and resting, fingers open,
  tool climbed and returned home. That is what makes episode length mean
  the same thing it means in the demonstrations.
* The settle **holds the last commanded gripper** rather than forcing the
  fingers open. A policy that ends still gripping the block keeps gripping
  it and fails honestly, instead of having the harness let go for it.

The bands are not invented. `src/rl/reference.py` already held them as
constants measured from all 221 recorded episodes.

### Validating the instrument before trusting it

The same argument `test_harness.py` and `test_rl_env.py` make. A metric that
has not been scored on a known-good policy is aimed at nothing.

**Positive control, the scripted oracle, 60 trials:**

| | jitter 0, n=20 | jitter 1.0, n=40 |
|---|---|---|
| strict | **100.0%** | **100.0%** |
| every individual gate | 100% | 100% |
| lift | 78 mm (demos 80) | 78 mm |
| episode length | 284 (demos 295) | 283 |
| trajectory score | 0.988 | 0.988 |
| block moved after release | 0.06 mm | 0.06 mm |

These reproduce the documented oracle figures exactly, including the
0.06 mm disturbance and the 0.991-vs-0.988 trajectory score. The instrument
agrees with the one policy already known to be correct.

**Negative control, the same oracle at 3× demonstration speed.** Identical
waypoints, identical controller, `step_m` raised from the measured per-phase
speeds to a flat 12 mm/step:

| | result |
|---|---|
| loose `check_success` | **100.0%** |
| **strict** | **0.0%** |
| `in_time` | 0.0% (205 steps, band is 227 to 367) |
| `released_low` / `lifted_off` / `home` / `undisturbed` | 85% each |
| trajectory score | 0.594 against the oracle's 0.988 |

**A policy can score 100% on the old criterion and 0% on this one.** That is
the entire case for the rewrite, demonstrated rather than argued, and it is
the same 100%-while-wrong signature the RL teachers produced.

### Incidental fix: render resolution can no longer drift

`config.py` warns that the training and evaluation resolution must match,
because the backbone is fully convolutional and accepts any size silently.
A mismatch once cost this project 60% success down to 0.4% with no error
raised. It then stated the constant, and `rollout.py` **stated it a second
time**, hardcoded. Two copies of a constant that must agree is that bug
waiting to happen, and this work is about to introduce a second resolution.

`rollout.py` now imports it, and `strict.use_checkpoint_render_size` reads
the resolution out of the checkpoint's own `config.json` and renders at
that. The constant cannot disagree with the weights any more, because it is
no longer a constant.

---

## S1. The real baseline: 77.0% strict against 91.0% loose

`act_oracle_v1` checkpoint 20,000, the policy behind the 86.8% headline,
100 trials on seed 51.

| | |
|---|---|
| loose `check_success` | **91.0%** |
| **strict pick and place** | **77.0%** |

The loose number reproduces the previously reported 92% on this seed to
within one trial, which is the harness agreeing with itself. **The strict
number is 14 points lower.**

Failure structure, which is more useful than the rate:

| what happened | count |
|---|---|
| passed every gate | 77 |
| failed **only** the timing band | 8 |
| failed **only** the release criterion | 4 |
| hard failures, ran the full 600 steps | 9 |
| other combinations | 2 |

### The 9 hard failures are all the same failure

| steps | lift | final distance | block start x |
|---|---|---|---|
| 600 | 8.8 mm | 27.1 cm | 0.630 |
| 600 | 30.6 mm | 15.2 cm | 0.638 |
| 600 | 16.1 mm | 16.4 cm | 0.635 |
| 600 | 9.4 mm | 27.1 cm | 0.629 |
| 600 | 7.7 mm | 17.1 cm | 0.605 |
| 600 | 62.9 mm | 5.1 cm | 0.613 |
| 600 | 14.0 mm | 15.2 cm | 0.609 |
| 600 | 9.0 mm | 15.7 cm | 0.473 |
| 600 | 12.0 mm | 27.0 cm | 0.501 |

Seven of nine never lifted the block past 31 mm. The gripper closed at a
normal frame and opened again a few steps later. **These are grasps that
never happened**, which is exactly the failure mode Stage 0 measured at 62%
with zero near misses, now confirmed on the best policy under a harness that
can see it. Six of the nine started at `x >= 0.605`, the far edge of the
block distribution, matching the far-region deficit (73.7% strict against
81.4% near).

---

## S2. Two defects in my own instrument, found by using it

Recorded because the gates are the foundation for everything after this, and
both would have quietly distorted every number.

### The timing band was rejecting near-perfect trajectories

`in_time` was set to the demonstrations' p10 to p90, 227 to 367 steps. **By
construction 20% of the demonstrations fall outside their own p10 to p90.** The
gate was demanding behaviour tighter than the data it was measured from.

On ACT it threw out 8 episodes. Their trajectory scores against the
demonstration profile:

```
397 steps → 0.947    372 → 0.903    218 → 0.998    221 → 0.998
398 steps → 0.914    380 → 0.846    377 → 0.953    391 → 0.823
```

Two of them scored **0.998**, essentially indistinguishable from a
demonstration, and were failed for being nine steps quicker than a
percentile.

Measured over all 221 recorded episodes:

| | value |
|---|---|
| episode length | min **195**, max **452** |
| trajectory score | min **0.856**, 100% at or above 0.85 |

So the band became **195 to 452 steps**, and a ninth gate `demo_like` was
added at **trajectory score ≥ 0.85**. Both are now calibrated so that
**every one of the 221 recorded demonstrations passes them.**

Widening the length band alone would have been wrong: the rushed oracle
finishes in 205 steps, inside 195 to 452. What separates it is trajectory
*shape*, not duration, and it scores 0.594. The two gates together reject
it and neither does alone. Re-verified: oracle **100%**, rushed oracle
**0% strict / 100% loose**.

### The release velocity was measuring solver noise

`released_low` requires the block under 50 mm/s when the fingers open. It
was computed as a one-step finite difference of the recorded block height.
At 30 Hz, 2 mm of numerical settling reads as 60 mm/s.

Four of the five failures were episodes whose tool was correctly 25 mm below
the release ceiling and whose block was at rest, failed on 59, 61 and
75 mm/s of arithmetic. The trace now records the simulator's own block
velocity instead.

---

## S3. The resolution hypothesis is dead. Measured, not argued.

The plan approved before any of this was: re-render at higher resolution,
larger backbone, auxiliary loss. The reasoning was that the 1,006-episode
dataset peaked at 4.3 epochs and gained only 3 points, which is not a
data-limited model, and that "62% missed grasps with zero near misses" reads
as a policy that cannot see the block.

**That reasoning was half right, and the expensive half was wrong.**

`src/scripts/perception_probe.py` trains a small convolutional regressor on
exactly the images the policy is given, with one job: predict where the
block is. Labels are free, because before the gripper first closes the block
has not moved, so every pre-grasp frame is labelled by its episode's
recorded `block_start_pos`.

**First attempt, and why it proved nothing.** 250 episodes, 6 frames each,
10 epochs: 17.1 mm median error. That looks like a confirmation of the
hypothesis. It was not, because the loss was still falling steeply at the last
epoch on 1,200 training frames. It measured an undertrained probe, not the
images.

**Second attempt, and the confound in it.** 800 episodes, 10 frames each,
40 epochs: **2.0 mm median**. But the frames were sampled across the whole
approach, and the oracle drives the arm *to the block*, so in late frames
the tool sits directly above it. A probe can read the block position off the
arm's own pose without ever seeing the block. That number was not safe
either.

**The control.** Sample **frame 0 only**. At frame 0 the arm is at the
identical home keyframe in every episode, so the only thing that differs
between images is the block. 800 episodes, held out by episode, 60 epochs:

| resolution | median | mean | p90 | over 10 mm |
|---|---|---|---|---|
| **160 × 128, frame 0 only** | **3.1 mm** | 3.7 mm | 7.2 mm | **2.5%** |

The oracle positions itself to 6 mm before closing, and the fingers span
40 mm around a 44 mm block, so a lateral error past about 10 mm makes the
grasp a coin flip. **3.1 mm median with 2.5% past 10 mm is not the
bottleneck.**

**The block position is present in the 160 × 128 images with margin.
Re-rendering at higher resolution would have cost a day and bought
nothing.** The plan changes here: resolution and backbone size are dropped,
and the auxiliary-supervision half of the approved plan becomes the whole of
it.

The question is no longer "can the policy see the block". It is "why does a
policy that is given a 3 mm-legible block position still close its fingers
on empty table 9% of the time". That is a representation and objective
problem, not a sensing one.

---

## S4. Where the failures are, and why it is not a coverage problem

The nine hard failures clustered at the far edge of the block range. Binning
the 100 strict trials by block start x, against the same bins for the
training data:

| block start x | ACT delivered | oracle training episodes | human episodes |
|---|---|---|---|
| 0.460 to 0.490 | 95.8% | 18.0% | 19.5% |
| 0.490 to 0.520 | 90.0% | 19.9% | 23.1% |
| 0.520 to 0.550 | **100.0%** | 14.9% | 21.7% |
| 0.550 to 0.580 | **100.0%** | 16.0% | 13.6% |
| 0.580 to 0.610 | 86.7% | 14.1% | 12.7% |
| **0.610 to 0.640** | **66.7%** | 17.1% | 9.5% |

**The far bin is not undersampled in the oracle data.** It holds 17.1% of
the 800 oracle episodes, essentially the uniform share. This is the opposite
of the near-region deficit found at 221 episodes, which *was* a coverage
problem and was fixed by collecting 70 targeted episodes.

The human data is thin there, 9.5%, which is one more argument for dropping
it. But the oracle data alone already covers the region evenly and the
policy still fails a third of the time in it.

### Is the far edge harder to see?

Re-running the frame-0 perception probe, binned the same way:

| block start x | probe median | probe p90 | n |
|---|---|---|---|
| 0.460 to 0.490 | 2.7 mm | 6.5 mm | 14 |
| 0.490 to 0.520 | 3.8 mm | 5.9 mm | 12 |
| 0.520 to 0.550 | **1.2 mm** | 2.0 mm | 8 |
| 0.550 to 0.580 | **1.5 mm** | 4.2 mm | 15 |
| 0.580 to 0.610 | 2.7 mm | 5.7 mm | 15 |
| **0.610 to 0.640** | **4.7 mm** | **11.0 mm** | 16 |

The far bin is the worst for the probe too, about three times the
mid-range error, and the only bin whose p90 crosses 10 mm. The per-bin
counts are 8 to 16, so this is directional rather than precise.

But the size of the effect matters. **4.7 mm is still well inside the grasp
tolerance**, and the probe manages it while ACT fails 33% of trials in the
same region. So degraded sensing is a contributing factor and not the
mechanism. The reading that fits both measurements is that the far edge
carries a slightly weaker signal, and ACT, which is not required to
represent block position at all, amplifies that weakness into a missed
grasp, while a network explicitly trained to extract it does not.

That is an argument for auxiliary supervision specifically, and it predicts
the gain should be largest in the far bin. Written down before the run.

---

## S5. Plan of record for the next two runs, written before execution

Two variables changed against the 86.8% policy, and **they are changed one
at a time**, because this project has already paid for the alternative: RL
run v8 changed gamma and the demo ratio in the same 900,000-step run and
could attribute nothing afterwards.

| run | dataset | action vector | what it isolates |
|---|---|---|---|
| **A** `act_oracle_only` | 800 oracle episodes, no human | 8 dims | whether dropping the 221 human demonstrations helps or hurts |
| **B** `act_oracle_aux` | the same 800 episodes | 11 dims, block x/y/yaw appended | whether auxiliary supervision helps, holding the data fixed |

Run A is also the oracle-only ablation the README listed as the outstanding
next step, so it is owed regardless.

### Why the auxiliary target is three extra action dimensions

ACT already regresses a chunk of actions under an L1 loss. Appending the
true block x, y and yaw makes them three more regression targets on the same
chunk, which requires **no change to LeRobot's ACT implementation at all**.
At rollout the harness reads `action[:3]`, `action[3:7]` and `action[7]`, so
the extra columns are ignored without any code needing to know they exist.

It is a training target and never an input. The student's observation is
untouched and still carries no privileged state, so the project rule that a
policy must not skip perception still holds.

The block poses were recovered for the 800 already-collected episodes by
replaying their recorded actions through the simulator, since the oracle
both drove and labelled them so the recorded action is exactly what was
executed. Verified rather than assumed: joint drift against the recorded
trajectory was **0.00000 rad**, bit-exact, and any episode over 0.05 rad
would have been rejected rather than silently mislabelled.

### Predictions, written down now

* **Run A: 75 to 83% strict.** Removing the human quarter of the data should
  help the `home` and `in_time` gates, because human episodes do not contain
  the two-stage retreat, and should cost a little diversity.
* **Run B: 5 to 12 points above Run A**, concentrated in the far bin and in
  `delivered`. If auxiliary supervision does anything, that is where.
* **Neither reaches 97%.** Behaviour cloning on a fixed dataset has never
  closed a gap this size in this project, and the remaining failures are
  compounding-error failures on states the training data does not contain.

### Recovery, decided in advance

| outcome | reading | next step |
|---|---|---|
| B beats A by a clear margin | the representation was the constraint, as S3 argued | keep aux, then DAgger on top of it |
| B ≈ A | the loss already forced enough of the block position | drop aux, spend the budget on DAgger and far-edge data |
| B worse than A | 3 of 11 action dims is too much weight on the auxiliary term | retry with the aux columns scaled down, or abandon |
| A worse than the mixed 77% | the human data carried diversity the oracle lacks | go back to the mix and keep aux |
| far bin does not move in either | far-edge failure is geometry, not representation | targeted far-edge oracle collection, the intervention that already fixed the near region |

**The step after these two is DAgger regardless of outcome**, because the
residual failures are grasps that never happened on states the training
distribution does not contain, and that is the one thing more demonstrations
of correct behaviour cannot fix. The earlier DAgger attempt failed on
dataset composition (gripper open fraction 0.960 against the
demonstrations' 0.557, and 29.4% of frames from failed episodes), and both
are now measured before conversion rather than diagnosed after.

---

## S6. The training labels never had the demonstration speed profile

The composition gate, run before conversion as the plan required, failed the
800-episode oracle dataset on a check that had never been applied to it:

```
                               dataset       demos     delta
gripper_open_fraction            0.488       0.557    -0.069
failed_frame_share               0.037       0.000     0.037
mean_trajectory_score            0.832       0.994    -0.162
min_trajectory_score             0.000       0.856    -0.856  <-- OUT OF TOLERANCE
p05_trajectory_score             0.628       0.974    -0.345
```

The two checks that broke the DAgger dataset both passed comfortably. A
third, never previously measured on a *dataset*, did not: **the training
data scores 0.832 against the demonstration profile, with a 5th percentile
of 0.628, while all 221 demonstrations score at or above 0.856.**

The dataset would fail its own `demo_like` gate.

### Where it goes wrong

Per-metric, over 200 episodes:

| metric | band score |
|---|---|
| approach steps | 0.884 |
| carry steps | 0.929 |
| carry height | 1.000 |
| jerk | 1.000 |
| retreat height | 1.000 |
| approach speed | 0.986 |
| carry speed | 0.935 |
| **retreat steps** | **0.456** |
| **retreat speed** | **0.233** |

| phase | collected | demonstrations |
|---|---|---|
| approach | 2.28 mm/step | 1.85 (band 1.33 to 2.49) |
| carry | 2.86 mm/step | 3.27 (band 2.67 to 3.81) |
| retreat | **2.74 mm/step** | **4.78 (band 4.17 to 5.17)** |

Every collected phase sits near 2.7 mm/step. The demonstrations are
deliberately **slow to position, quicker to carry, quickest to leave**. The
collected data is flat.

### The cause

`collect_dagger.py` asked the oracle for a label with `oracle.label()`
regardless of who was driving.

`label()` is the **reactive** labeller, and it has to be: DAgger needs an
answer for states the student reaches that the oracle's own trajectory never
visits, so it recomputes from the current world state with no memory. It
places the target a fixed 12 mm ahead of the tool. **A fixed lead produces
one speed**, whatever the phase.

`__call__` is the phase machine. It rate-limits the target to
`PHASE_STEP_M`, which is 1.85 / 3.27 / 4.78 mm per step, the measured
demonstration speeds. It scores 0.987.

When the oracle is driving there is no student to be reactive for, so the
reactive labeller was pure loss. The unused helper `oracle_label()`, which
calls `__call__`, was sitting in the same file having been written and never
wired up.

**This is why `demo_like` is ACT's worst gate at every single checkpoint**
(52% to 80% across the sweep). ACT was imitating its labels faithfully. The
labels were the problem.

### A second bug found while fixing the first

`__call__` read the bare constants `GRASP_OFFSET`, `HOVER_Z` and
`TRANSIT_Z`, and took its speed straight from `PHASE_STEP_M`. **Only
`label()` ever used the per-episode jitter.** So the phase machine was fully
deterministic: the same block pose produced a byte-identical episode every
time.

That is exactly the degenerate demonstrator the jitter was added to prevent,
and the jitter had silently never applied to it. It also explains why the
oracle scored identically at jitter 0 and jitter 1.0 in the S0 controls:
the jitter was doing nothing.

Both are fixed. `__call__` now uses `jit_grasp`, `jit_hover`, `jit_transit`
and scales its per-phase speed by `jit_speed`.

### Re-validated after the change

| | strict | trajectory | steps |
|---|---|---|---|
| phase machine, jitter 0 | **100.0%** (n=20) | 0.987 | 281 |
| phase machine, jitter 1.0 | **97.5%** (n=40) | 0.969 | 288 |

The 2.5% cost of jitter matches the 1.8% previously documented, and is the
price of a demonstrator that does not repeat itself. Failed episodes are
excluded at conversion.

### The corrected data

Re-collected with the phase machine, 8-episode check:

| | old collection | **new collection** | demos |
|---|---|---|---|
| trajectory score | 0.832 | **0.982** | 0.994 |

**+0.150 on the metric ACT fails most.** Collection of 1,000 episodes is
running; at roughly 3 seconds an episode it is about 50 minutes, against the
day that re-rendering at higher resolution would have cost for a hypothesis
that turned out to be false.

### The baseline, re-scored under the corrected gates

The saved per-episode measurements let the seed-51 run be re-scored without
re-running it:

| | act_oracle_v1 @ 20,000, seed 51 |
|---|---|
| original p10 to p90 gates | 77.0% |
| **recalibrated gates** | **81.0%** |

81.0% is a lower bound: `released_low` cannot be recomputed from the saved
rows because the old run did not record the simulator's block velocity, and
four of its five failures were solver noise.

Remaining failures by gate, out of 100:

| gate | failures |
|---|---|
| `demo_like` | 15 |
| `in_time` | 10 |
| `delivered` | 9 |
| `released_low` | 9 |
| `home` | 9 |
| `lifted` | 8 |

`demo_like` and `in_time` head the list, and those are exactly what the
label-profile fix in S6 attacks. `delivered` at 9 is the hard core of missed
grasps, which needs the auxiliary target and then DAgger.

### The reactive labeller, fixed for the DAgger stage

DAgger cannot use the phase machine, because it has to answer for states the
oracle's own trajectory never reaches. So `label()` was given a per-phase
lead scale, the only control it has over speed, set to the ratio of each
demonstration speed to the 2.7 mm/step a fixed 12 mm lead produces.

Driving the arm with the labeller itself, 25 trials, no rendering:

| | before | after |
|---|---|---|
| trajectory score | 0.832 | **0.950** |
| `demo_like` | n/a | 92% |
| strict | n/a | 80% |

Still short of the phase machine's 0.987, and its remaining failures are
`in_time` (3 episodes under 195 steps, 2 over 452) and one episode that
knocked the block 250 mm on the way out. A reactive labeller with no phase
latching can dither. Good enough to stop poisoning the profile, not yet good
enough to be a DAgger teacher; revisited before that stage.

### A third defect in the harness, found by auditing it against itself

`episode_complete` ended an episode once the tool reached
`RETREAT_Z - 0.010`, which is 0.490. The `home` gate requires the tool to
have reached `RETREAT_Z`, which is 0.500.

So an episode could be declared finished at 0.492 and then fail `home` for
never having reached 0.500, **a failure the harness created by stopping the
policy one step early**, not one the policy earned. It never showed on the
oracle, which retreats to 0.518, so the positive control could not have
caught it. It would have quietly cost ACT some fraction of its `home` gate.

The rule, now checked across every condition: the termination test must be
at least as strict as the gate it feeds, or the harness manufactures
failures.

| condition | terminates at | gate requires | |
|---|---|---|---|
| retreat height | ≥ 0.500 | ≥ 0.500 | equal |
| home distance | ≤ 0.060 | ≤ 0.060 | equal |
| block resting | < 0.010 | < 0.020 | terminate stricter |
| block distance | < 0.050 | < 0.050 | equal |
| fingers open | > 0.020 | > 0.020 | equal |

### A fourth: the release frame was the first open, not the last

`reference.phases` takes the first gripper close and the first open after
it. That is correct for a clean demonstration and wrong for a policy that
grasps, misses, opens, and grasps again, a mode one RL teacher hit in 72 of
100 episodes.

On such an episode the "release" is the failed grasp, so `released_low`,
`lifted_off`, `home` and `undisturbed` were all being measured at a frame in
the middle of the approach, hundreds of steps and tens of centimetres away
from where the block was actually put down.

The release is now the frame after the **last** one where the block was
genuinely inside a closed gripper, which is robust to both chatter and
re-grasping. If the block is still held on the final frame there is no
release and the gates fail, which is correct.

### Consequence: the sweep numbers predate two of these fixes

The checkpoint sweep ran with the recalibrated gates but **before** the
termination-height fix and the release-frame fix. Its 72.5% for checkpoint
20,000 is therefore a slight underestimate, and the baseline is re-measured
on the final harness before Run A and Run B are compared to it.

Recorded rather than silently corrected, because the sweep's *ranking* is
still the useful part and it is unaffected: every checkpoint was scored the
same way.

---

## S7. The corrected dataset

1,000 episodes collected with the phase machine, jitter 1.0, in 43.8 minutes.
The oracle solved 997 of them; the 3 failures are dropped at conversion.

The composition gate, old dataset against new:

| | old 800 | **new 1,000** | demonstrations |
|---|---|---|---|
| mean trajectory score | 0.832 | **0.974** | 0.994 |
| p05 trajectory score | 0.628 | **0.925** | 0.974 |
| mean episode length | 302.6 | **289.0** | 295.5 |
| sd episode length | 85.8 | 43.3 | 52.9 |
| gripper open fraction | 0.488 | 0.488 | 0.557 |
| share of frames from failed episodes | 0.037 | **0.006** | 0.000 |
| episodes with a recorded block pose | 0.000 | **1.000** | n/a |

Excluding the 3 failed episodes, over the 997 that are kept:

| | value |
|---|---|
| mean trajectory score | 0.977 |
| p01 / p05 | 0.917 / 0.925 |
| **fraction clearing the 0.85 `demo_like` gate** | **99.8%** |

The old dataset would have failed its own `demo_like` gate at the 5th
percentile. This one clears it in 998 episodes out of 1,000.

The gate still reports `min_trajectory_score` out of tolerance. That is the
3 failed episodes scoring 0, and `--exclude-failed` removes them; the gate
deliberately measures the raw directory before that filter is applied.

The remaining gap to the demonstrations is the gripper open fraction, 0.488
against 0.557. That is a consequence of phase proportions rather than a
defect: oracle episodes spend relatively more time in the closed-gripper
carry. It was 0.488 in the dataset that produced 86.8%, so it is not new,
and it is nowhere near the 0.960 that broke DAgger.

### Harness re-validated on the final code

Four defects were fixed in the harness after its first validation, so both
controls were re-run against the final version:

| | loose | **strict** | trajectory |
|---|---|---|---|
| oracle, jitter 1.0, n=40 | 100.0% | **97.5%** | 0.969 |
| oracle at 3× speed, n=20 | 90.0% | **0.0%** | 0.569 |

The separation is intact.

### How much weight the auxiliary target actually gets

ACT normalises actions with mean/std and takes an L1 loss over the chunk, so
every action dimension is unit variance and contributes equally. Three
auxiliary dimensions out of eleven is **27% of the loss**. That is a lot for
an auxiliary term, and it is the most likely way this experiment fails.

Per-column spread in the new dataset before normalisation:

| column | mean | std |
|---|---|---|
| x | 0.5413 | 0.0349 |
| y | 0.0614 | 0.1058 |
| z | 0.4749 | 0.0338 |
| qw | 0.0000 | **0.00000** |
| qx | 0.9731 | 0.0241 |
| qy | 0.0339 | 0.2268 |
| qz | 0.0000 | **0.00000** |
| gripper | 0.0196 | 0.0200 |
| block_x | 0.5412 | 0.0413 |
| block_y | 0.0624 | 0.1184 |
| block_yaw | 0.0510 | 0.4646 |

Two of the eight existing action dimensions, `qw` and `qz`, are **exactly
constant**, because the tool is always gripper-down, so those quaternion components
are always zero. They carry no information and contribute nothing to the
loss, which means the effective action vector was never 8 dimensions. Not a
defect, and it predates all of this, but it is worth knowing when reasoning
about loss weighting.

A caveat on `block_yaw`, written before the run: the commanded wrist yaw is
already in the action through `qy`, so yaw is partly redundant supervision,
and a cube's yaw is only visible as a subtle edge orientation folded into a
quarter turn. It is the auxiliary column most likely to inject gradient
noise rather than signal. If Run B underperforms Run A, dropping to x and y
alone is the first thing to try before abandoning the idea.

A second caveat, and the honest one: **the action chunk already carries a
lot of block position implicitly.** Predicting 32 steps of approach requires
knowing where the arm is going. What the auxiliary target adds is that the
block position becomes required *explicitly and at every frame*, including
frames where the commanded target is still far from the block. Whether that
is worth 27% of the loss is exactly what Run B measures.

### Definitive baseline, final harness

`act_oracle_v1` @ 20,000, screening seed 61, 40 trials, all four harness
fixes applied:

| | |
|---|---|
| loose `check_success` | 92.5% |
| **strict** | **72.5%** |

Unchanged from the sweep's 72.5%, so neither the termination-height nor the
release-frame fix moved this checkpoint on this seed.

| gate | pass |
|---|---|
| `demo_like` | **72.5%** |
| `home` | 80.0% |
| `in_time` | 80.0% |
| `undisturbed` | 85.0% |
| `released_low` | 87.5% |
| `lifted_off` | 90.0% |
| `delivered` / `lifted` / `carried` | 92.5% |

Trajectory score 0.897 against the oracle's 0.991, mean 371 steps against
the demonstrations' 295, mean lift 66 mm against 80.

**`demo_like` is the single worst gate**, and the training data it learned
from scored 0.832. ACT is doing slightly better than its own labels, which
is what regression to the mean looks like, and it is capped by them.

The two reference points for Run A and Run B:

| | strict |
|---|---|
| baseline, seed 51, n=100 | 81.0% |
| **baseline, seed 61, n=40** | **72.5%** |

Seed 61 is the comparison that matters, because it is the screening seed
both new runs will be swept on.

### What the corrected data can and cannot fix, quantified before the run

Decomposing the 40 baseline trials on seed 61:

| | count |
|---|---|
| episodes that ran the full 600 steps and never finished | 8 |
| of those, that never delivered the block at all | 3 |
| **that delivered the block but never retreated home** | **5** |
| episodes with a trajectory score under 0.85 | 11 |
| **passing every gate except `in_time` and `demo_like`** | **31 / 40 = 77.5%** |

Two things follow.

**Every `in_time` failure is a 600-step timeout**, not a marginal length.
None were under 195 and none were between 453 and 599. So after the band
recalibration the timing gate is no longer rejecting near-misses at all; it
is reporting episodes that never finished.

**5 of 40 delivered the block and then failed to go home.** That is 12.5% of
trials lost to the phase the old training data represented worst. The
retreat was the phase whose speed was 1.7× too slow, and 22% of the mixed
dataset was human episodes with no retreat at all.

So 77.5% is the ceiling the data fix alone can reach on this seed, and only
if it fixes `demo_like` and the retreat timeouts perfectly. **The remaining
22.5% is substance**: 3 missed grasps, and quality failures on
`released_low`, `lifted_off` and `undisturbed` in otherwise good episodes.

Prediction restated against the seed-61 baseline of 72.5%:

* **Run A: 80 to 88%.** Above the naive 77.5% ceiling only because the
  corrected retreat should also help `released_low`, `lifted_off` and
  `undisturbed`, which are all measured at or after the release.
* **Run B: 0 to 8 points above Run A**, in `delivered` and the far bin.
* **Neither reaches 97%.** The missed grasps need states the training
  distribution does not contain, which is DAgger.

---

## S8. DAgger design, written before Run A and B report

DAgger is the step after these two regardless of their outcome, because the
residual failures are grasps that never happened on states the training
distribution does not contain, and no quantity of correct demonstrations
fixes that. It is also the intervention that has already failed once in this
project, at 11% and then at 76%, so the failure causes are enumerated here
with what has changed for each.

| what went wrong before | status |
|---|---|
| labels led the tool by 41.9 mm, up to 119.8, against the demonstrations' 11.8 | fixed when `label()` was made reactive; still true |
| the latched phase machine stalled in DESCEND, so the label said "keep the gripper open" all episode, giving 96% gripper-open frames | fixed; `label()` has no latched state |
| 29.4% of frames came from episodes where the demonstrator failed | now measured by `dataset_gate` before conversion, and episodes are capped |
| labels had a flat speed profile | fixed in S6 by the per-phase lead scale, 0.832 → 0.950 |

Three rules for the collection:

**Cap episode length at about 300 steps.** A flailing episode runs to the
600-step cap and contributes twice the frames of a good one, so a handful of
them dominate the loss. This is how 12.5% failed episodes became 29.4% of
frames last time.

**Do not gate DAgger data on trajectory score.** When the student drives
badly the oracle's labels *should* look unlike a demonstration. That is
recovery behaviour, and it is the entire reason for collecting it. The gate
checks that apply are gripper open fraction and failed-frame share. This is
a deliberate exception, recorded so it is not mistaken for an oversight.

**Mix, do not replace.** Roughly 300 DAgger episodes against the 997 oracle
episodes, about 23%. The README's own reading of the earlier failure is that
failure coverage may only pay once the base data is strong, which is
consistent with it hurting on a weak base and is testable now that the base
is much stronger.

---

## S9. Failure: both conversions killed by the system, out of memory

Both LeRobot conversions were launched in parallel and **both were killed at
exactly episode 455 of 997** when the machine ran out of RAM. Nothing was
salvageable; the partial datasets were deleted.

### What caused it. First answer, and it was wrong.

The obvious reading was that two conversions running concurrently on a 32 GB
machine, plus six redundant shell waiters I had spawned polling the same
file, had exhausted RAM. The README even warned about it: *"32 GB already
reached 89% with eight dataloader workers."*

**That diagnosis was wrong, and measuring it is what showed so.**

Re-running one conversion alone with a memory monitor:

| episodes converted | python RSS | available RAM |
|---|---|---|
| 0 | 805 MB | 10,289 MB |
| 61 | 848 MB | 8,022 MB |
| 121 | 857 MB | 5,616 MB |

**The conversion's own memory is flat.** 805 to 857 MB while 4.7 GB
disappeared. Whatever was consuming memory was not the thing I had blamed.

Sampling the process table twice, 45 seconds apart:

```
09:32:08 avail=5640MB  AWPerformance.SCSubAgent=4711MB  python=666MB
09:32:55 avail=4789MB  AWPerformance.SCSubAgent=5049MB  python=674MB
```

`AWPerformance.SCSubAgent`, a Dell/Alienware SupportAssist telemetry
sub-agent, was growing at roughly **450 MB per minute** while the conversion
sat flat at 670 MB. It reached 5.1 GB. That is what exhausted the machine
and got the background jobs killed, and it had nothing to do with this
project.

Two conversions in parallel and six idle waiters were still wasteful and
still worth fixing, but they were not the cause. **The first explanation
was the one that fit my own actions, which is exactly why it needed
checking against a measurement rather than accepted.**

### How it resolved

Stopping the telemetry process was blocked by the permission system, which
is correct, since it is the user's machine and not part of this project.

It turned out not to be necessary. The leak is **cyclical rather than
monotonic**: the agent is trimmed back periodically, dropping to 631 MB at
one point. Available memory over the rest of the run oscillated between 4.2
and 10.0 GB rather than falling to zero, and the single conversion completed
all 997 episodes cleanly.

So the earlier double failure was the leak peaking while two conversions
happened to be running. One conversion at a time has enough headroom to
survive the peaks; two did not.

### What the layout actually is, measured afterwards

| | |
|---|---|
| `data/` parquet at 455 episodes | 2.4 GB |
| `images/` directory | 6 MB, scratch for the in-flight episode only |
| projected full dataset | about 5.3 GB |

Images are embedded into the parquet, not left as loose PNGs, so disk was
never the issue and is not a reason to avoid a second conversion.

### What changed

**Conversions run one at a time**, with a monitor watching free memory and
python RSS rather than assuming it will be fine. This is what made the real
cause visible, and it is worth keeping for that reason alone.

**One waiter per thing waited on.** Six processes polling one file is pure
overhead and it contributed to the failure.

### Result

997 episodes, 287,178 frames, action dimension 8. Converted cleanly in one
pass once it was the only heavy job running.

### And a plan change that follows from it

Converting both variants costs roughly three hours sequentially. Rather than
spend that up front, **the plain 8-dimensional dataset is converted first
and Run A is trained on it.** Whether the auxiliary variant is worth its own
conversion is then decided from Run A's failure breakdown:

* if Run A's residual failures are missed grasps, auxiliary supervision is
  aimed at exactly that and is worth the conversion
* if they are something else, the time goes to DAgger instead

This is not skipping the control. Run A **is** the control, and the decision
about Run B is simply deferred until there is evidence about what it needs
to fix. The cost is that the auxiliary experiment, if run, arrives later.

---

## S10. Run A: 92.5% and 95.0% strict on the two screening seeds

Trained on the corrected 997-episode oracle-only dataset. Nothing else
changed: same ACT config, same resolution, same chunk size, 8-dimensional
actions, no auxiliary target.

### The sweep, screening seed 61, 40 trials per checkpoint

| step | strict | loose | gap | trajectory |
|---|---|---|---|---|
| 10,000 | 75.0% | 85.0% | 10.0 | 0.870 |
| 12,500 | 47.5% | 65.0% | 17.5 | 0.780 |
| 15,000 | 75.0% | 80.0% | 5.0 | 0.855 |
| 17,500 | 65.0% | 70.0% | 5.0 | 0.832 |
| 20,000 | 67.5% | 85.0% | 17.5 | 0.863 |
| 22,500 | 80.0% | 90.0% | 10.0 | 0.876 |
| 25,000 | 65.0% | 82.5% | 17.5 | 0.836 |
| 27,500 | 85.0% | 87.5% | 2.5 | 0.888 |
| **30,000** | **92.5%** | **92.5%** | **0.0** | **0.914** |

The 27-point swings between adjacent checkpoints are exactly the
non-monotonicity this project already documented, and they are the reason
selection happens on a sweep rather than a single checkpoint.

### The headline is not the rate, it is the gap

| | loose | strict | gap |
|---|---|---|---|
| baseline, `act_oracle_v1` @ 20,000 | 92.5% | 72.5% | **20.0** |
| **Run A @ 30,000** | 92.5% | **92.5%** | **0.0** |

**Identical loose success. Twenty points of strict success.** Raw delivery
did not improve at all; what changed is that every delivery is now an actual
pick and place. The baseline was putting the block on the target 92.5% of
the time and doing it properly only 72.5% of the time. That difference was
invisible to every number this project reported before this work.

### Confirmed on a second screening seed

| seed | strict | loose |
|---|---|---|
| 61 | 92.5% | 92.5% |
| 62 | **95.0%** | 97.5% |
| **pooled, n=80** | **93.75%** | 95.0% |

Gate detail on seed 62: `delivered`, `lifted`, `carried`, `lifted_off`,
`home`, `undisturbed`, `in_time` and `demo_like` all at 97.5%;
`released_low` at 95.0%. Mean lift 79 mm against the demonstrations' 80.
Block moved 0.52 mm after release on average. Trajectory score 0.932.

### The far-edge deficit is gone

| block start x | baseline delivered | **Run A strict** |
|---|---|---|
| x < 0.53 | n/a | 92.3% |
| x >= 0.53 | n/a | **96.3%** |
| far bin 0.610 to 0.640, baseline | **66.7%** | n/a |

The far region was the baseline's worst, at 66.7% delivered in the top bin,
and S4 established it was not a coverage problem. It is now the *stronger*
half. That is consistent with the S4 reading: the far edge carries a
slightly weaker perceptual signal, and a policy trained on coherent labels
stops amplifying that into a missed grasp.

### Prediction versus outcome

Predicted **80 to 88%**. Measured **92.5% and 95.0%**. The prediction was
too low, and the reason is instructive: I estimated a 77.5% ceiling from
"episodes passing every gate except `in_time` and `demo_like`", treating the
other gates' failures as independent substance. They were not. The retreat
being wrong in the training data was causing `released_low`, `lifted_off`
and `undisturbed` failures too, because all three are measured at or after
the release. Fixing one cause fixed four gates.

### Still rising at the end of training

92.5% at 30,000 is the **last** checkpoint, and the trend over the final
three is 80.0 → 85.0 → 92.5. Previous runs on this task peaked at 20,000.
Training has been extended to 60,000 rather than accepting an endpoint as a
peak, and checkpoint 30,000 is backed up to `outputs/_keep/` first, because
selecting the last checkpoint of a run is exactly the mistake the sweep
exists to prevent.

### The extension past 30,000 did not help. Stopped.

Training was resumed toward 60,000 on the reasoning that 92.5% was the last
checkpoint and the trend over the final three was rising. Rather than wait
three hours to find out, the checkpoints that had already been written were
swept as soon as they existed:

| step | strict | loose | trajectory |
|---|---|---|---|
| **30,000** | **92.5%** | 92.5% | 0.914 |
| 32,500 | 67.5% | 77.5% | 0.852 |
| 35,000 | 57.5% | 60.0% | 0.771 |
| 37,500 | 75.0% | 82.5% | 0.859 |

**It fell 25 points immediately and did not recover.** The rising trend over
22,500 → 27,500 → 30,000 was the same non-monotonic swinging this run shows
everywhere else, not a curve heading somewhere. Reading three consecutive
rising points as a trend was over-reading noise, on a task where adjacent
checkpoints have differed by 27 points all along.

Training was stopped at 40,000 and the budget redirected to reporting. All
checkpoints are kept.

A note on cost: the extension also slowed from 3.4 to 1.56 steps per second
as it ran, which would have made 60,000 a three and a half hour wait for
what the first three post-30,000 checkpoints already answered in thirty
minutes. Sweeping checkpoints as they appear, rather than after the run, is
the cheaper order and is now the default.

---

## S11. The headline measurement

Checkpoint 30,000, **seeds 71 to 76, 100 trials each, 600 trials total**.
None of these seeds has been used for any selection decision at any point in
this project. Seeds 51 to 56 are retired for contamination; 61 and 62 were used
for the sweep and the confirmation, so they are excluded too.

### Result: 92.3% ± 2.6% strict over 600 trials

| | loose | **strict** |
|---|---|---|
| baseline `act_oracle_v1` (the 86.8% policy) | 92.5% | 72.5% |
| **Run A `act_oracle_v2` @ 30,000** | **96.7%** | **92.3%** |

Per seed: 95 / 95 / 91 / 94 / 88 / 91.

| gate | pass |
|---|---|
| `undisturbed` | 98.0% |
| `carried` / `lifted_off` | 97.5% |
| `lifted` | 97.0% |
| `delivered` | 96.7% |
| `home` | 96.5% |
| `in_time` | 96.0% |
| `demo_like` | 94.5% |
| `released_low` | 94.3% |

Mean lift 80 mm, exactly the demonstrations' 80. Trajectory score 0.930
against the oracle's 0.991. Block moves 0.22 mm after release on average.
Regional split 94.0% near and 91.3% far, against a baseline whose worst far
bin was 66.7%.

**The old headline was 86.8% on the loose criterion. This is 96.7% loose and
92.3% strict.** Both numbers moved, but the strict one is the one that
describes the task.

### Where the remaining 46 failures are

| | count |
|---|---|
| fail **only** `released_low` | 13 |
| fail **only** `demo_like` | 9 |
| hard failures, every gate down, 600 steps | ~21 |
| other combinations | 3 |

The 13 release failures are marginal rather than gross: the tool is at 0.4543
against a 0.450 limit, four millimetres too high, with the block descending
at 102 mm/s. That is a small genuine drop, not a place, since the block bottom
sits about 10 mm above the table when the fingers open, but it is nothing
like the 95 mm drop the gate was built to catch.

So the residual is roughly half quality and half genuine task failure.

---

## S12. Negative result: replanning more often makes it much worse

The release happens inside a 32-step action chunk, so the policy commits to
about a second of open-loop motion across the descent. The obvious idea is
to replan more often. It is inference-only and needs no retraining.

Checkpoint 30,000, screening seed 61, 40 trials:

| `n_action_steps` | strict | loose | trajectory | mean lift |
|---|---|---|---|---|
| 8 | **32.5%** | 52.5% | 0.681 | 54 mm |
| 16 | 85.0% | 87.5% | 0.896 | 73 mm |
| **32 (default)** | **92.5%** | 92.5% | 0.914 | 79 mm |

**Monotonically worse the more often it replans**, and at 8 it collapses.
This reproduces the Stage 1 finding, 88% to 0% as replanning went from
every 32 steps to every step, and extends it: that was measured with
temporal ensembling, and this shows plain chunk shortening does the same
thing. Whatever ACT gains from committing to a long chunk on this task, it
loses immediately when interrupted.

Chunk size stays at 32. A *larger* chunk was not tested because it needs
retraining, and it is the one direction this result argues for.

---

## S13. DAgger, and the defect the gate caught before it trained

300 episodes with the 92.3% policy driving and the oracle relabelling every
state it reached. The student solved 285 of them, so this round is mostly
on-policy coverage of *correct* behaviour rather than failure states, which
is still the right thing to collect, because the student's own release
states are exactly where it opens the fingers four millimetres too high, and
the oracle relabels those with "keep descending".

### The gate result

| check | DAgger v2 | demonstrations | the set that broke DAgger in Stage 3 |
|---|---|---|---|
| gripper open fraction | **0.549** | 0.557 | **0.960** |
| share of frames from failed episodes | **0.084** | 0.000 | **0.294** |
| **never released** | **1.000** | 0.000 | n/a |
| mean episode length | 238.6 | 295.5 | 206 |

**The two checks that destroyed the earlier DAgger dataset both passed.**
Gripper open fraction is within 0.008 of the demonstrations, against 0.960
last time, and that is the phase-scaled reactive labeller from S6 working.

**And a third check, which had never been run on a DAgger dataset before,
failed completely.**

### Every episode ended with the gripper commanded shut

`never_released` was **1.000**. Not one of the 300 episodes contained an
open-gripper label after the grasp.

The cause is an interaction between two reasonable-looking pieces of code.
The label is asked for *before* the driver moves, because otherwise it would
describe a state the student never saw. And the student path broke out of
the loop the moment `check_success` fired. So on the step where the
student's release lets the block land, the label recorded was computed while
the block was still held, and it says "gripper closed", and then the
episode ends.

A policy trained on that learns to approach, grasp, carry, descend, and
never let go. **That is exactly what RL teacher v2 was rejected for**, at a
reported 99.8% success with the block ending in mid-air in 37 of 100
episodes.

It would not have shown up in training loss, and it would not have shown up
in the loose success rate. It showed up in one line of arithmetic comparing
the dataset against the demonstrations, run before conversion, which is the
entire reason that step exists.

### The fix

Both drivers now continue until the arm is clear and back at home, which is
what the oracle path already did and for the same stated reason. Episode
lengths went from 186 to 280 steps to 265 to 364, the retreat now being included.

The chain that follows re-runs the gate and **aborts before conversion** if
`never_released` is still above 0.15, rather than trusting the fix.

### Cost

About forty minutes: 300 collected episodes discarded and a conversion
killed at 29 episodes. Cheap against the eighty minutes of training and a
full evaluation cycle that the Stage 3 DAgger dataset cost before anyone
looked at it.

### The gate aborted again, on a different and worse defect

The retreat fix worked: `never_released` fell from 1.000 to 0.280. The
abort threshold caught the remainder, and the remainder turned out to be a
separate bug that was doing more damage than the first one.

`never_released` was now **exactly equal** to `never_grasped`, both 0.280.
Measured directly on the 300 episodes:

| | |
|---|---|
| episodes whose **oracle label never commands a close** | **84 / 300** |
| of those, the block was airborne (student had lifted it) | **74** |
| of those, the student **succeeded** | **69** |

So in 84 episodes the label said *"keep approaching the block"* from start to
finish while the block was already in the gripper being carried to the
target. Not missing labels but **confidently wrong ones**, which is worse,
because behaviour cloning has no way to discount them.

### Cause: latched state that was documented as removed

`label()` decides whether the block is held from
`closed = float(self.grip) < 0.02`, **the oracle's own last commanded
gripper**. The oracle only commands a close once the tool reaches its own
computed grasp pose within 6 mm in z and 10 mm in xy.

When the oracle drives, that always happens. **When a student drives, it
grasps from wherever it likes**, so the oracle may never latch closed, and
`held` then stays false for the rest of the episode.

`docs/RL_PROCESS.md` states that a latched phase machine "rules out a
latched machine entirely" and that everything in `label()` is decided from
the current world state. That was the intent and it is not what the code
did. One latched variable survived, and it only misbehaves in the exact
situation DAgger creates.

### The fix, and why it does not re-open the old deadlock

The commit message on the original test is a warning worth respecting:
defining "held" by lift height alone deadlocks when the oracle drives, since
lifting is only commanded once held and held would only become true once
lifted. That scored 0 of 20.

So the world-state test is added as an **OR**, not a replacement:

```python
held = (closed and dist < 0.035) or (near and off_table)
```

The first clause preserves oracle-driven behaviour exactly. The second
recognises a block that is off the table and next to the tool regardless of
who closed the fingers, which cannot deadlock because the first clause still
carries the oracle-driven case.

### Regression check on both paths

| | before | after |
|---|---|---|
| phase machine (collection path), strict | 100.0% | **100.0%** |
| phase machine, trajectory | 0.976 | 0.976 |
| reactive labeller driving, strict | 80.0% | **93.3%** |
| reactive labeller driving, trajectory | 0.950 | **0.982** |

The collection path is unchanged, as intended. The labeller improved
sharply on its own account, because the same latching bug had been degrading
it whenever it drifted off its own grasp pose.

**Two DAgger datasets were discarded to find these.** Both were caught by
the composition gate before any training, which is the whole argument for
running it.

### Third attempt: the gate passed

| check | v2 | v3 | **v4** | demonstrations | the Stage 3 set that failed |
|---|---|---|---|---|---|
| `never_grasped` | 0.280 | 0.280 | **0.023** | 0.000 | n/a |
| `never_released` | **1.000** | 0.280 | **0.027** | 0.000 | n/a |
| gripper open fraction | 0.549 | 0.651 | **0.530** | 0.557 | 0.960 |
| frames from failed episodes | 0.084 | 0.078 | **0.078** | 0.000 | 0.294 |

**The first DAgger dataset in this project to clear its composition checks.**
The two defects that produced v2 and v3 were both invisible to training loss
and to the success rate, and both would have been diagnosed after the fact
as "DAgger does not work on this task", which is what happened the first
time, in Stage 3, and was recorded as a property of DAgger rather than of a
labelling bug.

### DAgger fine-tuning made it much worse, and the experiment was confounded

Fine-tuned from the 92.3% checkpoint on the merged 1,282-episode dataset,
learning rate halved to 5e-5, 15,000 steps. Screening seed 61, 40 trials:

| step | strict | loose | gap | trajectory |
|---|---|---|---|---|
| 2,500 | 57.5% | 85.0% | 27.5 | 0.777 |
| 5,000 | 52.5% | 95.0% | 42.5 | 0.821 |
| 7,500 | 52.5% | 87.5% | 35.0 | 0.799 |
| 10,000 | **37.5%** | 92.5% | **55.0** | 0.806 |
| 12,500 | 57.5% | 75.0% | 17.5 | 0.805 |
| 15,000 | 57.5% | 92.5% | 35.0 | 0.804 |

Against the base policy's 92.3% strict. **Loose success survived; quality
collapsed**. The gap reopened to as much as 55 points and `demo_like` fell
to 38%. This is the same signature the whole project keeps producing: a
policy that puts the block on the target while doing something else.

**The obvious conclusion is that the DAgger data is bad. The measurements do
not support it.**

The DAgger labels are close to the oracle's:

| dataset | trajectory score | approach | carry | retreat |
|---|---|---|---|---|
| `oracle_v2`, phase machine | 0.976 | 1.76 | 2.72 | 3.87 |
| `dagger_v4`, student-driven | **0.951** | 1.56 | 2.70 | 3.65 |
| demonstrations | 0.994 | 1.85 | 3.27 | 4.78 |

A 0.951 dataset cannot explain a policy at 0.80.

**The control was already on record.** Continuing training from checkpoint
30,000 on the *original oracle-only data* collapsed the same way: 92.5% at
30,000, then 67.5%, 57.5%, 75.0%. And here, DAgger fine-tuning was already
down to 57.5% after only 2,500 steps.

So the collapse is a property of **training onward from that checkpoint**,
not of the DAgger data. Checkpoint 30,000 is a sharp optimum, since adjacent
checkpoints differ by 25 points on this task, and a fresh optimizer at any
appreciable learning rate walks off it immediately.

**This experiment cannot answer whether DAgger helps.** Recording it as
"DAgger failed" would repeat exactly the mistake Stage 3 made, and the two
labelling bugs found earlier today suggest Stage 3's own DAgger verdict
deserves the same suspicion.

The clean test is training from scratch on the merged dataset, which is what
runs next. The merged data is already converted, so it costs one training
run rather than a whole pipeline.


---

## Summary of everything measured

### The result

| policy | loose | **strict** | n |
|---|---|---|---|
| scripted oracle, jitter 1.0 | 100.0% | 97.5% | 40 |
| previously reported best (`act_oracle_v1`) | 92.5% | **72.5%** | 40 |
| **`act_oracle_v2` @ 30,000** | **96.7%** | **92.3% ± 2.6%** | **600** |

Reported on seeds 71 to 76, none used for any selection decision.

### What actually moved the number

Nothing that was planned. The approved plan was higher resolution, a bigger
backbone and auxiliary supervision. **None of those were used.**

| intervention | outcome |
|---|---|
| higher image resolution | **not needed**, the block is legible to 3.1 mm at 160×128 |
| bigger backbone | not tried, for the same reason |
| auxiliary block-pose target | built and validated end to end, **never needed** |
| oracle-only data | used |
| **correcting the label speed profile** | **this was the whole thing** |
| more frequent replanning | **negative**, monotonically worse |
| DAgger fine-tuning | **negative**, and confounded |
| DAgger from scratch, controlled | **negative**, 92.5% -> 65.0% |
| larger action chunk (64) | **negative**, 92.3% -> 90.2% on clean seeds |
| ResNet34 backbone | +0.7 points, 0.6 SE, not a difference |
| **ensembling two policies that fail in different places** | **92.3% -> 97.0%** |

The 20-point gain came from noticing that the training labels had been
collected with the reactive DAgger labeller instead of the rate-limited
phase machine, which flattened the demonstrations' slow-fast-slow speed
profile into one constant speed and made the retreat 1.7× too slow.

### Defects found, in order

| # | where | what |
|---|---|---|
| 1 | `eval/rollout.py` | success criterion could not see a shove, a drop or a hover, and the loop broke before the retreat |
| 2 | `eval/strict.py` | timing band was p10 to p90, rejecting trajectories scoring 0.998 |
| 3 | `eval/strict.py` | release velocity measured solver noise, not the block |
| 4 | `eval/strict.py` | termination height 10 mm under the gate it fed, manufacturing failures |
| 5 | `eval/strict.py` | release frame was the first gripper opening, not the last |
| 6 | `scripts/collect_dagger.py` | plain collection used the reactive labeller, flattening the speed profile |
| 7 | `eval/oracle.py` | phase machine ignored every jitter field, so it was a deterministic demonstrator |
| 8 | `scripts/to_lerobot.py` | dims read from metadata that the oracle collector never wrote |
| 9 | `scripts/collect_dagger.py` | student path broke at success, so no episode contained a release label |
| 10 | `eval/oracle.py` | `label()` kept latched state, mislabelling 84/300 DAgger episodes |

Five of these were in the evaluation harness itself, found by using it.
Three were found by the composition gate before anything trained.

### What would be tried next

1. ~~The clean DAgger test~~. **Done, and negative.** Adding 285 DAgger
   episodes to 997 oracle episodes cost 27 points under an otherwise
   identical training run. The leading explanation is that a chunked policy
   cannot use per-step corrections as a 32-step plan; testing that needs
   single-step DAgger or oracle-rollout relabelling.
2. **A larger action chunk.** Replanning less often is the one direction the
   `n_action_steps` result argues for, and it needs retraining to test.
3. **Filtering the training data by strict pass**, since the jittered oracle
   is itself only 97.5% strict and about 2.5% of episodes teach imperfect
   behaviour.
4. **The auxiliary block-pose target**, which is built, validated and unused.

### What 97% would require

The remaining 46 failures in 600 trials are roughly half quality and half
genuine task failure. The quality half is dominated by releases four
millimetres too high. The teacher itself scores 97.5% strict under the same
jitter the data was collected with, so **the demonstrator is close to the
target and the student is 5 points behind it.** Closing that is a question
of matching the teacher more exactly, not of finding a better teacher.

### The clean DAgger test: it genuinely hurts

Trained from scratch on the merged dataset, identical procedure to Run A:
same 30,000 steps, same learning rate, same batch size, same architecture.
**Only the dataset differs.** Screening seed 61, 40 trials:

| step | strict | loose | trajectory |
|---|---|---|---|
| 15,000 | 32.5% | 52.5% | 0.612 |
| 17,500 | 40.0% | 77.5% | 0.702 |
| 20,000 | 52.5% | 75.0% | 0.763 |
| 22,500 | 55.0% | 85.0% | 0.789 |
| 25,000 | 55.0% | 65.0% | 0.760 |
| **27,500** | **65.0%** | 82.5% | 0.837 |
| 30,000 | 62.5% | 82.5% | 0.843 |

| dataset | episodes | best strict |
|---|---|---|
| oracle only | 997 | **92.5%** |
| oracle + DAgger | 1,282 | **65.0%** |

**Adding 285 DAgger episodes to 997 oracle episodes cost 27 points.** This
is the controlled comparison the fine-tuning run could not provide, and it
is unambiguous. `demo_like` is the worst gate at every single checkpoint,
exactly as it was for the fine-tuned model.

### Why, and it is not the label quality

The DAgger labels score 0.951 against the demonstration profile, against the
oracle data's 0.976. A 2.5-point difference in the data cannot produce a
10-point difference in the trained policy's trajectory score.

The likely mechanism is specific to **chunked** policies. ACT is trained on
(observation at t → the 32 actions from t to t+31). In oracle data those 32
actions are the trajectory that actually follows the observation. In DAgger
data they are 32 oracle *corrections*, each computed from a state the
*student* reached, so the labelled chunk describes a path that is never the
one the images go on to show. The policy is asked to predict a coherent
32-step plan from targets that were never executed as a plan.

That would also explain why DAgger hurt in Stage 3 on a weaker base, and why
it hurts here on a much stronger one, without needing the "style mixing"
explanation the README previously offered.

It is a hypothesis, and the experiment that would test it is single-step
DAgger, chunk size 1, or relabelling with the oracle's own rollout from
each student state rather than a per-step correction. Neither was run. It is
recorded as the leading explanation, not as a finding.

### What this does not say

It does **not** say the DAgger implementation is broken. Two labelling bugs
were found and fixed today, and the dataset passed every composition check
afterwards, including the two that destroyed the Stage 3 attempt. This is a
clean negative result about DAgger data mixed into a chunked-policy training
set on this task, not another bug.

**The 92.3% oracle-only policy remains the deliverable.**

---

## S14. Longer action chunk, the one direction the evidence points

`n_action_steps` measured 92.5 / 85.0 / 32.5 percent at 32 / 16 / 8. The
policy gets monotonically worse the more often it replans, and 32 was the
ceiling because that is the `chunk_size` it was trained with. The obvious
question, what happens with a longer chunk, cannot be answered at
inference time and needs a retrain.

Training ACT with `chunk_size=64`, `n_action_steps=64`, on the same 997
corrected oracle-only episodes. Every other hyperparameter identical to
Run A, so this is one variable.

### Prediction, written before the sweep

**60 to 90% strict, most likely below Run A's 92.5%.** Two arguments pull
opposite ways and the negative one looks stronger.

For: the replanning result is monotonic across three points and the
demonstrations average 295 steps, so a 64-step chunk is still only a fifth
of an episode. Committing further might continue to help.

Against: a 64-step chunk is 2.1 seconds of open-loop motion, which spans an
entire phase boundary. The grasp happens around step 105 and the release
around step 236. With 64-step chunks the policy must commit to a plan that
crosses those transitions blind, and both are the moments millimetres
matter. The `released_low` gate is already the worst one at 94.3%, and it is
measured exactly at a phase boundary.

There is also a reason the monotonic trend may not extrapolate: shortening
the chunk hurts because the policy was *trained* at 32 and shortening it at
inference breaks that assumption. Retraining at 64 removes the mismatch, so
the mechanism behind the trend does not obviously carry over.

### Why data quality is not the next lever

Worth stating, because filtering the training data to strict-clean episodes
was the obvious alternative and it was rejected on an argument rather than
tried.

The jittered oracle scores **97.5% strict** and the student scores 92.3%.
If data quality were the binding constraint, the student would sit near its
teacher. It is five points below. So the student is not saturating the
quality that is already in the data, and removing the imperfect 2.5% of
episodes would be optimising something that is not binding.

That points at student-side capacity or imitation fidelity, which is what
chunk size and backbone size address.

### Result: chunk 64 matches on one screening seed and beats on the other

| step | strict | loose | trajectory |
|---|---|---|---|
| 20,000 | 80.0% | 82.5% | 0.878 |
| 22,500 | 75.0% | 85.0% | 0.895 |
| 25,000 | 60.0% | 65.0% | 0.793 |
| **27,500** | **92.5%** | 92.5% | **0.928** |
| 30,000 | 70.0% | 77.5% | 0.869 |

Confirmed on the second screening seed:

| policy | seed 61 | seed 62 | pooled n=80 |
|---|---|---|---|
| chunk 32 @ 30,000 | 92.5% | 95.0% | 93.75% |
| **chunk 64 @ 27,500** | 92.5% | **100.0%** | **96.25%** |

40 of 40 on seed 62. Trajectory score 0.964 against chunk 32's 0.932, mean
episode length 286 steps against the demonstrations' 295, block moved
0.10 mm after release.

**The prediction was 60 to 90% and most likely below chunk 32.** It came in
at the top of the range and slightly above. The reasoning that a 64-step
chunk would have to commit blind across the grasp and release phase
boundaries was sound but did not dominate, and the counter-argument in the
same prediction turned out to be the operative one: the `n_action_steps`
trend was a train-inference *mismatch* effect, not evidence that shorter
horizons are intrinsically worse, so it did not extrapolate. Retraining at
64 removes the mismatch, and the result is a wash on one seed and better on
the other.

A 600-trial report on the clean seeds is running to see whether the
2.5-point screening advantage survives.

### And the clean seeds overturned it. Chunk 32 stays.

600 trials, seeds 71 to 76:

| | screening seeds 61+62, n=80 | **clean seeds 71 to 76, n=600** |
|---|---|---|
| chunk 32 @ 30,000 | 93.75% | **92.3% ± 2.6%** |
| chunk 64 @ 27,500 | **96.25%** | **90.2% ± 1.1%** |

**The screening advantage reversed completely.** Chunk 64 looked 2.5 points
better on the seeds used to choose its checkpoint and came out 2.1 points
worse on seeds that had never been used for anything.

Per seed: 91 / 91 / 90 / 91 / 88 / 90.

This is the single cleanest demonstration in the whole project of why the
seed protocol exists. Chunk 32 shrank by 1.5 points from screening to
reporting; chunk 64 shrank by **6.0**. The difference is that its checkpoint
27,500 was chosen as the best of five, and seed 62 happened to hand it 40 of
40. I wrote up "chunk 64 is the better policy" off that pair of screening
numbers, and it was wrong.

Two other things the clean report shows:

| | chunk 32 | chunk 64 |
|---|---|---|
| strict | **92.3%** | 90.2% |
| per-seed spread | ± 2.6 | **± 1.1** |
| mean lift | **80 mm** | 73 mm |
| near region, x < 0.53 | **94.0%** | 85.0% |
| far region, x >= 0.53 | 91.3% | **93.4%** |

Chunk 64 is markedly **more consistent** across seeds and markedly worse in
the near region, where the arm folds in toward the base. A longer blind
commitment costing most where the geometry is tightest is a coherent story,
but it is a story fitted after the fact and is not being claimed as a
finding.

The difference of 2.1 points sits at about 1.8 standard errors, so this is
"chunk 64 is not better", not "chunk 64 is worse".

**Chunk 32 at 30,000 remains the deliverable at 92.3%.**

---

## S15. ResNet34 backbone: 93.0%, the new best, and a plateau

The surviving half of the originally approved plan. Same 997 corrected
oracle-only episodes, same chunk size 32, same everything except the vision
backbone: ResNet18 to ResNet34.

Sweep, screening seed 61:

| step | strict | loose | trajectory | steps |
|---|---|---|---|---|
| 20,000 | 90.0% | 92.5% | 0.938 | 320 |
| 22,500 | 57.5% | 60.0% | 0.796 | 421 |
| **25,000** | **92.5%** | 97.5% | **0.955** | **299** |
| 27,500 | 80.0% | 82.5% | 0.896 | 340 |
| 30,000 | 80.0% | 87.5% | 0.900 | 337 |

**600 trials on clean seeds 71 to 76: 93.0% ± 1.3% strict, 95.0% loose.**
Per seed 93 / 92 / 95 / 91 / 94 / 93.

Selection was done on seed 61 only and the report run straight afterwards.
The seed-62 confirmation step was deliberately skipped: after chunk 64, a
second screening number would only have invited the same over-claim.

### Against the previous best

| | chunk 32, ResNet18 | **ResNet34** |
|---|---|---|
| strict, 600 trials | 92.3% ± 2.6 | **93.0% ± 1.3** |
| loose | 96.7% | 95.0% |
| trajectory score | 0.930 | **0.948** |
| per-seed spread | ± 2.6 | **± 1.3** |
| near region, x < 0.53 | **94.0%** | 87.2% |
| far region, x >= 0.53 | 91.3% | **96.7%** |

**+0.7 points, which is 0.6 standard errors.** That is not a difference.
ResNet34 is the new best number and it is not a better policy in any sense
this sample size can establish. What it does show is a **halved per-seed
spread** and the highest trajectory score of any student, both of which are
consistent with the capacity argument without confirming it.

The regional profiles are the interesting part and they are near mirror
images. ResNet18 is 6.8 points better near the base; ResNet34 is 5.4 points
better far from it. Both were trained on the same data with the same
uniform block distribution.

### Three configurations, one plateau

| policy | strict, 600 clean trials |
|---|---|
| ResNet18, chunk 32 | 92.3% ± 2.6 |
| ResNet18, chunk 64 | 90.2% ± 1.1 |
| ResNet34, chunk 32 | **93.0% ± 1.3** |

Three architecture and horizon variations spanning 2.8 points, with standard
errors around 1. **Architecture is not the binding constraint.** Whatever is
holding the student at roughly 93% is not addressed by more capacity or a
longer horizon, which is the same conclusion Stage 3b reached about data
scale and for the same kind of reason: the thing being varied moves the
number less than the noise.

The teacher is at 97.5% strict under the jitter its data was collected with.
The student is four and a half points behind it, and three different
students are behind it by about the same amount.

---

## S16. Ensembling the two: 97.5%, and a contamination problem with it

The three single policies plateau at 92 to 93%, but their **regional profiles
are near mirror images**: ResNet18 is 6.8 points better near the base,
ResNet34 is 5.4 points better far from it, on the same data with the same
uniform block distribution. Two models that fail in different places is the
only situation where averaging them can beat either.

`src/eval/ensemble.py` averages the target pose of `act_oracle_v2` @ 30,000
and `act_r34` @ 25,000. Two details:

**The gripper is not averaged.** It is binary in the data and the evaluation
calls anything under 0.02 closed, so averaging a closed command with an open
one lands exactly on the threshold, ambiguous precisely when the models
disagree about the grasp or release frame. They vote instead, and on
disagreement the previous command is held.

**The quaternion is averaged and renormalised.** Both models are
gripper-down at all times, so the two are always close and this approximates
proper interpolation well.

### Result on seeds 71 to 76

| | |
|---|---|
| **strict** | **97.5% ± 0.8%** |
| loose | 98.7% |
| per seed | 98 / 97 / 97 / 99 / 97 / 97 |

| gate | pass |
|---|---|
| `undisturbed` | 99.3% |
| `carried` | 99.0% |
| `delivered` | 98.7% |
| `lifted` | 98.5% |
| `lifted_off` / `home` | 98.3% |
| `in_time` | 98.2% |
| `released_low` | 98.0% |
| `demo_like` | 97.8% |

Mean lift **80 mm**, exactly the demonstrations'. Trajectory score 0.959.
Regional split 97.0% near and 97.8% far, the first policy in this project
with no regional weakness at all.

### Why that number is not yet the claim

**The decision to build this ensemble was informed by clean-seed data.** The
complementary regional profiles that motivated it came from the 600-trial
reports of chunk 32 and ResNet34 on seeds 71 to 76, and then the ensemble was
reported on those same seeds.

Nothing was numerically tuned on them. No hyperparameter, no checkpoint, no
threshold was chosen by looking at seeds 71 to 76. But the *existence* of this
experiment was suggested by them, and that is the same category of error
this project has already paid for twice: once when seed 51 drove checkpoint
selection and was then included in the 86.8% headline, and once this week
when chunk 64 looked 2.5 points better on its screening seeds and came out
2.1 points worse on clean ones.

The rule the project uses is that reporting seeds must be untouched by any
decision. These are not, so a fresh set is being run: **seeds 81 to 86, never
used for anything.** Both numbers will be reported.

If the fresh number holds near 97.5%, the target is met. If it drops the way
chunk 64's did, then this is another selection artefact and the honest
headline stays at 93.0%.

### The fresh seeds held. 97.0% over 1,200 trials.

| seed set | strict | loose | per seed |
|---|---|---|---|
| 71 to 76, idea-contaminated | 97.5% ± 0.8% | 98.7% | 98/97/97/99/97/97 |
| **81 to 86, never used for anything** | **96.5% ± 1.8%** | 97.7% | 96/100/96/96/97/94 |
| **pooled, n=1,200** | **97.00%** | 98.17% | sd 1.5 across 12 seeds |

**95% confidence interval 96.0% to 98.0%.**

The shrinkage from the contaminated set to the fresh one is **1.0 point**,
against chunk 64's 6.0-point reversal. That is the difference between an
idea suggested by held-out data and a hyperparameter fitted to it, and it
is small enough to be noise.

### What the claim is, precisely

**97.0% strict pick-and-place over 1,200 trials, 95% CI 96.0 to 98.0.**

The target of 97% is met at the point estimate and the interval straddles
it. Separating 97% from 96% with confidence would need more trials. This
README already estimated 1,500 to 2,000 for that class of claim, and 1,200
is short of it. So the honest statement is that the policy is at
approximately 97%, not that it is provably above it.

### What it costs

The ensemble runs **two policies per step**, so inference is twice a single
ACT: about 5.6 s per evaluation trial against 2.8 s. On real hardware at
30 Hz that is the number that decides whether it deploys at all, and it is
the one real drawback of this result.

Both members are already trained and on disk. Nothing else changed: same
data, same criterion, same seeds protocol.

### Against the teacher

| | strict |
|---|---|
| scripted oracle, jitter 1.0 (the demonstrator) | 97.5% |
| **ensemble student** | **97.0%** |
| best single student | 93.0% |

**The student has effectively caught its teacher.** The oracle's own 2.5%
strict failure rate under the jitter its data was collected with is now the
same order as the student's, which means further gains need a better
demonstrator rather than a better student: collect at lower jitter, or
filter the training set to strict-clean episodes. Both are listed in "what
would be tried next" and neither was run.

### Gate detail, fresh seeds

| gate | pass |
|---|---|
| `undisturbed` | 98.8% |
| `carried` | 98.5% |
| `lifted` | 98.3% |
| `delivered` / `lifted_off` | 97.7% |
| `home` / `in_time` | 97.5% |
| `released_low` | 97.3% |
| `demo_like` | 96.5% |

Mean lift 79 mm against the demonstrations' 80. Trajectory score 0.955
against the oracle's 0.991. Block moves 0.14 mm after release. Regional
split 95.2% near and 97.3% far.

---

## S17. What the remaining failures actually are

Three of 100 trials failed on held-out seed 92. All three are recorded by
the gates as "the block never rose past 20 mm", which the earlier failure
taxonomy called a grasp that never happened. That description is correct and
tells you nothing about the mechanism, and several different mechanisms
produce it: closing on empty air, closing correctly and having the block
slip, knocking the block away during the approach, or never attempting a
close at all.

`src/scripts/diagnose_failures.py` replays specific trials and records what
the normal trace does not: the measured finger width after closing, the
tool-to-block offset at the moment of closing, and the wrist-to-block yaw
misalignment. Block placements come from the same generator in the same
order, so any trial index can be reproduced without running the ones before
it.

### The moment the fingers close

| trial | lateral | height above block centre | yaw | fingers settle | lift | |
|---|---|---|---|---|---|---|
| 0 | 4.2 mm | **−2.2 mm** | 2.6° | 30.4 mm | 81.4 mm | passed |
| 1 | 7.7 mm | −1.2 mm | 2.1° | 30.2 mm | 79.9 mm | passed |
| 2 | 5.7 mm | −1.4 mm | 2.3° | 30.4 mm | 76.8 mm | passed |
| 3 | 6.3 mm | −0.6 mm | 3.9° | 30.8 mm | 80.0 mm | passed |
| 4 | 3.6 mm | −2.4 mm | 0.2° | 29.9 mm | 82.7 mm | passed |
| **11** | 16.8 mm | **+21.6 mm** | 12.1° | 18.2 mm | 6.4 mm | **failed** |
| **21** | 12.2 mm | **+21.5 mm** | 18.1° | 9.3 mm | 19.7 mm | **failed** |
| **96** | 10.8 mm | **+21.7 mm** | 8.2° | 8.4 mm | 9.0 mm | **failed** |

| | passed | failed |
|---|---|---|
| lateral offset | 5.5 mm | 13.3 mm |
| height above block centre | **−1.6 mm** | **+21.6 mm** |
| yaw misalignment | 2.2° | 12.8° |

### The answer

**It is not that the gripper could not pick the block up.** The gripper
closed in the wrong place.

The block's half-height is 22.0 mm. A successful grasp closes with the tool
1 to 2 mm *below* the block's centre, so the fingers straddle it and settle
at about 30 mm, the width of the block between them. All three failures
closed at **+21.6 mm**, which is exactly level with the **top face**. The
fingers came together above the block, caught its top edge or nothing at
all, and settled at 8 to 18 mm. The block was shoved 3 to 12 mm sideways in
the process.

The consistency is the striking part: +21.6, +21.5, +21.7 mm. That is not
scatter, it is a single reproducible mode. **The policy skipped the descent
and closed at hover height.** Lateral offset and yaw were also unconverged
at that moment, 2.4× and 6× worse than a successful grasp, which is what
being at the wrong point in the trajectory looks like.

So the residual failure is a **timing failure, not a perception failure or a
hardware limit**, which is consistent with the perception probe locating the
block to 3.1 mm from these same images.

### Why the policy commits early, and why the obvious fix is already ruled out

ACT predicts 32 actions from one observation and executes all of them open
loop, about a second. The descent and the close sit either side of a phase
boundary. If the chunk containing the close is predicted from an observation
taken at hover height and the policy underestimates how many steps the
descent needs, the close fires while the arm is still high and **nothing can
correct it until the chunk ends.**

The obvious remedy is to replan more often, and that was measured: 92.5% at
32 action steps, 85.0% at 16, 32.5% at 8. It makes things much worse.
A longer chunk was also measured and is no better. So the fix is not a
horizon setting, and the useful directions are elsewhere:

* the demonstrator itself is only 97.5% strict under the jitter its data was
  collected with, so some of this timing sloppiness is in the labels
* a grasp-phase-only correction, which is what the residual RL stage was
  rescoped to do and has never been run

### An incidental observation

Successful trials show a **second gripper close** at around step 410 to 490,
with the tool 260 to 280 mm from the block and 70 mm above it, closing on
air and moving the block 0.0 mm. The policy shuts its fingers again after
returning home. It is harmless, since the block is already placed and
undisturbed, and it is invisible to every gate, because the release frame
is taken from the last moment the block was actually held. It is worth
knowing about before anyone reads a raw gripper trace and is puzzled by it.
