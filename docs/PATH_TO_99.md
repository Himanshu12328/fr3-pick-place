# Past 97%, and what the 97% number was actually measuring

A running log, in the same form as `PATH_TO_97.md`. Every step, every
prediction written down before the run that tests it, and every failure.
Nothing is removed when it turns out to be wrong; the wrong entries are the
point of the document.

The target set for this session: **99.99% strict pick and place**. It is
worth saying immediately that 99.99% is a claim about one failure in ten
thousand, and that bounding it below at 99.99% with 95% confidence requires
roughly **30,000 consecutive clean trials**. The reporting protocol carried
over from `PATH_TO_97.md` is six seeds of a hundred, twice — 1,200 trials —
which can bound nothing tighter than **99.75%**. Both numbers are reported
throughout, and no sentence in this document claims 99.99% on 1,200 trials.

---

## S18. The teacher was never 97.5%. It is 99.7%.

### The claim that was load-bearing

`PATH_TO_97.md` S16 closed the previous session with this:

> **The student has effectively caught its teacher.** The oracle's own 2.5%
> strict failure rate under the jitter its data was collected with is now
> the same order as the student's, which means further gains need a better
> demonstrator rather than a better student — collect at lower jitter, or
> filter the training set to strict-clean episodes.

That conclusion set the entire direction of what to do next. Two of the four
items in "what would be tried next" follow from it, and both are about
fixing the data.

It rests on one number: the scripted oracle at jitter 1.0 scoring **97.5%
strict**. Which is `39/40`. Forty trials.

This project's own measurement protocol, written in the handover document
under "learned the hard way", says **20 trials is ±11% and 100 is ±4%**.
Forty trials is about ±8. A 97.5% point estimate from 40 trials is
consistent with anything from roughly 87% to 100%, and it was used to
conclude that a 97.0% student had converged on its teacher.

### Measured properly

Same oracle, same jitter, same gates, same harness. 1,000 trials across
four seeds:

| | strict | loose | n |
|---|---|---|---|
| oracle, jitter 1.0, as previously reported | 97.5% | 100.0% | **40** |
| **oracle, jitter 1.0, remeasured** | **99.7%** | 99.9% | **1,000** |

Per seed 99.2 / 99.6 / 100.0 / 100.0. Three strict failures in a thousand.

| gate | pass |
|---|---|
| `delivered` / `lifted` / `carried` | 99.9% |
| every other gate | 99.7% |

Mean lift 78 mm against the demonstrations' 80, trajectory score 0.973,
block moved after release 0.04 mm mean and 0.14 mm worst.

### What this changes

**The student has not caught its teacher. It is 2.7 points behind it.**

| | strict | n |
|---|---|---|
| teacher, jitter 1.0 | **99.7%** | 1,000 |
| student, the 97.0% ensemble | **97.0%** | 1,200 |

The gap is not noise: 99.7% and 97.0% on 1,000 and 1,200 trials do not
overlap at any reasonable confidence.

So the two data interventions the previous session recommended are aimed at
a problem that does not exist. The labels are not the limiting factor — they
already demonstrate 99.7% behaviour under exactly the jitter they were
collected with. Collecting at lower jitter would improve a demonstrator
that fails 3 times in 1,000, to close a gap of 27 in 1,000 that lives
entirely in the student. Filtering the training set to strict-clean
episodes would discard about 0.3% of it, not the 2.5% the estimate implied.

This is the fourth time in this project that a conclusion turned out to rest
on a sample too small to support it, and the second time specifically that a
40-trial number drove a decision. The pattern is consistent enough to be
worth stating as a rule: **a number that is going to decide what gets built
next does not get measured on 40 trials**, however cheap the run was.

The useful consequence is that there are 2.7 points of headroom reachable
without touching the data, and the ceiling to aim at is 99.7% rather than
97.5%.

---

## S19. What the residual failures are, measured over 200 trials rather than 3

`PATH_TO_97.md` S17 diagnosed three failures on held-out seed 92 and found
one reproducible mode: the fingers close **+21.6 mm above the block's
centre**, exactly level with its top face, instead of 1 to 2 mm below it.
The policy skips the descent and closes at hover height. That diagnosis is
correct and this section does not overturn it. It was measured on three
trials, and three trials cannot say which of the numbers that accompanied it
generalise.

So the instrument was rebuilt to record every gripper close of every trial,
and run in a mode that changes nothing, over 200 trials on seeds 61, 62, 91
and 92 — all of them already spent on screening or video in previous
sessions, so no reporting seed informed anything below.

The monitor-mode run reproduces the headline exactly: **97.00% strict on
200 trials**, against 97.0% on 1,200.

### The height of the close separates completely

| at the first close | passed strict (n=194) | closed on air (n=4) |
|---|---|---|
| height above block centre | −3.27 … **+1.55** mm | **+20.74** … +30.71 mm |
| finger settle width | **29.90** … 33.94 mm | 15.35 … **19.11** mm |

Median height for a passing trial is −0.60 mm and p95 is +0.64. Every
success puts the tool a millimetre or two *below* the block's centre so the
fingers straddle it; every air close is level with the top face. Nothing in
200 trials lands between +1.55 and +20.74 mm, and nothing lands between
19.11 and 29.90 mm of finger width.

The S17 mode is confirmed, on 194 successes instead of 5, and it is
separable by a threshold on a quantity a real arm reads off its own joint
encoders.

### The lateral offset does not separate, and that matters

S17 also reported that failures sat at 13.3 mm lateral offset against 5.5 mm
for successes — "2.4× larger" — and that yaw was 6× worse. On three trials
both were true. On 200:

| lateral offset at first close | |
|---|---|
| passed strict (n=194) | min 0.95, median 5.71, p95 12.47, **max 19.04 mm** |
| failed strict (n=5) | 10.97, 12.08, 12.47, 17.31, **19.19 mm** |

Complete overlap. **17 of the 194 passing trials have a larger lateral
offset than the tightest failure.** A veto on lateral alignment would have
refused seventeen grasps that worked in order to catch five that did not.

This is the most useful negative result of the session, because the plan
approved before it was measured was to estimate the block's position from
the cameras — the perception probe locates it to 3.1 mm — and gate the grasp
on lateral and yaw alignment as well as height. That component is not
needed. The lateral offset is a **symptom** of being at the wrong point in
the trajectory, exactly as S17 suspected, not an independent cause, and
gating on a symptom that overlaps costs more than it buys.

What is left is a rule that reads only the commanded gripper channel, the
tool height from forward kinematics, and the finger encoders. **No
perception, no privileged simulator state, nothing a real FR3 does not
report about itself.** That is a stronger position than the one this session
set out to build, and it was arrived at by measuring rather than by
argument.

### Two failures in 200 are not grasp-timing failures at all

Of the six failures in the 200-trial monitor run:

| | n | what happened |
|---|---|---|
| closed on air at hover height | 4 | the S17 mode |
| never commanded a close at all | 1 | 600 steps, no grasp frame, block nudged 3.4 mm |
| grasped and carried, failed the retreat | 1 | lifted 62.4 mm, then failed `lifted_off`, `home`, `in_time` |

The last two are outside anything a grasp-timing fix can reach. The
never-closed trial cannot be seen by a rule that triggers on a close,
because it is the absence of one; it gets a separate and much weaker rule,
below. The retreat failure is a different defect entirely and nothing in
this session addresses it.

So **at most 4 of 6 failures per 200 trials are addressable here**, which
puts a floor of about 1% on what the intervention can leave behind, unless
the two remaining modes are rarer than one in two hundred makes them look.

### Two constants in the existing diagnostic were wrong

Found by using it at scale, which is how five of the defects in
`PATH_TO_97.md` were found.

`diagnose_failures.py` sets `FINGER_ON_BLOCK_M = 0.012` and reasons to it
from geometry: a pair of fingers on a 44 mm block "settles near 22 mm each
and a pair closed on air goes to nearly zero". The geometry is right and the
conclusion is wrong. Air closes settle at **15.35 to 19.11 mm**, not near
zero, because the fingers catch the block's top edge on the way past. At
12 mm that constant calls **every one of the four air closes a successful
grasp**.

The same script reads the finger width 25 steps after the close. Measured,
the two populations separate at 12 and not before: a real grasp is resolved
one step after the command, already at 29.9 mm or wider, while an air close
is still travelling and at 8 steps the widest of them is 33.5 mm —
indistinguishable from a grasp. Twelve steps is the first point of clean
separation and 25 is twice as long as the wait needs to be.

---

## S20. The instrument, widened to admit a recovery, and the cost of doing so

A policy that closes on nothing, notices, reopens and grasps properly on the
second attempt has performed the task. It has also taken longer than any of
the 221 demonstrations, because none of them ever needed a second attempt.
Two of the nine gates therefore reject it, and they reject it for the wrong
reason.

**`in_time`** bands the episode at 195 to 452 steps, measured so that every
one of the 221 demonstrations passes. A recovery adds an approach. The upper
bound is now widened by `RECOVERY_ALLOWANCE_STEPS = 105` per recovery the
runtime layer performed — one measured approach phase, because that is what
a recovery repeats. It is deliberately one measured phase rather than a
round number, and deliberately not generous.

**`demo_like`** scores the trajectory's shape per phase, and
`reference.phases` takes the first close and the first open after it. On a
recovery episode the first close is the *failed* attempt, so `carry`
measures the handful of steps before the fingers reopened and `retreat`
measures everything from there to the end. The score collapses, and it
collapses without saying anything about the trajectory that actually carried
the block.

`strict.evaluate_trace` already resolves the phase frames the robust way —
from the last moment the block was genuinely in a closed gripper — precisely
because seven of its nine gates had this same problem, and one RL teacher in
Stage 2 dropped and re-grasped in 72 of 100 episodes. `reference.describe`
now accepts those frames instead of recomputing them from the first close.

**This is a change to the task definition and it is reported as one.** Every
result below carries the rate under the original unwidened band alongside
the widened one, and the fraction of trials that used a recovery, so the two
can never be confused. The other seven gates are untouched.

### Proven inert on the data the gates were calibrated against

The property a gate has to have is that it requires behaviour the
demonstrations actually contain. So the change was checked against all 221
of them:

| | |
|---|---|
| demonstrations whose gripper closes more than once | **0 of 221** |
| trajectory score, first-close phases vs last-held phases | **max difference 0.00e+00** |
| demonstrations whose score changed at all | **0** |

Bit-identical. Every number `reference.py` has ever reported still holds,
and the gates' calibration is preserved exactly.

### The harness is not bit-reproducible, and this was not previously recorded

Monitor mode returns the inner policy's action untouched, so it should
reproduce the stored per-trial rows of the 97.0% run exactly. It does not,
quite. Comparing 8 trials of seed 81 against the stored rows: gate verdicts,
grasp frames, episode lengths and lifts all identical, but trajectory scores
differ by up to 4.6e-3, release heights by up to 0.8 mm, and one trial's
release frame moved by a single step.

The control that settles it is running monitor mode **twice** and comparing
it to itself. The same code, the same seed, produces the same differences —
including the identical single-step release-frame flip. This is
nondeterminism in GPU inference, not an effect of the wrapper. Monitor mode
is a genuine no-op, and it reproduces the headline at 97.00% over 200
trials.

It is worth recording for its own sake. Per-trial verdicts in this project
are reproducible only up to a trajectory-score wobble of about 5e-3 and a
release-height wobble of about a millimetre. Nothing measured so far sat
close enough to a gate boundary for that to matter. A claim at 99.99% would
mean one failure in ten thousand, and at that point a trial sitting within a
millimetre of the `released_low` ceiling can flip between runs. **Any rate
past about 99.9% in this project has a reproducibility floor under it**, and
that floor has now been measured rather than assumed.

---

## S21. The runtime layer, and what it is not allowed to do

`src/eval/supervisor.py` wraps any policy or ensemble. Three rules:

| rule | reads | does |
|---|---|---|
| **close veto** | commanded gripper, tool height | refuses a close more than 12 mm above where a resting block's centre is, and flushes the action chunk |
| **failure detector** | finger encoders | 12 steps after an honoured close, if the fingers settled under 24 mm they are on nothing: reopen, flush, let the policy replan |
| **stall flush** | step count | no grasp by step 160, which is 35 steps past the latest grasp that has ever worked: flush the chunk |

Every input is proprioception. The block's pose is read in monitor mode for
analysis and **no rule consults it**; the height rule uses the table
geometry, because a block rests on the table at a known height until it is
picked up.

**The layer may withhold a close and it may flush a plan. That is all it may
do.** Every target pose the arm is driven to is the policy's own output,
unmodified. There is no scripted approach, no hand-written descent and no
waypoint in the file. The constraint is the point: the result has to stay a
statement about a learned policy, and a layer that issued its own motion
commands would quietly turn it into a statement about a state machine with a
policy attached.

The flush needs a cooldown and the reason is measured. Clearing the chunk
every step is exactly `n_action_steps=1`, which this project measured at
32.5% against 92.5%. A chunked policy has to be allowed to execute its plan.
So a flush is an event and not a policy: it happens when a close is refused
or a grasp is found to have failed, and then not again for 24 steps.

The veto disarms permanently once a grasp is confirmed. Vetoing during the
carry would open the fingers and drop the block.

### Predictions, written before the screening run

Four configurations on the four burned seeds, 200 trials each: monitor (the
97.00% baseline), veto alone, retry alone, both. Thresholds at 5, 8 and
12 mm for the veto.

1. **Retry alone gains almost nothing.** This is the strongest prediction
   available and it is falsifiable. The four air-close failures already
   attempt **37 to 45 closes each** without any help — the policy reopens
   and re-closes dozens of times on its own and never recovers, because it
   keeps commanding the close at hover height. A rule that reopens the
   fingers and asks the same policy to replan is adding one more of
   something that has already been tried forty times in the same episode.
   Predicted: 97.0% to 97.5%, the gain coming from the stall flush catching
   the never-closed trial rather than from the retry.
2. **The veto is the whole fix.** It is the one rule that removes the
   *option* the policy keeps taking. Predicted 98.0% to 99.0%: four of six
   failures addressed, two untouchable.
3. **Both is at best marginally better than veto alone**, because once the
   close cannot fire at the wrong height there is little left for the
   detector to detect.
4. **Threshold ordering: 12 ≥ 8 ≥ 5.** All three separate the observed
   populations, so the difference between them is only how often they refuse
   a grasp that would have worked. 5 mm is 3.45 mm from the worst passing
   close and risks livelock; 12 mm sits above the one demonstrated high
   capture at +8.94 mm.
5. **The ceiling for this session is 99.7%, not 99.99%**, because that is
   what the demonstrator scores and nothing here improves the labels.
   Exceeding the teacher would require the supervisor to fix failures the
   teacher itself commits, which is possible in principle — the teacher has
   no supervisor — but is not what any rule above is designed for.

### Outcome: one prediction right, one badly wrong

| configuration | strict, 200 trials | failures |
|---|---|---|
| monitor (baseline) | **97.00%** | 6 |
| veto, threshold 5 mm | **97.00%** | 6 |
| veto, threshold 8 mm | **97.00%** | 6 |
| veto, threshold 12 mm | **97.00%** | 6 |
| retry | **97.00%** | 6 |

**Prediction 1 was right and for the right reason.** Retry gains nothing.
The detector fired on exactly the four air-close trials and not one
recovered. The stall flush did not rescue the never-closed trial either.

**Prediction 2 was wrong.** The veto was supposed to be the whole fix at 98
to 99%. It is worth zero points. And it is not that it failed to fire: it
refused **36 closes across the four right trials, up to 29 in a single
episode**, and they all still failed.

**Prediction 3 is right by accident** — both configurations match veto alone
because both are worth nothing. **Prediction 4 is void**: the three
thresholds are indistinguishable because the threshold was never what
mattered.

The result that makes the rest of the session: **every configuration fails
the same six trials.** Not six of the same number, the same six. That is
what it looks like when the outcome is decided by the block's initial pose
before the episode starts, and nothing a runtime layer withholds or flushes
can touch it.

---

## S22. The arm is not mistimed. It is jammed.

> **Two claims in this section do not survive S24.** The jam itself,
> measured per step, is correct and is the finding the rest of the document
> builds on. But the lateral offset at descent onset (n=2) does not
> separate on 200 trials, and the under-rotated wrist is *necessary to* but
> not *sufficient for* the jam — one of the four jams happens 0.8 degrees
> from perfect alignment. S25 gives the account that covers all four.

Refusing a close does not unjam an arm, so the veto's zero raised a question
the failure taxonomy could not answer: when the policy commands a close at
hover height, is it **asking** for a descent it does not get, or not asking?

Those call for completely different work. If the command descends and the
tool does not follow, the policy is blameless and the controller, the gains
or a joint limit is at fault. If the command never descends, the policy
genuinely believes the grasp happens at hover, and no amount of vetoing or
replanning helps.

`src/scripts/probe_stuck_grasp.py` replays a trial recording the commanded
target height and the achieved tool height separately. Per step, at the
moment of the descent:

| | failing trial 11 | passing trial 0 |
|---|---|---|
| deepest **commanded** target above block centre | **-3.54 mm** | -16.53 mm |
| steps commanded below +5 mm | 15 | 164 |
| deepest **achieved** tool | **+21.34 mm** | -5.12 mm |
| tool descent rate while commanded deep | **-0.01 to -0.08 mm/step** | **-1.2 to -1.47 mm/step** |
| finger width | 40.19 mm, wide open | 40.00 mm, wide open |
| total orientation tracking error | 0.60 deg | 0.47 deg |
| wrist q7 | +0.204 rad (limit +-3.02) | +0.546 rad |

**The policy asks for the right descent and the arm does not go.** It sits
at +21.9 mm — the block's half height, so exactly its top face — moving
twenty to a hundred times slower than a free descent, with the fingers wide
open. That is contact, not sluggishness.

And it is not the controller's fault either, which was the other half of the
question: the commanded *orientation* is tracked to 0.4 to 0.6 degrees and
both wrist joints sit at a fifth of their range. The arm delivers exactly
the pose it is asked for. It is asked for a pose that ends with a fingertip
on top of the block.

### The yaw, and the correction to S17

| trial | yaw error at deepest descent | commanded yaw error | achieved depth | outcome |
|---|---|---|---|---|
| 11 | **12.04 deg** | 11.78 deg | +21.34 mm | failed |
| 21 | **18.00 deg** | 17.99 deg | +21.27 mm | failed |
| 0 | **2.21 deg** | 2.21 deg | -5.12 mm | passed |
| 1 | **1.55 deg** | 1.55 deg | -3.63 mm | passed |

Commanded yaw error equals achieved yaw error to a quarter of a degree, so
this is what the policy asked for. At 12 to 18 degrees off, open fingers
cannot straddle a 44 mm cube; one comes down on its top face and the descent
stops dead.

S17 reported the yaw error too, at "6x worse" on three trials, and read it
as a symptom of being at the wrong point in the trajectory. It is the cause.
The height at which the fingers close, which S17 read as the defect, is the
symptom: the fingers close at +21.6 mm because that is where the arm
stopped.

### Two hypotheses killed on the way

**The ensemble's quaternion averaging is not the cause.** `ensemble.py`
averages the two members' quaternions and renormalises, justified by "both
models are gripper-down at all times, so the two are always close" — an
assumption a 90-degree-symmetric object could break, with two members
choosing alignments a quarter turn apart and the average landing 45 degrees
from both. Measured, the members' yaw disagreement is **1.93 and 2.17
degrees** in the failing trials against 0.50 and 0.27 in the passing ones.
They agree. They agree *because they average the modes the same way*, which
is also why no amount of ensembling fixes this.

**Lateral offset at the close does not separate**, as recorded in S19, but
at **descent onset** it does: 17.4 mm in the failing trial against 3.4 mm in
the passing one. Both are true and they are not in conflict — by the time
the fingers close, a failing trial has already shoved the block. The earlier
measurement was taken at the wrong instant.

---

## S23. The root cause: a discontinuous regression target at the cube's symmetry boundary

> **This section is wrong and it is kept deliberately.** The mechanism it
> proposes is refuted in S24 by the same measurement extended from 6
> failures to 199 trials, and the correct account is in S25. The
> association it reports between the block's yaw offset and failure is
> real; the explanation it gives for that association is not. Read S24 and
> S25 before acting on anything below.

Block x and y do not predict failure. Block **yaw** does, and it does so
overwhelmingly. Folding the block's initial yaw into its distance from the
nearest 90-degree face alignment, over the 200-trial monitor run:

| block yaw offset from nearest face alignment | failures / trials | rate |
|---|---|---|
| 0-15 deg | **0 / 57** | 0.00% |
| 15-30 deg | **0 / 65** | 0.00% |
| 30-40 deg | 5 / 90 | **5.56%** |
| 40-45 deg | 5 / 40 | **12.50%** |

All six failures sit at or above **31.8 degrees**, mean 38.9, three of them
above 42.8. Passing trials average 23.8. If yaw were irrelevant, the chance
of all six landing in the 39% of trials above 30 degrees is
**0.39^6 = 0.0035**.

Five configurations were run over these same 200 placements. They are five
measurements of the same trials, not 1,000 independent ones, and the honest
statistic is the 200-trial one above. What the repetition establishes is
that all five configurations fail the same six trials, which is the stronger
point.

### The mechanism, and it is not the labels

A cube is symmetric every 90 degrees, so four gripper alignments grasp it
equally well and the demonstrator has to pick one. Checked against all 997
oracle episodes, the demonstrator's choice is perfect:

| oracle's commanded gripper yaw relative to the block, folded to (-45, 45] | |
|---|---|
| mean | **-0.000 deg** |
| standard deviation | **0.016 deg** |
| worst episode | 0.371 deg |
| episodes beyond 5 deg | **0 of 997** |

And the training set is not short of examples near the boundary either: 248
episodes between 30 and 40 degrees and 100 between 40 and 45, a third of the
data.

So neither label quality nor coverage explains it. The problem is the
**representation of the target**. The policy must output an absolute
quaternion, and the absolute label is a *sawtooth* in the block's yaw: for a
block at 44 degrees the aligned gripper yaw is +44, for a block at 46 it is
-44. Two nearly identical images, labels 88 degrees apart.

A network regressing that under an L1 loss cannot represent the jump. It
smears, and it smears exactly where the two branches are closest to equally
likely — which is why the failure rate climbs monotonically from zero at
0-15 degrees to 12.5% at 40-45, and why the resulting yaw errors are 12 to
18 degrees rather than random.

This also explains, at last, why the teacher is at 99.7% and the student at
97.0%. The scripted oracle computes one alignment from the block's pose and
never averages anything. The gap between them is not sloppiness, not a
shortage of data and not a small backbone. It is one variable with a
discontinuous target.

It retires a question the handover has carried since Stage 2:

> **Multimodal task variant** — approach from left/right 50/50. The axis ACT
> and diffusion were designed to differ on; the current single-mode task
> cannot separate them on capability.

The task was multimodal the whole time. Not by design, through the cube's
symmetry, and it is precisely where the L1-regressed policy fails.

---

## S24. S23 is wrong. What the yaw actually does, measured at scale.

S23 above says the residual failure is an L1-regressed policy smearing
across a discontinuous target at the cube's 45 degree symmetry boundary.
That section stays as written, because this document's rule is that wrong
entries are the point of it. It is wrong, and so is part of what replaced
it. Three claims in this session's earlier sections came from samples of
two to six trials and did not survive being measured over two hundred.

### Wrong turn 1: the symmetry boundary

If the mechanism were averaging between two branches that are equally
likely near 45 degrees, the commanded yaw error would be near zero below
30 degrees of offset and rise sharply above it. Measured over 199 trials
with a recorded close, it does not. The ratio of error to offset is flat:

| block yaw offset | n | mean commanded yaw error | error / offset |
|---|---|---|---|
| 0-5 deg | 17 | 0.81 deg | 0.265 |
| 5-10 | 16 | 0.99 | 0.123 |
| 10-15 | 24 | 1.78 | 0.143 |
| 15-20 | 24 | 2.45 | 0.136 |
| 20-25 | 20 | 2.77 | 0.125 |
| 25-30 | 22 | 3.19 | 0.115 |
| 30-35 | 29 | 3.65 | 0.111 |
| 35-40 | 25 | 3.29 | 0.088 |
| 40-45 | 22 | 5.50 (worst 18.27) | 0.129 |

Least squares through the origin gives **error = 0.115 x offset**, r =
0.476, residual sd 2.35 degrees. That is **shrinkage toward the mean**, a
constant proportional under-rotation, and it is a different thing from
smearing at a boundary. The failures concentrate at large offsets because a
proportional error grows with the rotation required, not because 45 degrees
is special.

### Wrong turn 2: the lateral offset at descent onset

S22 reports 17.4 mm of lateral offset at descent onset in a failing trial
against 3.4 mm in a passing one, and reads it as the earlier lateral
measurement having been taken at the wrong instant. On two trials that was
true. On 200:

| lateral offset at descent onset | |
|---|---|
| passed strict, n=194 | median 5.10, p95 10.15, **max 21.37 mm** |
| the four jams | 5.3, 10.6, 13.8, **17.5 mm** |

**Ninety-one of the 194 passing trials have a larger lateral offset than
the tightest jam.** It does not separate at either instant. The n=2
observation was noise.

### Wrong turn 3: the yaw error as the cause

This is the important one. Isolating the four jams from the other two
failure modes — one trial never commanded a close at all, one grasped and
carried the block and failed the retreat — and measuring at descent onset:

| quantity at descent onset | passing max (n=194) | the four jams | separates? |
|---|---|---|---|
| tool height above block centre | 21.04 mm | 21.2, 21.4, 21.6, 21.6 | **yes** |
| lateral offset | 21.37 mm | 5.3, 10.6, 13.8, 17.5 | no |
| commanded yaw error | 18.73 deg | **0.8**, 11.7, 18.4, 27.6 | **no** |
| block yaw offset | 44.79 deg | 34.9, 35.8, 43.9, 44.3 | no |
| ensemble member disagreement | 4.85 mm | 1.6, 2.3, 3.6, 4.6 | no |

**One of the four jams happens with the wrist 0.8 degrees from perfect
alignment.** So an under-rotated wrist cannot be the whole mechanism, and
S22's "at 12 to 18 degrees off, open fingers cannot straddle a 44 mm cube"
is not a sufficient explanation of the four.

The one quantity that separates perfectly is the tool's height, and it is
nearly a tautology: descent onset is the first step the commanded target
goes below the block's centre plus 12 mm, and at that step a jammed trial's
tool is already parked at the top face while a healthy trial's tool is at
about 19 mm and still moving. It detects the jam rather than predicting it.
As a **detector** it is exact and proprioceptive, which is worth knowing;
as an explanation it is circular.

### What the block's yaw does, stated at the strength the evidence supports

Not a clean cause, and more than nothing:

* Of 122 trials with a block yaw offset below 30 degrees, **zero failed.**
* Of 78 above 30 degrees, six failed, 7.7%.
* All six failures, across all six supervisor configurations, are the same
  six trials, and every one has an offset of at least 31.8 degrees.
* If yaw were irrelevant, the chance of all six landing in the 39% of
  trials above 30 degrees is 0.39^6, about 0.0035.

And 46 of the 194 passing trials sit above the tightest failing offset, so
a large offset is a **risk factor and not a determinant**. The honest
statement is that the risk of a jam rises steeply with how far the wrist
has to turn from the pose it starts every episode in, and that neither the
policy's yaw error, its lateral error, nor its members' disagreement picks
out which of those trials will jam.

### Why the wrist is under-rotated: the yaw is not in the pictures

`PATH_TO_97.md` S3 is titled "The resolution hypothesis is dead. Measured,
not argued." It is right about what it measured — the block's **position**
is legible to 3.1 mm at 160x128 — and nobody measured its **yaw**.

Measured now, on the one view that cannot leak. Every episode starts with
the arm at the same home keyframe, so with the arm there the only thing
that differs between two images is the block; any later frame lets a probe
read the yaw off the wrist the oracle has already rotated into alignment.
8,000 randomly posed blocks, 6,500 for training and 1,500 held out, target
(sin 4*theta, cos 4*theta) so that the four physically identical
orientations carry one label:

| resolution | median error | mean | p90 |
|---|---|---|---|
| 160x128 | **22.27 deg** | 22.36 | 40.59 |
| 320x256 | **21.37 deg** | 22.08 | 40.49 |
| chance | 22.5 | | |

**The block's yaw is not recoverable from these three camera views at the
start of the approach, at either resolution.** Not weakly recoverable —
chance.

That reframes the shrinkage rather than excusing it. A regressor that
cannot see its target and is trained under an L1 loss should hedge toward
the mean, and hedging toward the mean is exactly what an 11.5%
proportional under-rotation is. The policy is not failing to learn
something the images contain; for the first part of the approach it is
guessing, and it stops guessing only once the wrist cameras are close
enough to see the block's faces.

An earlier attempt at this estimator, trained on recorded frames from the
first 35% of each approach, reported a 6.9 degree median and looked
usable. It was reading the arm. By 35% of the way in, the wrist has begun
rotating toward alignment, and the leak the sampling window was supposed to
prevent was inside the window. The synthetic home-pose measurement is the
one to trust, and it is much worse.

### Corrections to earlier sections, collected

| claim | where | status |
|---|---|---|
| teacher is 97.5% strict | `PATH_TO_97.md` S16 | **wrong**, 39/40; it is 99.7% over 1,000 |
| the student has caught its teacher | `PATH_TO_97.md` S16 | **wrong**, it is 2.7 points behind |
| the residual is a timing failure | `PATH_TO_97.md` S17 | **wrong**; the policy commands the descent correctly and the arm jams |
| the close at +21.6 mm is the defect | `PATH_TO_97.md` S17 | it is the **symptom**; the fingers close where the arm stopped |
| lateral offset is 2.4x worse in failures | `PATH_TO_97.md` S17 | true of 3 trials, **does not generalise** |
| resolution is ruled out | `PATH_TO_97.md` S3 | true for **position**, never tested for **yaw**, and yaw is at chance |
| the ensemble averages modes and that causes it | S22 above | **wrong**, members agree to 2 degrees |
| the cause is the symmetry boundary | S23 above | **wrong**, the error/offset ratio is flat |
| lateral offset at descent onset separates | S22 above | **wrong** on 200 trials |
| the under-rotated wrist is the cause of the jams | S22 above | **insufficient**; one of four jams is at 0.8 degrees |

---

## S25. The mechanism, stated once, with the arithmetic

Four sections of this document proposed a cause and three of them were
wrong. The reason they were wrong the same way is worth naming: each
proposed a **single** quantity, and the jam is governed by two.

### The geometry

The gripper descends with its fingers open. Each finger face sits 40 mm
from the tool centre. The block is a 44 mm cube, so a cube square to the
fingers presents 22 mm of half-extent along the closing axis, and a cube at
yaw error `e` presents

    extent = 22 * (cos e + sin e)

which is 22.0 mm at 0 degrees, 26.1 at 12, and 27.8 at 18. With the tool
offset laterally by `d`, the near finger has

    clearance = 40 - (d + extent)

millimetres of room to pass the block's top face. Negative clearance means
a finger is over the block and the descent cannot complete. Small positive
clearance means it completes only if nothing else goes slightly wrong.

### Measured at descent onset, over 200 trials

| clearance | passed strict (n=194) | the four jams |
|---|---|---|
| min | −8.83 mm | **−3.49 mm** |
| p5 / values | 6.99 mm | −3.49, 1.56, 3.93, **5.05** |
| median | 12.08 mm | |
| max | 16.81 mm | |

| threshold | jams caught | passing trials caught |
|---|---|---|
| clearance < 0 mm | 1 of 4 | 3 of 194 |
| clearance < 2 mm | 2 of 4 | 3 of 194 |
| clearance < 4 mm | 3 of 4 | 3 of 194 |
| **clearance < 6 mm** | **4 of 4** | **7 of 194** |
| clearance < 8 mm | 4 of 4 | 24 of 194 |

Every jam has under 6 mm of clearance. Eleven trials in 200 are in that
band and four of them jam: **36% against a 2% base rate, an eighteen-fold
elevation.** It is the strongest predictor found in this session by a wide
margin, and it is still not a determinant — three trials descend with
*negative* computed clearance and succeed, because the policy keeps
refining both quantities during the descent and because a lateral offset
measured as a magnitude is the worst case rather than the component that
actually matters.

### Why this explains the four wrong turns

**Why neither lateral offset nor yaw error separates alone.** They trade
off inside one sum. A jam needs `d + extent` near 40, and that is reachable
from either end: seed 61 trial 23 jams at 13.78 mm of lateral offset with
the wrist 0.8 degrees from perfect, and seed 61 trial 18 jams at 5.26 mm of
lateral offset with 29.69 mm of extent. Conditioning on one term while the
other varies is exactly how to see no separation in either.

**Why the block's yaw offset correlates without determining anything.** A
larger offset means a larger commanded yaw error, through the 0.115
proportional shrinkage, which means a larger extent. It raises one term of
the sum. It does not set the other.

**Why snapping the wrist yaw with the true block orientation did not
help,** which is the prediction this section was written to explain before
the run finished. Snapping reduces `extent` toward 22 mm and leaves `d`
untouched. For seed 92 trial 11 that buys 4.0 mm and turns a −3.49 mm
clearance into +0.52, still marginal. For seed 61 trial 23 it buys
**0.3 mm**, because at 0.8 degrees of yaw error the extent is already
22.29 mm and there is nothing to recover. A fix aimed at one term of a
two-term budget cannot clear a budget that is short on the other.

**Why no runtime layer helps.** Clearance is computable, and a rule could
refuse to descend below 6 mm of it and force a replan. Two things stop
that being a deliverable here. It needs the block's lateral position, which
the perception probe supplies to 3.1 mm, **and** its yaw, which is at chance
from these cameras at both 160x128 and 320x256. And the action available on
detection is a flush, which returns the same plan from the same
observation — measured, across six configurations, as worth exactly zero.

### What would actually move it

Both terms are the policy's own output, and both are the policy's to
improve. The evidence says where the room is:

* **`d`, the lateral offset.** The block's position *is* legible, to 3.1 mm
  against a p5 clearance of 7 mm. A policy that placed the tool as
  accurately as the images allow would have several millimetres more
  clearance on every trial. The auxiliary block-position target built and
  validated in `PATH_TO_97.md` S5 and never used is aimed exactly here.
* **`e`, the yaw error.** Not fixable from these three views, because the
  yaw is not in them. This is the first thing in this project to point at
  camera placement or a wrist view closer to the fingers rather than at the
  policy — and it is the one lever `PATH_TO_97.md` S3 ruled out on position
  evidence without testing yaw.

That is a concrete, quantitative pair of next steps derived from a measured
budget, which is a better position than the session started in, and it is
not the 99.99% the session set out to reach.

---

## S26. Fixing the yaw with the answer makes it worse

S25 was written before this run finished, and it predicted the shape of the
result: snapping reduces one term of a two-term clearance budget, so it
should rescue the jams that are short on extent and do nothing for the ones
short on lateral offset.

`probe_yaw_ceiling.py` corrects the residual wrist-yaw error using the
block's true orientation from the simulator, from the moment the commanded
target drops within 40 mm of grasp height until the grasp is confirmed. 200
trials, the same four seeds:

| | strict | failures |
|---|---|---|
| baseline, monitor mode | **97.00%** | 6 |
| **wrist yaw corrected with the true block orientation** | **96.00%** | **8** |

It is a point worse. 20,799 corrections were applied, the largest 44.93
degrees.

### Which trials changed, and it is exactly the predicted split

| trial | baseline | with the yaw corrected | clearance budget |
|---|---|---|---|
| seed 61 trial 18 | jam | **fixed** | extent 29.69 mm — snapping buys 7.7 mm |
| seed 91 trial 37 | failed the retreat | **fixed** | — |
| seed 61 trial 23 | jam | still a jam | extent 22.29 mm at 0.8 deg — snapping buys **0.3 mm** |
| seed 92 trial 11 | jam | still a jam | clearance −3.49 mm; snapping buys 4.0, still marginal |
| seed 92 trial 21 | jam | still a jam | |
| seed 91 trial 25 | never closed | still never closes | |
| seed 61 trial 9 | passed | **broken**, fails `demo_like`, lift 84.9 mm | |
| seed 62 trial 35 | passed | **broken**, fails `released_low`, lift 85.0 mm | |
| seed 62 trial 48 | passed | **broken**, fails `demo_like`, lift 85.8 mm | |
| seed 91 trial 26 | passed | **broken**, fails `demo_like`, lift 92.2 mm | |

Two of the four jams are rescued and both are the ones with a large extent
term, exactly as the budget says. The two that remain include the 0.8
degree case, for which a perfect yaw is worth a third of a millimetre.

And four trials that worked are broken, all of them with a healthy 85 to 92
mm lift — so the grasp still succeeded and what failed was the shape of the
trajectory or the release. Overwriting the commanded orientation walks the
policy off the distribution it was trained on, and it pays for the two
rescues with four new failures at a different gate.

**This closes the yaw line of enquiry.** The residual is not "the policy
gets the wrist angle wrong"; handed the exact answer, with no perception
error at all, the policy ends up worse. Any real fix has to improve both
terms of the clearance budget from inside the policy, not correct one of
them from outside it.

---

## S27. The headline, remeasured on twelve seeds that have never been used

Every number in this session's screening came from seeds 61, 62, 91 and 92,
all spent in earlier sessions. The reporting protocol carried over from
`PATH_TO_97.md` is six seeds of a hundred, twice, on seeds untouched by any
decision. Seeds 101 to 106 and 111 to 116 have never been used for
anything.

Monitor mode, so the policy's actions are untouched and this is the same
ensemble the 97.0% headline was measured on.

| seed set | strict | loose | per seed |
|---|---|---|---|
| 101-106, never used | **95.33% ± 1.25%** | 97.17% | 95 / 97 / 94 / 95 / 97 / 94 |
| 111-116, never used | **96.17% ± 1.67%** | 97.67% | 97 / 96 / 98 / 98 / 94 / 94 |
| **pooled, n=1,200** | **95.75%** | 97.42% | sd 1.53 across 12 seeds |

**95% confidence interval 94.61 to 96.89.**

### Against the reported 97.0%

| seed set | strict | n |
|---|---|---|
| 71-76 and 81-86, as reported in `PATH_TO_97.md` | 97.00% | 1,200 |
| **101-106 and 111-116, this session** | **95.75%** | 1,200 |
| **all 24 seeds pooled** | **96.38%** | **2,400** |

**The 97.0% does not replicate.** The fresh set comes in 1.25 points lower.
The standard error of that difference is 0.76 points, so z = 1.64 and it is
**not** significant at the 5% level: this is consistent with noise, and it
is not evidence of a defect in the earlier measurement.

What it does change is the best estimate. Over 2,400 trials on 24 seeds the
policy is at **96.38%, 95% CI 95.63 to 97.12**. The interval still contains
97% at its upper edge, and the point estimate is below it. The honest
headline is **96.4% over 2,400 trials**, not 97.0%, and `PATH_TO_97.md`'s
own qualification — "the claim is approximately 97%, not provably above" —
reads better in hindsight than the number next to it.

It is worth being precise about what went wrong and what did not. Nothing
was fitted to seeds 71-86; the previous session checked that carefully and
reported both a contaminated and a clean set. But twelve seeds of a hundred
give a standard error near 0.5 points, and this project has now watched two
separate 1,200-trial measurements of the same policy land 1.25 points
apart. **Twelve hundred trials is not enough to quote a strict rate to the
tenth of a point**, which is the same lesson S18 records about forty.

### Gates, pooled over the 1,200 fresh trials

| gate | pass |
|---|---|
| `carried` / `undisturbed` | 98.25% |
| `lifted` | 98.00% |
| `delivered` | 97.42% |
| `lifted_off` | 97.25% |
| `home` | 97.17% |
| `in_time` | 97.00% |
| `released_low` | 96.67% |
| `demo_like` | 96.00% |

Mean lift 79 mm against the demonstrations' 80, trajectory score 0.954
against the oracle's 0.991.

**Zero recoveries were used**, because monitor mode performs none, so the
widened episode-length band is inert here and `strict` and
`strict_success_unwidened` agree to the digit on all 1,200 trials. That is
the check that the task-definition change of S20 costs nothing when nothing
triggers it.

---

## S28. Where this leaves the 99.99% target

It is not reached, it is not close, and the session produced a clearer
account of why than it did of how.

| | strict | n |
|---|---|---|
| target | 99.99% | would need ~30,000 clean trials to bound |
| scripted oracle, the demonstrator | **99.70%** | 1,000 |
| ensemble, all 24 seeds | **96.38%** | 2,400 |

### What was measured, in one table

| intervention | outcome |
|---|---|
| remeasuring the teacher | **97.5% → 99.7%**; the earlier figure was 39/40 |
| close veto, 5 / 8 / 12 mm | **97.00%**, no change at any threshold |
| grasp-failure detector and replan | **97.00%**, no change |
| both together | **97.00%**, no change |
| stall flush | no change; the never-closed trial still never closes |
| third ensemble member | **97.0% → 92.0%**, a 5-point regression |
| correcting the wrist yaw with the true block orientation | **97.0% → 96.0%**, worse |
| camera-only yaw estimator | **at chance** at 160x128 and 320x256 |
| remeasuring the headline on 12 unused seeds | **97.0% → 95.75%** |

Nine measurements, and not one of them improves the policy. Two of them
correct a previously reported number downward, and the rest close off
directions.

### What is now known that was not

1. The demonstrator is at 99.7%, so 2.7 points of headroom exist and they
   belong to the student, not the data. The two data interventions the
   previous session recommended are aimed at nothing.
2. The residual failure is a **geometric clearance event**, not a timing
   failure: `clearance = 40 - (lateral offset + 22(cos e + sin e))`, and
   every jam in 200 trials sits under 6 mm of it against a median of 12.
3. It is governed by **two** terms that trade off, which is why four
   separate single-variable explanations — timing, lateral offset, symmetry
   boundary, wrist yaw — each looked right on a handful of trials and each
   failed at scale.
4. The block's **yaw is not in the images** at the start of the approach, at
   either 160x128 or 320x256. `PATH_TO_97.md` S3 ruled resolution out on
   position evidence and this is the first measurement of yaw.
5. No runtime layer restricted to withholding a close and flushing a plan
   can touch any of it, measured across six configurations that fail the
   same six trials.
6. Twelve hundred trials cannot separate 96% from 97%.

### What would be tried next, in the order the evidence supports

1. **The auxiliary block-position target**, built and validated in S5 of
   `PATH_TO_97.md` and never used. It aims at the lateral term, which is the
   term that can be improved: position is legible to 3.1 mm and the p5
   clearance is 7 mm, so a policy as accurate as its own images allow gains
   millimetres on every trial. This is the one intervention this project has
   already built, has never run, and now has a measured reason to run.
2. **Camera placement, or a wrist view closer to the fingers.** The yaw term
   cannot be improved from these three views because the information is not
   in them. This is the first evidence in the project pointing at the rig
   rather than the policy.
3. **A policy class that does not regress a single orientation**, now that
   the task is known to be multimodal through the cube's symmetry rather
   than hypothetically multimodal in some future variant.

And one measurement worth doing before any of them, because it is cheap and
it bounds the rest: **the clearance budget of the oracle**. The teacher
fails 3 times in 1,000 under the same gates. If its clearances are far
larger than the student's, the budget is the whole story and item 1 is the
right next run. If they are similar, something else separates 99.7% from
96.4% and none of the three items above is aimed at it.

---

## S29. Recorded episodes

`logs/videos_jam/`, twelve episodes of held-out seed 92 recorded through
the strict harness, so each one includes the retreat and the settle rather
than stopping when the block touches down.

| file | what it shows |
|---|---|
| `strict_000_pass.mp4` … `strict_010_pass.mp4` | eleven consecutive clean pick and places |
| **`strict_011_fail.mp4`** | **the jam.** The gripper arrives with 3.5 mm of *negative* clearance, a finger comes down on the block's top face, and the arm sits there while the policy keeps commanding a descent 25 mm below it |

Trial 11 is the trial the per-step traces in S22 are taken from, and the one
the left panel of `docs/yaw_jam.png` plots. It is worth watching before
reading S25, because the arithmetic describes something that is completely
obvious once seen: the hand is simply resting on the box it is trying to
pick up.

There is no video of a recovery, because there are none to record. The
grasp-failure detector fires on exactly the right trials and the policy
never recovers from any of them — see the table in S21.

Regenerate with:

```powershell
python -m src.scripts.eval_supervised --mode monitor --seeds 92 --trials 12 `
  --video-dir logs/videos_jam --video-n 12 --max-steps 900 --tag videos_jam
```

---

## S30. The teacher's clearance budget, which closes the account

S28 named this as the measurement to make before any of the three proposed
next steps, because it decides whether the clearance budget is the whole
story or only part of it. It is cheap: the oracle needs no images, so 400
trials run on the CPU without touching the GPU.

The scripted oracle, jitter 1.0, through the same harness with the same
instrumentation: **99.75% strict, one failure in 400**, consistent with the
99.7% over 1,000 in S18.

| at descent onset | **teacher**, n=400 | **student**, n=200 |
|---|---|---|
| clearance, **minimum** | **+6.40 mm** | **−8.83 mm** |
| clearance, p1 | 10.17 mm | −3.00 mm |
| clearance, p5 | 11.94 mm | 5.08 mm |
| clearance, median | 14.91 mm | 12.03 mm |
| **trials under 6 mm** | **0 of 400 (0.00%)** | **12 of 200 (6.00%)** |
| jams | **0** | 4 |

**The teacher's worst trial in 400 has more clearance than the student's
fifth percentile.** It never once enters the band under 6 mm where every
student jam occurs, and it never jams.

That is the account of the gap between 99.7% and 96.4%, and it is entirely
a statement about a distribution's tail rather than its centre. The median
clearances differ by only 2.88 mm. What differs is the worst case:

| | teacher | student |
|---|---|---|
| lateral offset, median | 3.09 mm | 5.21 mm |
| lateral offset, **max** | **11.60 mm** | **21.37 mm** |
| commanded yaw error, median | **0.00 deg** | 2.14 deg |
| commanded yaw error, **max** | **0.00 deg** | **33.05 deg** |

The oracle's yaw error is zero at every percentile, which is what a
demonstrator that computes the alignment from the block's pose looks like
and confirms the label audit in S23 from the other direction. Its lateral
offset is not zero — it is jittered deliberately — but its worst is half
the student's.

### What this settles

1. **The clearance budget is the whole story.** No third mechanism is
   needed to explain why the teacher is at 99.7% and the student at 96.4%.
   The teacher stays out of the danger band; the student enters it on 6% of
   trials and jams on a third of those.
2. **The 6 mm threshold is not fitted to the student.** It was derived from
   4 jams against 194 passes, and the teacher independently never goes
   below 6.40 mm across 400 trials and never jams. A threshold that
   separates in one population and is respected by a second, unseen one is
   doing better than curve-fitting.
3. **Both terms have to improve, and only one of them can.** The student's
   lateral maximum is 9.8 mm worse than the teacher's and its yaw maximum
   is 33 degrees worse. The lateral term is learnable — position is legible
   to 3.1 mm. The yaw term is not, from these three views, at either
   resolution tested. So an auxiliary position target is the right next
   run, and it is not sufficient on its own.
4. **This is the number to beat.** Not 99.99%, and not the teacher's
   99.7% either, until the yaw term has somewhere to come from. It is
   "raise the student's 5th-percentile clearance from 5.08 mm toward the
   teacher's 11.94", which is a target a training run can be scored
   against directly, after one evaluation, without waiting for a
   1,200-trial success rate that cannot resolve a point anyway.

---

## S31. The budget across three policies, and what it does not explain

`src/scripts/clearance_report.py` computes the budget from any monitor-mode
run. Three policies, the same four seeds for the students, 400 trials for
the demonstrator:

| | strict | jams | grasped then failed | **p5 clearance** | median | min | under 6 mm |
|---|---|---|---|---|---|---|---|
| teacher, oracle jitter 1.0 | **99.75%** | **0** | 0 | **11.94 mm** | 14.91 | **+6.40** | **0.00%** |
| student, 2-policy ensemble | 97.00% | 4 | 1 | 5.08 mm | 12.03 | −8.83 | 6.00% |
| student, `act_oracle_v2` alone | 92.00% | 4 | 12 | 1.83 mm | 10.22 | −12.87 | 17.50% |

The fifth-percentile clearance orders all three in the same order as their
success rates, across a range from 92% to 99.75%. That is a good sign for
using it as a tuning signal, and it is not the whole story.

### What it explains

**The teacher-student gap.** The demonstrator never enters the band under
6 mm across 400 trials and never jams; both students enter it and both jam
four times in 200. The threshold was derived from one population and is
independently respected by a second, unseen one.

### What it does not explain

**The ensemble's advantage over a single policy.** Ensembling raised the p5
clearance by 3.25 mm, from 1.83 to 5.08, and the jam count did not move:
**four jams in 200 either way.** The five points ensembling gained came
almost entirely from somewhere else — trials that grasped the block and
then failed later fell from **12 to 1**.

So the previous session's account of why ensembling worked, that two
policies fail in different *regions*, is not contradicted, but the
mechanism is more specific than the clearance budget: averaging two
policies' target poses mostly improved what happens **after** the grasp,
and barely touched the grasp itself.

That also means the two students' residual failures are differently
composed. `act_oracle_v2` alone fails 16 times in 200, of which 12 are
post-grasp and 4 are jams. The ensemble fails 6 times, of which 4 are
jams. **Ensembling removed the post-grasp failures and left the jams
untouched**, which is exactly why the residual 3% looked like one
reproducible mode when the previous session went looking for it: by then it
was one mode, because the other had been eliminated.

### A caution about the band

Entering the band is necessary and not sufficient, and the two students
disagree about how sufficient:

| | trials under 6 mm | of which jammed |
|---|---|---|
| 2-policy ensemble | 12 | 4 (33%) |
| `act_oracle_v2` alone | 35 | 4 (11%) |

Four jams out of twelve against four out of thirty-five, on counts of four.
These are not distinguishable at this sample size and the difference should
not be read as real. What both agree on is the direction: no jam in any run
has a clearance above 6 mm, and the demonstrator never goes there.

### The consequence for the next run

The metric to score a training run against is the **fifth-percentile
clearance**, because a success rate cannot resolve a point without
thousands of trials — S27 watched two clean 1,200-trial measurements of one
unchanged policy land 1.25 points apart — whereas p5 clearance is a
continuous quantity measurable on 200 and it spans 1.83 to 11.94 across
policies whose success rates span 92 to 99.75.

The target is explicit: **raise the ensemble's p5 clearance from 5.08 mm
toward the demonstrator's 11.94**, and watch the jam count rather than the
success rate to see whether it helped.

---

## S32. How well the clearance budget tracks the success rate

Five monitor-mode runs, 2,000 trials in total, spanning success rates from
92% to 99.75%:

| run | strict | n | **p5 clearance** | median | jam rate | under 6 mm |
|---|---|---|---|---|---|---|
| teacher, oracle jitter 1.0 | 99.75% | 400 | **11.94 mm** | 14.91 | 0.00% | 0.00% |
| ensemble x2, seeds 61-92 | 97.00% | 200 | 5.08 mm | 12.03 | 2.00% | 6.00% |
| ensemble x2, seeds 111-116 | 96.17% | 600 | 3.91 mm | 11.44 | 1.00% | 12.00% |
| ensemble x2, seeds 101-106 | 95.33% | 600 | 3.99 mm | 11.91 | 1.67% | 9.00% |
| `act_oracle_v2` alone | 92.00% | 200 | 1.83 mm | 10.22 | 2.00% | 17.50% |

| | |
|---|---|
| Pearson r, p5 clearance vs strict rate | **+0.90** |
| Pearson r, p5 clearance vs jam rate | **−0.86** |
| Spearman, p5 vs strict | +0.90 |

### What that supports, and what it does not

It supports using p5 clearance as the signal a training run is scored on
**instead of** a success rate, for changes of the size a retrain produces.
The range is wide — 1.83 to 11.94 mm across policies spanning eight points
of success — so a real improvement should be unmistakable.

Two honest limits.

**The teacher carries most of the correlation.** It is far outside the
students' range in both variables, and a correlation computed across four
students alone is weaker, Spearman 0.8 on four points. Five runs is five
runs.

**It does not resolve differences under about a point.** The two fresh
600-trial seed sets differ by 0.84 points of success rate — 96.17% against
95.33% — and p5 clearance orders them the *wrong way round*, 3.91 mm
against 3.99. An 0.08 mm difference in a fifth percentile estimated from
600 trials is noise, and so is an 0.84-point difference in a success rate
from the same trials. Neither metric can see a difference that small, which
is consistent with S27 rather than a strike against the budget.

So: use it to tell whether a retrain moved the thing the failures depend
on, and do not use it to rank two policies that are already close.

### And one consistency check worth noting

The fresh seed sets are harder than the screening seeds by both measures at
once. Success falls from 97.00% to 95.33% and 96.17%, and p5 clearance
falls from 5.08 mm to 3.99 and 3.91, while the fraction of trials entering
the danger band rises from 6% to 9% and 12%. The two quantities moved
together across a seed change that nothing else explains, which is a
modest independent sign that the budget is measuring the thing that
actually decides these trials.

---

## S33. The run in flight, and a confound caught before it cost four hours

The intervention S28 and S30 point at: **auxiliary supervision of the
block's position**, aimed at the lateral term of the clearance budget,
which is the term with room in it. Built and validated in `PATH_TO_97.md`
S5, listed in "what would be tried next", never run.

### Setup

| | |
|---|---|
| data | `data/oracle_v2`, the same 997 oracle episodes `act_oracle_v2` was trained on |
| aux columns | **block x and y only** |
| action | 8 dimensions widened to 10 |
| config | byte-identical to `act_oracle_v2`: chunk 32, `n_action_steps` 32, lr 1e-4, backbone lr 1e-5, batch 64, 30,000 steps |
| baseline | `act_oracle_v2` @ 30,000, measured this session at 92.00% strict, p5 clearance 1.83 mm, 4 jams in 200 |

The yaw is deliberately **not** supervised. S24 measured it at chance from
these cameras at both resolutions tried, and a target the input does not
contain teaches the representation nothing while adding an unlearnable
term to the loss that competes with the terms that can be learned.

### The confound, and it was nearly paid for

The first conversion produced **1,000** episodes. The baseline's dataset has
**997**. The difference is episodes 0543, 0671 and 0991, the three where
the demonstrator itself failed, which `--exclude-failed` drops and which
the baseline conversion had dropped.

Three episodes in a thousand is 0.3%, and it would have been easy to wave
through. It is not 0.3% of the frames: a failed demonstration runs to the
600-step cap, so those three contribute 1,800 frames, and every one of them
teaches the arm to do something that did not work.

Had the run gone ahead, an aux model that underperformed would have had two
explanations and no way to separate them. `PATH_TO_97.md` S13 already
records what that costs — "DAgger fine-tuning made it much worse, and the
experiment was confounded" — so the conversion was thrown away and redone
with `--exclude-failed`. Ninety minutes against four hours of an
uninterpretable result.

The dataset now has 997 episodes and the **only** difference from the
baseline's is the two extra action columns.

### Predictions, written before the run

1. **The aux target raises the p5 clearance.** The mechanism is direct:
   position is legible to 3.1 mm and the aux target forces the
   representation to carry it. Predicted p5 from 1.83 mm to somewhere in
   3 to 6 mm, which is the ensemble's current range from a single policy.
2. **The jam count falls by less than the clearance suggests.** The
   clearance budget has two terms and this improves one. The yaw term is
   untouched and cannot be improved from these cameras. Predicted 4 jams in
   200 to 2 or 3, not to 0.
3. **The strict rate moves by about a point and will not be measurable as
   such.** From 92.00% to somewhere near 93%, which needs far more than 200
   trials to separate. This is exactly why p5 clearance is the metric being
   watched.
4. **It will not reach the demonstrator's 11.94 mm.** The demonstrator has
   zero yaw error by construction; the student cannot, because the
   information is not in its images.

If prediction 1 fails — if the p5 clearance does not move — then the
lateral error is not a representation problem either, and the budget's
remaining explanation is the yaw term alone, which would make camera
placement the only lever left in the project.

---

## S34. 98.83%. The auxiliary position target, and why it worked twice over

The run S33 launched, scored against the four predictions written before it.

### The single policy

| | strict | p5 clearance | min clearance | under 6 mm | jams | grasped then failed |
|---|---|---|---|---|---|---|
| `act_oracle_v2` @30k, the baseline | 92.00% | 1.83 mm | **−12.87 mm** | 35 of 200 | 4 | 12 |
| `act_aux_xy` @30k | 93.50% | **6.33 mm** | +1.41 mm | 8 of 200 | 5 | 8 |
| `act_aux_xy` @25k | — | 5.46 mm | +1.15 mm | 12 of 200 | 3 | 4 |

| # | prediction | outcome |
|---|---|---|
| 1 | p5 clearance rises from 1.83 mm into 3 to 6 mm | **right**, 6.33 mm, slightly better than predicted |
| 2 | jams fall from 4 to 2 or 3, not 0 | **wrong**, 5 jams at 30k, 3 at 25k |
| 3 | strict rate moves about a point, unmeasurable as such | **right**, 92.00% to 93.50% |
| 4 | it will not reach the teacher's 11.94 mm | **right**, 6.33 mm |

The lateral term moved exactly as designed. Maximum lateral offset fell
from **29.75 mm to 14.38**, and the clearance minimum went from −12.87 mm
to **+1.41** — the negative tail that produced the jams is gone.

**Prediction 2 failed for an instructive reason.** The yaw got worse:
median commanded yaw error 2.31° to 4.10°, p95 6.04° to 10.96°. Two extra
regression targets on the same action head bought position accuracy by
spending orientation accuracy. The budget has two terms and improving one
degraded the other, which is why the jam count did not follow the
clearance.

### Checkpoint selection, and the sweep tax paid again

Screening on seeds 61 and 62, 40 trials per checkpoint: 86.25% at 20k,
**100.00% at 25k**, 96.25% at 27.5k. On never-used seeds 121-124 the 25k
checkpoint scores **96.50%**, a 3.5-point regression — squarely inside the
5-to-7-point sweep tax this project has now measured five times.

### The finding that mattered: the failures are disjoint

Paired on the identical 200 trials of seeds 121-124:

| | strict | failures |
|---|---|---|
| `act_oracle_v2` @30k | 94.00% | 12 |
| `act_aux_xy` @25k | 96.50% | 7 |

| | count |
|---|---|
| baseline failed, aux passed | 12 |
| baseline passed, aux failed | 7 |
| **failed by both** | **0** |

Exact McNemar on the 19 discordant pairs: two-sided **p = 0.36**. The
+2.50-point difference is **not significant**, and on its own the aux
policy is not demonstrably the better model.

What is significant is *where* each fails, and this is not a rate:

| | block yaw offset of its failures |
|---|---|
| baseline | 26.4, 31.5, 35.8, 36.6, 39.2, 39.2, 40.0, 40.1, 40.9, 42.1, 43.3, 43.9 |
| aux | **3.7, 12.2, 12.6, 17.3, 17.3, 21.1, 30.6** |

The baseline fails where the wrist has furthest to turn, exactly as S24 and
S25 describe. The aux policy fails nowhere near there — it has eliminated
that mode and acquired a different one at low yaw offsets. **Zero trials in
200 are failed by both.**

That is precisely the condition `ensemble.py` was written for: "two models
that fail in different places is the only situation where averaging them
can beat either." Here it is not a regional split but a disjoint one.

### The result

`act_oracle_v2` @30,000 averaged with `act_aux_xy` @25,000:

| seed set | strict | loose | per seed |
|---|---|---|---|
| 121-124, the seeds that suggested the experiment | 98.50% | 99.00% | 98 / 100 / 100 / 96 |
| **131-136, never used for anything** | **98.33% ± 0.75%** | 98.50% | 100/98/98/98/98/98 |
| **141-146, never used for anything** | **99.33% ± 0.75%** | 99.50% | 98/100/100/100/99/99 |
| **pooled clean, n=1,200** | **98.83%** | 99.00% | sd 0.90 across 12 seeds |

**98.83% strict, 95% confidence interval 98.23 to 99.44.**

The idea-contaminated set came in at 98.50% and the clean sets at 98.83%,
so the shrinkage this time is **negative**: nothing was lost on fresh
seeds, against the 1.25 points S27 lost and the 6.0 points the chunk-64
experiment lost in `PATH_TO_97.md` S14.

| | strict | n |
|---|---|---|
| previous best, `act_oracle_v2` + `act_r34`, all 24 seeds | 96.38% | 2,400 |
| **this ensemble, 12 never-used seeds** | **98.83%** | **1,200** |
| improvement | **+2.46 points**, se 0.49, **z = 5.00** | |
| scripted demonstrator | 99.70% | 1,000 |

The improvement is significant at any conventional level, and the student
is now **0.87 points behind its teacher** rather than 2.7.

| gate | pass |
|---|---|
| `carried` | 99.67% |
| `lifted` | 99.42% |
| `undisturbed` | 99.25% |
| `delivered` / `lifted_off` / `home` | 99.00% |
| `released_low` / `in_time` | 98.92% |
| `demo_like` | 98.83% |

Mean lift **81 mm** against the demonstrations' 80. Trajectory score 0.976
against the oracle's 0.991.

### The clearance budget tracked all of it

| policy | strict | **p5 clearance** | under 6 mm | jams |
|---|---|---|---|---|
| `act_oracle_v2` alone | 92.00% | 1.83 mm | 17.50% | 4 / 200 |
| old ensemble, v2 + r34 | 97.00% | 5.08 mm | 6.00% | 4 / 200 |
| **new ensemble, v2 + aux_xy** | **98.83%** | **7.08–7.32 mm** | 2.50–3.00% | 6 / 1,200 |
| scripted demonstrator | 99.70% | 11.94 mm | 0.00% | 0 / 400 |

Monotone in both directions across four policies and a 7.7-point range of
success. S32 proposed p5 clearance as the signal to tune on because a
success rate cannot resolve a point without thousands of trials; it then
predicted the direction and rough size of this result before the success
rate could confirm it.

### One bug, found by trying the experiment

`EnsemblePolicy` could not combine the two policies at all. An
aux-supervised policy emits a 10-vector, a plain one an 8-vector, and
`np.stack` raised `ValueError: all input arrays must have the same shape`
several frames deep, naming nothing. The auxiliary columns are a training
target and never an action — the harness reads `action[:3]`, `[3:7]` and
`[7]` and ignores the rest — so the members are now truncated to eight
dimensions before averaging.

### What is left

Fourteen failures in 1,200. Six are jams, and the two clearance minima
among them are −1.88 mm and +1.31 mm, so the two-term budget is still what
decides them. The yaw term remains unimprovable from these three camera
views, and the aux target made it slightly worse in exchange for the
position gain it delivered.

So the ceiling of this approach is close. **99.99% is not reachable by more
of this**, and the demonstrator's own 99.70% is now less than a point away.
Past that, the yaw has to come from somewhere, and S24 says it is not in
these images at any resolution tried.
