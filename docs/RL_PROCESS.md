# The RL teacher: implementation, iteration and findings

A complete account of the reinforcement learning work in this project: what
was built, every run that was made, what each one taught, and where it ended
up. Written so the next person does not repeat any of it.

The short version: an RL teacher was trained to 98.8% success and then
repeatedly rejected on quality grounds, because success rate could not see
what it was actually doing. Five separate reward exploits were found, four
of them by policies that were already reporting 99% or better. The scripted
oracle, built as a measuring instrument rather than a policy, ended up
outperforming every learned teacher on every dimension that mattered.

---

## 1. Why an RL teacher at all

Behaviour cloning had plateaued. The scaling curve across 71, 151 and 221
demonstrations reads 58.7% → 77.2% → 83.5%, which is logarithmic, so
reaching 95% by teleoperation alone would take well over a thousand
episodes.

The deeper reason is in the project's own evidence. ACT's teacher-forced fit
error fell from 26 mm to 6.8 mm between training step 5k and 55k while its
success rate did not move at all. The policy can imitate. It cannot recover
once it leaves the demonstration manifold. That is compounding error, and
the only cure is training on the states the policy actually visits.

Stage 0 then made the target precise:

* **The environment permits 100%.** A privileged scripted oracle scored
  600/600. Every point of the gap to ACT's 83.5% is learnable.
* **62% of ACT's failures are grasps that never happened**, 18% are blocks
  knocked aside, 20% are drops. **Zero of 300 failures were near misses.**

So the teacher's job was specific: demonstrate recovery from botched
approaches, which a human operator who rarely botches approaches cannot
demonstrate.

---

## 2. What was built

```
src/rl/env.py          Gymnasium environment, privileged, render-free
src/rl/reward.py       The staged dense reward
src/rl/reference.py    The demonstration trajectory profile, and scoring
src/rl/buffer.py       Replay buffer that blends in frozen oracle data
src/rl/seed.py         Collects oracle rollouts as buffer transitions
src/eval/oracle.py     The scripted privileged pick-and-place

src/scripts/train_teacher.py      SAC training, resumable
src/scripts/build_demo_buffer.py  Builds the frozen oracle buffer once
src/scripts/eval_teacher.py       600-trial evaluation with quality gates
src/scripts/trajectory_report.py  Scores trajectories against the demos
src/scripts/teacher_demos.py      Videos, contact sheets, height traces
src/scripts/test_rl_env.py        Validates the env against the oracle
```

### The environment

**Privileged observation, 38 dimensions.** Joint state, end effector pose,
target pose, and the true block pose. This is the opposite of the rule the
datasets follow, and it is deliberate and confined to this file. The teacher
exists to relabel actions for a vision student; the student never sees any
of it. Excluding block pose from the student is what stops it skipping
perception. Including it here is what makes the teacher cheap, because a
privileged observation needs no cameras and therefore no rendering, which is
the expensive half of the simulator.

**Delta actions, absolute targets.** The agent emits a small change to the
target pose; the environment integrates it into an absolute target before
handing it to the impedance controller. That mirrors what the DualSense
integrator does during collection, where sticks produce deltas and the
recorded action is the absolute target. The integrated absolute target is
exposed every step as `info["absolute_action"]` in the dataset's own
8-element layout, so Stage 3 can relabel a student's states with no
conversion. A teacher emitting actions in a different space would need a
translation layer, and translation layers are where silent divergence lives.

**No rendering**, which is what buys 4,753 policy steps per second across 20
workers.

### Validating the environment before trusting it

`test_rl_env.py` drives the environment with the scripted oracle and checks
it still solves the task. This is the same argument `test_harness.py` makes
about the evaluation harness. The environment is a *second* implementation
of the collection dynamics, with its own action integration, stepping loop
and termination, and if it had drifted then a teacher trained inside it
would be solving a different task from the one being measured, with nothing
raising an error.

It has passed at 100% after every change made to it.

---

## 3. The reward, and the five exploits it survived

Every exploit below produced a policy that looked like it was working, and
every fix felt complete until the next one was found.

### 1. Loitering

`+2.0` per step for holding the block. Carrying it for a full 600-step
episode paid about +1000 against +100 for finishing, so the optimal policy
picks the block up and never puts it down.

Caught by arithmetic before training.

### 2. Repeat grasping

The grasp bonus fired on every transition into holding, so grasp, release,
grasp again paid +25 each time.

Caught the same way. Fixed by paying it once per episode, which is why
`grasped_once` is threaded through from the environment rather than derived
from the previous step alone.

### 3. Shoving, which reached a reported 99.0%

After the first two fixes every per-step term was negative and every bonus
was one-time, which felt airtight. It was not.

`check_success` asks only that the block end within 5 cm of the target and
*resting*, and **a block slid across the table satisfies both**. The
transport shaping was gated on a genuine lift, so shoving earned no shaping,
but the completion bonus was never gated. Shoving paid the full +100 and,
being quicker than picking, also paid less time penalty.

**That teacher scored 99.0% over 600 trials while never holding the block in
two thirds of them.** Mean lift was 8 mm against the oracle's 78 mm.

It announced itself and the signal was ignored. The risk register already
said that a teacher earning far more than the oracle's reference return has
found the reward rather than the task. It earned 119 against 104. That was
explained away as "it finishes faster so it pays less time penalty", which
is mechanically true and misses the cause entirely: it finished faster
*because it was not picking the block up*.

Fixed with two guards, since one had already proven insufficient: the
completion bonus requires a genuine lift, and any block motion while the
block is not held costs 100 per metre.

### 4. Never releasing, which reached a reported 99.8%

The shoving fix worked: picked-and-placed went from ~33% to 99.3%. But
plotting block height against time showed the block ending in mid-air.

| | measured over 100 episodes |
|---|---|
| reported success | 100/100 |
| gripper still closed at episode end | **98/100** |
| ended with block gripped and airborne | **37/100** |
| dropped and re-grasped at least once | **72/100** |

`check_success` allows 20 mm of resting tolerance while holding needs
10 mm, so a block hovering in that window counts as held and resting at the
same time. Neither the criterion nor the reward ever asked for a release.

**This one was caught only by watching ten videos.** It was invisible in the
success rate and invisible in the picked-and-placed metric built to catch
the previous exploit.

### 5. Dropping instead of placing

The release fix required the gripper to be open and the block near resting.
It never required the gripper to open *low*.

| | measured |
|---|---|
| EE height at release | 0.515 m, **95 mm above the block** |
| block downward speed at release | 303 mm/s, max 1255 |
| steps from release to episode end | **0.7** |

The gripper opened while the arm was high, the block fell 95 mm, and the
criterion fired the instant it landed. A drop satisfied a place test.

Fixed by requiring three things: gripper open, block within 5 mm of resting
and moving under 50 mm/s, and the tool descended below 0.45 before
releasing.

### The lesson

`check_success` is under-specified for training. It does not require a pick,
it does not require a release, its 20 mm tolerance admits a hover, and it is
applied at the instant the episode ends rather than after the block settles.
Using it as a reward target invites the optimiser to find each of those
gaps in turn, and it did.

`eval_teacher.py` now reports five numbers, and the gap between them is the
diagnosis:

| measure | what a gap reveals |
|---|---|
| success | nothing on its own |
| picked | shoving |
| placed | never letting go |
| gripper open at end | never letting go |
| still placed after a 60-step settle | released too near the edge |

---

## 4. The trajectory problem

With the exploits closed, teacher v3 reached **98.8% still-placed over 600
trials** and looked nothing like the demonstrations doing it.

| | demonstrations | teacher v3 |
|---|---|---|
| approach, start to grasp | 105 steps | not separable |
| carry, grasp to release | 131 steps | not separable |
| retreat, release to end | 59 steps | **absent** |
| total | 295 steps | **73** |
| mean target speed | 3.06 mm/step | ~6 sustained |
| lift above grasp | 80 mm | 31 mm |

The demonstrations have a deliberate speed profile the teacher ignored
entirely. Across normalised episode time it reads

```
1.9  1.8  1.9  2.8  4.7  4.1  2.4  1.6  4.7  4.8
```

slow to position, quick to lift and carry, slow again to descend and
release, quick to leave.

### Two causes, both self-inflicted

**Every per-step reward term was negative.** That was the correct fix for
the loitering exploit, but it means the optimal policy is always the fastest
one: speed is never traded against anything. A flat time penalty has no
interior optimum.

**The episode ended at release.** No retreat phase existed because the
environment terminated before one could. The teacher was not declining to
lift the arm clear; it was never asked.

### The metric, validated before use

`src/rl/reference.py` holds the demonstration profile as measured constants
and scores any policy against it in bands rather than at points, because the
demonstrations themselves span a band.

The oracle was scored *before* anything was optimised against the metric. If
the metric had not rated the oracle highly, it would have been the wrong
target and every run after it aimed at nothing.

| oracle version | trajectory score |
|---|---|
| single speed, retreat straight up | 0.882 |
| per-phase demonstration speeds, retreat toward home | 0.986 |
| two-stage retreat (lift off, then home) | **0.991** |

The first score was not a metric bug. It correctly caught that the oracle's
retreat did not match the demonstrations, and fixing the *oracle* fixed the
score.

---

## 5. Every run

| run | change | outcome |
|---|---|---|
| v1 | first reward | 99.0% raw, **shoving in 2/3 of episodes** |
| v2 | shove guards | 99.8% raw, **never released the gripper** |
| v3 | release required | **98.8% settled**, trajectory nothing like the demos |
| v4 | trajectory shaping + retreat phase | **0 to 5%**, shut the gripper on step 0 |
| v5 | RLPD fix | killed at 100k; demo buffer found 20× undersized |
| v6 | demo buffer sized correctly | **97.8% settled**, trajectory score 0.507 |
| v7 | heavier speed weights | abandoned, worse than v6 at the same point |
| v8 | gamma 0.997, demo ratio 0.75 | **0% at 900k** |
| v9 | gentle place + retreat to home | 4.2% at 360k |
| v10 | two-stage retreat + disturbance penalty | **0% at 660k** |

### What each failure taught

**v4: trajectory shaping outvoted the task.** `W_SPEED` was 0.05 with no
cap. A policy at the 7 mm/step ceiling during approach deviates 5.15 from
the 1.85 target, which squared and weighted is 1.33 per step against a reach
shaping term of about 0.15. The dominant gradient was "move at 1.85 mm/step"
rather than "go to the block", and the policy shut the gripper on step zero
so it could never straddle the block, then ran out the clock for 1.2 million
steps. Shaping should nudge a policy that is already solving the task; it
must never outvote the task.

**v5: three bugs in the RLPD implementation.**

1. *FIFO eviction.* The plan said RLPD; what was implemented was writing
   oracle transitions into SAC's ordinary buffer before training. That
   buffer holds a million transitions and the run pushed 1.2 million more
   through it, so the 50,000 oracle transitions were diluted immediately and
   **evicted entirely** around step 950,000. Real RLPD keeps demonstrations
   in a separate buffer that never evicts and draws a fixed share of every
   gradient batch from it.
2. *Buffer sized 20× too small.* Stable Baselines treats `buffer_size` as a
   count of transitions and divides by `n_envs` internally.
   `make_demo_buffer` divided as well. The buffer wrapped and kept only the
   last nine oracle episodes, and half of every gradient batch was drawn
   from those nine for an entire run. **The training log said "50,000
   transitions from 176 oracle episodes", which was true of what the
   collector sent and false of what the buffer kept.** Only unpickling the
   saved buffer showed the difference. `build_demo_buffer.py` now reports
   what was kept and warns if it wrapped.
3. *A `None` field.* Stable Baselines 2.9 added a `discounts` field to
   `ReplayBufferSamples` which is `None` unless n-step returns are in use,
   so concatenating blindly across the tuple fails.

**v8: raising gamma did not work, and the experiment could not say why.**
The diagnosis was that at `gamma=0.99` the effective horizon is about 100
steps against a 295-step episode, so the agent is myopic and collecting the
terminal bonus 150 steps earlier is worth roughly 4.5× more in present
value. That may still be true. But the run changed `gamma` from 0.99 to
0.997 **and** `demo_ratio` from 0.5 to 0.75 at once, and demo ratio 0.75
leaves only a quarter of each batch as online data. Two variables, one
900,000-step run, no attribution. That was avoidable.

**v9 and v10: each quality fix made the task harder to explore into.**

| | place criterion | retreat criterion | result |
|---|---|---|---|
| v6 | loose | height only | 100% |
| v9 | gentle, low release | reach home | 4.2% at 360k |
| v10 | + staged retreat, disturbance penalty | | 0% at 660k |

The task became a ~280-step chain where the terminal bonus pays only if
approach, grasp, lift, carry, low descent, gentle release, vertical lift-off
and return-to-home all succeed. At `gamma=0.99` that bonus is invisible from
the start of an episode, so exploration has to find the whole chain from
dense shaping alone. It does not.

---

## 6. Infrastructure findings

**Five consecutive background training runs were killed with no error**, GPU
idle and 14 GB of RAM free each time. Two died at exactly the same point,
500 of 2,500 lockstep steps, about 45 seconds in. The common factor was
seeding inside the training process, which had 20 SubprocVecEnv workers and
20 more local environments alive simultaneously.

Two workarounds, both of which are better engineering anyway:

* `build_demo_buffer.py` collects the oracle data once in its own
  short-lived process and writes it to disk. It has never failed.
* Long runs were split into roughly eight-minute foreground segments chained
  with `--resume`. Foreground segments were never killed.

Background runs have been reliable since seeding was decoupled.

**Checkpoints must carry the replay buffer.** Saving the policy alone is
close to useless for resuming an off-policy run: SAC's competence lives as
much in the buffer as in the weights, and reloading against an empty buffer
re-enters the warmup phase. `ResumableCheckpoint` writes both.

**Throughput.** 20 workers reach 4,753 environment steps per second with no
rendering, but SAC training runs at 150 to 235 because gradient steps are the
constraint, not the simulator.

---

## 7. Where it ended up

The scripted oracle, built in Stage 0 purely as a measuring instrument,
outperforms every learned teacher on every dimension that matters.

| | oracle | best RL teacher (v6) | demos |
|---|---|---|---|
| success, 600 trials | **100.0%** | 97.8% | n/a |
| retreated properly | **100.0%** | partial | n/a |
| trajectory score | **0.991** | 0.507 | 1.0 |
| episode length | 284 | 80 | 295 |
| carry speed | 2.74 | 7.25 | 3.27 |
| block nudged after release | **0.06 mm** | n/a | n/a |

Every trajectory metric except retreat speed is a perfect band match.

The oracle is closed-loop on the true block pose, recomputing its waypoints
from the current block position every step, so it handles perturbed states.
That is what DAgger relabelling in Stage 3 actually requires, and it is the
teacher's real job.

Its parameters were not invented. Grasp offset, wrist yaw, hover height,
transit height and release pose were all extracted from the 221 recorded
episodes, and its per-phase speeds are the measured demonstration speeds.

### The remaining RL idea

RLPD seeds the **critic** well and leaves the **actor** to discover the
chain, which is the part that fails on a 280-step task. The standard remedy
is to pretrain the actor by behaviour cloning on the oracle data, then let
RL fine-tune from there. 60,000 oracle transitions are already on disk.

### It was tried. Behaviour cloning works; the fine-tuning destroys it.

`src/rl/pretrain.py` regresses the actor's output onto the oracle's actions
before any environment step. Two details mattered: SAC's actor is
tanh-squashed, so targets are inverse-tanh transformed and clipped short of
the asymptote, because the binary gripper channel sits on the boundary
constantly; and only the actor is pretrained, since the critics learn real
returns from the demonstration buffer anyway.

**Behaviour cloning alone reaches the goal that reinforcement learning never
found.**

| BC epochs | MSE | success | retreated | steps |
|---|---|---|---|---|
| 6 | 0.079 | 0% | 0% | 600 |
| 60 | 0.016 | **70%** | **70%** | 381 |
| 150 | 0.008 | 40% | 0% | 600 |
| 60, repeated | 0.017 | **10%** | **10%** | 571 |

Two things follow. 150 epochs overfits and is worse than 60. And the same
60-epoch configuration produced 70% once and 10% another time, so the
apparently good result was substantially luck on a ten-episode check.
Behaviour cloning on this data is high variance and not a reliable teacher
either.

**Reinforcement learning then destroys whatever the cloning achieved.**
Starting from a pretrained actor, `rl_teacher_v11` fell to 0% within 50,000
steps and stayed there for the remaining 600,000. SAC's entropy term and its
bootstrapped critic push the actor off the cloned solution long before the
critic is accurate enough to pull it back.

`--bc-only` exists so a cloned policy can be kept without that happening.

---

## 8. Conclusion

Ten training runs. The best learned teacher, v6, reached 97.8% success under
criteria that were then correctly rejected as not matching the recorded
demonstrations. Every subsequent quality requirement made the task harder to
explore into, and none of the later runs learned it at all.

**The scripted oracle is the Stage 3 teacher.** It was built in Stage 0 as a
measuring instrument and ended up outperforming every learned policy on
every dimension that matters.

### 20-trial visual evaluation

```powershell
python -m src.scripts.teacher_demos --oracle --episodes 20 --tail 50
```

| | result |
|---|---|
| succeeded | **20/20** |
| genuine lift | **20/20** |
| lift height | 78 to 80 mm (demonstrations: 80 mm) |
| episode length | 247 to 329 steps (demonstrations: 227 to 367) |
| block nudged after release | 0.06 mm |

![Oracle block height traces](oracle_block_height.png)

The height traces are the evidence. Flat on the table, one clean rise to
78 mm, a flat plateau through the carry, one clean descent, and **flat
afterwards** because the two-stage retreat leaves the block where it was
put. Compare that to the teacher that scored 99.0% while shoving, whose
traces never left the table, or the one that scored 99.8% while ending in
mid-air.

### Against the full 600-trial protocol

| | oracle | best RL teacher | demonstrations |
|---|---|---|---|
| success, 600 trials | **100.0%** | 97.8% | n/a |
| retreated properly | **100.0%** | partial | n/a |
| trajectory score | **0.991** | 0.507 | 1.0 |
| episode length | 284 | 80 | 295 |
| carry speed | 2.74 | 7.25 | 3.27 |
| block nudged after release | **0.06 mm** | 17.4 mm | n/a |

### What the RL work produced anyway

The teacher itself is not the only output, and the rest is kept:

* an environment validated against the oracle at 100%, reusable for Stage 4
* a trajectory scoring metric validated to 0.991 on a known-good policy
* an evaluation harness whose five quality gates each exist because a policy
  exploited their absence
* a demonstration profile measured from all 221 episodes
* the oracle itself, which only became a demonstration-faithful controller
  because the RL work kept exposing ways it was not one

### If the RL route is resumed

Two things to try, in order, and one variable at a time:

1. **`gamma=0.995` alone**, holding `demo_ratio` at 0.5. The myopia
   diagnosis was never cleanly tested, because the one run that changed
   gamma also changed the demo ratio.
2. **Cap the action space.** Drop `MAX_DELTA_M` from 7 mm to about 4 mm. The
   policy saturates whatever ceiling it is given, so lowering the ceiling
   enforces demonstration speed directly instead of trying to persuade the
   optimiser out of rushing.

Reward shaping should not be the next thing tried. It has been tried five
times.
