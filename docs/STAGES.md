<!-- Moved out of README.md. The README keeps the current result, the
     understanding needed to read it, and how to reproduce it; everything
     else lives here. -->

# Plan of action, and what each stage actually produced

The full stage-by-stage record: what was planned, what was measured,
and where the two differed. This is the narrative the README used to
carry inline.


The goal is 97% success **on the strict criterion**, which is not the
criterion most of this section was written against. Everything below
Stage 3b was planned and executed before `src/eval/strict.py` existed, and
its target of "95% success" means 95% on the loose criterion, which the
current policy already exceeds at 96.7%.

The live plan, and every result under the strict criterion, is in
**[docs/PATH_TO_97.md](PATH_TO_97.md)**. What follows is the record of
how the project got here.

**Current position: 92.3% ± 2.6% strict, 600 trials, seeds never used for
selection.**

### Why more demonstrations will not get there

The scaling curve reads 58.7, then 77.2, then 83.5 across 71, 151 and 221
episodes. That is logarithmic. Reaching 95% by teleoperation alone would
take well over a thousand episodes. Operator speed also drifted from 74 to
122 mm/s across the existing 221, so each new episode is noisier than the
one before it.

The deeper reason is already in the results above. ACT's teacher-forced fit
error fell from 26 mm to 6.8 mm between step 5k and 55k, and its success
rate did not move at all. The policy can imitate. It cannot recover once it
leaves the demonstration manifold. That is compounding error. The one cure
is training on the states the policy actually visits.

The target also changes shape as it gets higher. At 95% there are 30
failures in 600 trials. At 99% there are 6. The last stretch is not about
average competence. It is about removing a few rare failure modes. Nobody
knows what those modes are yet, because this repo has no failure taxonomy.
Stage 0 builds one before anything else starts.

### Platform decisions

Everything runs on native Windows. No WSL, no cloud, no second machine.

| Decision | Choice | Reason |
|---|---|---|
| Simulator | MuJoCo, unchanged | The harness is only trustworthy because collection, training and evaluation share one environment and one success criterion. `test_harness.py` proves it by replaying demonstrations at 10 out of 10. Changing engines throws that away |
| Isaac Sim | Not used | It runs on Windows, but it swaps in PhysX. That is a different contact model, in a task where contact is the hard part. It would also mean rewriting the impedance controller, and every number above would stop being comparable. Revisit it only for sim-to-real |
| MJX and JAX | Not available | `jax-cuda12-plugin` publishes no Windows wheel, so JAX on Win32 is CPU only. This is a platform fact, not a preference |
| Parallelism | 20 worker processes | The RL stage is state only and renders nothing, so 24 CPU cores are enough. GPU vectorisation is not on the critical path |
| GPU fallback | `mujoco-warp` 3.11 | Installs on Windows through NVIDIA Warp. Same engine and same scene file, so the harness stays valid. Held in reserve |
| Privileged state | Teacher only | The RL teacher and the critics see the true block pose during training. The deployed student stays vision only, so the rule that a policy must not skip perception still holds |

### Stage 0. Measure the ceiling, name the failures

One to two days. No training.

Write a scripted oracle that reads the true block pose and runs a fixed
approach, grasp, lift, transport and release. Run it for 600 trials on the
evaluation distribution. This measures what the environment itself allows.
It has never been measured.

At the same time, instrument `run_trial` to log the block z trajectory, the
gripper state and the end effector path. Then classify every failure
automatically into one of five buckets: never grasped, grasped then
dropped, near miss at 5 to 10 cm, timeout, or block knocked out of reach.

Also benchmark the environment. Steps per second, state only and rendered,
from one process up to twenty.

**This stage gates everything after it.** If near misses dominate, the
problem is precision, and residual RL is enough. If grasp failures
dominate, the problem is perception, and distillation matters most. If
timeouts and drift dominate, the problem is compounding error, and DAgger
is the answer. Running the later stages without this is aiming blind.

New files: `src/eval/oracle.py`, `src/eval/failure_analysis.py`,
`src/scripts/measure_ceiling.py`, `src/scripts/classify_failures.py`.

#### The ceiling is 100%

```powershell
python -m src.scripts.measure_ceiling
```

| Metric | Value |
|---|---|
| Success | **600 / 600, 100.0%** |
| Seeds | 51 to 56, 100 trials each |
| Near region, x < 0.53 | 100.0%, n=238 |
| Far region, x >= 0.53 | 100.0%, n=362 |
| Steps to complete | mean 152, against a 600 step budget |

Not one failure in 600 trials. By the rule of three the true failure rate
sits under 0.5% at 95% confidence, so the ceiling is at least 99.5%.

**Every point of the 16.5 point gap between ACT at 83.5% and a solved task
is learnable.** Nothing in the contact model, the gripper geometry, the
workspace or the step budget forbids 99%. The target set out at the top of
this plan is available, and the earlier caveat that it might not be is now
withdrawn.

Two further readings. The near region, which cost both policies 14 to 21
points before the biased collection, is not intrinsically harder: the
oracle scores 100% there too, which independently confirms the coverage
explanation reached earlier by a completely different route. And the oracle
finishes in 152 steps against a 600 step budget, so timeouts are not a
structural constraint. A policy that times out is dithering, not running
out of road.

The oracle earns this only because it was built from the demonstrations
rather than invented. Its grasp pose, approach height, transit height,
release pose and wrist yaw were all extracted from the 221 recorded
episodes, and its target is rate limited to 4 mm per step, inside the 74 to
122 mm/s the operator produced. It drives the same absolute target pose
interface through the same impedance controller. An oracle that teleported
its target or used inverse kinematics would have measured the ceiling of a
different system.

What it does not show: that a closed-loop policy driven by pixels can reach
100%. It shows that the control path and the physics permit it. The
remaining gap is perception and closed-loop behaviour.

#### The failures are grasps, and there are no near misses at all

```powershell
python -m src.scripts.classify_failures --run act_v4 --checkpoint 020000
```

ACT v4 @ 020000, 300 trials on seeds 51 to 53, scoring 83.3%. That matches
the 83.5% headline, so this is the same policy behaving normally.

| Failure mode | n | of all trials | of failures | near x<0.53 | far x>=0.53 |
|---|---|---|---|---|---|
| never grasped | 31 | 10.3% | **62.0%** | 7 | 24 |
| dropped in transit | 10 | 3.3% | 20.0% | 4 | 6 |
| knocked away | 9 | 3.0% | 18.0% | 1 | 8 |
| **near miss** | **0** | **0%** | **0%** | 0 | 0 |

All 50 failures hit the 600 step limit. Not one ended early.

**There are no near misses. Zero in 300 trials.** Not a single failure was
a block delivered close to the target but outside the 5 cm radius. Once
this policy has the block in hand and is carrying it, it finishes. The
5 cm success radius is not what is costing points, and neither is fine
positioning at release.

That result retires an assumption this plan was built on. Stage 4 proposed
residual RL to supply "fine precision correction" on an already competent
policy. There is no imprecision to correct. A bounded correction of a
couple of centimetres, applied to a policy that fails by never picking the
block up, is aimed at a failure mode that does not occur. Stage 4 is
rewritten below.

**80% of failures are grasp-phase failures**, counting never grasped
together with knocked away, since both are a grasp attempt that fouled or
missed. Add the drops and 100% of failures happen at or before the moment
the block is secured. Fixing only the grasp phase would take this policy
from 83.3% to about 96.7%.

The timeouts complete the picture from the other side. The oracle finishes
in 152 steps. Every ACT failure burns all 600 and delivers nothing. These
policies are not running out of road, they are dithering in front of the
block, which is the same open-loop commitment problem the Stage 1 chunk
sweep exposed from a different angle.

One incidental finding. Failures now concentrate in the **far** region, 24
of 31 missed grasps at x >= 0.53. The 70 near-biased episodes did their job
and then some. The weak region has moved, and a further biased collection
would now need to point the other way.

#### Throughput, and the two risks it settles

```powershell
python -m src.scripts.bench_env
```

Policy steps per second, 33 physics steps each, on 24 cores.

| Workers | State only | Scaling | 3 cameras | Scaling |
|---|---|---|---|---|
| 1 | 358 | 1.0x | 195 | 1.0x |
| 4 | 1,424 | 4.0x | 697 | 3.6x |
| 8 | 2,645 | 7.4x | 936 | 4.8x |
| 16 | 4,399 | 12.3x | 891 | 4.6x |
| 20 | **4,753** | 13.3x | 934 | 4.8x |

**Stage 2 is far cheaper than feared.** Five million state-only policy
steps take about 18 minutes at 4,753 steps per second. Twenty million take
about 70 minutes. The overnight budget set aside for the RL teacher is not
the constraint, which means the sample budget can be generous rather than
carefully rationed.

**Multi-process offscreen rendering works.** Eight processes each holding
their own WGL context ran without incident. That risk is closed. Rendering
does plateau at about 930 steps per second from eight workers onward, which
is GPU contention rather than a failure, so Stage 3 should use eight
rendering workers and not twenty.

The controller cost is real but not binding. A single worker manages 358
policy steps per second, which is 11,800 physics steps per second including
33 Python controller calls each doing a Jacobian and a mass matrix solve.
That is slower than MuJoCo alone would run, so the earlier suspicion was
correct in direction. It simply does not matter at this scale. Scaling
falls off above 16 workers, 13.3x rather than 20x, which is most likely
BLAS threads competing across processes and would be worth pinning
`OMP_NUM_THREADS=1` per worker if it ever becomes the limit.

### Stage 1. Cash in the free wins. Done, and it did not work

One day. No retraining. Both settings are inference-time only, so every
configuration ran against `act_v4` @ 020000, the checkpoint already on disk.

Temporal ensembling was the standard fix for the compounding error
diagnosed above, and the expectation written here beforehand was 86 to 90%.
That expectation was wrong. Nothing beat the baseline.

```powershell
python -m src.scripts.ensemble_sweep --run act_v4 --checkpoint 020000
```

| Config | Success | vs baseline | se | flips | s/trial |
|---|---|---|---|---|---|
| baseline, chunk 32 | **88.0%** | | | | 1.8 |
| temporal ensembling, coeff 0.05 | 85.0% | -3.0 | 3.9 | 15 | 8.7 |
| temporal ensembling, coeff 0.00 | 83.0% | -5.0 | 4.1 | 17 | 6.9 |
| temporal ensembling, coeff 0.01 | 81.0% | -7.0 | 4.4 | 19 | 6.2 |
| n_action_steps 16 | 85.0% | -3.0 | 3.6 | 13 | 3.3 |
| n_action_steps 8 | 61.0% | -27.0 | 6.2 | 39 | 5.2 |
| n_action_steps 4 | 26.0% | -62.0 | 8.0 | 64 | 8.4 |
| n_action_steps 1 | 0.0% | -88.0 | 9.4 | 88 | 19.7 |

100 trials on seed 41, held apart from the seeds reserved for reporting.
Every configuration saw identical block placements, so the comparison is
paired and the differences are McNemar deltas. Only the trials where two
configurations disagreed carry information, which is why `se` tracks the
flip count rather than n.

All three ensembling coefficients land below baseline. The best of them is
3.0 points down with a standard error of 3.9, so it cannot be told apart
from no change. What can be said is that there is no evidence of a gain in
any of them.

**The chunk length sweep is the real finding.** Success collapses
monotonically as the policy re-plans more often: 88, 85, 61, 26, 0. This is
the opposite of what the roadmap assumed. Re-planning with fresher
observations should help a policy that suffers from stale plans. Here it
destroys it.

Instrumenting a `n_action_steps=1` rollout shows why. The arm is not stuck.
Joint velocity is non-zero throughout and the impedance controller tracks
its target to within 2 to 12 mm. What happens is that the arm dithers. It
descends about 40 mm over 105 policy steps and then starts rising again
before it ever reaches the block. Re-planning every step from a state that
looks mid-motion makes the policy re-issue the early part of a chunk over
and over, so it never commits to the trajectory. Effective progress falls
far enough that the 600 step budget expires.

#### Why temporal ensembling did not work

**It never had headroom to find.** Ensembling forces `n_action_steps=1`, so
it starts from the same catastrophe as the control above, at 0%. Its
averaging then reconstructs an approximation of the committed trajectory
and claws back to 81 to 85%. Executing the chunk open loop already gives
88% directly. So ensembling is not adding a correction on top of the
baseline. It is paying a large price to rebuild most of what the baseline
had for free. There was never a gain available above 88%, only a shortfall
to recover.

**Averaging drags the commanded target backward, measurably.** The action
is an absolute Cartesian target, and it leads the end effector. That lead
is what pulls the arm forward. Ensembling averages 32 predictions made at
32 different times, many of them from when the arm was further back, which
pulls the average toward where the arm has already been:

| Setting | Target lead | Target advance per step |
|---|---|---|
| baseline, chunk 32 | 11.79 mm | 2.66 mm |
| temporal ensembling 0.01 | 10.61 mm | 2.30 mm |
| temporal ensembling 0.05 | 10.52 mm | 2.17 mm |

The lead shrinks by about 10% and the target advances about 18% more
slowly. That is a real effect and it points the right way, but it is too
small on its own to explain a 3 to 7 point drop. Episodes run roughly 18%
longer against a 600 step budget they were finishing in 250 to 350. Stating
it plainly: the direction is confirmed, the magnitude is not sufficient,
and the residual is within the noise of the measurement.

**Why the ALOHA result does not transfer.** This part is interpretation
rather than measurement. Temporal ensembling was introduced for an arm
taking absolute joint position commands directly, where jitter between
successive chunks becomes real jerk and averaging removes it. Here the
action is a target pose handed to a Cartesian impedance controller running
at 1 kHz with a 1.5 Hz bandwidth. That controller is already a low pass
filter, and a well characterised one. The smoothing ensembling offers is
largely redundant with smoothing the plant performs anyway, while its cost
in target lead is not. A technique that pays for smoothing it does not need
should be expected to come out behind.

**What this says about the policy.** Its competence depends on open-loop
commitment to a 32 step motion primitive, not on closed-loop feedback. That
is a sharper statement of the compounding error problem than the fit-error
result gave, and it carries a warning for Stage 4: a residual correction
applied per step is operating in exactly the regime where this policy falls
apart. Stage 4 should correct at chunk boundaries, or be validated against
this table before it is trusted.

Two mechanisms were proposed and both were tested and rejected, so they are
recorded here to save the next person the same detour. Chunk element 0 is
not a stay-put command, it already leads the end effector by about 11 mm.
And ensembling does not smear the binary gripper: it produces 5 intermediate
gripper frames in 200 against the baseline's 1.

No configuration was promoted to a six seed run, because none of them
earned one. Confirming a negative point estimate across six seeds would
cost about four hours and change nothing.

### Stage 2. Teacher. Done, and the winner is the scripted oracle

The goal was an RL teacher that could demonstrate recovery from botched
approaches, which Stage 0 identified as 62% of ACT's failures. Eleven
training runs later the teacher is the scripted oracle from Stage 0, which
outperforms every learned policy on every dimension that matters.

Full account in [docs/RL_PROCESS.md](RL_PROCESS.md), run-by-run log in
[docs/STAGE2_TRAJECTORY_LOG.md](STAGE2_TRAJECTORY_LOG.md). Summary here.

| | oracle | best RL teacher | demonstrations |
|---|---|---|---|
| success, 600 trials | **100.0%** | 97.8% | n/a |
| retreated properly | **100.0%** | partial | yes |
| trajectory score vs demos | **0.991** | 0.507 | 1.0 |
| episode length | 284 | 80 | 295 |
| carry speed, mm/step | 2.74 | 7.25 | 3.27 |
| block nudged after release | **0.06 mm** | 17.4 mm | n/a |

```powershell
python -m src.scripts.test_rl_env --trials 20          # validate the env
python -m src.scripts.trajectory_report --oracle       # score vs the demos
python -m src.scripts.teacher_demos --oracle --episodes 20 --tail 50
```

![Oracle block height traces](oracle_block_height.png)

Twenty of twenty, every one a genuine 78 mm lift. The traces are the point:
flat on the table, one clean rise, a flat plateau through the carry, one
clean descent, and flat afterwards because the arm lifts straight up before
translating home. A shove would never leave the table. A drop would end in
mid-air. Both of those happened, and both reported better than 99%.

#### Five reward exploits, four of them reporting over 99%

Each fix felt complete until the next was found. This is the single most
transferable finding in the project.

| exploit | what it reported | what it was doing |
|---|---|---|
| loitering | n/a, caught early | picking the block up and never releasing |
| repeat grasping | n/a, caught early | grasp, release, grasp to farm the bonus |
| **shoving** | **99.0%** | sliding the block, never holding it in 2/3 of episodes |
| **never releasing** | **99.8%** | gripper shut in 98 of 100 episodes, block left in mid-air |
| **dropping** | high | opening the gripper 95 mm up and letting the block fall |

`check_success` is under-specified as a training target. It does not require
a pick, does not require a release, its 20 mm resting tolerance admits a
hover, and it is applied at the instant the episode ends rather than after
the block settles. The optimiser found every one of those gaps in turn.

**The success rate saw none of it.** Nor did the picked-and-placed metric
built to catch the previous exploit catch the next one. What caught the
fourth was plotting block height against time and watching ten videos.

`eval_teacher.py` now reports five numbers, and the gap between them is the
diagnosis: success, picked, placed, gripper-open-at-end, and still-placed
after a 60-step settle. The last is the honest one.

#### Why the RL teacher lost

It did work under loose criteria: 97.8% still-placed over 600 trials. It
then failed every quality requirement in turn, and each fix made the task
harder to explore into.

| | place criterion | retreat criterion | result |
|---|---|---|---|
| v6 | loose | height only | 100% |
| v9 | gentle, low release | reach home | 4.2% at 360k |
| v10 | + two-stage retreat, disturbance penalty | | 0% at 660k |

The finished task is a chain of roughly 280 steps through approach, grasp,
lift, carry, low descent, gentle release, vertical lift-off and return to
home, and the terminal bonus pays only if every link holds. Exploration
never completes it once.

Behaviour cloning the actor on oracle data reaches 70% where reinforcement
learning reached 0%, but the same configuration produced 10% on a repeat, so
it is not reliable either. Reinforcement fine-tuning from a cloned actor
destroys it within 50,000 steps.

#### What the RL work produced anyway

The teacher was not the only output, and the rest is in use:

* a Gymnasium environment validated against the oracle at 100%, reusable
  for Stage 4
* a trajectory metric validated at 0.991 on a known-good policy, which is
  how the demonstration profile became measurable at all
* five evaluation gates that each exist because a policy exploited their
  absence
* the demonstration profile itself, measured across all 221 episodes
* the oracle, which only became demonstration-faithful because the RL work
  kept exposing ways it was not: per-phase speeds, a retreat that returns
  home, and a two-stage lift-off that leaves the block undisturbed

### Stage 3. Distil to vision. Done, and it did not work

**Teacher DAgger, unattended.** The vision student drives; the oracle is
asked at every step what the correct action would have been. The student's
observations are paired with the oracle's answers, so the labels describe
correct behaviour on the student's own state distribution, which is exactly
what behaviour cloning never gets to see.

This is aimed directly at what Stage 0 measured. 62% of ACT's failures are
grasps that never happened and 18% are blocks knocked aside. Together that
is 80% of everything that goes wrong, and all of it happens in states no
demonstration contains, because a human operator who rarely botches an
approach cannot demonstrate recovering from one.

```powershell
python -m src.scripts.collect_dagger --run act_v4 --checkpoint 020000 --episodes 200
python -m src.scripts.to_lerobot --raw data/pick_place_v1_small data/dagger_v1 `
  --repo-id local/fr3_dagger_v1
```

First collection: **200 episodes, 175 of which the student solved (87.5%)**,
so roughly 25 carry the failure states plus 175 where the oracle corrects
the student's drift step by step. 9.5 minutes.

New files: `src/scripts/collect_dagger.py`. `to_lerobot.py` now merges
several raw directories, and refuses to merge sources whose image shapes
disagree.

#### Why these labels can be mixed with the recorded demonstrations

Only because the oracle was built *from* those demonstrations. Its grasp
pose, wrist yaw, hover height, transit height, release pose and per-phase
speeds were all measured from the 221 episodes, and it scores 0.991 against
their trajectory profile. Both halves of the dataset look like the same
operator.

That was not true earlier. The RL teacher ran four times faster than the
demonstrations with a 31 mm carry against their 80 mm, and mixing its labels
with recorded human actions would have put two conflicting styles into one
dataset. Stage 2 had to be finished properly before Stage 3 could start.

#### Two traps, both hit and both worth recording

**The documented resolution trap, walked into anyway.** DAgger collection
first rendered at the collection resolution of 640 x 480 for a student
trained at 160 x 128. `crop_shape` is applied in pixels and the vision
backbone accepts any size silently, so the student went effectively blind
and scored **0 out of 3** against its true 83.5%. Rendering at training
resolution restored it to 8 of 8. This is the same failure already in the
Gotchas section, met again in a new place.

DAgger episodes are therefore written at training resolution and **must
never be re-rendered**. `rerender.py` reconstructs frames by replaying
recorded actions, and the recorded actions here are the oracle's labels
rather than what the student executed, so a replay would visit entirely
different states from the ones the stored images show.

**The oracle was a bad labeller as written.** Its `__call__` integrates its
own target from its own previous target, which is correct when the oracle is
driving and wrong when a student is. Measured during student-driven
rollouts, its labels led the real end effector by **41.9 mm on average and
up to 119.8 mm**, against the 11.8 mm the demonstrations lead by. A student
trained on those would learn to command targets far ahead of its own arm.

`ScriptedOracle.label()` places the target a fixed 12 mm ahead of where the
arm actually is, in the direction of the current waypoint, with phase
advancement still keyed off the real arm. Validated two ways: label step
size is 1.99 mm/step and consistent with the arm's own motion, and driving
the task with `label()` alone scores 19 of 20, so the labels produce correct
control rather than merely plausible numbers.

#### Result: DAgger made the policy worse

**76% at the best checkpoint, against the 83.5% baseline.** Training was
carried to 30,000 steps and the success rate plateaued from 15,000 onward.

| checkpoint | 100 trials | loss |
|---|---|---|
| 5,000 | 31.0% | 0.111 |
| 10,000 | 61.0% | 0.084 |
| 15,000 | 75.0% | 0.072 |
| 17,500 | **76.0%** | 0.066 |
| 20,000 | 63.0% | 0.060 |
| 27,500 | 72.0% | 0.052 |
| 30,000 | **76.0%** | 0.049 |

Loss fell smoothly and monotonically across every one of those rows while
success went 31, 61, 75, 76, 63, 72, 76. This is the clearest example yet of
a finding already in this README: loss does not predict success. The wandb
curve looked like steady progress at exactly the point success dropped
twelve points.

```powershell
python -m src.scripts.collect_dagger --run act_v4 --checkpoint 020000 --episodes 200
python -m src.scripts.to_lerobot --raw data/pick_place_v1_small data/dagger_v1 `
  --repo-id local/fr3_dagger_v1
lerobot-train --config_path=... --wandb.enable=true --wandb.disable_artifact=true `
  --num_workers=12
python -m src.scripts.sweep_checkpoints --run act_dagger_v1 --type act
```

#### An unexplained y asymmetry

The best checkpoint scores 63.3% for blocks starting at y < -0.04 and 88.2%
above it, a 25 point gap that is stable across checkpoints. This README
states elsewhere that the y split is flat, 70 to 80 percent either side
across all twelve earlier runs. It arrived with the DAgger data.

The obvious cause was tested and is false. Failures during DAgger collection
do not cluster at low y: those episodes had 89.0% student success against
86.2% above the split, and contributed *less* failure padding, 25.0% of
frames against 33.2%. So the asymmetry is not an artefact of imbalanced
collection, and no other explanation has been established.

#### What is most likely wrong

Not established, but the two candidates worth an ablation each:

**Failure padding.** An episode where the student never grasps runs to the
full 600 step cap, against 206 for a success. So 25 episodes out of 200,
12.5%, contribute 29.4% of all DAgger frames, and each is one stuck
situation sampled six hundred times. Those are the states DAgger exists to
cover, but at that weight they may simply be teaching the policy to dither.

**Style mixing.** The oracle labels are correct control, verified by driving
the task at 19 of 20 standalone. But correct is not the same as identical to
the operator, and a dataset containing two subtly different ways of doing
the task may be worse than either alone.

#### Two process failures worth recording

**The sweep was run at 15 trials instead of the standard 20 and
over-reported by 24 points**, calling 17,500 a 100% checkpoint where 100
trials gave 76%. The documented sweep bias in this project is 5 to 7 points.
Cutting the trial count to fit a time limit destroyed the sweep's ability to
rank checkpoints at all, which is the only thing it is for.

**The first DAgger dataset was collected, merged and trained on for 80
minutes before anyone checked the gripper distribution.** Episode counts,
frame counts, image shapes and action means all matched the demonstrations.
The gripper open fraction was 0.960 against 0.575, and that single number
was the whole story. It is now checked before conversion.

#### Human DAgger, if the automated pass leaves a residue

Operator time is the scarce resource, so it goes last and goes to the
hardest cases. Run 100 trials, take over on the failures, log the
corrections.

Takeover is bumpless, which keeps the intervention mode small. Actions are
absolute target poses and the teleop integrator can be seeded from the
policy's current target, so control changes hands with no jump in the action
stream. That falls out of a design decision made for other reasons during
collection.

Expected: 92 to 97%.

### Stage 3b. Oracle demonstrations at scale. 86.8%

Plan of record, written before execution so the result can be read against
the prediction rather than after it.

#### Why this and not something else

Only one thing has ever moved the number in this project: **data scale**.
58.7 → 77.2 → 83.5 percent across 71, 151 and 221 episodes. Architecture did
not move it, inference-time tricks did not, reinforcement learning did not,
and DAgger moved it backwards.

The bottleneck on data has always been the operator. Teleoperation is slow
and tiring, and the recorded speed drifted from 74 to 122 mm/s across 221
episodes as the operator got faster. **The oracle removes that bottleneck.**
It solves the task 100% of the time at 0.991 trajectory similarity to the
recorded demonstrations, and produces an episode every few seconds.

The scaling curve suggests roughly six points per doubling. 221 to 800 is
about 1.9 doublings.

#### Steps

1. **Add per-episode diversity to the oracle.** Hover height ±10 mm, transit
   height ±15 mm, grasp offset ±3 mm, per-phase speed ±20%, target lead
   ±3 mm. A deterministic demonstrator produces degenerate data; the
   operator never did the same thing twice.
2. **Verify the randomised oracle still scores near 100%** and holds
   trajectory similarity above 0.97. If diversity breaks it, reduce it.
3. **Collect 800 oracle episodes**, oracle driving and labelling.
4. **Gate before conversion: gripper open fraction against the
   demonstrations' 0.557.** This single number was the whole story behind
   the first broken DAgger dataset, which trained for 80 minutes before
   anyone looked at it. It is now checked first.
5. **Train two variants**, mixed (221 human + 800 oracle) and oracle-only
   (800). Running both isolates style mixing from data quality in one pass
   rather than needing a separate ablation.
6. **Sweep at 25 trials**, not 15. Cutting the trial count to fit a time
   limit made the last sweep over-report by 24 points and destroyed its only
   function. Then confirm the peak on six seeds of 100.

**Prediction: 88 to 93%.** Written down in advance. The curve is
logarithmic and oracle data is less diverse than human data even with noise
injected.

#### Result: 86.8%, the first thing that beat the baseline

**86.8% ± 3.7% over 600 trials** at checkpoint 20,000, against the
83.5% ± 4.2% baseline. Predicted 88 to 93% before running, so the outcome
landed just under the low end of the prediction.

| seed | 51 | 52 | 53 | 54 | 55 | 56 | mean |
|---|---|---|---|---|---|---|---|
| success | 92 | 82 | 89 | 84 | 84 | 90 | **86.8%** |

| | ACT baseline | **oracle mixed** | oracle teacher |
|---|---|---|---|
| episodes | 221 | **1,006** | n/a |
| frames | 65,302 | **298,351** | n/a |
| success, 600 trials | 83.5% ± 4.2% | **86.8% ± 3.7%** | 100% |
| near region, x < 0.53 | 87.8% | 95.3% | 100% |
| far region, x >= 0.53 | 80.7% | 89.5% | 100% |

#### How much of that improvement is real

Less than the headline suggests, and the honest reading matters more than
the number.

Seed 51 drove every checkpoint decision in this run, so including it is
selection-contaminated. **On the five held-out seeds alone it is 85.8%.**
The gain is +3.3 points with seed 51 and +2.3 without, against a standard
error on the difference of about 2.0 points. That is 1.2 to 1.6 standard
errors: suggestive, not significant.

Separating 86.8% from 83.5% properly needs roughly 1,400 trials, by the same
arithmetic this README already applies elsewhere. What can be said without
qualification is that this is the first of four approaches that did not land
*below* the baseline, and that its regional splits are balanced rather than
hiding a weakness.

#### The training curve, and why single checkpoints are worthless here

Measured on seed 51 during training, then on a clean screening seed:

| checkpoint | seed 51 | screening seed 61 |
|---|---|---|
| 5,000 | 57% | |
| 10,000 | 59% | |
| 13,750 | 53% | |
| 15,000 | 78% | |
| 17,500 | 70% | |
| 18,750 | | 71.7% |
| **20,000** | **92%** | **90.0%** |
| 21,250 | | 86.7% |
| 22,500 | | 80% |
| 23,750 | | 63.3% |
| 25,000 | | 85.0% |

**Adjacent checkpoints 1,250 steps apart differ by up to 27 points.** 53% at
13,750 and 78% at 15,000. 90% at 20,000 and 63.3% at 23,750. Loss fell
smoothly and monotonically across every one of those rows.

A protocol slip is recorded here rather than hidden: seeds 52 and 53 were
used at one point to compare checkpoints, which contaminates them as
reporting seeds by exactly the mechanism this README describes as sweep
bias. Screening moved to dedicated seeds afterwards. The six-seed number
above was measured before that happened and is unaffected.

#### What made the difference

The oracle dataset differs from the failed DAgger dataset in ways that were
measured before training, not diagnosed after:

| | DAgger | oracle |
|---|---|---|
| gripper open fraction, demos are 0.557 | **0.960 then 0.710** | **0.488** |
| episodes ending in failure | 12.5% | 1.8% |
| share of frames from failed episodes | **29.4%** | **3.5%** |
| frames per successful episode, demos are 302 | 206 | **295** |
| episode length | | 303 ± 86 vs 295 ± 53 |

The gate that catches this is one line comparing the gripper open fraction
against the demonstrations, and it now runs before conversion. It was not
run on the first DAgger dataset, which trained for eighty minutes before
anyone looked.

Per-episode jitter in the oracle mattered too. A deterministic demonstrator
produces the same episode every time for a given block pose, so the policy
can fit the demonstrator rather than the task. Randomising hover height,
transit height, grasp offset, speed and target lead within the ranges the
recorded episodes actually span cost 1.8% of the oracle's own success rate
and doubled the episode-length spread.

#### Where the peak sits, and what that says about scale

The peak is at 20,000 steps, which on this dataset is **4.3 epochs**. The
83.5% baseline peaked at 20,000 steps on 65k frames, roughly 20 epochs.

So the larger dataset peaked at a fifth of the exposure, in the same number
of gradient steps. That is not what a data-limited model looks like. It
suggests the remaining gap is not something more demonstrations will close,
and that the next constraint is elsewhere: model capacity, image resolution,
or the information actually present in three 160x128 views.

#### Recovery plans, decided in advance

| Outcome | What it means | Recovery |
|---|---|---|
| both variants below 83.5% | oracle behaviour is hard to imitate from pixels despite being correct control | measure teacher-forced fit error on oracle labels against human labels. Higher error on oracle labels means the 160x128 images do not contain what those labels require, and no quantity of oracle data will fix it |
| oracle-only good, mixed bad | style mixing, confirming the DAgger diagnosis | use oracle-only, or curriculum from human to oracle |
| mixed good, oracle-only bad | human data carries diversity the oracle lacks | keep the mix and push diversity injection harder |
| 84 to 88%, marginal | real but small gain | add DAgger failure states back, capped at 300 steps. Failure coverage may only pay once the base data is strong, which would also explain why DAgger hurt on a weak base |
| above 90% | worked | Stage 5 reporting at 1500+ trials, then reconsider Stage 4 |
| y asymmetry reappears | it is a property of oracle labels, not of DAgger | the oracle is 100% at low y, so the gap must be perceptual. Check camera coverage and occlusion there |

**If Plan A fails, stop.** Record it as a fourth negative result, leave the
83.5% policy and the 100% oracle as the deliverables, and document the
remaining options without executing them.

#### The risk, stated up front

This is the fourth attempt to beat 83.5%. Temporal ensembling gave nothing,
reinforcement learning gave nothing usable, and DAgger made the policy
worse. Plan A is the best remaining bet because it attacks the only lever
that has ever worked here, but a fourth failure is a real possibility and
saying so now is cheaper than discovering it in four hours.

### Stage 4. Residual RL, rewritten after Stage 0

About one week. Overnight runs. Conditional, and lower priority than it was.

The original plan here was a bounded correction supplying fine precision on
an already competent policy. **Stage 0 killed that premise.** There were
zero near misses in 300 trials. Nothing is landing slightly wrong. A policy
that fails by never picking the block up has no precision error for a
2 cm correction to fix.

What survives is the residual idea aimed at the grasp instead. If Stage 3
leaves a residue of missed grasps, learn a correction on the approach and
closing pose specifically, keyed to the phase before the gripper shuts,
rather than a uniform correction across the whole episode.

Two constraints Stage 0 and Stage 1 place on how it may be built:

* **Do not correct per step.** Stage 1 measured this policy collapsing from
  88% to 0% as re-planning went from every 32 steps to every step. A
  per-step residual operates in exactly that regime. Corrections have to be
  applied at chunk boundaries, and any implementation has to be checked
  against the Stage 1 table before it is trusted.
* **Score it on grasp success, not on final success.** The failure being
  targeted is upstream of delivery, so an aggregate rate will move slowly
  and hide whether the correction did anything.

Run this only if Stage 3 lands short of target. If distillation removes the
missed grasps, and it is aimed directly at them, this stage has nothing
left to do.

Expected: unquantified until Stage 3 reports.

### Stage 5. Report at a sample size that carries the claim

One to two days. Unattended.

Six seeds of 100 trials resolves to about 1.5 points at n=600. That is
enough to claim 95%. It is not enough to claim 99%. Separating 99% from 97%
needs roughly 1,500 to 2,000 trials.

At ACT's 2.8 s per trial that is about 1.5 hours. At diffusion's 26 s per
trial it is over twelve. That is one more reason to make ACT the vehicle
for the endgame, on top of the 5.9 point lead it already holds.

The rule from the earlier stages still stands. Sweep to find a checkpoint.
Use fresh seeds to report it.

### Expected outcome, against what happened

| Milestone | Predicted | Measured |
|---|---|---|
| Stage 1, temporal ensembling | 86 to 90% | no gain |
| Stage 2, teacher (privileged) | n/a | **100%**, the scripted oracle |
| Stage 3, DAgger distillation | 92 to 97% | **76%**, negative |
| Stage 3b, oracle data at scale | 88 to 93% | **86.8%** loose, just under |
| Stage 3e, corrected labels | 80 to 88% strict | **92.3%** strict, over |
| Stage 3f, DAgger done properly | 0 to 8 points up | **27 points down** |

Two of the six predictions landed inside their band. The Stage 3e miss is
the instructive one: the ceiling was estimated at 77.5% by counting
episodes that failed only the timing and trajectory gates and treating the
rest as independent substance. They were not independent. The bad retreat in
the training data was also causing the release, lift-off and disturbance
failures, because all three are measured at or after the release. One cause,
four gates, and the estimate was low by 15 points.

**The strict criterion did not change what the environment permits.** The
oracle still solves the task 600 times out of 600 under the loose criterion
and 97.5% of the time under the strict one with jitter applied. Every
remaining point is a policy problem, not a physics one, but the teacher is
now only 5 points ahead of the student rather than 27, so the remaining work
is matching it rather than replacing it.

95% looks reachable, and Stage 0 has now established that 99% is not
blocked by the environment. The oracle solves this task 600 times out of
600, so every remaining point is a policy problem rather than a physics
one. What is still unproven is that a vision policy can close that gap,
which is what Stages 2 and 3 are for.

### Risks

**Controller throughput. Measured, not binding.** The Python 1 kHz
controller does cost more than MuJoCo alone would, as suspected. It does
not matter at this scale: 20 workers reach 4,753 state-only policy steps
per second, so a five million step run finishes in about 18 minutes.
`mujoco-warp` stays unneeded. Scaling tails off above 16 workers and
`OMP_NUM_THREADS=1` per worker is the first thing to try if it ever binds.

**Offscreen rendering across processes. Resolved.** Eight processes each
holding their own WGL context rendered three cameras without incident.
Throughput plateaus at about 930 policy steps per second from eight
workers on, which is GPU contention rather than failure, so Stage 3 should
use eight rendering workers rather than twenty.

**Memory.** 32 GB already reached 89% with eight dataloader workers. Twenty
RL workers are individually small, since they hold no renderer and no
images. The DAgger stage runs training and rendering together, and it will
need watching.

**Reward hacking. This happened, three times.** See Stage 2 above. The
mitigation written here originally was wrong in an instructive way: it
said to score the teacher on the unmodified `check_success` criterion,
but `check_success` is itself satisfied by a block shoved across the
table, so it certifies the exact hack it was supposed to catch. The
working guard is the picked-and-placed rate reported beside the raw one.
The rest of the advice held: the reward curve did give it away, and it
was ignored.

**Non-monotonic training.** Both architectures already swing violently
during training. ACT fell to 20% at 10k steps with genuinely degraded fit
error. Assume the same for RL and DAgger checkpoints. Never select on a
single one.

---

