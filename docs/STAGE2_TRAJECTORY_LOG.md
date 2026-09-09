# Stage 2 trajectory work: iteration log

Making the RL teacher move the way the demonstrations do, without losing the
98.8% success it already had.

**Hard constraint:** still-placed success >= 97% over 600 trials.
**Objective:** trajectory similarity to the 221 recorded episodes.

---

## The problem

Teacher v3 solved the task 98.8% of the time and looked nothing like the
data doing it.

| | Demonstrations | Teacher v3 |
|---|---|---|
| approach, start to grasp | 105 steps (p10 75, p90 145) | not separable |
| carry, grasp to release | 131 steps (p10 96, p90 166) | not separable |
| retreat, release to end | 59 steps (p10 50, p90 71) | **absent** |
| total | 295 steps (p10 227, p90 367) | **73** |
| mean target speed | 3.06 mm/step (92 mm/s) | ~6 sustained |
| lift above grasp | 80 mm | 31 mm |

The demonstrations have a deliberate speed profile that the teacher ignored
entirely. Across normalised episode time it reads

```
1.9  1.8  1.9  2.8  4.7  4.1  2.4  1.6  4.7  4.8
```

slow to position, quick to lift and carry, slow again to descend and
release, quick to leave. Per phase: approach 1.85, carry 3.27, retreat 4.78
mm/step.

## Two root causes, both mine

**Every per-step reward term was negative.** That was the fix for the
loitering exploit and it was correct for that, but it means the optimal
policy is always the fastest one. Speed was never traded against anything. A
flat time penalty has no interior optimum.

**The episode ended at release.** There was no retreat phase because the
environment terminated before one could exist. The teacher was not declining
to lift the arm clear; it was never asked.

## What changed

**`src/rl/reference.py`** holds the demonstration profile as measured
constants and scores any policy against it. Bands, not points, because the
demonstrations themselves span a band.

**Reward.** The flat time penalty is replaced by a quadratic cost on
deviation from the demonstrated speed *for the current phase*, which has an
interior optimum where a flat penalty has none. Added: a jerk penalty
against step-to-step speed change above the demonstrations' 0.98 mm/step^2,
a charge on horizontal progress made while carrying below transit height
(which is what forces lift-then-move instead of the 31 mm skim), a pull
toward retreat height once the block is down, and a terminal cost on episode
length outside 255 to 335 steps.

**Environment.** Terminates on retreat complete, not on release and
certainly not on `check_success`. Every earlier termination rule cut the
episode off before the behaviour it was meant to elicit could happen.

**Oracle.** Now moves at the demonstrated per-phase speeds and retreats
toward the home pose rather than straight up, because the demonstrations
travel 282 mm during retreat, not the 99 mm a vertical lift would take.

**Oracle-seeded replay buffer.** The original Stage 2 plan called for this
and it was skipped, which cost four reward revisions. The oracle already
performs the trajectory the demonstrations show, so its episodes go into
SAC's buffer before training starts.

## Metric validation

Scored the oracle before optimising anything against the metric. If the
metric did not rate the oracle highly then the metric would be wrong and
every run after it would be aimed at the wrong target.

| Metric version | Oracle score |
|---|---|
| oracle retreating straight up | 0.882 (retreat duration 0.52, retreat speed 0.32) |
| oracle retreating toward home | **0.986** |

The first score was not a metric bug. It correctly caught that the oracle's
retreat did not match the demonstrations, and fixing the oracle fixed the
score. That is the metric behaving as intended.

---

## Iterations

### Run 1: `rl_teacher_v4`

First run with the trajectory-shaped reward and the retreat phase.
1.2M steps, 20 workers, replay buffer seeded with 2,500 lockstep steps of
oracle data (50,000 transitions, roughly 180 episodes).

```powershell
python -m src.scripts.train_teacher --steps 1200000 --workers 20 `
  --seed-oracle 2500 --learning-starts 1000 --run rl_teacher_v4
python -m src.scripts.trajectory_report --run rl_teacher_v4 --model latest
python -m src.scripts.eval_teacher --run rl_teacher_v4 --model latest
```

**Result: failed. 0 to 5% success for the whole 1.2 million steps.** Returns
ran -200 to -566 against the oracle's +131, and every episode hit the 600
step cap. 108 minutes for nothing usable.

The trajectory report said exactly what went wrong:

| metric | policy | demos |
|---|---|---|
| approach steps | **0.5** | 105 |
| retreat steps | 515.7 | 59 |
| retreated | **0%** | 100% |
| episode length | 600 (cap) | 295 |

**The policy shut the gripper on step zero** and held it shut, so it could
never straddle the block, and then ran out the clock. It never learned the
task at all. Trajectory score 0.458.

Two bugs, both mine.

**1. The speed penalty outvoted the task.** `W_SPEED` was 0.05 with no cap.
A policy at the 7 mm/step ceiling during approach deviates 5.15 from the
1.85 target, which squared and weighted is 1.33 per step, against a reach
shaping term of about 0.15 per step. The dominant gradient was therefore
"move at 1.85 mm/step", not "go to the block". Trajectory shaping should
nudge a policy that is already solving the task; it must never outvote the
task itself.

Fixed: `W_SPEED` 0.02 capped at 0.4 per step, `W_JERK` 0.02 capped at 0.2.
The cap matters as much as the weight.

**2. The oracle seeding did almost nothing, and I described it wrongly.**
The plan said RLPD. What was actually implemented was writing oracle
transitions into SAC's ordinary FIFO buffer before training. That buffer
holds one million transitions and the run pushed 1.2 million more through
it, so the 50,000 oracle transitions were diluted immediately and **evicted
entirely** around step 950,000. For most of the run the anchor the plan
depended on was not in the buffer.

RLPD keeps demonstrations in a separate buffer that never evicts and draws
a fixed share of every gradient batch from it. Fixed in
`src/rl/buffer.py`: a `MixedReplayBuffer` that samples half of each batch
from a frozen oracle buffer.

A third, smaller bug surfaced while wiring that up: Stable Baselines 2.9
added a `discounts` field to `ReplayBufferSamples` which is `None` unless
n-step returns are in use, so concatenating blindly across the tuple fails
on it.

### Run 2: `rl_teacher_v5`

Both fixes. Shortened to 500k steps as a validation rather than another
1.2M commitment, because run 1 cost 108 minutes to learn nothing and the
same mistake twice would cost most of the budget.

```powershell
python -m src.scripts.train_teacher --steps 500000 --workers 20 `
  --seed-oracle 2500 --demo-ratio 0.5 --learning-starts 1000 --run rl_teacher_v5
```

**Result: stopped externally at 100k of 500k steps, cause unknown.** This
was the second unexplained kill of a training run; both happened at a
checkpoint save, which writes a 336 MB replay buffer, but neither reported
an error.

The signal up to that point was better than run 1 at the same stage:

| steps | success | return |
|---|---|---|
| 25,000 | 0.0% | -204.6 |
| 50,000 | 2.5% | -185.1 |
| 75,000 | 0.0% | -119.5 |
| 100,000 | 0.0% | **-91.6** |

Returns climbing steadily, where run 1 at 100k was at -322 and getting
worse. Seeding reported correctly: 176 oracle episodes, all 176 completing
the retreat.

**But inspecting the checkpoint before resuming found a third bug, and it
had been silently crippling the fix.** The frozen demo buffer held 2,500
transitions, not 50,000.

Stable Baselines' `ReplayBuffer` treats `buffer_size` as a count of
transitions and divides by `n_envs` internally. `make_demo_buffer` divided
by `n_envs` as well, so the buffer was built a factor of 20 too small. It
then wrapped, keeping only the last nine oracle episodes, and half of every
gradient batch was drawn from those nine for the whole run.

Worth noting how this was caught. The training log said "50,000 transitions
from 176 oracle episodes", which was true of what the seeding function
*sent* and false of what the buffer *kept*. Only unpickling the saved buffer
and reading `buffer_size` off it showed the difference.

### Run 3: `rl_teacher_v6`

Demo buffer sizing fixed and verified by assertion before launching.
900k steps.

**Killed at 45 seconds, twice, at exactly 500 of 2500 lockstep steps.** Then
a third launch died with an empty log, before its first print. GPU was
completely idle and 14 GB of RAM was free each time, and no error was ever
reported. Four background training launches in a row were terminated
externally.

The common factor was seeding inside the training process, which had 20
SubprocVecEnv workers and 20 more local environments alive simultaneously
before a single gradient step. Two fixes:

* `src/scripts/build_demo_buffer.py` builds the oracle buffer once in its
  own short-lived process and writes it to disk. It completed without
  trouble, which confirms the collection was never the problem. It also
  reports what the buffer **kept** rather than what the collector **sent**,
  the exact distinction that hid the sizing bug, and warns if it wrapped.
* Training moved to the foreground in roughly eight minute segments chained
  with `--resume`. Foreground runs were never killed.

**Result with those in place: 97.8% still-placed over 600 trials.** The hard
constraint is met. Reached at 810k steps after nine chained segments.

| steps | success | return | episode length |
|---|---|---|---|
| 90k | 0.0% | -186 | 600 |
| 240k | 6.7% | -91 | 575 |
| 330k | 23.3% | -62 | 500 |
| 450k | 53.3% | +8 | 358 |
| 510k | 66.7% | +26 | **272** |
| 660k | 100.0% | +98 | 98 |
| 810k | 100.0% | +96 | 77 |

Look at the episode length column. It passes **through** the demonstration
band at 510k and keeps going. The policy briefly moved the way the
demonstrations do and then accelerated away from it as it got better.

### Final scores

| | v3, before this work | **v6** | demos |
|---|---|---|---|
| still-placed success, 600 trials | 98.8% | **97.8%** | n/a |
| trajectory score | not measured | **0.507** | 1.0 |
| retreat phase exists | **no** | **yes, 100%** | yes |
| releases the block | 99.7% | 99.5% | yes |
| lift above resting | 22 mm | **90 mm** | 80 mm |
| carry height | skimming | **0.514** | 0.499 |
| episode length | 73 | 80 | 295 |
| carry speed | n/a | 7.25 mm/step | 3.27 |

**What was fixed:** the phase structure. The teacher now genuinely
approaches, grasps, lifts clear, carries, descends, releases and retreats.
Lift height went from 22 mm to 90 mm, slightly above the demonstrations' own
80 mm, and both carry and retreat heights now exceed the recorded values.
The retreat phase exists at all, which it did not before.

**What was not fixed:** speed and duration. The policy saturates the
7 mm/step ceiling during carry and finishes in 80 steps against 295.

### The cause of the remaining gap, identified

It is not the shaping weights. It is **gamma**.

At `gamma=0.99` the effective horizon is about 100 steps against a 295 step
episode, so the agent is myopic. Collecting the +140 terminal bonus 150
steps earlier is worth roughly 4.5 times more in present value
(`0.99^-150`). Discounting was paying the policy to rush far harder than any
speed penalty was paying it not to, which is why raising `W_SPEED` from 0.02
to 0.05 in run 4 did not help and simply slowed learning.

The evidence is in the length column above. The policy is slowest when it is
worst and accelerates as it improves, which is what a myopic agent does as
it gets better at reaching the terminal bonus.

**The proposed fix was `gamma=0.997`**, so the horizon matches the episode
length. It was implemented, exposed as `--gamma`, and then tested to
convergence in run 5 below. **It did not work.** Read that section before
acting on this one.

### Next step

Train with `--gamma 0.997 --demo-ratio 0.75` for roughly 900k steps, about
80 minutes of chained foreground segments. Everything needed is in place and
validated. That is the single change most likely to close the remaining gap,
and it is a one-line change rather than more reward engineering.

```powershell
python -m src.scripts.build_demo_buffer --steps 2500 --envs 20
python -m src.scripts.train_teacher --steps 90000 --workers 20 `
  --demo-buffer data/oracle_demo_buffer.pkl --demo-ratio 0.75 --gamma 0.997 `
  --buffer-size 400000 --checkpoint-every 9999999 --run rl_teacher_v8
# then repeat with --resume until about 900k steps
```

### Run 5: `rl_teacher_v8`. The gamma fix was wrong.

Ran to the full 900k steps with `--gamma 0.997 --demo-ratio 0.75`, which is
exactly what the previous section recommended.

**It never learned the task.** 0% at 900k, best 3.3% across the whole run,
returns oscillating between -190 and -371 with no trend, every episode
hitting the 600 step cap.

| steps | v8, gamma 0.997 | v6, gamma 0.99 |
|---|---|---|
| 225k | 0% | ~3% |
| 405k | 0% | ~40% |
| 585k | 0% | 93% |
| 765k | 3.3% | 100% |
| 900k | **0%** | 100% (at 810k) |

**The recommendation in the section above was wrong.** Raising gamma did not
close the trajectory gap; it made the task unlearnable. The likely reason is
that a longer horizon means much higher variance in credit assignment across
a 300 step episode, and SAC could not propagate the terminal bonus back that
far.

**A confound, and it was avoidable.** The run changed `gamma` from 0.99 to
0.997 *and* `demo_ratio` from 0.5 to 0.75 at the same time, because that
pair was what the recommendation named. Demo ratio 0.75 leaves only a
quarter of each gradient batch as online data, which on its own would slow
learning substantially. So this experiment cannot say which change caused
the failure, and 80 minutes bought an ambiguous answer instead of a clean
one. Two variables, one run, no attribution.

The diagnosis that discounting drives the rushing may still be correct. What
is now known is that gamma 0.997 paired with demo ratio 0.75 does not work.

### Revised next step

Reward shaping has now failed twice at this, and gamma has failed once.
Prefer the change the optimiser cannot defeat:

**Cap the action space.** Drop `MAX_DELTA_M` in `src/rl/env.py` from 7 mm to
about 4 mm per step. The demonstrations average 3.06 mm/step and the policy
saturates whatever ceiling it is given, so lowering the ceiling enforces
demonstration speed directly rather than trying to persuade the optimiser
out of rushing. It also leaves credit assignment untouched, which is what
gamma broke.

Expect episode length to roughly double from v6's 80 steps toward the
227-367 band. Retrain from scratch, since the action scale changes what
every stored transition means.

If a discount change is still wanted afterwards, test `gamma=0.995` alone,
holding `demo_ratio` at 0.5, so the result is attributable.

### If that does not work

The oracle already scores 0.986 trajectory similarity at 100% success, and
is closed-loop on the true block pose so it handles perturbed states, which
is what DAgger relabelling in Stage 3 actually needs. It is a legitimate
Stage 3 teacher and costs nothing further.

---

## Reference numbers

| | value |
|---|---|
| oracle, trajectory score | 0.986 |
| oracle, episode length | 272 steps |
| oracle, still-placed success | 100% |
| teacher v3, still-placed success | 98.8% |
| teacher v3, episode length | 73 steps |
| demonstrations, episode length | 295 steps (227 to 367) |
