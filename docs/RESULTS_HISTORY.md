<!-- Moved out of README.md. The README keeps the current result, the
     understanding needed to read it, and how to reproduce it; everything
     else lives here. -->

# Earlier results, all on the loose criterion

These numbers predate Stage 3c, which replaced the success criterion.
They are **not comparable** to any strict number: `check_success` is
satisfied by a shove, a drop and a hover. They are kept because the
scaling behaviour and the architecture comparison they establish are
still the evidence behind several later decisions.

### Earlier results, all on the loose criterion

Everything from here to the end of this section predates the strict
criterion and is kept for the record. Read every number in it as "put the
block near the target by any means", not as "performed a pick and place".
The one policy from this era that was re-measured, `act_oracle_v1`, lost
20 points when scored properly.

#### 151 demonstrations

| Policy | Best checkpoint | Success rate | Per-seed |
|---|---|---|---|
| ACT | 20,000 steps | **77.2% ± 5.0%** | 78 / 74 / 85 / 70 / 77 / 79 |
| Diffusion Policy | 20,000 steps | **71.3% ± 5.9%** | 80 / 63 / 69 / 68 / 74 / 74 |

Pooled, that is 463 out of 600 for ACT and 428 out of 600 for diffusion.
The 5.9 point gap has a standard error of 2.5 points. At this sample size
ACT is ahead by a meaningful margin. This is the first time the two
architectures separate in this project.

Both policies saw identical block placements on each seed, so the comparison
is paired.

This result applies to this task, this dataset and these checkpoints. It is
not a general claim about the two architectures.

#### 221 demonstrations, near-region bias

The 151-episode results below showed both policies were 14 to 21 points
weaker when the block started near the base (`x < 0.53`). Two explanations
were possible: the region was under-sampled, or it was genuinely harder to
control. 70 additional episodes were collected with `--bias-near`,
restricting placements to `x < 0.53`, and mixed with the original 151 for a
221-episode dataset. Evaluation still samples the full range uniformly, so
this changes only what the policy trains on, not what it is scored on.

| Policy | Best checkpoint | Success rate | Per-seed |
|---|---|---|---|
| ACT | 20,000 steps | **83.5% ± 4.2%** | 80 / 80 / 90 / 80 / 86 / 85 |
| Diffusion Policy | 50,000 steps | **83.2% ± 4.4%** | 83 / 81 / 78 / 89 / 80 / 88 |

**The near-region gap closed for both policies — reversed for ACT, flattened for diffusion.**

| Region | ACT, 151 eps | ACT, 221 eps | Diffusion, 151 eps | Diffusion, 221 eps |
|---|---|---|---|---|
| x < 0.53 m (near) | 67.9% (n=209) | **87.8%** (n=238) | 57.9% (n=209) | **82.4%** (n=238) |
| x ≥ 0.53 m (far) | 82.1% (n=391) | 80.7% (n=362) | 78.5% (n=391) | 83.7% (n=362) |
| gap | -14.2 | **+7.1** | -20.6 | **-1.3** |

Adding 70 episodes to only the near region moved only the near region — the
far side held steady within noise for both policies. That is close to a
controlled experiment for "was this a coverage problem," and the answer is
yes, not a control-geometry limit.

Diffusion's near-region result carries a second finding: it was not just
weak before, it was wildly unstable there. Across the six 151-episode seeds
its near-region rate ranged 41.9 to 77.8 percent, a 36-point spread on the
same checkpoint. At 221 episodes that range is 78.6 to 87.8 percent, a
9.2-point spread. The extra data fixed the mean and the variance.

This also recalibrates the sweep-overestimate finding for both policies: the
checkpoint-selection sweeps (n=20) picked 90% for ACT at step 20,000 and 90%
for diffusion at step 50,000; the six-seed numbers are 83.5% and 83.2%, gaps
of 6.5 and 6.8 points, both consistent with the 5-to-7-point pattern already
established below.

![ACT success vs training step, 221 episodes](sweep_act_v4.png)
![Diffusion success vs training step, 221 episodes](sweep_diffusion_v4.png)

#### Data scaling

| Dataset | ACT | Diffusion Policy |
|---|---|---|
| 71 episodes | 58.7% ± 8.1% | 62.7% ± 3.8% |
| 151 episodes | 77.2% ± 5.0% | 71.3% ± 5.9% |
| 221 episodes | **83.5% ± 4.2%** | **83.2% ± 4.4%** |

Doubling the dataset from 71 to 151 moved both policies. ACT gained 18.5
points and diffusion gained 8.6. Going from 151 to 221 — mostly a targeted
addition to one region rather than general coverage — moved ACT a further
6.3 points and diffusion 11.9. Diffusion gained more from the same 70
episodes, consistent with its pre-intervention near-region deficit being
larger to begin with (20.6 points against ACT's 14.2).

It also changed when they converge. On 71 episodes ACT peaked at 5,000 steps
and diffusion at 40,000. On 151 episodes both peak at 20,000.

The ordering flipped too. At 71 episodes diffusion led by 4 points, which
was well inside the noise. At 151 episodes ACT leads by 5.9 points with six
seeds behind it. The earlier ordering was never real. Three seeds establish
very little.

![ACT success vs training step, 151 episodes](sweep_act_v3.png)

The diffusion counterpart of that plot was referenced by the README for
months as `sweep_diffusion_v3.png` and has never been in the repository —
only the v2 and v4 sweeps were committed. The reference is removed rather
than pointed at a different run's plot, which would mislabel it. The
numbers it showed are in the table above.

#### Other differences between the two

**ACT is 9× cheaper to run.** It takes 2.8 s per evaluation trial against
26 s for diffusion. ACT does one transformer forward pass per action chunk.
Diffusion does 100 DDPM denoising steps every 8 actions. On real hardware at
30 Hz that difference decides whether the policy runs at all.

**ACT imitates more precisely.** Teacher-forced mean position error is
4.2 mm against 6.1 mm. Grasp-phase error is 1.5 to 3.0 mm against 5.3 to
6.4 mm. ACT's gripper timing is within one frame with no spurious toggles.
Diffusion's predictions oscillate, and it opens and closes the gripper
several times near release where the demonstration opens once.

#### Known limitations, as they stood then

**Both policies are weaker near the base.** Splitting all 600 trials per
policy by where the block started:

| Region | ACT | Diffusion | n |
|---|---|---|---|
| x < 0.53 m | 67.9% | 57.9% | 209 |
| x ≥ 0.53 m | 82.1% | 78.5% | 391 |
| **gap** | **14.2** | **20.6** | |

Both policies do worse when the block sits closer to the base. Diffusion is
worse than ACT there. The gap survives six seeds for each policy, so it is
not sampling noise.

Diffusion's near-region performance is also very unstable. Across the six
seeds it scored 41.9, 44.4, 50.0, 69.0, 71.0 and 77.8 percent. That is a
36 point range on the same checkpoint. A policy that swings that far
depending on which placements it draws is not one to deploy.

There were two possible causes, needing different fixes. The near region
may simply have been under-represented — the block distribution is centred
at x = 0.55 while the split sits at 0.53, so only about 35% of placements
fell below it. Or the near region may be harder to control, since the arm
folds in closer to the base there.

**Resolved: it was coverage, not control geometry, for both policies.** See
[221 demonstrations, near-region bias](#221-demonstrations-near-region-bias)
above — 70 episodes collected with `--bias-near` took ACT's near-region
success from 67.9% to 87.8% and diffusion's from 57.9% to 82.4%, while the
untouched far region held steady for both. Diffusion's near-region result
was also wildly unstable before (36-point spread across seeds) and is not
after (9.2-point spread).

```powershell
fr3-collect --bias-near
```

**The y split is flat.** Both policies score 70 to 80 percent on either side
across all twelve runs. An earlier idea that the operator-side region was
harder came from eyeballing failure logs. Twelve seeds killed it.

**Training success is not monotonic.** This happens in both architectures.
ACT scored 80% at 5k steps, then **20% at 10k**, then 65% at 15k, then 80 to
85% from 20k onward. Diffusion scored 80% at 20k, then **45% at 30k**, then
recovered to 65 to 70%.

The ACT 10k checkpoint had a teacher-forced fit error of 16 to 18 mm, where
the checkpoints either side had a few millimetres. So the model really did
pass through a bad phase. It was not just rolling out badly. A
single-checkpoint evaluation at either of those steps would have reported
the whole pipeline as broken.

**Sweep peaks over-report by 5 to 7 points.** This was measured four times:
65 → 58.7, 70 → 62.7, 85 → 77.2, 80 → 71.3. The sweep is the right way to
find a good checkpoint. It is the wrong way to report how good that
checkpoint is, because the checkpoint gets chosen on the same placements it
is then scored on. Every headline number here comes from re-evaluating the
chosen checkpoint on fresh seeds.

**Fit quality does not predict task success.** On the 71-episode dataset,
ACT's teacher-forced error fell from 26 mm to 6.8 mm between step 5k and
55k. Its success rate did not move at all over that range.

Training loss is an even weaker signal. Diffusion's loss declined smoothly
while its success rate swung between 10% and 80%. Only rollout evaluation
finds the peak. That is why the harness exists, and why it is validated
against recorded demonstrations before any policy is trusted.

---

