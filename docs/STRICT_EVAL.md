# The strict pick-and-place criterion

What counts as a success in this project, and why the previous answer was
wrong.

## The short version

`check_success` asks whether the block ended up near the target and roughly
at table height. It does not ask whether the robot picked it up, whether it
let go, whether it let go gently, or whether it did any of it at a speed a
person would recognise. Every ACT success rate in this repository before
this file existed, 58.7% through 86.8%, used that criterion.

`src/eval/strict.py` replaces it with nine gates. A trial passes only if all
nine hold.

## The gates

| gate | requirement |
|---|---|
| `delivered` | block within 5 cm of the target and resting, checked **after a 60-step settle** |
| `lifted` | block reached at least 40 mm above its resting height |
| `carried` | some frame had the fingers commanded shut, the block airborne, and the block within 60 mm of the tool |
| `released_low` | at the frame the fingers opened: tool below z = 0.45, block within 5 mm of resting, block moving under 50 mm/s |
| `lifted_off` | tool rose at least 30 mm before it translated more than 20 mm sideways |
| `home` | tool ended within 60 mm of the home xy, having reached z ≥ 0.50 after release |
| `undisturbed` | block moved no more than 5 mm between release and the end of the settle |
| `in_time` | episode length between 195 and 452 steps |
| `demo_like` | trajectory score ≥ 0.85 against the measured demonstration profile |

Every one of them exists because a policy failed it. The provenance is in
`docs/RL_PROCESS.md`: five reward exploits, four of them found in policies
already reporting 99% or better.

## How a trial is run

Two differences from `rollout.run_trial`, both necessary for the gates to be
measurable at all.

**The episode does not stop when the block arrives.** It ends when the whole
sequence is complete: block delivered and resting, fingers open, tool
climbed and back within 60 mm of home. The old harness broke out of the loop
the instant `check_success` fired, which made episode length meaningless and
meant the entire retreat, a third of what a demonstration contains, was
never simulated and never scored.

**The settle holds the last commanded gripper** rather than forcing the
fingers open. A policy that ends still gripping the block keeps gripping it
and fails honestly, instead of having the harness let go on its behalf.

## Where the numbers come from

Nothing here is invented. `src/rl/reference.py` holds the demonstration
profile as constants measured from all 221 recorded episodes, and the two
loosest gates are calibrated so that **every one of those 221 episodes
passes**:

| | measured over 221 demonstrations |
|---|---|
| episode length | min 195, max 452 |
| trajectory score against own profile | min 0.856, 100% at or above 0.85 |

A gate tighter than that would be rejecting behaviour a human actually
performed. An earlier version of this file used the p10 to p90 length band and
did exactly that; see `docs/PATH_TO_97.md` section S2.

## Validation

A criterion that has not been scored on a known-good policy is aimed at
nothing. The same argument `test_harness.py` and `test_rl_env.py` make.

| policy | loose | **strict** |
|---|---|---|
| scripted oracle, jitter 0 | 100.0% | **100.0%** |
| scripted oracle, jitter 1.0 | 100.0% | **100.0%** |
| oracle driven at 3× demonstration speed | 100.0% | **0.0%** |

The negative control is the point. Identical waypoints, identical
controller, only the speed changed: it satisfies the old criterion on every
single trial and fails the new one on every single trial.

## Running it

```powershell
python -m src.scripts.eval_strict --oracle --trials 20
python -m src.scripts.eval_strict --checkpoint outputs/<run>/checkpoints/<step>/pretrained_model --trials 100 --seeds 71 72 73 74 75 76
python -m src.scripts.strict_sweep --run <run> --trials 40 --seed 61
```

Render resolution is read out of the checkpoint's own config, not from
`config.py`, so a policy trained at one resolution cannot be silently
evaluated at another.

## Seed protocol

Selection and reporting seeds are disjoint, because the 86.8% result was
contaminated when seed 51 drove the checkpoint choice and was then included
in the headline.

| purpose | seeds |
|---|---|
| screening and checkpoint selection | 61, 62, 63 |
| reporting, never used for selection | 71, 72, 73, 74, 75, 76 |

Seeds 51 to 56 are retired from reporting: they were used for selection
during earlier stages and cannot be uncontaminated.
