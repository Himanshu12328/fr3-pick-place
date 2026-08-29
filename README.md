# FR3 Pick-and-Place: Simulation to Trained Policy

An end-to-end Physical AI pipeline built on Windows: a Cartesian impedance
controller for a 7-DOF Franka FR3 in MuJoCo, DualSense teleoperation for
demonstration collection, and imitation learning with ACT and Diffusion
Policy — plus an evaluation harness that reports task success rate rather
than loss.

Everything runs natively on Windows. No WSL, no dual boot, no cloud.

---

## Results

Block spawned uniformly in an 18 × 18 cm region; success is the block
resting within 5 cm of the target. Every figure below is **three seeds of
100 trials each**, on placements not used to select the checkpoint.

### Data scaling

| Dataset | Policy | Best checkpoint | Success rate | Per-seed |
|---|---|---|---|---|
| 151 episodes | ACT | 20,000 steps | **79.0% ± 5.6%** | 78 / 74 / 85 |
| 71 episodes | ACT | 5,000 steps | 58.7% ± 8.1% | 66 / 60 / 50 |
| 71 episodes | Diffusion Policy | 40,000 steps | 62.7% ± 3.8% | 67 / 60 / 61 |

**Doubling the dataset moved ACT by 20 points.** 71 → 151 demonstrations
took it from 58.7% to 79.0%, a gap well clear of the ~10-point standard
error on the difference. It also moved the convergence point: on 71
episodes ACT peaked at 5,000 steps and stayed flat through 60,000; on 151
it needs 20,000. More data raised both the ceiling and the compute needed
to reach it.

This is the most actionable finding in the project. Before collecting the
extra 80 episodes, failures were distributed roughly uniformly across the
workspace, which suggested capacity rather than coverage was the limit —
and the scaling result confirms it.

![ACT success vs training step, 151 episodes](docs/sweep_act_v3.png)

### ACT versus Diffusion Policy

Measured on the 71-episode dataset, where both were trained:

| Policy | Success rate |
|---|---|
| Diffusion Policy | 62.7% ± 3.8% |
| ACT | 58.7% ± 8.1% |

The 4-point gap sits inside the standard error of the difference (±4%), so
**these two policies are not distinguishable on this task**. Separating 63%
from 59% with confidence would need roughly 1,400 trials each.

What does separate them:

**ACT is 9× cheaper at inference.** 2.8 s versus 26 s per evaluation
trial: one transformer forward pass per action chunk, against 100 DDPM
denoising steps every 8 actions. On real hardware at 30 Hz that gap
decides whether the policy runs at all.

**ACT imitates more precisely.** Teacher-forced mean position error 4.2 mm
versus 6.1 mm, grasp-phase error 1.5–3.0 mm versus 5.3–6.4 mm, gripper
timing within one frame with no spurious toggles. Diffusion's predictions
visibly oscillate and it opens and closes the gripper several times near
release where the demonstration opens once.

**ACT's seed variance is wider.** 8.1 points against 3.8, and 5.6 points
on the larger dataset.

![Diffusion success vs training step](docs/sweep_diffusion_v2.png)

### Characterised limitations

**Near-workspace weakness.** Splitting the 151-episode ACT results by where
the block started, pooled over three seeds:

| Region | Success | n |
|---|---|---|
| x < 0.53 m | ~68% | 93 |
| x ≥ 0.53 m | ~84% | 207 |
| y < −0.04 m | ~79% | 141 |
| y ≥ −0.04 m | ~79% | 159 |

The 16-point x gap is consistent in direction across all three seeds and
clears the noise floor. The y split is flat, as it was on the smaller
dataset. Note the near region also has fewer demonstrations — the block
distribution is centred at x = 0.55 while the split is at 0.53 — so this
is as likely a coverage gap as a control one, and it says where the next
demonstrations should go.

**Training success is not monotonic.** The 151-episode ACT run scored
80% at 5k, **20% at 10k**, 65% at 15k, then 80–85% from 20k on. The 10k
checkpoint's teacher-forced fit error was 16–18 mm against a few
millimetres either side, so the model genuinely passed through a bad
phase rather than merely rolling out badly. A single-checkpoint evaluation
at 10k would have reported this pipeline as broken.

**Selection bias, measured twice.** Sweep peaks over-report by 5–7 points
consistently: 65% → 58.7% and 70% → 62.7% on the 71-episode runs, 85% →
79.0% on the 151-episode run. The sweep is the right way to *find* a good
checkpoint and the wrong way to *report* how good it is. Every headline
figure here comes from re-evaluating the chosen checkpoint on fresh seeds.

**Fit quality does not predict task success.** ACT's teacher-forced error
fell from 26 mm to 6.8 mm between step 5k and 55k on the 71-episode
dataset while its success rate did not move at all. Training loss is an
even weaker signal: diffusion's declined monotonically while success
oscillated between 10% and 70%. Only rollout evaluation locates a peak,
which is why the harness exists and why it is validated against recorded
demonstrations before any policy is trusted.

---

## Requirements

- Windows 10/11
- NVIDIA GPU. Built and tested on an RTX 5080 (Blackwell, sm_120)
- Python 3.11
- A DualSense controller, for collecting new demonstrations
- ~80 GB free disk for a 150-episode dataset at 640 × 480

---

## Setup

### 1. Python

Install Python 3.11 from python.org — not the Microsoft Store build, which
sandboxes paths and breaks venv activation. Check "Add python.exe to PATH".

### 2. Clone and create the environment

```powershell
git clone https://github.com/<your-username>/fr3-pick-place.git
cd fr3-pick-place
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
```

If activation is blocked:

```powershell
Set-ExecutionPolicy -Scope CurrentUser -ExecutionPolicy RemoteSigned
```

### 3. PyTorch first, alone

Install PyTorch before anything else, so no other package pulls in a
wrong-CUDA build as a dependency.

```powershell
pip install --upgrade pip
pip install "torch==2.10.0" "torchvision==0.25.0" --index-url https://download.pytorch.org/whl/cu128
```

**Blackwell GPUs require the cu128 index.** Standard PyPI wheels contain no
sm_120 kernels. Verify both of these before continuing:

```powershell
python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_capability())"
python -c "import torch; a=torch.randn(4096,4096,device='cuda'); print((a@a).sum().item())"
```

You need `(12, 0)` from the first and an actual number from the second.
`cuda.is_available()` returning True does not prove the kernels exist — the
matmul does.

### 4. Install the project

```powershell
pip install -e ".[train]"
```

Confirm the stack imports together:

```powershell
python -c "import torch, numpy, mujoco, lerobot; print(torch.__version__, numpy.__version__, mujoco.__version__, lerobot.__version__)"
```

### 5. Robot model

```powershell
git clone https://github.com/google-deepmind/mujoco_menagerie
```

Or set `MENAGERIE_DIR` if you keep it elsewhere.

### 6. Enable Developer Mode

Settings → System → For developers → Developer Mode **on**. LeRobot writes a
symlink when saving checkpoints, and Windows refuses symlinks without it.
Training will otherwise crash at the first checkpoint, hours in.

### 7. Route Python to the discrete GPU

Settings → System → Display → Graphics → Add desktop app → browse to
`.venv\Scripts\python.exe` → Options → **High performance**.

This is not optional and it is not obvious. Without it, MuJoCo's offscreen
renderer falls back to Microsoft's software rasterizer: **48 ms per frame
instead of 0.8 ms**, a 60× penalty. The interactive viewer works fine either
way, because windowed OpenGL follows a different routing path — so "the
viewer is smooth" is not evidence that headless rendering is on the GPU.
Measure it, don't assume it.

### 8. Build the scene

```powershell
fr3-build-scene
python -m src.scripts.preview_cameras
```

Generates `models/fr3_pick_place.xml` from Menagerie's bare FR3 by adding a
parallel-jaw gripper, a table, a graspable block, and three cameras. The
verify output should report `nq=16 nv=15 nu=7`.

---

## Pipeline

### Phase 1 — Impedance controller

Cartesian impedance control with dynamically-consistent nullspace posture
regulation:

```
tau = J^T (Kp * pose_error - Kd * ee_velocity)     task space
    + N (Kp_n * posture_error - Kd_n * qvel)       nullspace
    + qfrc_bias                                    gravity + Coriolis
    - qfrc_passive                                 model damping
```

Damping is recomputed each step from the task-space inertia
(`Kd = 2ζ√(Λ·Kp)`), so the damping ratio stays at ζ regardless of arm
configuration. Apparent tip inertia varies by more than 10× across the
workspace, which is why fixed damping gains feel right in one pose and
mushy in another. Adding the 0.8 kg gripper changed the measured bandwidth
and overshoot by under half a point at every damping ratio — the
inertia-scaled damping absorbed the mass change, which is what it is for.

**Interactive feel test:**

```powershell
python -m src.scripts.impedance_hold
```

`WASD`/`QE` move the target, `[`/`]` change stiffness, `n` toggles the
nullspace term. Push the elbow while watching the tip: with nullspace on,
the elbow reconfigures and the tip stays put.

**Quantitative characterisation:**

```powershell
python -m src.scripts.tune_step
```

Step response per axis, frequency sweep, stiffness sweep, damping-ratio
sweep, and a joint-friction comparison. Plots to `logs/`.

| Use | Kp (N/m) | ζ | Bandwidth | Overshoot |
|---|---|---|---|---|
| Transit | 800 | 0.7 | 1.5 Hz | 3.2% |
| Contact | 300 | 1.0 | 0.5 Hz | 0.8% |

![Damping ratio sweep](docs/zeta_sweep_kp800.png)

### Phase 2 — Teleoperation and data collection

```powershell
python -m src.scripts.test_pad      # verify controller mapping first
fr3-collect
```

**Controls:** left stick moves in x/y, L2/R2 move down/up, right stick
pitches and yaws, L1/R1 roll, Square/Cross close and open the gripper.
D-pad Up starts recording, Right saves, Down discards.

Stick deflections are integrated into an **absolute target pose**, and that
pose is what gets logged as the action. This matters: the impedance
controller consumes target poses, so the same controller serves both data
collection and policy inference without modification. Logging velocities
would require a different interface at inference time.

The gripper is **binary**. A rigid cube has no use for intermediate
openings, a continuous target gets averaged across demonstrations into a
gradual closure that clips the block, and binary matches what VLA action
heads are pretrained on.

**Validate before collecting hundreds:**

```powershell
fr3-validate
python -m src.scripts.inspect_episode --batch 4
```

`validate_data.py` reports per-episode speed and maximum per-step jump.
Watch these across a session: over 151 episodes the mean speed drifted from
~74 mm/s to ~103 mm/s as the operator got faster, and `maxjump` settled at
exactly 7.00 mm — both sticks at full deflection — meaning the later
demonstrations are almost entirely bang-bang diagonal moves. The 20-point
scaling gain shows this did not hurt, but it is the kind of drift worth
noticing while it is happening rather than afterwards.

![Dataset trajectories](docs/dataset_trajectories.png)

### Phase 3 — Training

Re-render at training resolution first. This is not optional for
throughput:

```powershell
fr3-rerender
fr3-to-lerobot
```

`rerender.py` replays each episode's recorded actions through the simulator
and re-renders every camera at 128 × 160. Because the state arrays hold all
joint angles, the scene at any frame is exactly reconstructible — no data
is re-collected. Joint drift after replay stays under 0.005 rad across a
400-step episode, on all 151.

The reason this matters: at 640 × 480 the dataloader spent **0.97 s per
step** decoding and resizing PNGs while the GPU update took 0.13 s. The GPU
idled 88% of the time. After re-rendering, `data_s` drops to ~0.1 and
training runs 7× faster.

**Train ACT:**

```powershell
$env:PYTHONIOENCODING="utf-8"

lerobot-train `
  --dataset.repo_id=local/fr3_pick_place_small `
  --dataset.root=.\data\lerobot\local_fr3_pick_place_small `
  --dataset.image_transforms.enable=true `
  --policy.type=act `
  --policy.device=cuda `
  --policy.use_amp=true `
  --policy.chunk_size=32 `
  --policy.n_action_steps=32 `
  --policy.n_obs_steps=1 `
  --policy.kl_weight=1.0 `
  --policy.optimizer_lr=1e-4 `
  --policy.optimizer_lr_backbone=1e-5 `
  --policy.push_to_hub=false `
  --batch_size=64 `
  --steps=30000 `
  --save_freq=5000 `
  --output_dir=.\outputs\act_v3 `
  --job_name=act_v3
```

`PYTHONIOENCODING=utf-8` is needed because LeRobot's help text and policy
names contain non-cp1252 characters that crash the Windows console.

**ACT's default learning rate of 1e-5 is too low at this data scale.** It
was tuned for ALOHA's bimanual tasks with hundreds of episodes and 100k+
steps. On 71 episodes, 1e-5 reached 43 mm fit error at 20k steps; 1e-4
reached 26 mm at 5k. Keep the backbone at 1e-5 to preserve the pretrained
ImageNet features.

**Train Diffusion Policy:**

```powershell
lerobot-train `
  --dataset.repo_id=local/fr3_pick_place_small `
  --dataset.root=.\data\lerobot\local_fr3_pick_place_small `
  --dataset.image_transforms.enable=true `
  --policy.type=diffusion `
  --policy.device=cuda `
  --policy.use_amp=true `
  --policy.crop_shape="[115,144]" `
  --policy.horizon=16 `
  --policy.n_action_steps=8 `
  --policy.n_obs_steps=2 `
  --policy.down_dims="[256,512,1024]" `
  --policy.optimizer_lr=1e-4 `
  --policy.push_to_hub=false `
  --batch_size=64 `
  --steps=50000 `
  --save_freq=10000 `
  --output_dir=.\outputs\diffusion_v2 `
  --job_name=diffusion_v2
```

---

## Evaluation

The harness runs the policy in the same simulator, with the same impedance
gains, the same block distribution, and the same success criterion used to
label the demonstrations. That shared definition is what makes the number
comparable to anything.

**Validate the harness itself first:**

```powershell
python -m src.scripts.test_harness
```

Replays recorded actions from ten episodes. All ten should succeed. If they
do not, the harness has diverged from the collection environment and every
success rate it reports would be wrong in the same invisible way.

**Check the plumbing before spending time on trials:**

```powershell
fr3-eval --run act_v3 --checkpoint 020000 --type act --check
```

Compares the policy's prediction against a recorded action on a training
frame. Expect a few millimetres. Predictions confined to [-1, 1] mean the
postprocessor is not being applied and the action is still normalised.

**Find the best checkpoint:**

```powershell
fr3-sweep --run act_v3 --type act --trials 20
```

**Then re-evaluate that checkpoint on fresh seeds** — the sweep peak is
optimistic by 5–7 points because the checkpoint was chosen on the same
placements it was scored on:

```powershell
python -m src.scripts.region_analysis --run act_v3 --checkpoint 020000 --type act --trials 100 --seed 41
python -m src.scripts.region_analysis --run act_v3 --checkpoint 020000 --type act --trials 100 --seed 42
python -m src.scripts.region_analysis --run act_v3 --checkpoint 020000 --type act --trials 100 --seed 43
```

`region_analysis` also splits the result by where the block started, which
is how the near-workspace weakness above was found.

**Diagnose a failing policy:**

```powershell
python -m src.scripts.replay_predict --run act_v3 --checkpoint 020000 --type act
```

Feeds a recorded episode's observations through the policy and plots the
predicted action trajectory against the recorded one, with per-phase error
and gripper timing. Teacher-forced, so it measures fit rather than
closed-loop behaviour — but it separates "the policy did not learn the
task" from "the policy learned it and drifts in rollout", which a success
rate alone cannot.

![Predicted vs recorded actions](docs/replay_act_v2_035000_episode_0000.png)

---

## Repository layout

```
src/
  config.py                    Paths and shared constants
  controllers/impedance.py     Cartesian impedance controller
  teleop/dualsense.py          Pad reader and target-pose integrator
  data/recorder.py             Episode buffering and disk format
  data/task.py                 Block randomisation and success criterion
  eval/rollout.py              Rollout harness
  eval/policy_wrapper.py       LeRobot policy + processor adapter
  scripts/                     Entry points, one concern each
tests/                         Invariants whose violation is silent
docs/                          Result plots referenced by this README
```

---

## Data format

Episodes are written as raw directories first, then converted to
LeRobotDataset offline. Conversion is deliberately separate: LeRobot's
dataset API changes between versions, and a collection session should never
be lost to a library upgrade.

```
observation.state    (16,)  7 joint pos, 7 joint vel, 2 finger pos
observation.images   three RGB streams
action               (8,)   target position 3, target quaternion 4, gripper 1
```

Design choices made for cross-architecture reuse:

- **Absolute target poses**, not velocities, so one controller serves
  collection and inference
- **Binary gripper**, matching what VLA action heads expect
- **Task string per episode**, required for language-conditioned policies
- **Workspace bounds in metadata**, so normalisation is reproducible
- **Block pose excluded from observations** — available in simulation but
  not on hardware, and a policy that reads it learns to skip perception

---

## Gotchas

Each of these cost real debugging time. All produced plausible-looking
wrong results rather than errors.

**Evaluation resolution must match training resolution.** The vision
backbone is fully convolutional and accepts any input size silently, but
`crop_shape` is applied in pixels. Rendering at 640 × 480 for a policy
trained at 128 × 160 turns a whole-scene 90% crop into a 115 × 144 patch of
dead centre — the policy goes effectively blind and runs on proprioception
alone. This dropped measured success from 60% to 0.4%. The tell was
identical rollout distances across different checkpoints and different
architectures.

**MuJoCo's API moved.** `data.qM` was renamed to `data.M`, still holds the
sparse packed lower triangle rather than the dense matrix, and `mj_fullM`'s
signature changed from `(model, dst, src)` to `(model, data, dst)`.
`mass_matrix()` in `impedance.py` handles all three variants.

**`qfrc_bias` does not include passive joint forces.** It holds only
`C(q,q̇)q̇ + g(q)`. MuJoCo's joint damping and springs live in
`qfrc_passive` and are added independently, so leaving them uncompensated
means the model's damping stacks on the controller's and the achieved
damping ratio is not the requested ζ.

**Dry friction cannot be cancelled feedforward.** Menagerie's FR3 sets
~6 Nm of total `frictionloss`, producing a constant ~6 N opposing force and
22 mm of steady-state error at Kp=300. It is resolved by the constraint
solver, not readable as a force, so it is zeroed for controller
characterisation. On hardware it is real and sets a minimum usable
stiffness: `Kp × acceptable_error > 6 N`.

**DualSense mappings differ from the SDL documentation.** On this pad L1/R1
are buttons 9 and 10, not 4 and 5 — indices 4 and 5 are Share and the PS
button. Yaw also needs its sign flipped for front-of-arm operation, despite
the geometric argument that rotation should read the same from either
viewpoint. Verify both empirically with `test_pad.py` and `teleop_test.py`.

**Windows console encoding.** LeRobot prints policy names containing `π`,
which crashes cp1252. Set `PYTHONIOENCODING=utf-8`.

---

## Roadmap

- [x] Cartesian impedance controller, characterised on the full scene
- [x] DualSense teleoperation and demonstration collection
- [x] Evaluation harness, validated against recorded demonstrations
- [x] ACT and Diffusion Policy on 71 episodes — 58.7% and 62.7%,
      indistinguishable
- [x] Multi-seed evaluation with selection bias accounted for
- [x] Data scaling — 151 episodes takes ACT to **79.0% ± 5.6%**
- [ ] Diffusion Policy on the 151-episode dataset, to check whether the
      scaling gain is architecture-independent
- [ ] More demonstrations at x < 0.53 m, where the current policy is ~16
      points weaker and the block distribution is thinner
- [ ] Multimodal task variant, comparing ACT and diffusion on demonstration
      multimodality — the axis these architectures were designed to differ
      on, which the current single-mode task does not test
- [ ] π0 fine-tuning
- [ ] RL with Isaac Lab parallel environments
- [ ] Sim-to-real transfer

---

## Acknowledgements

Robot model from [MuJoCo Menagerie](https://github.com/google-deepmind/mujoco_menagerie).
Training and dataset infrastructure from [LeRobot](https://github.com/huggingface/lerobot).

## License

MIT