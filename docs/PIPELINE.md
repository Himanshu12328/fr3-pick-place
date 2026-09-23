<!-- Moved out of README.md. The README keeps the current result, the
     understanding needed to read it, and how to reproduce it; everything
     else lives here. -->

# The pipeline, phase by phase


### Phase 1: Impedance controller

This is a Cartesian impedance controller with nullspace posture regulation:

```
tau = J^T (Kp * pose_error - Kd * ee_velocity)     task space
    + N (Kp_n * posture_error - Kd_n * qvel)       nullspace
    + qfrc_bias                                    gravity + Coriolis
    - qfrc_passive                                 model damping
```

Damping is recomputed every step from the task-space inertia, using
`Kd = 2ζ√(Λ·Kp)`. This keeps the damping ratio at ζ no matter how the arm
is configured. Apparent tip inertia varies by more than 10× across the
workspace. That is why fixed damping gains feel right in one pose and mushy
in another.

Adding the 0.8 kg gripper changed the measured bandwidth and overshoot by
less than half a point at every damping ratio. The inertia-scaled damping
absorbed the mass change, which is exactly what it is for.

**Interactive feel test:**

```powershell
python -m src.scripts.impedance_hold
```

Use `WASD` and `QE` to move the target. Use `[` and `]` to change stiffness.
Press `n` to toggle the nullspace term. Push the elbow while watching the
tip. With nullspace on, the elbow moves and the tip stays put.

**Quantitative characterisation:**

```powershell
python -m src.scripts.tune_step
```

| Use | Kp (N/m) | ζ | Bandwidth | Overshoot |
|---|---|---|---|---|
| Transit | 800 | 0.7 | 1.5 Hz | 3.2% |
| Contact | 300 | 1.0 | 0.5 Hz | 0.8% |

![Damping ratio sweep](zeta_sweep_kp800.png)

### Phase 2: Teleoperation and data collection

```powershell
python -m src.scripts.test_pad      # check the controller mapping first
fr3-collect
fr3-collect --bias-near             # sample only x < 0.53, the weak region
```

**Controls.** Left stick moves in x and y. L2 and R2 move down and up. Right
stick pitches and yaws. L1 and R1 roll. Square and Cross close and open the
gripper. D-pad Up starts recording, Right saves, Down discards.

Stick deflections are integrated into an **absolute target pose**. That pose
is what gets logged as the action.

This choice matters. The impedance controller takes target poses as input.
So the same controller works for both data collection and policy inference,
with no changes. If you logged velocities instead, you would need a
different interface at inference time.

The gripper is **binary**, either fully open or fully closed. A rigid cube
has no use for intermediate openings. A continuous target gets averaged
across demonstrations into a gradual closure that clips the block. And
binary matches what VLA action heads are pretrained on.

**Validate before you collect hundreds:**

```powershell
fr3-validate
python -m src.scripts.inspect_episode --batch 4
```

`validate_data.py` reports the speed and maximum per-step jump of every
episode. Watch these across a session.

Over 151 episodes the mean speed drifted from about 74 mm/s to about
103 mm/s as the operator got faster. `maxjump` settled at exactly 7.00 mm,
which means both sticks at full deflection. So the later demonstrations are
almost all bang-bang diagonal moves.

The scaling gain shows this did not hurt. But it is the kind of drift worth
noticing while it happens, not afterwards.

![Dataset trajectories](dataset_trajectories.png)

### Phase 3: Training

Re-render at training resolution first. This is not optional if you care
about throughput:

```powershell
fr3-rerender
fr3-to-lerobot
```

`rerender.py` replays each episode's recorded actions through the simulator
and re-renders every camera at 128 × 160. The state arrays hold all joint
angles, so the scene at any frame can be reconstructed exactly. No data gets
re-collected. Joint drift after replay stayed under 0.005 rad across a
400-step episode, on all 151 episodes.

Here is why this matters. At 640 × 480 the dataloader spent **0.97 s per
step** decoding and resizing PNGs, while the GPU update took only 0.13 s.
The GPU sat idle 88% of the time. After re-rendering, `data_s` drops to
about 0.1 and training runs 7× faster.

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

You need `PYTHONIOENCODING=utf-8` because LeRobot prints policy names
containing characters that crash the Windows cp1252 console.

**ACT's default learning rate of 1e-5 is too low for this data scale.** That
default was tuned for ALOHA's bimanual tasks, with hundreds of episodes and
100k or more steps. On 71 episodes, 1e-5 reached 43 mm fit error at 20k
steps. A rate of 1e-4 reached 26 mm at 5k. Keep the backbone at 1e-5 so the
pretrained ImageNet features survive.

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
  --steps=60000 `
  --save_freq=10000 `
  --output_dir=.\outputs\diffusion_v3 `
  --job_name=diffusion_v3
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

This replays recorded actions from ten episodes. All ten should succeed. If
they do not, the harness has drifted away from the collection environment.
Every success rate it reports would then be wrong in the same invisible way.

**Check the plumbing before you spend time on trials:**

```powershell
fr3-eval --run act_v3 --checkpoint 020000 --type act --check
```

This compares the policy's prediction against a recorded action on a
training frame. Expect a few millimetres of error. If the predictions sit
between -1 and 1, the postprocessor is not being applied and the action is
still normalised.

**Find the best checkpoint:**

```powershell
fr3-sweep --run act_v3 --type act --trials 20
```

**Then re-evaluate that checkpoint on fresh seeds.** The sweep peak is
optimistic by 5 to 7 points, because the checkpoint was chosen on the same
placements it was scored on. Six seeds of 100 trials is what the headline
numbers above are built from:

```powershell
python -m src.scripts.region_analysis --run act_v3 --checkpoint 020000 --type act --trials 100 --seed 41
```

`region_analysis` also splits the result by where the block started. That is
how the near-workspace weakness was found.

**Diagnose a failing policy:**

```powershell
python -m src.scripts.replay_predict --run act_v3 --checkpoint 020000 --type act
```

This feeds a recorded episode's observations through the policy. It plots
the predicted action trajectory against the recorded one, with per-phase
error and gripper timing.

It is teacher-forced, so it measures fit rather than closed-loop behaviour.
But it separates two very different problems: the policy never learned the
task, versus the policy learned it and drifts during rollout. A success rate
alone cannot tell those apart.

![Predicted vs recorded actions](replay_act_v2_035000_episode_0000.png)

---


## Data format


Episodes get written as raw directories first. They are converted to
LeRobotDataset later, as a separate offline step.

That separation is deliberate. LeRobot's dataset API changes between
versions. A collection session should never be lost to a library upgrade.

```
observation.state    (16,)  7 joint pos, 7 joint vel, 2 finger pos
observation.images   three RGB streams
action               (8,)   target position 3, target quaternion 4, gripper 1
```

Design choices that make the data reusable across architectures:

* **Absolute target poses** instead of velocities, so one controller serves
  both collection and inference
* **Binary gripper**, which is what VLA action heads expect
* **A task string per episode**, needed for language-conditioned policies
* **Workspace bounds in the metadata**, so normalisation is reproducible
* **No block pose in the observation.** It is available in simulation but
  not on hardware, and a policy that reads it learns to skip perception

---

