# Runbook: collect, gate, convert, train, evaluate

The order matters. Two of these steps exist because skipping them has
already cost this project a dataset and a training run.

## 1. Collect

```powershell
python -m src.scripts.collect_dagger --driver oracle --episodes 1000 `
    --jitter 1.0 --seed 900 --out oracle_v2
```

`--out` is joined to `DATA_ROOT`, so pass `oracle_v2` and not
`data/oracle_v2` or the episodes land in `data/data/oracle_v2`.

`--driver oracle` labels with the **phase machine**, which is rate limited
to the measured per-phase demonstration speeds. `--driver student` labels
with the reactive `label()`, which is what DAgger needs. Using the reactive
labeller for plain collection is what flattened the speed profile of the
first 800-episode oracle dataset; see `docs/PATH_TO_97.md` section S6.

`--jitter 1.0` costs about 2.5% of the oracle's own success rate and is
worth it. A deterministic demonstrator produces the same episode every time
for a given block pose, so the policy fits the demonstrator rather than the
task.

Roughly 24 episodes per minute. **Do not run evaluations alongside it** —
they are CPU bound on the same cores and drop collection to a fifth of that.

## 2. Gate, before converting anything

```powershell
python -m src.scripts.dataset_gate --data data/oracle_v2
```

Non-negotiable. The first DAgger dataset had the gripper open in 96% of
frames against the demonstrations' 55.7%, and 29.4% of its frames came from
episodes where the demonstrator itself failed. It converted cleanly, trained
for eighty minutes and produced an 11% policy. Every check here is one line
of arithmetic that would have caught it.

The three that matter:

| check | demonstrations | a dataset that failed |
|---|---|---|
| gripper open fraction | 0.557 | 0.960 |
| share of frames from failed episodes | 0.000 | 0.294 |
| trajectory score, 5th percentile | 0.974 | 0.628 |

## 3. Convert

```powershell
python -m src.scripts.to_lerobot --raw data/oracle_v2 `
    --repo-id local/fr3_oracle_v2 --exclude-failed

python -m src.scripts.to_lerobot --raw data/oracle_v2 `
    --repo-id local/fr3_oracle_v2_aux --exclude-failed --with-block-pose
```

`--exclude-failed` drops the roughly 2% of episodes the jittered oracle did
not solve. Recorded teleoperation contains only successes because the
operator discarded the rest, and machine collection should match.

`--with-block-pose` appends the true block x, y and yaw to the action vector
as auxiliary regression targets, making it 11 dimensions. It is a training
target and never an input; the observation is unchanged. Requires block
poses in the raw episodes, which the collector now records and
`add_block_pose.py` can recover for older ones.

The two conversions can run concurrently.

## 4. Train

```powershell
$env:PYTHONIOENCODING="utf-8"
lerobot-train `
  --dataset.repo_id=local/fr3_oracle_v2 `
  --dataset.root=./data/lerobot/local_fr3_oracle_v2 `
  --dataset.image_transforms.enable=true `
  --policy.type=act --policy.device=cuda --policy.use_amp=true `
  --policy.chunk_size=32 --policy.n_action_steps=32 --policy.n_obs_steps=1 `
  --policy.kl_weight=1.0 `
  --policy.optimizer_lr=1e-4 --policy.optimizer_lr_backbone=1e-5 `
  --policy.push_to_hub=false `
  --batch_size=64 --steps=30000 --save_freq=2500 `
  --output_dir=./outputs/act_oracle_v2 --job_name=act_oracle_v2
```

Keep the backbone at 1e-5 so the pretrained ImageNet features survive. ACT's
default 1e-5 for the whole model is too low at this data scale; it reached
43 mm fit error at 20k steps where 1e-4 reached 26 mm at 5k.

## 5. Select a checkpoint, on a screening seed

```powershell
python -m src.scripts.strict_sweep --run act_oracle_v2 --trials 40 --seed 61
```

Never select on a single checkpoint. Adjacent checkpoints 1,250 steps apart
have differed by 27 points on this task while the loss fell smoothly and
monotonically throughout.

## 6. Report, on seeds never used for selection

```powershell
python -m src.scripts.eval_strict `
  --checkpoint outputs/act_oracle_v2/checkpoints/<step>/pretrained_model `
  --trials 100 --seeds 71 72 73 74 75 76 --quiet
```

| purpose | seeds |
|---|---|
| screening and checkpoint selection | 61, 62, 63 |
| reporting | 71, 72, 73, 74, 75, 76 |

Seeds 51–56 are retired from reporting. They were used for selection in
earlier stages, which is how the 86.8% result ended up contaminated.

## Sanity checks that are cheap and worth repeating

```powershell
python -m src.scripts.eval_strict --oracle --jitter 1.0 --trials 40
python -m src.scripts.perception_probe --data data/oracle_v2 --frames 1 --frac-hi 0.0 --epochs 60
```

The first should stay at or above 97% strict. If it does not, the harness or
the scene changed and no policy number is comparable to an earlier one.
