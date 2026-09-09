#!/usr/bin/env bash
# Trains an ACT policy and resumes it if the process dies.
#
# This project has lost training runs to silent kills before: five
# consecutive background runs died with no error, GPU idle and 14 GB of RAM
# free, two of them at exactly the same step. The workaround that worked was
# chaining shorter segments with resume, so that is what this does
# automatically rather than relying on someone noticing.
#
# Usage:
#   src/scripts/train_act.sh <job_name> <repo_id> [steps] [batch]
#
# Example:
#   src/scripts/train_act.sh act_oracle_v2 local/fr3_oracle_v2 30000 64

set -u

JOB="${1:?job name required}"
REPO="${2:?repo id required}"
STEPS="${3:-30000}"
BATCH="${4:-64}"

ROOT="./data/lerobot/$(echo "$REPO" | tr '/' '_')"
OUT="./outputs/${JOB}"
LOG="logs/train_${JOB}.txt"

export PYTHONIOENCODING=utf-8
mkdir -p logs

# How many steps the run has actually completed, read from the checkpoint
# rather than from the log, because the log can be truncated by a kill.
completed_steps() {
  local f="${OUT}/checkpoints/last/training_state/training_step.json"
  if [ -f "$f" ]; then
    python -c "import json;print(json.load(open(r'$f'))['step'])" 2>/dev/null || echo 0
  else
    echo 0
  fi
}

attempt=0
while true; do
  done_steps=$(completed_steps)
  if [ "$done_steps" -ge "$STEPS" ]; then
    echo "=== ${JOB}: reached ${done_steps}/${STEPS} steps, finished ==="
    break
  fi

  attempt=$((attempt + 1))
  if [ "$attempt" -gt 20 ]; then
    echo "=== ${JOB}: giving up after 20 attempts at step ${done_steps} ==="
    exit 1
  fi

  if [ "$done_steps" -gt 0 ]; then
    echo "=== ${JOB}: attempt ${attempt}, resuming from step ${done_steps} ===" | tee -a "$LOG"
    lerobot-train \
      --config_path="${OUT}/checkpoints/last/pretrained_model/train_config.json" \
      --resume=true >> "$LOG" 2>&1
  else
    echo "=== ${JOB}: attempt ${attempt}, starting fresh ===" | tee -a "$LOG"
    lerobot-train \
      --dataset.repo_id="${REPO}" \
      --dataset.root="${ROOT}" \
      --dataset.image_transforms.enable=true \
      --policy.type=act \
      --policy.device=cuda \
      --policy.use_amp=true \
      --policy.chunk_size=32 \
      --policy.n_action_steps=32 \
      --policy.n_obs_steps=1 \
      --policy.kl_weight=1.0 \
      --policy.optimizer_lr=1e-4 \
      --policy.optimizer_lr_backbone=1e-5 \
      --policy.push_to_hub=false \
      --batch_size="${BATCH}" \
      --steps="${STEPS}" \
      --save_freq=2500 \
      --output_dir="${OUT}" \
      --job_name="${JOB}" >> "$LOG" 2>&1
  fi

  after=$(completed_steps)
  echo "=== ${JOB}: process exited at step ${after} ===" | tee -a "$LOG"

  # A run that exits without advancing a single step is not going to be
  # fixed by retrying it. Stop rather than spin.
  if [ "$after" -le "$done_steps" ] && [ "$attempt" -gt 1 ]; then
    echo "=== ${JOB}: no progress since last attempt, stopping ===" | tee -a "$LOG"
    exit 1
  fi
done
