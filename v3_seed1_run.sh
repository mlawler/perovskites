#!/usr/bin/env bash
# v3 with a different seed — sanity check on whether the per-formulation
# residual pattern (e.g. LiBOB +0.4 offset) is real architectural behavior
# or a single-seed artifact. Identical to v3_long_run.sh except for --seed.
#
# Side benefit: train.py now saves a checkpoint at every val-improvement
# epoch into runs/<name>/checkpoints/best_epXXX.pt. Use eval_trajectory.py
# afterwards to replay the per-formulation residual evolution.
#
# Wall time estimate: ~5.7 hours, finishes around 4 am if started at ~10:30 pm.
set -euo pipefail

cd "$(dirname "$0")"

NAME="${NAME:-v3_long_100_seed1}"

caffeinate -is python3 train_scilm.py \
  --mode full \
  --name "$NAME" \
  --seed 1 \
  --d-model 128 \
  --n-layers 4 \
  --n-heads 4 \
  --d-ff 512 \
  --lr 1e-3 \
  --dropout 0.3 \
  --lambda-scalar 0.3 \
  --batch-size 32 \
  --weight-decay 0.1 \
  --min-pairs-masked 1 \
  --max-pairs-masked 2 \
  --n-epochs 100 \
  "$@"
