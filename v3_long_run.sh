#!/usr/bin/env bash
# Third major SciLM run: 100 epochs from scratch with the v2 winner config.
# Goal: read whether 40 epochs was at the bottom of the loss curve, or whether
# significantly longer training keeps improving val_total.
#
# *Not* a continuation of v2_winner_full — the cosine LR schedule needs the
# full horizon set at the start to give a clean answer.
#
# Wall time estimate: ~204 s/epoch × 100 = ~5.7 hours on CPU.
# Best val checkpoint is saved as runs/<name>/model_best.pt during training
# (final-epoch weights are still saved as model.pt).
#
# Usage:  ./v3_long_run.sh
#         ./v3_long_run.sh --name v3_long_100_rerun
set -euo pipefail

cd "$(dirname "$0")"

NAME="${NAME:-v3_long_100}"

caffeinate -is python3 train_scilm.py \
  --mode full \
  --name "$NAME" \
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
