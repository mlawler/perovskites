#!/usr/bin/env bash
# Second major SciLM run: 40-epoch full training with the winning config from
# the v2_overnight sweep (trial 012_small_lr1e-03, best_val_total=0.313).
#
# Usage:  ./v2_full_run.sh
#         ./v2_full_run.sh --name v2_winner_full_rerun       # override name
#         ./v2_full_run.sh --device cpu                      # override device
#
# Any flag passed here is forwarded to train_scilm.py and overrides the
# defaults below (argparse uses the last value for repeated flags).
set -euo pipefail

cd "$(dirname "$0")"

NAME="${NAME:-v2_winner_full}"

# caffeinate -is keeps the Mac awake (and disk spun up) for the duration of
# the run; matches what we used for v2_overnight.
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
  --n-epochs 40 \
  "$@"
