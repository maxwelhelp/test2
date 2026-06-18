#!/usr/bin/env bash
set -euo pipefail

cd "$(git rev-parse --show-toplevel 2>/dev/null || pwd)"

python simple_butterfly_matrix_v4_tape_lane/tape_lane_transport.py \
  --data-root "${DATA_ROOT:-../architecture_builder/data/speechcommands}" \
  --epochs "${EPOCHS:-5}" \
  --train-limit "${TRAIN_LIMIT:-12000}" \
  --val-limit "${VAL_LIMIT:-2000}" \
  --batch-size "${BATCH_SIZE:-128}" \
  --eval-batch-size "${EVAL_BATCH_SIZE:-256}" \
  --device "${DEVICE:-cuda}" \
  --amp "${AMP:-fp16}" \
  --dim "${DIM:-96}" \
  --evidence-cells "${EVIDENCE_CELLS:-48}" \
  --lanes "${LANES:-4}" \
  --cells-per-lane "${CELLS_PER_LANE:-12}" \
  --tape-steps "${TAPE_STEPS:-12}" \
  --pair-slots "${PAIR_SLOTS:-12}" \
  --lr "${LR:-5e-4}" \
  --workers "${WORKERS:-4}" \
  --pin-memory \
  --log-every "${LOG_EVERY:-50}" \
  --no-save-checkpoints \
  --out-dir "${OUT_DIR:-simple_butterfly_matrix_v4_tape_lane/runs/speechcommands_5ep}"
