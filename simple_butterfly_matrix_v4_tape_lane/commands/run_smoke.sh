#!/usr/bin/env bash
set -euo pipefail

cd "$(git rev-parse --show-toplevel 2>/dev/null || pwd)"

python simple_butterfly_matrix_v4_tape_lane/tape_lane_transport.py \
  --synthetic \
  --epochs "${EPOCHS:-1}" \
  --train-limit "${TRAIN_LIMIT:-512}" \
  --val-limit "${VAL_LIMIT:-256}" \
  --batch-size "${BATCH_SIZE:-64}" \
  --eval-batch-size "${EVAL_BATCH_SIZE:-128}" \
  --device "${DEVICE:-cuda}" \
  --amp "${AMP:-fp16}" \
  --dim "${DIM:-96}" \
  --evidence-cells "${EVIDENCE_CELLS:-48}" \
  --lanes "${LANES:-4}" \
  --cells-per-lane "${CELLS_PER_LANE:-12}" \
  --tape-steps "${TAPE_STEPS:-12}" \
  --pair-slots "${PAIR_SLOTS:-12}" \
  --workers "${WORKERS:-2}" \
  --log-every "${LOG_EVERY:-10}" \
  --no-save-checkpoints \
  --out-dir "${OUT_DIR:-simple_butterfly_matrix_v4_tape_lane/runs/smoke}"
