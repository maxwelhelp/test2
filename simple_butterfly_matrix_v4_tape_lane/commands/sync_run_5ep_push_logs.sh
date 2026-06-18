#!/usr/bin/env bash
set -euo pipefail

ROOT="$(git rev-parse --show-toplevel)"
cd "$ROOT"

echo "[sync] repo root: $ROOT"
echo "[sync] pulling latest main/current branch"
git pull --ff-only

RUN_ID="${RUN_ID:-$(date +%Y%m%d_%H%M%S)}"
OUT_DIR="${OUT_DIR:-simple_butterfly_matrix_v4_tape_lane/runs/speechcommands_5ep_${RUN_ID}}"
mkdir -p "$OUT_DIR"

echo "[run] output dir: $OUT_DIR"
echo "[run] starting 5 epoch TapeLaneRouter test"

set +e
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
  --out-dir "$OUT_DIR" 2>&1 | tee "$OUT_DIR/train.log"
RUN_STATUS=${PIPESTATUS[0]}
set -e

echo "[git] status after run" | tee -a "$OUT_DIR/train.log"
git status --short | tee -a "$OUT_DIR/train.log"

# Never push checkpoints by accident.
find "$OUT_DIR" -type f \( -name '*.pt' -o -name '*.pth' -o -name '*.ckpt' \) -delete

git add \
  "$OUT_DIR/metrics.csv" \
  "$OUT_DIR"/analysis_epoch_*.json \
  "$OUT_DIR/final_report.json" \
  "$OUT_DIR/REPORT_TO_CHATGPT.txt" \
  "$OUT_DIR/train.log" 2>/dev/null || true

if ! git diff --cached --quiet; then
  git commit -m "add tape-lane router 5ep logs ${RUN_ID}"
  git push
  echo "[done] logs pushed to GitHub: $OUT_DIR"
else
  echo "[done] no log changes to commit"
fi

exit "$RUN_STATUS"
