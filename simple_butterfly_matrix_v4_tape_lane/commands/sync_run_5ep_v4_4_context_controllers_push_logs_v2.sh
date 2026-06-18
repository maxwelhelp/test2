#!/usr/bin/env bash
set -euo pipefail

cd "$(git rev-parse --show-toplevel 2>/dev/null || pwd)"

git pull --ff-only

TS="$(date +%Y%m%d_%H%M%S)"
REPORT_DIR="${OUT_DIR:-simple_butterfly_matrix_v4_tape_lane/agent_reports/v4_4_context_controllers_${TS}}"
mkdir -p "$REPORT_DIR"

MAIN="simple_butterfly_matrix_v4_tape_lane/tape_lane_transport_v4_4_context_controllers.py"
if [ ! -f "$MAIN" ]; then
  echo "[v4.4 sync] missing $MAIN" | tee "$REPORT_DIR/run_status.txt"
  echo "Implement v4.4 context controllers after v4.3 canonical main smoke/5ep is stable." | tee -a "$REPORT_DIR/run_status.txt"
  echo "Read: simple_butterfly_matrix_v4_tape_lane/V4_4_CONTEXT_CONTROLLERS_IMPLEMENTATION_BRIEF.md" | tee -a "$REPORT_DIR/run_status.txt"
  exit 2
fi

DATA_ROOT="${DATA_ROOT:-../architecture_builder/data/speechcommands}"
EPOCHS="${EPOCHS:-5}"
TRAIN_LIMIT="${TRAIN_LIMIT:-12000}"
VAL_LIMIT="${VAL_LIMIT:-2000}"
BATCH_SIZE="${BATCH_SIZE:-192}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-512}"
WORKERS="${WORKERS:-6}"
AMP="${AMP:-fp16}"
DEVICE="${DEVICE:-cuda}"
SEED="${SEED:-42}"
LOG_EVERY="${LOG_EVERY:-100}"
MAX_TRAIN_BATCHES="${MAX_TRAIN_BATCHES:-0}"
MAX_VAL_BATCHES="${MAX_VAL_BATCHES:-0}"
CONTEXT_ALPHA_MIN="${CONTEXT_ALPHA_MIN:-0.01}"
CONTEXT_ALPHA_MAX="${CONTEXT_ALPHA_MAX:-0.10}"
COMPARE_TO="${COMPARE_TO:-v4.3_canonical_same_seed_config}"

python -m py_compile "$MAIN"

echo "[v4.4 sync] run -> $REPORT_DIR" | tee "$REPORT_DIR/speed_config.txt"
set +e
python "$MAIN" \
  --data-root "$DATA_ROOT" \
  --epochs "$EPOCHS" \
  --train-limit "$TRAIN_LIMIT" \
  --val-limit "$VAL_LIMIT" \
  --batch-size "$BATCH_SIZE" \
  --eval-batch-size "$EVAL_BATCH_SIZE" \
  --workers "$WORKERS" \
  --max-train-batches "$MAX_TRAIN_BATCHES" \
  --max-val-batches "$MAX_VAL_BATCHES" \
  --seed "$SEED" \
  --device "$DEVICE" \
  --amp "$AMP" \
  --context-alpha-min "$CONTEXT_ALPHA_MIN" \
  --context-alpha-max "$CONTEXT_ALPHA_MAX" \
  --compare-to "$COMPARE_TO" \
  --pin-memory \
  --log-every "$LOG_EVERY" \
  --no-save-checkpoints \
  --out-dir "$REPORT_DIR" 2>&1 | tee "$REPORT_DIR/train.log"
RUN_STATUS=${PIPESTATUS[0]}
set -e

echo "[v4.4 sync] run_status=$RUN_STATUS" | tee "$REPORT_DIR/run_status.txt"

find "$REPORT_DIR" -maxdepth 1 \( -name '*.json' -o -name '*.csv' -o -name '*.txt' -o -name '*.log' \) -print -exec git add {} +
if git diff --cached --name-only | grep -E '\.(pt|pth|ckpt|safetensors)$'; then
  echo "[v4.4 sync] ERROR: checkpoint staged"
  exit 1
fi
if ! git diff --cached --quiet; then
  git commit -m "Add v4.4 context controller run logs $TS"
  git push
fi
exit "$RUN_STATUS"
