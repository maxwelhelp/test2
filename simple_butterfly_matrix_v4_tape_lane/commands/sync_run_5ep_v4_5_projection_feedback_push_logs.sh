#!/usr/bin/env bash
set -euo pipefail

cd "$(git rev-parse --show-toplevel 2>/dev/null || pwd)"

git pull --ff-only

TS="$(date +%Y%m%d_%H%M%S)"
REPORT_DIR="${OUT_DIR:-simple_butterfly_matrix_v4_tape_lane/agent_reports/v4_5_projection_feedback_${TS}}"
mkdir -p "$REPORT_DIR"

MAIN="simple_butterfly_matrix_v4_tape_lane/tape_lane_transport_v4_5_projection_feedback.py"
COUNCIL="simple_butterfly_matrix_v4_tape_lane/projection_council_v2.py"

if [ ! -f "$MAIN" ]; then
  echo "[v4.5 sync] missing $MAIN" | tee "$REPORT_DIR/run_status.txt"
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
COMPARE_TO="${COMPARE_TO:-v4.4_no_scout_same_seed_config}"
SCOUT_FEEDBACK_SCALE="${SCOUT_FEEDBACK_SCALE:-0.015}"
ONLINE_SCOUT_SCALE="${ONLINE_SCOUT_SCALE:-0.020}"
ONLINE_SCOUT_GATE_INIT="${ONLINE_SCOUT_GATE_INIT:--3.0}"
ONLINE_SCOUT_USE_COST="${ONLINE_SCOUT_USE_COST:-0.0005}"
SCOUT_SKIP_COST_SCALE="${SCOUT_SKIP_COST_SCALE:-0.001}"
SKIP_COLLAPSE_TARGET="${SKIP_COLLAPSE_TARGET:-4.0}"

# If no feedback path was provided, generate feedback from the latest v4.4 report.
if [ -z "${SCOUT_FEEDBACK_PATH:-}" ]; then
  LATEST_V44="$(ls -td simple_butterfly_matrix_v4_tape_lane/agent_reports/v4_4_context_controllers_* 2>/dev/null | head -1 || true)"
  if [ -n "$LATEST_V44" ] && [ -d "$LATEST_V44" ]; then
    echo "[v4.5 sync] build scout feedback from $LATEST_V44" | tee "$REPORT_DIR/scout_feedback_source.txt"
    python -m py_compile "$COUNCIL"
    python "$COUNCIL" --repo-root . --report-dir "$LATEST_V44" --dim "${SCOUT_DIM:-24}" --max-bias "${SCOUT_MAX_BIAS:-0.03}" | tee -a "$REPORT_DIR/scout_feedback_source.txt"
    SCOUT_FEEDBACK_PATH="$(ls -t "$LATEST_V44"/scout_runtime_feedback_epoch_*.json | head -1)"
  else
    echo "[v4.5 sync] WARNING: no v4.4 report found; running without scout feedback" | tee "$REPORT_DIR/scout_feedback_source.txt"
    SCOUT_FEEDBACK_PATH=""
  fi
fi

python -m py_compile "$MAIN"

echo "[v4.5 sync] run -> $REPORT_DIR" | tee "$REPORT_DIR/speed_config.txt"
echo "SCOUT_FEEDBACK_PATH=$SCOUT_FEEDBACK_PATH" | tee -a "$REPORT_DIR/speed_config.txt"
echo "SCOUT_FEEDBACK_SCALE=$SCOUT_FEEDBACK_SCALE" | tee -a "$REPORT_DIR/speed_config.txt"
echo "ONLINE_SCOUT_SCALE=$ONLINE_SCOUT_SCALE" | tee -a "$REPORT_DIR/speed_config.txt"
echo "ONLINE_SCOUT_GATE_INIT=$ONLINE_SCOUT_GATE_INIT" | tee -a "$REPORT_DIR/speed_config.txt"
echo "ONLINE_SCOUT_USE_COST=$ONLINE_SCOUT_USE_COST" | tee -a "$REPORT_DIR/speed_config.txt"
echo "SCOUT_SKIP_COST_SCALE=$SCOUT_SKIP_COST_SCALE" | tee -a "$REPORT_DIR/speed_config.txt"
echo "SKIP_COLLAPSE_TARGET=$SKIP_COLLAPSE_TARGET" | tee -a "$REPORT_DIR/speed_config.txt"

set +e
SCOUT_FEEDBACK_PATH="$SCOUT_FEEDBACK_PATH" \
SCOUT_FEEDBACK_SCALE="$SCOUT_FEEDBACK_SCALE" \
ONLINE_SCOUT_SCALE="$ONLINE_SCOUT_SCALE" \
ONLINE_SCOUT_GATE_INIT="$ONLINE_SCOUT_GATE_INIT" \
ONLINE_SCOUT_USE_COST="$ONLINE_SCOUT_USE_COST" \
SCOUT_SKIP_COST_SCALE="$SCOUT_SKIP_COST_SCALE" \
SKIP_COLLAPSE_TARGET="$SKIP_COLLAPSE_TARGET" \
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

echo "[v4.5 sync] run_status=$RUN_STATUS" | tee "$REPORT_DIR/run_status.txt"
find "$REPORT_DIR" -maxdepth 1 \( -name '*.json' -o -name '*.csv' -o -name '*.txt' -o -name '*.log' \) -print -exec git add {} +
git add "$MAIN" "$COUNCIL" "simple_butterfly_matrix_v4_tape_lane/commands/sync_run_5ep_v4_5_projection_feedback_push_logs.sh"
if git diff --cached --name-only | grep -E '\.(pt|pth|ckpt|safetensors)$'; then
  echo "[v4.5 sync] ERROR: checkpoint staged"
  exit 1
fi
if ! git diff --cached --quiet; then
  git commit -m "Add v4.5 projection feedback run logs $TS"
  git push
fi
exit "$RUN_STATUS"
