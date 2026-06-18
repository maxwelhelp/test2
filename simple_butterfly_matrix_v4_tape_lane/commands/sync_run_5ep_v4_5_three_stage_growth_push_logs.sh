#!/usr/bin/env bash
set -euo pipefail

# v4.5 staged program growth runner.
# Starts from a tiny 2-step seed, not a hard-coded first/middle/last program:
#   step 0: input/evidence/work step, weak task-aware prior only
#   step 1: output/aggregation/head-ready step, weak task-aware prior only
# Then the growth planner proposes where to insert the next step based on weak
# trace signals. This avoids paying for 12 exact tape steps before extra depth is justified.

cd "$(git rev-parse --show-toplevel 2>/dev/null || pwd)"

git pull --ff-only

TS="$(date +%Y%m%d_%H%M%S)"
REPORT_DIR="${OUT_DIR:-simple_butterfly_matrix_v4_tape_lane/agent_reports/v4_5_two_step_growth_${TS}}"
mkdir -p "$REPORT_DIR"

MAIN="simple_butterfly_matrix_v4_tape_lane/tape_lane_transport_v4_5_projection_feedback.py"
GROWTH_PLANNER="simple_butterfly_matrix_v4_tape_lane/program_growth_planner_v1.py"

DATA_ROOT="${DATA_ROOT:-../architecture_builder/data/speechcommands}"
EPOCHS="${EPOCHS:-7}"
TRAIN_LIMIT="${TRAIN_LIMIT:-12000}"
VAL_LIMIT="${VAL_LIMIT:-2000}"
BATCH_SIZE="${BATCH_SIZE:-256}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-512}"
WORKERS="${WORKERS:-6}"
AMP="${AMP:-fp16}"
DEVICE="${DEVICE:-cuda}"
SEED="${SEED:-42}"
LOG_EVERY="${LOG_EVERY:-100}"
MAX_TRAIN_BATCHES="${MAX_TRAIN_BATCHES:-0}"
MAX_VAL_BATCHES="${MAX_VAL_BATCHES:-0}"

# Small exact program first. This is the speed knob.
TAPE_STEPS="${TAPE_STEPS:-2}"
PAIR_SLOTS="${PAIR_SLOTS:-4}"
CELLS_PER_LANE="${CELLS_PER_LANE:-10}"
EVIDENCE_CELLS="${EVIDENCE_CELLS:-40}"

# Scout stays weak. Online scout is trainable and not crushed by a default gate cost.
SCOUT_FEEDBACK_PATH="${SCOUT_FEEDBACK_PATH:-}"
SCOUT_FEEDBACK_SCALE="${SCOUT_FEEDBACK_SCALE:-0.006}"
SCOUT_AUX_LAMBDA="${SCOUT_AUX_LAMBDA:-0.001}"
ONLINE_SCOUT_SCALE="${ONLINE_SCOUT_SCALE:-0.010}"
ONLINE_SCOUT_GATE_INIT="${ONLINE_SCOUT_GATE_INIT:--2.2}"
ONLINE_SCOUT_USE_COST="${ONLINE_SCOUT_USE_COST:-0.0}"
ONLINE_SCOUT_GATE_TARGET="${ONLINE_SCOUT_GATE_TARGET:-0.10}"
SCOUT_SKIP_COST_SCALE="${SCOUT_SKIP_COST_SCALE:-0.001}"
SKIP_COLLAPSE_TARGET="${SKIP_COLLAPSE_TARGET:-3.5}"

python -m py_compile "$MAIN"

echo "[v4.5 growth] run -> $REPORT_DIR" | tee "$REPORT_DIR/speed_config.txt"
echo "TAPE_STEPS=$TAPE_STEPS" | tee -a "$REPORT_DIR/speed_config.txt"
echo "PAIR_SLOTS=$PAIR_SLOTS" | tee -a "$REPORT_DIR/speed_config.txt"
echo "CELLS_PER_LANE=$CELLS_PER_LANE" | tee -a "$REPORT_DIR/speed_config.txt"
echo "EVIDENCE_CELLS=$EVIDENCE_CELLS" | tee -a "$REPORT_DIR/speed_config.txt"
echo "SCOUT_FEEDBACK_PATH=$SCOUT_FEEDBACK_PATH" | tee -a "$REPORT_DIR/speed_config.txt"
echo "SCOUT_FEEDBACK_SCALE=$SCOUT_FEEDBACK_SCALE" | tee -a "$REPORT_DIR/speed_config.txt"
echo "SCOUT_AUX_LAMBDA=$SCOUT_AUX_LAMBDA" | tee -a "$REPORT_DIR/speed_config.txt"
echo "ONLINE_SCOUT_SCALE=$ONLINE_SCOUT_SCALE" | tee -a "$REPORT_DIR/speed_config.txt"
echo "ONLINE_SCOUT_GATE_INIT=$ONLINE_SCOUT_GATE_INIT" | tee -a "$REPORT_DIR/speed_config.txt"
echo "ONLINE_SCOUT_USE_COST=$ONLINE_SCOUT_USE_COST" | tee -a "$REPORT_DIR/speed_config.txt"
echo "ONLINE_SCOUT_GATE_TARGET=$ONLINE_SCOUT_GATE_TARGET" | tee -a "$REPORT_DIR/speed_config.txt"
echo "SCOUT_SKIP_COST_SCALE=$SCOUT_SKIP_COST_SCALE" | tee -a "$REPORT_DIR/speed_config.txt"
echo "SKIP_COLLAPSE_TARGET=$SKIP_COLLAPSE_TARGET" | tee -a "$REPORT_DIR/speed_config.txt"

set +e
SCOUT_FEEDBACK_PATH="$SCOUT_FEEDBACK_PATH" \
SCOUT_FEEDBACK_SCALE="$SCOUT_FEEDBACK_SCALE" \
SCOUT_AUX_LAMBDA="$SCOUT_AUX_LAMBDA" \
ONLINE_SCOUT_SCALE="$ONLINE_SCOUT_SCALE" \
ONLINE_SCOUT_GATE_INIT="$ONLINE_SCOUT_GATE_INIT" \
ONLINE_SCOUT_USE_COST="$ONLINE_SCOUT_USE_COST" \
ONLINE_SCOUT_GATE_TARGET="$ONLINE_SCOUT_GATE_TARGET" \
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
  --tape-steps "$TAPE_STEPS" \
  --pair-slots "$PAIR_SLOTS" \
  --cells-per-lane "$CELLS_PER_LANE" \
  --evidence-cells "$EVIDENCE_CELLS" \
  --compare-to "v4.5_two_step_growth seed T=$TAPE_STEPS" \
  --profile-speed \
  --pin-memory \
  --log-every "$LOG_EVERY" \
  --no-save-checkpoints \
  --out-dir "$REPORT_DIR" 2>&1 | tee "$REPORT_DIR/train.log"
RUN_STATUS=${PIPESTATUS[0]}
set -e

echo "[v4.5 growth] run_status=$RUN_STATUS" | tee "$REPORT_DIR/run_status.txt"

if [ "$RUN_STATUS" -eq 0 ] && [ -f "$GROWTH_PLANNER" ]; then
  python -m py_compile "$GROWTH_PLANNER"
  python "$GROWTH_PLANNER" --report-dir "$REPORT_DIR" | tee "$REPORT_DIR/growth_planner.log" || true
fi

find "$REPORT_DIR" -maxdepth 1 \( -name '*.json' -o -name '*.csv' -o -name '*.txt' -o -name '*.log' \) -print -exec git add {} +
git add "$MAIN" "$GROWTH_PLANNER" "simple_butterfly_matrix_v4_tape_lane/commands/sync_run_5ep_v4_5_three_stage_growth_push_logs.sh" 2>/dev/null || true
if git diff --cached --name-only | grep -E '\.(pt|pth|ckpt|safetensors)$'; then
  echo "[v4.5 growth] ERROR: checkpoint staged"
  exit 1
fi
if ! git diff --cached --quiet; then
  git commit -m "Add v4.5 two-step growth run logs $TS"
  git push
fi
exit "$RUN_STATUS"
