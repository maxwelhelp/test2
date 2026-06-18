#!/usr/bin/env bash
set -euo pipefail

cd "$(git rev-parse --show-toplevel 2>/dev/null || pwd)"

echo "[v4.3 sync] git pull"
git pull --ff-only

TS="$(date +%Y%m%d_%H%M%S)"
REPORT_DIR="${OUT_DIR:-simple_butterfly_matrix_v4_tape_lane/agent_reports/v4_3_min_heart_${TS}}"
mkdir -p "$REPORT_DIR"

DATA_ROOT="${DATA_ROOT:-../architecture_builder/data/speechcommands}"
EPOCHS="${EPOCHS:-5}"
TRAIN_LIMIT="${TRAIN_LIMIT:-12000}"
VAL_LIMIT="${VAL_LIMIT:-2000}"
AMP="${AMP:-fp16}"
BATCH_SIZE="${BATCH_SIZE:-128}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-256}"
WORKERS="${WORKERS:-4}"
SEED="${SEED:-42}"
DEVICE="${DEVICE:-cuda}"
LOG_EVERY="${LOG_EVERY:-50}"
COMPARE_TO="${COMPARE_TO:-v4.2_fixed_guided same seed/config or baseline_missing}"

echo "[v4.3 sync] validate"
bash simple_butterfly_matrix_v4_tape_lane/commands/validate_v4_3_min_heart.sh

echo "[v4.3 sync] run -> $REPORT_DIR"
python simple_butterfly_matrix_v4_tape_lane/tape_lane_transport_v4_3_min_heart.py \
  --data-root "$DATA_ROOT" \
  --epochs "$EPOCHS" \
  --train-limit "$TRAIN_LIMIT" \
  --val-limit "$VAL_LIMIT" \
  --batch-size "$BATCH_SIZE" \
  --eval-batch-size "$EVAL_BATCH_SIZE" \
  --workers "$WORKERS" \
  --seed "$SEED" \
  --device "$DEVICE" \
  --amp "$AMP" \
  --dim "${DIM:-96}" \
  --evidence-cells "${EVIDENCE_CELLS:-48}" \
  --lanes "${LANES:-4}" \
  --cells-per-lane "${CELLS_PER_LANE:-12}" \
  --tape-steps "${TAPE_STEPS:-12}" \
  --pair-slots "${PAIR_SLOTS:-12}" \
  --lr "${LR:-5e-4}" \
  --structured-init-strength "${STRUCTURED_INIT_STRENGTH:-0.35}" \
  --read-prior-mode "${READ_PRIOR_MODE:-structured}" \
  --read-prior-strength "${READ_PRIOR_STRENGTH:-0.35}" \
  --route-prior-mode "${ROUTE_PRIOR_MODE:-weak_flow}" \
  --route-prior-strength "${ROUTE_PRIOR_STRENGTH:-0.20}" \
  --boundary-route-strength "${BOUNDARY_ROUTE_STRENGTH:-0.25}" \
  --context-primitive-scale "${CONTEXT_PRIMITIVE_SCALE:-0.15}" \
  --class-lane-prior-mode "${CLASS_LANE_PRIOR_MODE:-soft_cover}" \
  --class-lane-init-strength "${CLASS_LANE_INIT_STRENGTH:-0.10}" \
  --lane-prior-strength "${LANE_PRIOR_STRENGTH:-0.20}" \
  --lambda-route-offdiag-outside-boundary "${LAMBDA_ROUTE_OFFDIAG_OUTSIDE_BOUNDARY:-0.012}" \
  --lambda-boundary-budget "${LAMBDA_BOUNDARY_BUDGET:-0.006}" \
  --lambda-late-input-read "${LAMBDA_LATE_INPUT_READ:-0.012}" \
  --lambda-memory-write-cost "${LAMBDA_MEMORY_WRITE_COST:-0.003}" \
  --lambda-memory-overwrite "${LAMBDA_MEMORY_OVERWRITE:-0.004}" \
  --lambda-detail-head-shortcut "${LAMBDA_DETAIL_HEAD_SHORTCUT:-0.010}" \
  --lambda-skip-cost "${LAMBDA_SKIP_COST:-0.0}" \
  --lambda-operator-complexity "${LAMBDA_OPERATOR_COMPLEXITY:-0.0}" \
  --detail-head-shortcut-target "${DETAIL_HEAD_SHORTCUT_TARGET:-0.42}" \
  --boundary-peak-threshold "${BOUNDARY_PEAK_THRESHOLD:-0.35}" \
  --late-input-start "${LATE_INPUT_START:-0.45}" \
  --late-input-tau "${LATE_INPUT_TAU:-0.12}" \
  --max-candidates-per-epoch "${MAX_CANDIDATES_PER_EPOCH:-8}" \
  --compare-to "$COMPARE_TO" \
  --pin-memory \
  --log-every "$LOG_EVERY" \
  --no-save-checkpoints \
  --out-dir "$REPORT_DIR" 2>&1 | tee "$REPORT_DIR/train.log"

STATUS="simple_butterfly_matrix_v4_tape_lane/AGENT_STATUS.md"
cat > "$STATUS" <<EOF_STATUS
# Agent Status

Last run: v4.3_min_heart
Timestamp: $TS
Report dir: $REPORT_DIR
Command: sync_run_5ep_v4_3_min_heart_push_logs.sh

Expected artifacts:
- metrics.csv
- analysis_epoch_XXX.json
- trace_feedback_epoch_XXX.json
- candidate_suggestions_epoch_XXX.json
- REPORT_TO_CHATGPT.txt
- final_report.json
- train.log

No checkpoints should be committed.
EOF_STATUS

echo "[v4.3 sync] git add logs only"
find "$REPORT_DIR" -maxdepth 1 \( -name '*.json' -o -name '*.csv' -o -name '*.txt' -o -name '*.log' \) -print -exec git add {} +
git add "$STATUS"

echo "[v4.3 sync] ensure no checkpoint files staged"
if git diff --cached --name-only | grep -E '\.(pt|pth|ckpt|safetensors)$'; then
  echo "[v4.3 sync] ERROR: checkpoint file staged"
  exit 1
fi

if git diff --cached --quiet; then
  echo "[v4.3 sync] no logs to commit"
else
  git commit -m "Add v4.3 min heart run logs $TS"
  git push
fi

echo "[v4.3 sync] done: $REPORT_DIR"
