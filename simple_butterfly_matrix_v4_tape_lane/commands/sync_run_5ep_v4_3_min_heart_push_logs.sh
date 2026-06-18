#!/usr/bin/env bash
set -euo pipefail

cd "$(git rev-parse --show-toplevel 2>/dev/null || pwd)"

TS="$(date +%Y%m%d_%H%M%S)"
REPORT_DIR="${OUT_DIR:-simple_butterfly_matrix_v4_tape_lane/agent_reports/v4_3_canonical_${TS}}"
mkdir -p "$REPORT_DIR"

DATA_ROOT="${DATA_ROOT:-../architecture_builder/data/speechcommands}"
EPOCHS="${EPOCHS:-5}"
TRAIN_LIMIT="${TRAIN_LIMIT:-12000}"
VAL_LIMIT="${VAL_LIMIT:-2000}"
AMP="${AMP:-fp16}"
BATCH_SIZE="${BATCH_SIZE:-192}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-512}"
WORKERS="${WORKERS:-6}"
SEED="${SEED:-42}"
DEVICE="${DEVICE:-cuda}"
LOG_EVERY="${LOG_EVERY:-100}"
MAX_TRAIN_BATCHES="${MAX_TRAIN_BATCHES:-0}"
MAX_VAL_BATCHES="${MAX_VAL_BATCHES:-0}"
COMPARE_TO="${COMPARE_TO:-v4.3_audit_fixed same seed/config or baseline_missing}"
MAIN_FILE="simple_butterfly_matrix_v4_tape_lane/tape_lane_transport_v4_3_min_heart.py"

echo "[v4.3 sync] validate"
bash simple_butterfly_matrix_v4_tape_lane/commands/validate_v4_3_min_heart.sh

echo "[v4.3 sync] gradient sanity"
OUT="$REPORT_DIR/grad_sanity.json" bash simple_butterfly_matrix_v4_tape_lane/commands/grad_sanity_v4_3_min_heart.sh | tee "$REPORT_DIR/grad_sanity.log"

echo "[v4.3 sync] run canonical main -> $REPORT_DIR"
echo "[v4.3 sync] speed cfg: batch=$BATCH_SIZE eval_batch=$EVAL_BATCH_SIZE workers=$WORKERS log_every=$LOG_EVERY max_train_batches=$MAX_TRAIN_BATCHES max_val_batches=$MAX_VAL_BATCHES" | tee "$REPORT_DIR/speed_config.txt"
set +e
python "$MAIN_FILE" \
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
  --memory-write-warmup-epochs "${MEMORY_WRITE_WARMUP_EPOCHS:-2}" \
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
RUN_STATUS=${PIPESTATUS[0]}
set -e

echo "[v4.3 sync] run_status=$RUN_STATUS" | tee "$REPORT_DIR/run_status.txt"

STATUS="simple_butterfly_matrix_v4_tape_lane/AGENT_STATUS.md"
if [ "$RUN_STATUS" -eq 0 ]; then
  RUN_LABEL="pass"
  REMAINING="- no runtime canonicalizer in standard sync; context-controller is intentionally separate"
else
  RUN_LABEL="fail"
  REMAINING="- inspect $REPORT_DIR/train.log and $REPORT_DIR/run_status.txt"
fi
cat > "$STATUS" <<EOF_STATUS
# Agent Status

Current stage: v4.3 canonical main stabilization
Entrypoint: $MAIN_FILE
Runtime wrappers: not used by standard sync
Canonicalizer: deprecated / not used in runtime
Smoke status: $RUN_LABEL

Last run: v4.3_canonical_main
Timestamp: $TS
Report dir: $REPORT_DIR
Command: sync_run_5ep_v4_3_min_heart_push_logs.sh
Run status: $RUN_STATUS

Canonical main:
- canonicalizer: deprecated / not used in runtime
- entrypoint: $MAIN_FILE
- wrapper: not used by standard sync
- still observer-only: candidate deploy=false, no editor auto-deploy

Speed config:
- batch_size: $BATCH_SIZE
- eval_batch_size: $EVAL_BATCH_SIZE
- workers: $WORKERS
- log_every: $LOG_EVERY
- max_train_batches: $MAX_TRAIN_BATCHES
- max_val_batches: $MAX_VAL_BATCHES

Expected artifacts:
- grad_sanity.json
- grad_sanity.log
- speed_config.txt
- metrics.csv
- analysis_epoch_XXX.json
- trace_feedback_epoch_XXX.json
- candidate_suggestions_epoch_XXX.json
- REPORT_TO_CHATGPT.txt
- final_report.json
- train.log
- run_status.txt

Known remaining issues:
$REMAINING

Next step:
- only after stable 5ep, test context-controller separately

No checkpoints should be committed.
EOF_STATUS

if [ -f "$REPORT_DIR/final_report.json" ]; then
  python - "$REPORT_DIR/final_report.json" "$STATUS" <<'PY'
import json, sys
src, dst = sys.argv[1], sys.argv[2]
try:
    data = json.load(open(src, 'r', encoding='utf-8'))
    tr = data.get('last_trace_feedback') or {}
    route = tr.get('route') or {}
    head = tr.get('head') or {}
    memory = tr.get('memory') or {}
    flags = list(tr.get('collapse_flags') or [])
    bm = float(route.get('boundary_mean', 0.0) or 0.0)
    ent = float(route.get('entropy_mean', 0.0) or 0.0)
    detail = float(head.get('detail_attention_mass', 0.0) or 0.0)
    mem_w = float(memory.get('write_mean', 0.0) or 0.0)
    mem_c = float(memory.get('memory_consumer_proxy', memory.get('consumer_score', 0.0)) or 0.0)
    with open(dst, 'a', encoding='utf-8') as f:
        f.write('\nSummary:\n')
        f.write(f"- best_acc: {float(data.get('best_acc', 0.0))*100:.2f}% @ epoch {data.get('best_epoch', 0)}\n")
        f.write(f"- boundary_mean: {bm:.4f}\n")
        f.write(f"- route_entropy: {ent:.4f}\n")
        f.write(f"- self_route_mass: {float(route.get('self_route_mass', 0.0) or 0.0):.4f}\n")
        f.write(f"- useful_transition_mass: {float(route.get('useful_transition_mass', 0.0) or 0.0):.4f}\n")
        f.write(f"- detail_attention_mass: {detail:.4f}\n")
        f.write(f"- memory_write/proxy: {mem_w:.4f}/{mem_c:.4f}\n")
        f.write(f"- collapse_flags: {','.join(flags) if flags else 'NONE'}\n")
except Exception as e:
    with open(dst, 'a', encoding='utf-8') as f:
        f.write(f'\nSummary parse failed: {e}\n')
PY
fi

echo "[v4.3 sync] done: $REPORT_DIR status=$RUN_STATUS"
exit "$RUN_STATUS"
