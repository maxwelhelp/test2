#!/usr/bin/env bash
set -euo pipefail

cd "$(git rev-parse --show-toplevel 2>/dev/null || pwd)"
BRANCH="${BRANCH:-codex-full-agent-plan}"
CORE_DIR="simple_butterfly_matrix_v4_tape_lane/v4_6_loop_core"
MAIN_FILE="$CORE_DIR/tape_lane_transport_v4_6_loop_core.py"
TS="$(date +%Y%m%d_%H%M%S)"
REPORT_DIR="${OUT_DIR:-simple_butterfly_matrix_v4_tape_lane/agent_reports/v4_6_loop_core_${TS}}"
mkdir -p "$REPORT_DIR"

DATA_ROOT="${DATA_ROOT:-./data/speechcommands}"
CLASSES="${CLASSES:-yes,no,up,down,left,right,on,off,stop,go}"
EPOCHS="${EPOCHS:-5}"
TRAIN_LIMIT="${TRAIN_LIMIT:-12000}"
VAL_LIMIT="${VAL_LIMIT:-2000}"
BATCH_SIZE="${BATCH_SIZE:-128}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-256}"
WORKERS="${WORKERS:-4}"
SEED="${SEED:-42}"
DEVICE="${DEVICE:-cuda}"
AMP="${AMP:-fp16}"
DIM="${DIM:-128}"
STEPS="${STEPS:-8}"
LANES="${LANES:-4}"
NUM_PRIMITIVES="${NUM_PRIMITIVES:-8}"
PRIMITIVE_RANK="${PRIMITIVE_RANK:-32}"
LR="${LR:-5e-4}"
LOG_EVERY="${LOG_EVERY:-100}"
MAX_TRAIN_BATCHES="${MAX_TRAIN_BATCHES:-0}"
MAX_VAL_BATCHES="${MAX_VAL_BATCHES:-0}"
SYNTHETIC_DATA="${SYNTHETIC_DATA:-0}"
ALLOW_SYNTHETIC_FALLBACK="${ALLOW_SYNTHETIC_FALLBACK:-0}"

printf '[v4.6 sync] branch=%s report_dir=%s\n' "$BRANCH" "$REPORT_DIR" | tee "$REPORT_DIR/run_header.txt"

git checkout "$BRANCH"
git pull --rebase --autostash origin "$BRANCH"

printf '[v4.6 sync] validate\n'
bash "$CORE_DIR/commands/validate_v4_6_loop_core.sh" | tee "$REPORT_DIR/validate.log"

RUN_ARGS=(
  --data-root "$DATA_ROOT"
  --classes "$CLASSES"
  --epochs "$EPOCHS"
  --train-limit "$TRAIN_LIMIT"
  --val-limit "$VAL_LIMIT"
  --batch-size "$BATCH_SIZE"
  --eval-batch-size "$EVAL_BATCH_SIZE"
  --workers "$WORKERS"
  --seed "$SEED"
  --device "$DEVICE"
  --amp "$AMP"
  --dim "$DIM"
  --steps "$STEPS"
  --lanes "$LANES"
  --num-primitives "$NUM_PRIMITIVES"
  --primitive-rank "$PRIMITIVE_RANK"
  --lr "$LR"
  --gumbel-tau-start "${GUMBEL_TAU_START:-1.0}"
  --gumbel-tau-min "${GUMBEL_TAU_MIN:-0.2}"
  --gumbel-tau-decay "${GUMBEL_TAU_DECAY:-0.92}"
  --lambda-boundary-budget "${LAMBDA_BOUNDARY_BUDGET:-0.006}"
  --lambda-boundary-flatness "${LAMBDA_BOUNDARY_FLATNESS:-0.002}"
  --lambda-route-entropy "${LAMBDA_ROUTE_ENTROPY:-0.010}"
  --lambda-route-allowed "${LAMBDA_ROUTE_ALLOWED:-0.012}"
  --lambda-route-identity "${LAMBDA_ROUTE_IDENTITY:-0.008}"
  --lambda-primitive-uniform "${LAMBDA_PRIMITIVE_UNIFORM:-0.004}"
  --lambda-primitive-diversity "${LAMBDA_PRIMITIVE_DIVERSITY:-0.002}"
  --lambda-sign-balance "${LAMBDA_SIGN_BALANCE:-0.001}"
  --lambda-program-cost "${LAMBDA_PROGRAM_COST:-0.0005}"
  --lambda-memory-write-cost "${LAMBDA_MEMORY_WRITE_COST:-0.0005}"
  --boundary-min-peaks "${BOUNDARY_MIN_PEAKS:-1.0}"
  --boundary-max-peaks "${BOUNDARY_MAX_PEAKS:-4.0}"
  --boundary-peak-threshold "${BOUNDARY_PEAK_THRESHOLD:-0.35}"
  --max-train-batches "$MAX_TRAIN_BATCHES"
  --max-val-batches "$MAX_VAL_BATCHES"
  --log-every "$LOG_EVERY"
  --no-save-checkpoints
  --out-dir "$REPORT_DIR"
)

if [ "${PIN_MEMORY:-1}" != "0" ]; then
  RUN_ARGS+=(--pin-memory)
fi
if [ "$SYNTHETIC_DATA" != "0" ]; then
  RUN_ARGS+=(--synthetic-data)
fi
if [ "$ALLOW_SYNTHETIC_FALLBACK" != "0" ]; then
  RUN_ARGS+=(--allow-synthetic-fallback)
fi

cat > "$REPORT_DIR/speed_config.txt" <<EOF_CFG
batch_size=$BATCH_SIZE
eval_batch_size=$EVAL_BATCH_SIZE
workers=$WORKERS
device=$DEVICE
amp=$AMP
dim=$DIM
steps=$STEPS
primitive_rank=$PRIMITIVE_RANK
synthetic_data=$SYNTHETIC_DATA
allow_synthetic_fallback=$ALLOW_SYNTHETIC_FALLBACK
EOF_CFG

printf '[v4.6 sync] run main\n'
set +e
python "$MAIN_FILE" "${RUN_ARGS[@]}" 2>&1 | tee "$REPORT_DIR/train.log"
RUN_STATUS=${PIPESTATUS[0]}
set -e

echo "$RUN_STATUS" > "$REPORT_DIR/run_status.txt"
printf '[v4.6 sync] run_status=%s\n' "$RUN_STATUS"

STATUS="$CORE_DIR/AGENT_STATUS.md"
if [ "$RUN_STATUS" -eq 0 ]; then
  RUN_LABEL="pass"
  REMAINING="- read $REPORT_DIR/REPORT_TO_CHATGPT.txt and inspect structural flags"
else
  RUN_LABEL="fail"
  REMAINING="- inspect $REPORT_DIR/train.log and $REPORT_DIR/run_status.txt"
fi
cat > "$STATUS" <<EOF_STATUS
# Agent Status

Current stage: v4.6 loop core MVP
Entrypoint: $MAIN_FILE
Smoke status: $RUN_LABEL
Timestamp: $TS
Report dir: $REPORT_DIR
Run status: $RUN_STATUS

Closed-loop invariant:
- The controller decision affects execution, execution affects loss, loss gradient updates the controller decision path.

Expected artifacts:
- metrics.csv
- trace_epoch_XXX.json
- REPORT_TO_CHATGPT.txt
- final_report.json
- train.log
- run_status.txt

Known remaining issues:
$REMAINING

No checkpoints should be committed.
EOF_STATUS

if [ -f "$REPORT_DIR/final_report.json" ]; then
  python - "$REPORT_DIR/final_report.json" "$STATUS" <<'PY'
import json, sys
src, dst = sys.argv[1], sys.argv[2]
try:
    data = json.load(open(src, 'r', encoding='utf-8'))
    s = data.get('last_summary') or {}
    b = s.get('boundary') or {}
    r = s.get('route') or {}
    p = s.get('primitive') or {}
    m = s.get('memory') or {}
    flags = s.get('collapse_flags') or []
    with open(dst, 'a', encoding='utf-8') as f:
        f.write('\nSummary:\n')
        f.write(f"- best_acc: {float(data.get('best_acc', 0.0))*100:.2f}% @ epoch {data.get('best_epoch', 0)}\n")
        f.write(f"- boundary_mean/peaks: {float(b.get('mean', 0.0)):.4f}/{b.get('peak_count', 0)}\n")
        f.write(f"- route_entropy/self/useful: {float(r.get('entropy_mean', 0.0)):.4f}/{float(r.get('self_route_mass', 0.0)):.4f}/{float(r.get('useful_transition_mass', 0.0)):.4f}\n")
        f.write(f"- primitive_entropy/top1/neg_sign: {float(p.get('entropy_mean', 0.0)):.4f}/{float(p.get('top1_share', 0.0)):.4f}/{float(p.get('sign_negative_share', 0.0)):.4f}\n")
        f.write(f"- memory_write/read/influence: {float(m.get('write_norm_mean', 0.0)):.4f}/{float(m.get('read_norm_mean', 0.0)):.4f}/{float(m.get('read_influence_proxy', 0.0)):.4f}\n")
        f.write(f"- collapse_flags: {','.join(flags) if flags else 'NONE'}\n")
except Exception as e:
    with open(dst, 'a', encoding='utf-8') as f:
        f.write(f'\nSummary parse failed: {e}\n')
PY
fi

printf '[v4.6 sync] checkpoint guard before git add\n'
if find "$CORE_DIR" "$REPORT_DIR" -type f \( -name '*.pt' -o -name '*.pth' -o -name '*.ckpt' -o -name '*.safetensors' \) | grep .; then
  echo '[v4.6 sync] checkpoint-like file found; refusing to commit' >&2
  exit 4
fi

git add "$CORE_DIR" "$REPORT_DIR"
if git diff --cached --name-only | grep -E '\.(pt|pth|ckpt|safetensors)$'; then
  echo '[v4.6 sync] checkpoint-like file staged; refusing to commit' >&2
  git reset --cached "$CORE_DIR" "$REPORT_DIR" >/dev/null || true
  exit 5
fi

if git diff --cached --quiet; then
  printf '[v4.6 sync] nothing to commit\n'
else
  git commit -m "Add v4.6 loop core run artifacts"
  git push origin "$BRANCH"
fi

printf '[v4.6 sync] done: %s status=%s\n' "$REPORT_DIR" "$RUN_STATUS"
exit "$RUN_STATUS"
