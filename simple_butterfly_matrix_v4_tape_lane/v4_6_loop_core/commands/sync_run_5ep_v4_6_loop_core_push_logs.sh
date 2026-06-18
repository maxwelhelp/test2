#!/usr/bin/env bash
set -euo pipefail

cd "$(git rev-parse --show-toplevel 2>/dev/null || pwd)"
BRANCH="${BRANCH:-codex-full-agent-plan}"
CORE_DIR="simple_butterfly_matrix_v4_tape_lane/v4_6_loop_core"
MAIN_FILE="$CORE_DIR/tape_lane_transport_v4_6_1_grouped_core.py"
TS="$(date +%Y%m%d_%H%M%S)"
REPORT_DIR="${OUT_DIR:-simple_butterfly_matrix_v4_tape_lane/agent_reports/v4_6_1_grouped_real_${TS}}"
LATEST_REPORT="$CORE_DIR/LATEST_RUN_REPORT.md"
mkdir -p "$REPORT_DIR"

DATA_ROOT="${DATA_ROOT:-../architecture_builder/data/speechcommands}"
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
NUM_PRIMITIVES="${NUM_PRIMITIVES:-18}"
PRIMITIVE_RANK="${PRIMITIVE_RANK:-32}"
ROUTE_PRIOR_STRENGTH="${ROUTE_PRIOR_STRENGTH:-0.0}"
LR="${LR:-5e-4}"
LOG_EVERY="${LOG_EVERY:-100}"
MAX_TRAIN_BATCHES="${MAX_TRAIN_BATCHES:-0}"
MAX_VAL_BATCHES="${MAX_VAL_BATCHES:-0}"
SYNTHETIC_DATA="${SYNTHETIC_DATA:-0}"
ALLOW_SYNTHETIC_FALLBACK="${ALLOW_SYNTHETIC_FALLBACK:-0}"

printf '[v4.6.1 sync] branch=%s report_dir=%s\n' "$BRANCH" "$REPORT_DIR" | tee "$REPORT_DIR/run_header.txt"

if [ "$SYNTHETIC_DATA" != "0" ]; then
  echo '[v4.6.1 sync] SYNTHETIC_DATA is disabled for evidence runs. Use validate script RUN_SYNTHETIC_SMOKE=1 only.' | tee -a "$REPORT_DIR/run_header.txt" >&2
  exit 2
fi
if [ "$ALLOW_SYNTHETIC_FALLBACK" != "0" ]; then
  echo '[v4.6.1 sync] synthetic fallback is disabled for evidence runs.' | tee -a "$REPORT_DIR/run_header.txt" >&2
  exit 2
fi

git checkout "$BRANCH"
git pull --rebase --autostash origin "$BRANCH"

printf '[v4.6.1 sync] validate\n'
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
  --route-prior-strength "$ROUTE_PRIOR_STRENGTH"
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

cat > "$REPORT_DIR/speed_config.txt" <<EOF_CFG
batch_size=$BATCH_SIZE
eval_batch_size=$EVAL_BATCH_SIZE
workers=$WORKERS
device=$DEVICE
amp=$AMP
dim=$DIM
steps=$STEPS
num_primitives=$NUM_PRIMITIVES
primitive_rank=$PRIMITIVE_RANK
route_prior_strength=$ROUTE_PRIOR_STRENGTH
synthetic_data=disabled
synthetic_fallback=disabled
EOF_CFG

printf '[v4.6.1 sync] run main\n'
set +e
python "$MAIN_FILE" "${RUN_ARGS[@]}" 2>&1 | tee "$REPORT_DIR/train.log"
RUN_STATUS=${PIPESTATUS[0]}
set -e

echo "$RUN_STATUS" > "$REPORT_DIR/run_status.txt"
printf '[v4.6.1 sync] run_status=%s\n' "$RUN_STATUS"

STATUS="$CORE_DIR/AGENT_STATUS.md"
if [ "$RUN_STATUS" -eq 0 ]; then
  RUN_LABEL="pass"
  REMAINING="- inspect grouped/context/credit reports in $REPORT_DIR"
else
  RUN_LABEL="fail"
  REMAINING="- inspect $REPORT_DIR/train.log and $REPORT_DIR/run_status.txt; failed artifacts are not committed automatically"
fi
cat > "$STATUS" <<EOF_STATUS
# Agent Status

Current stage: v4.6.1 grouped real-data loop core
Entrypoint: $MAIN_FILE
Smoke status: $RUN_LABEL
Timestamp: $TS
Report dir: $REPORT_DIR
Run status: $RUN_STATUS
Latest compact report: $LATEST_REPORT

Evidence rules:
- synthetic data disabled in sync evidence run
- route_prior_strength default 0.0
- grouped primitive selector default 18 primitives

Closed-loop invariant:
- The controller decision affects execution, execution affects loss, loss gradient updates the controller decision path.

Known remaining issues:
$REMAINING
EOF_STATUS

echo "$LATEST_REPORT" > "$REPORT_DIR/latest_report_path.txt"

if [ -f "$REPORT_DIR/final_report.json" ]; then
  python - "$REPORT_DIR/final_report.json" "$STATUS" "$LATEST_REPORT" "$REPORT_DIR" <<'PY'
import csv, glob, json, os, sys
src, status_path, latest_path, report_dir = sys.argv[1:5]

def fnum(x, default=0.0):
    try:
        return float(x)
    except Exception:
        return default

def top_items(d, reverse=True, n=5):
    if not isinstance(d, dict):
        return []
    return sorted(d.items(), key=lambda kv: fnum(kv[1]), reverse=reverse)[:n]

try:
    data = json.load(open(src, 'r', encoding='utf-8'))
except Exception as e:
    data = {'_error': str(e)}
summary = data.get('last_summary') or {}
b = summary.get('boundary') or {}
r = summary.get('route') or {}
p = summary.get('primitive') or {}
m = summary.get('memory') or {}
flags = summary.get('collapse_flags') or []
metrics = summary.get('metrics') or {}
credit_files = sorted(glob.glob(os.path.join(report_dir, 'credit_ablation_epoch_*.json')))
credit = {}
if credit_files:
    try:
        credit = json.load(open(credit_files[-1], 'r', encoding='utf-8'))
    except Exception:
        credit = {}
credit_groups = credit.get('groups') or {}
credit_prims = credit.get('primitives') or {}
metrics_last = {}
metrics_path = os.path.join(report_dir, 'metrics.csv')
if os.path.exists(metrics_path):
    try:
        rows = list(csv.DictReader(open(metrics_path, 'r', encoding='utf-8')))
        metrics_last = rows[-1] if rows else {}
    except Exception:
        metrics_last = {}

pos_groups = top_items(credit_groups, True, 6)
neg_groups = top_items(credit_groups, False, 6)
pos_prims = top_items(credit_prims, True, 8)
neg_prims = top_items(credit_prims, False, 8)

best_acc = fnum(data.get('best_acc', metrics.get('best_acc', 0.0))) * 100.0
val_acc = fnum(metrics.get('val_acc', metrics_last.get('val_acc', 0.0))) * 100.0
train_acc = fnum(metrics.get('train_acc', metrics_last.get('train_acc', 0.0))) * 100.0
version = data.get('version', 'v4.6.1_grouped_core')

lines = []
lines.append('# Latest v4.6.1 Grouped Run Report')
lines.append('')
lines.append(f'- version: `{version}`')
lines.append(f'- report_dir: `{report_dir}`')
lines.append(f'- best_acc: **{best_acc:.2f}%** @ epoch {data.get("best_epoch", metrics.get("best_epoch", "?"))}')
lines.append(f'- last train/val acc: **{train_acc:.2f}% / {val_acc:.2f}%**')
lines.append(f'- collapse_flags: **{",".join(flags) if flags else "NONE"}**')
lines.append('')
lines.append('## Structural summary')
lines.append('')
lines.append(f'- boundary mean/std/peaks: `{fnum(b.get("mean")):.4f}` / `{fnum(b.get("std")):.4f}` / `{b.get("peak_count", 0)}` peaks={b.get("peaks", [])}')
lines.append(f'- route entropy/self/useful/disallowed: `{fnum(r.get("entropy_mean")):.4f}` / `{fnum(r.get("self_route_mass")):.4f}` / `{fnum(r.get("useful_transition_mass")):.4f}` / `{fnum(r.get("disallowed_route_mass")):.4f}`')
lines.append(f'- primitive entropy/top1/negative-sign: `{fnum(p.get("entropy_mean")):.4f}` / `{fnum(p.get("top1_share")):.4f}` / `{fnum(p.get("sign_negative_share")):.4f}`')
lines.append(f'- memory forget/write/read/influence: `{fnum(m.get("forget")):.4f}` / `{fnum(m.get("write_norm_mean")):.4f}` / `{fnum(m.get("read_norm_mean")):.4f}` / `{fnum(m.get("read_influence_proxy")):.4f}`')
lines.append('')
lines.append('## Credit ablation interpretation')
lines.append('')
lines.append('Positive delta CE means removing the component hurts, so the component was useful on this batch. Negative delta CE means removing it helped, so the component is suspicious/junk on this batch.')
lines.append('')
lines.append('### Useful groups')
lines += [f'- `{k}`: {fnum(v):+.6f}' for k, v in pos_groups] or ['- none']
lines.append('')
lines.append('### Suspicious groups')
lines += [f'- `{k}`: {fnum(v):+.6f}' for k, v in neg_groups] or ['- none']
lines.append('')
lines.append('### Useful primitives')
lines += [f'- `{k}`: {fnum(v):+.6f}' for k, v in pos_prims] or ['- none']
lines.append('')
lines.append('### Suspicious primitives')
lines += [f'- `{k}`: {fnum(v):+.6f}' for k, v in neg_prims] or ['- none']
lines.append('')
lines.append('## Artifacts')
lines.append('')
for name in ['REPORT_TO_CHATGPT.txt', 'final_report.json', 'metrics.csv', 'trace_epoch_%03d.json' % int(data.get('best_epoch', 0) or 0), os.path.basename(credit_files[-1]) if credit_files else 'credit_ablation_epoch_XXX.json', 'train.log']:
    lines.append(f'- `{report_dir}/{name}`')
lines.append('')
lines.append('## Current conclusion')
lines.append('')
if credit_groups:
    ch = fnum(credit_groups.get('channel', 0.0))
    mem = fnum(credit_groups.get('memory', 0.0))
    if ch > 0.05 and abs(mem) < 0.02:
        lines.append('The run is learning, but the main useful signal is still the channel group. Memory is not yet proven useful; it should be ablated/penalized if it stays near zero or negative.')
    elif mem > 0.03:
        lines.append('Memory shows positive ablation credit in this run, so it may be contributing. Confirm with a no-memory baseline before claiming it is useful.')
    else:
        lines.append('The loop is not proven as an intelligent program yet. Need frontend-only/full-loop/no-memory/no-route/step-ablation baselines.')
else:
    lines.append('No credit ablation was found. Need credit_ablation_epoch_XXX.json for usefulness analysis.')

text = '\n'.join(lines) + '\n'
open(latest_path, 'w', encoding='utf-8').write(text)
open(os.path.join(report_dir, 'RUN_SUMMARY.md'), 'w', encoding='utf-8').write(text)
with open(status_path, 'a', encoding='utf-8') as f:
    f.write('\nSummary:\n')
    f.write(f'- best_acc: {best_acc:.2f}%\n')
    f.write(f'- latest_report: {latest_path}\n')
    f.write(f'- run_summary: {report_dir}/RUN_SUMMARY.md\n')
    f.write(f'- useful_groups: {pos_groups[:3]}\n')
    f.write(f'- suspicious_groups: {neg_groups[:3]}\n')
PY
fi

printf '[v4.6.1 sync] cleanup pycache before git add\n'
find "$CORE_DIR" -type d -name "__pycache__" -prune -exec rm -rf {} +

printf '[v4.6.1 sync] checkpoint guard before git add\n'
if find "$CORE_DIR" "$REPORT_DIR" -type f \( -name '*.pt' -o -name '*.pth' -o -name '*.ckpt' -o -name '*.safetensors' \) | grep .; then
  echo '[v4.6.1 sync] checkpoint-like file found; refusing to commit' >&2
  exit 4
fi

if [ "$RUN_STATUS" -ne 0 ]; then
  git add "$STATUS" "$LATEST_REPORT" "$REPORT_DIR/latest_report_path.txt" "$REPORT_DIR/RUN_SUMMARY.md" 2>/dev/null || true
  git commit -m "Update v4.6.1 failed-run status" || true
  git push origin "$BRANCH" || true
  printf '[v4.6.1 sync] done: %s status=%s (failed run artifacts not committed)\n' "$REPORT_DIR" "$RUN_STATUS"
  exit "$RUN_STATUS"
fi

git add "$CORE_DIR" "$REPORT_DIR"
if git diff --cached --name-only | grep -E '\.(pt|pth|ckpt|safetensors|pyc)$'; then
  echo '[v4.6.1 sync] forbidden file staged; refusing to commit' >&2
  git reset --cached "$CORE_DIR" "$REPORT_DIR" >/dev/null || true
  exit 5
fi

if git diff --cached --quiet; then
  printf '[v4.6.1 sync] nothing to commit\n'
else
  git commit -m "Add v4.6.1 grouped real-data run artifacts"
  git push origin "$BRANCH"
fi

printf '[v4.6.1 sync] latest report: %s\n' "$LATEST_REPORT"
printf '[v4.6.1 sync] run summary: %s/RUN_SUMMARY.md\n' "$REPORT_DIR"
printf '[v4.6.1 sync] done: %s status=%s\n' "$REPORT_DIR" "$RUN_STATUS"
exit "$RUN_STATUS"
