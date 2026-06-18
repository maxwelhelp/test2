#!/usr/bin/env bash
set -euo pipefail

cd "$(git rev-parse --show-toplevel 2>/dev/null || pwd)"
BRANCH="${BRANCH:-codex-full-agent-plan}"
CORE_DIR="simple_butterfly_matrix_v4_tape_lane/v4_6_loop_core"
MAIN_FILE="$CORE_DIR/tape_lane_transport_v4_6_3_edge_program.py"
TS="$(date +%Y%m%d_%H%M%S)"
REPORT_DIR="${OUT_DIR:-simple_butterfly_matrix_v4_tape_lane/agent_reports/v4_6_3_edge_program_${TS}}"
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
LR="${LR:-5e-4}"
LOG_EVERY="${LOG_EVERY:-100}"
MAX_TRAIN_BATCHES="${MAX_TRAIN_BATCHES:-0}"
MAX_VAL_BATCHES="${MAX_VAL_BATCHES:-0}"
ENABLE_EDGE_CREDIT_PENALTY="${ENABLE_EDGE_CREDIT_PENALTY:-1}"
CREDIT_ABLATION_BATCH="${CREDIT_ABLATION_BATCH:-16}"
CREDIT_ABLATE_EDGES="${CREDIT_ABLATE_EDGES:-1}"
CREDIT_ABLATE_OUTPUTS="${CREDIT_ABLATE_OUTPUTS:-1}"

printf '[v4.6.3 sync] branch=%s report_dir=%s edge_penalty=%s\n' "$BRANCH" "$REPORT_DIR" "$ENABLE_EDGE_CREDIT_PENALTY" | tee "$REPORT_DIR/run_header.txt"

git checkout "$BRANCH"
git pull --rebase --autostash origin "$BRANCH"

printf '[v4.6.3 sync] validate base + py_compile edge entrypoint\n'
bash "$CORE_DIR/commands/validate_v4_6_loop_core.sh" | tee "$REPORT_DIR/validate.log"
python -m py_compile "$MAIN_FILE"

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
  --lambda-edge-gate-cost "${LAMBDA_EDGE_GATE_COST:-0.001}"
  --lambda-output-gate-cost "${LAMBDA_OUTPUT_GATE_COST:-0.0005}"
  --lambda-dead-step "${LAMBDA_DEAD_STEP:-0.002}"
  --dead-step-min-mass "${DEAD_STEP_MIN_MASS:-0.05}"
  --boundary-min-peaks "${BOUNDARY_MIN_PEAKS:-1.0}"
  --boundary-max-peaks "${BOUNDARY_MAX_PEAKS:-4.0}"
  --boundary-peak-threshold "${BOUNDARY_PEAK_THRESHOLD:-0.35}"
  --credit-ablation-batch "$CREDIT_ABLATION_BATCH"
  --credit-bad-group-scale "${CREDIT_BAD_GROUP_SCALE:-0.012}"
  --credit-bad-primitive-scale "${CREDIT_BAD_PRIMITIVE_SCALE:-0.008}"
  --credit-bad-step-scale "${CREDIT_BAD_STEP_SCALE:-0.008}"
  --credit-bad-edge-scale "${CREDIT_BAD_EDGE_SCALE:-0.008}"
  --credit-bad-output-scale "${CREDIT_BAD_OUTPUT_SCALE:-0.006}"
  --max-train-batches "$MAX_TRAIN_BATCHES"
  --max-val-batches "$MAX_VAL_BATCHES"
  --log-every "$LOG_EVERY"
  --no-save-checkpoints
  --out-dir "$REPORT_DIR"
)
if [ "${PIN_MEMORY:-1}" != "0" ]; then RUN_ARGS+=(--pin-memory); fi
if [ "$ENABLE_EDGE_CREDIT_PENALTY" != "0" ]; then RUN_ARGS+=(--enable-edge-credit-penalty); fi
if [ "$CREDIT_ABLATE_EDGES" = "0" ]; then RUN_ARGS+=(--no-credit-ablate-edges); fi
if [ "$CREDIT_ABLATE_OUTPUTS" = "0" ]; then RUN_ARGS+=(--no-credit-ablate-outputs); fi

cat > "$REPORT_DIR/speed_config.txt" <<EOF_CFG
batch_size=$BATCH_SIZE
eval_batch_size=$EVAL_BATCH_SIZE
workers=$WORKERS
device=$DEVICE
amp=$AMP
dim=$DIM
steps=$STEPS
lanes=$LANES
edges=$((LANES*LANES))
num_primitives=$NUM_PRIMITIVES
primitive_rank=$PRIMITIVE_RANK
edge_credit_penalty=$ENABLE_EDGE_CREDIT_PENALTY
credit_ablation_batch=$CREDIT_ABLATION_BATCH
synthetic_data=disabled
synthetic_fallback=disabled
EOF_CFG

printf '[v4.6.3 sync] run main\n'
set +e
python "$MAIN_FILE" "${RUN_ARGS[@]}" 2>&1 | tee "$REPORT_DIR/train.log"
RUN_STATUS=${PIPESTATUS[0]}
set -e

echo "$RUN_STATUS" > "$REPORT_DIR/run_status.txt"
STATUS="$CORE_DIR/AGENT_STATUS.md"
cat > "$STATUS" <<EOF_STATUS
# Agent Status

Current stage: v4.6.3 edge-conditioned program
Entrypoint: $MAIN_FILE
Run status: $RUN_STATUS
Report dir: $REPORT_DIR
Latest compact report: $LATEST_REPORT
Edge credit penalty: $ENABLE_EDGE_CREDIT_PENALTY

Closed-loop invariant:
- edge decisions affect execution
- execution affects CE loss
- delayed ablation credit can penalize bad edge/step/output mass next epoch
EOF_STATUS

if [ -f "$REPORT_DIR/final_report.json" ]; then
  python - "$REPORT_DIR/final_report.json" "$LATEST_REPORT" "$REPORT_DIR" "$ENABLE_EDGE_CREDIT_PENALTY" <<'PY'
import csv, glob, json, os, sys
src, latest, report_dir, penalty = sys.argv[1:5]
def fnum(x, d=0.0):
    try: return float(x)
    except Exception: return d
def top(d, rev=True, n=10):
    return sorted((d or {}).items(), key=lambda kv: fnum(kv[1]), reverse=rev)[:n]
data=json.load(open(src, encoding='utf-8'))
s=data.get('last_summary') or {}; m=s.get('metrics') or {}; b=s.get('boundary') or {}; r=s.get('route') or {}; p=s.get('primitive') or {}; mem=s.get('memory') or {}; edge=s.get('edge') or {}; out=s.get('output') or {}; flags=s.get('collapse_flags') or []
credit_files=sorted(glob.glob(os.path.join(report_dir,'credit_ablation_epoch_*.json')))
credit=json.load(open(credit_files[-1], encoding='utf-8')) if credit_files else {}
rows=list(csv.DictReader(open(os.path.join(report_dir,'metrics.csv'), encoding='utf-8'))) if os.path.exists(os.path.join(report_dir,'metrics.csv')) else []
last=rows[-1] if rows else {}
lines=['# Latest v4.6.3 Edge Program Report','']
lines += [f'- report_dir: `{report_dir}`', f'- version: `{data.get("version")}`', f'- edge_credit_penalty: `{penalty}`', f'- best_acc: **{100*fnum(data.get("best_acc")):.2f}%** @ epoch {data.get("best_epoch")}', f'- last train/val acc: **{100*fnum(m.get("train_acc", last.get("train_acc", 0))):.2f}% / {100*fnum(m.get("val_acc", last.get("val_acc", 0))):.2f}%**', f'- collapse_flags: **{",".join(flags) if flags else "NONE"}**']
lines += ['', '## Structural summary', f'- step boundary mean/std/peaks: `{fnum(b.get("mean")):.4f}` / `{fnum(b.get("std")):.4f}` / `{b.get("peak_count",0)}`', f'- route entropy/self/useful/disallowed: `{fnum(r.get("entropy_mean")):.4f}` / `{fnum(r.get("self_route_mass")):.4f}` / `{fnum(r.get("useful_transition_mass")):.4f}` / `{fnum(r.get("disallowed_route_mass")):.4f}`', f'- primitive entropy/top1/negative-sign: `{fnum(p.get("entropy_mean")):.4f}` / `{fnum(p.get("top1_share")):.4f}` / `{fnum(p.get("sign_negative_share")):.4f}`', f'- edge gate/boundary/write mean: `{fnum(edge.get("edge_gate_mean")):.4f}` / `{fnum(edge.get("edge_boundary_mean")):.4f}` / `{fnum(edge.get("edge_write_mean")):.4f}`', f'- output_gate_mean: `{fnum(out.get("output_gate_mean")):.4f}`', f'- memory forget/write/read/influence: `{fnum(mem.get("forget")):.4f}` / `{fnum(mem.get("write_norm_mean")):.4f}` / `{fnum(mem.get("read_norm_mean")):.4f}` / `{fnum(mem.get("read_influence_proxy")):.4f}`']
for title,key in [('Useful groups','groups'),('Suspicious groups','groups'),('Useful primitives','primitives'),('Suspicious primitives','primitives'),('Useful steps','steps'),('Suspicious steps','steps'),('Useful edges','edges'),('Suspicious edges','edges'),('Useful outputs','outputs'),('Suspicious outputs','outputs')]:
    rev=title.startswith('Useful')
    lines += ['', f'## {title}']
    items=top(credit.get(key) or {}, rev, 10)
    lines += [f'- `{k}`: {fnum(v):+.6f}' for k,v in items] or ['- none']
if credit.get('memory'):
    lines += ['', '## Memory ablation']
    lines += [f'- `{k}`: {fnum(v):+.6f}' for k,v in (credit.get('memory') or {}).items()]
lines += ['', '## Penalty/cost metrics']
for k in ['edge_gate_cost','output_gate_cost','dead_step_loss','edge_credit_penalty','bad_edge_loss','bad_output_loss']:
    lines.append(f'- `{k}`: {fnum(last.get(k)):.6f}')
lines += ['', '## Current conclusion', 'This is the first edge-conditioned version: primitive/operator choice is now made per source-target edge, with separate edge/write/boundary/output gates. Compare it against v4.6.2 and check whether useful edge/output credit appears without killing accuracy.']
text='\n'.join(lines)+'\n'
open(latest,'w',encoding='utf-8').write(text)
open(os.path.join(report_dir,'RUN_SUMMARY.md'),'w',encoding='utf-8').write(text)
PY
fi

find "$CORE_DIR" -type d -name "__pycache__" -prune -exec rm -rf {} +
if find "$CORE_DIR" "$REPORT_DIR" -type f \( -name '*.pt' -o -name '*.pth' -o -name '*.ckpt' -o -name '*.safetensors' -o -name '*.pyc' \) | grep .; then
  echo '[v4.6.3 sync] forbidden checkpoint/cache file found; refusing to commit' >&2
  exit 4
fi

if [ "$RUN_STATUS" -ne 0 ]; then
  git add "$STATUS" "$LATEST_REPORT" 2>/dev/null || true
  git commit -m "Update v4.6.3 failed-run status" || true
  git push origin "$BRANCH" || true
  exit "$RUN_STATUS"
fi

git add "$CORE_DIR" "$REPORT_DIR"
if git diff --cached --name-only | grep -E '\.(pt|pth|ckpt|safetensors|pyc)$'; then
  echo '[v4.6.3 sync] forbidden file staged; refusing to commit' >&2
  git reset --cached "$CORE_DIR" "$REPORT_DIR" >/dev/null || true
  exit 5
fi
if git diff --cached --quiet; then
  printf '[v4.6.3 sync] nothing to commit\n'
else
  git commit -m "Add v4.6.3 edge program run artifacts"
  git push origin "$BRANCH"
fi
printf '[v4.6.3 sync] latest report: %s\n' "$LATEST_REPORT"
printf '[v4.6.3 sync] done: %s status=%s\n' "$REPORT_DIR" "$RUN_STATUS"
exit "$RUN_STATUS"
