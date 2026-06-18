#!/usr/bin/env bash
set -euo pipefail
cd "$(git rev-parse --show-toplevel 2>/dev/null || pwd)"
OUT_DIR="${OUT_DIR:-simple_butterfly_matrix_v4_tape_lane/agent_reports/context_controller_smoke}"
rm -rf "$OUT_DIR"
CONTEXT_CONTROLLER_SCALE="${CONTEXT_CONTROLLER_SCALE:-0.15}" \
bash simple_butterfly_matrix_v4_tape_lane/commands/run_v4_3_context_controller.sh \
  --synthetic \
  --synthetic-length 512 \
  --epochs 1 \
  --train-limit 32 \
  --val-limit 16 \
  --batch-size 4 \
  --eval-batch-size 4 \
  --workers 0 \
  --device cpu \
  --amp fp32 \
  --dim 32 \
  --evidence-cells 12 \
  --lanes 4 \
  --cells-per-lane 4 \
  --tape-steps 3 \
  --pair-slots 4 \
  --max-train-batches 1 \
  --max-val-batches 1 \
  --log-every 1 \
  --no-save-checkpoints \
  --out-dir "$OUT_DIR"
python - "$OUT_DIR/final_report.json" <<'PY'
import json, sys
p = sys.argv[1]
data = json.load(open(p, 'r', encoding='utf-8'))
tr = data.get('last_trace_feedback') or {}
ctx = tr.get('context_controller') or {}
assert ctx.get('active') is True, ctx
assert 'last_candidate_suggestions' in data, data.keys()
print('[context smoke] ok', json.dumps({'best_acc': data.get('best_acc'), 'context_controller': ctx}, ensure_ascii=False))
PY
