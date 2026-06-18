#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

REPORT_ROOT="simple_butterfly_matrix_v4_tape_lane/agent_reports"
LATEST_DIR="${REPORT_DIR:-}"

if [[ -z "$LATEST_DIR" ]]; then
  LATEST_DIR="$(ls -td "$REPORT_ROOT"/v4_4_context_controllers_* 2>/dev/null | head -1 || true)"
fi

if [[ -z "$LATEST_DIR" || ! -d "$LATEST_DIR" ]]; then
  echo "[projection_scout] ERROR: no report directory found. Set REPORT_DIR=... or run v4.4 first." >&2
  exit 2
fi

echo "[projection_scout] repo=$REPO_ROOT"
echo "[projection_scout] report=$LATEST_DIR"

python -m py_compile simple_butterfly_matrix_v4_tape_lane/projection_scout_v1.py
python simple_butterfly_matrix_v4_tape_lane/projection_scout_v1.py \
  --repo-root . \
  --report-dir "$LATEST_DIR" \
  --top-k "${TOP_K:-2}"

echo "[projection_scout] generated files:"
ls -lh "$LATEST_DIR"/projection_scout_epoch_*.json "$LATEST_DIR"/scout_experience_memory.jsonl 2>/dev/null || true
