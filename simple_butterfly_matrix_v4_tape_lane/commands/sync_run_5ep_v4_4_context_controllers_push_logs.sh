#!/usr/bin/env bash
set -euo pipefail

cd "$(git rev-parse --show-toplevel 2>/dev/null || pwd)"

MAIN="simple_butterfly_matrix_v4_tape_lane/tape_lane_transport_v4_4_context_controllers.py"
if [ ! -f "$MAIN" ]; then
  echo "[v4.4 sync] missing $MAIN"
  echo "Implement v4.4 context controllers after v4.3 canonical main smoke/5ep is stable."
  echo "Read: simple_butterfly_matrix_v4_tape_lane/V4_4_CONTEXT_CONTROLLERS_IMPLEMENTATION_BRIEF.md"
  exit 2
fi

python -m py_compile "$MAIN"
python "$MAIN" "$@"
