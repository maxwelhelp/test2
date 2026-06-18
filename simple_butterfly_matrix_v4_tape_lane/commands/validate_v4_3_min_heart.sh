#!/usr/bin/env bash
set -euo pipefail

cd "$(git rev-parse --show-toplevel 2>/dev/null || pwd)"

FILE="simple_butterfly_matrix_v4_tape_lane/tape_lane_transport_v4_3_min_heart.py"
SYNC="simple_butterfly_matrix_v4_tape_lane/commands/sync_run_5ep_v4_3_min_heart_push_logs.sh"
PLAN="simple_butterfly_matrix_v4_tape_lane/V4_3_MIN_HEART_PLAN.md"

echo "[validate v4.3] py_compile"
python -m py_compile "$FILE"

echo "[validate v4.3] --help smoke"
python "$FILE" --help >/tmp/v4_3_min_heart_help.txt

echo "[validate v4.3] required implementation markers"
grep -q "route_offdiag_outside_boundary_cost" "$FILE"
grep -q "offdiag_mass_norm" "$FILE"
grep -q "boundary_budget_cost" "$FILE"
grep -q "detail_attention_mass" "$FILE"
grep -q "detail_head_shortcut_cost" "$FILE"
grep -q "candidate_suggestions_epoch" "$FILE"
grep -q "trace_feedback_epoch" "$FILE"
grep -q "sequence_nonflat_score" "$FILE"
grep -q "no backward" "$PLAN"
grep -q "heart_window = epoch" "$PLAN"

echo "[validate v4.3] no active old phase roles"
if grep -R "PHASES\|class_phase_logits\|phase_slot_matrix\|phase_balance" -n "$FILE"; then
  echo "[validate v4.3] ERROR: old hard phase code found"
  exit 1
fi

echo "[validate v4.3] sync script presence/correct target"
test -f "$SYNC"
grep -q "tape_lane_transport_v4_3_min_heart.py" "$SYNC"
grep -q -- "--no-save-checkpoints" "$SYNC"

echo "[validate v4.3] ok"
