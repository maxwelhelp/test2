#!/usr/bin/env bash
set -euo pipefail

cd "$(git rev-parse --show-toplevel 2>/dev/null || pwd)"

FILE="simple_butterfly_matrix_v4_tape_lane/tape_lane_transport_v4_3_min_heart.py"
CANON="simple_butterfly_matrix_v4_tape_lane/commands/canonicalize_v4_3_main.py"
WRAP="simple_butterfly_matrix_v4_tape_lane/commands/run_v4_3_min_heart_audit_fixed.sh"
CTX="simple_butterfly_matrix_v4_tape_lane/commands/run_v4_3_context_controller.sh"
SYNC="simple_butterfly_matrix_v4_tape_lane/commands/sync_run_5ep_v4_3_min_heart_push_logs.sh"
GRAD="simple_butterfly_matrix_v4_tape_lane/commands/grad_sanity_v4_3_min_heart.sh"
PLAN="simple_butterfly_matrix_v4_tape_lane/V4_3_MIN_HEART_PLAN.md"

echo "[validate v4.3] canonicalize main"
test -f "$CANON"
python "$CANON"

echo "[validate v4.3] py_compile"
python -m py_compile "$FILE"
python -m py_compile "$CANON"

echo "[validate v4.3] --help smoke"
python "$FILE" --help >/tmp/v4_3_min_heart_help.txt

echo "[validate v4.3] canonical main markers"
grep -q "v4.3_min_heart_canonical" "$FILE"
grep -q "def _route_extra_terms_from_routes" "$FILE"
grep -q "sequence_route_delta_mean" "$FILE"
grep -q "sequence_nonflat_score.*route_delta_mean.*prim_delta_mean.*upd_delta_mean.*read_delta_mean" "$FILE"
grep -q "read_delta_by_step" "$FILE"
grep -q "boundary_usefulness" "$FILE"
grep -q "self_route_mass" "$FILE"
grep -q "useful_transition_mass" "$FILE"
grep -q "ROUTE_IDENTITY_COLLAPSE" "$FILE"
grep -q "memory_write_warmup_epochs" "$FILE"
grep -q "memory_consumer_proxy" "$FILE"
grep -q "update_collapse_proxy" "$FILE"
grep -q "not_implemented" "$FILE"
grep -q "collapse_flags" "$FILE"
grep -q "duplicate_rate" "$FILE"
grep -q "attempted_candidates" "$FILE"
grep -q "route_offdiag_outside_boundary_cost" "$FILE"
grep -q "offdiag_norm" "$FILE"
grep -q "detail_head_shortcut_cost" "$FILE"
grep -q "candidate_suggestions_epoch" "$FILE"
grep -q "trace_feedback_epoch" "$FILE"
grep -q "no backward" "$PLAN"
grep -q "heart_window = epoch" "$PLAN"

echo "[validate v4.3] auxiliary wrapper files still present but not standard entrypoint"
test -f "$WRAP"
test -f "$CTX"
grep -q "deploy.*False" "$WRAP"
grep -q "deploy.*False" "$CTX"

echo "[validate v4.3] gradient sanity script markers"
test -f "$GRAD"
grep -q "backbone.route_logits" "$GRAD"
grep -q "backbone.boundary_logit" "$GRAD"
grep -q "backbone.read_group_logits" "$GRAD"
grep -q "backbone.write_gate_logit" "$GRAD"
grep -q "head.class_lane_logits" "$GRAD"

echo "[validate v4.3] no active old phase roles"
if grep -R "PHASES\|class_phase_logits\|phase_slot_matrix\|phase_balance" -n "$FILE" "$WRAP" "$CTX"; then
  echo "[validate v4.3] ERROR: old hard phase code found"
  exit 1
fi

echo "[validate v4.3] sync script presence/correct target"
test -f "$SYNC"
grep -q "canonicalize_v4_3_main.py" "$SYNC"
grep -q "tape_lane_transport_v4_3_min_heart.py" "$SYNC"
if grep -q "run_v4_3_context_controller.sh" "$SYNC"; then
  echo "[validate v4.3] ERROR: standard sync must not run context wrapper"
  exit 1
fi
grep -q "grad_sanity_v4_3_min_heart.sh" "$SYNC"
grep -q -- "--no-save-checkpoints" "$SYNC"
grep -q "RUN_STATUS" "$SYNC"

echo "[validate v4.3] ok"
