#!/usr/bin/env bash
set -euo pipefail

cd "$(git rev-parse --show-toplevel 2>/dev/null || pwd)"

FILE="simple_butterfly_matrix_v4_tape_lane/tape_lane_transport_v4_3_min_heart.py"
WRAP="simple_butterfly_matrix_v4_tape_lane/commands/run_v4_3_min_heart_audit_fixed.sh"
CTX="simple_butterfly_matrix_v4_tape_lane/commands/run_v4_3_context_controller.sh"
CTX_SMOKE="simple_butterfly_matrix_v4_tape_lane/commands/smoke_v4_3_context_controller.sh"
SYNC="simple_butterfly_matrix_v4_tape_lane/commands/sync_run_5ep_v4_3_min_heart_push_logs.sh"
GRAD="simple_butterfly_matrix_v4_tape_lane/commands/grad_sanity_v4_3_min_heart.sh"
PLAN="simple_butterfly_matrix_v4_tape_lane/V4_3_MIN_HEART_PLAN.md"

echo "[validate v4.3] py_compile"
python -m py_compile "$FILE"

echo "[validate v4.3] --help smoke"
python "$FILE" --help >/tmp/v4_3_min_heart_help.txt
bash "$CTX" --help >/tmp/v4_3_context_controller_help.txt || true

echo "[validate v4.3] required implementation markers in base file"
grep -q "route_offdiag_outside_boundary_cost" "$FILE"
grep -q "offdiag_norm" "$FILE"
grep -q "boundary_budget_cost" "$FILE"
grep -q "detail_attention_mass" "$FILE"
grep -q "detail_head_shortcut_cost" "$FILE"
grep -q "candidate_suggestions_epoch" "$FILE"
grep -q "trace_feedback_epoch" "$FILE"
grep -q "sequence_nonflat_score" "$FILE"
grep -q "no backward" "$PLAN"
grep -q "heart_window = epoch" "$PLAN"

echo "[validate v4.3] audit-fixed wrapper markers"
test -f "$WRAP"
grep -q "route_delta_by_step" "$WRAP"
grep -q "read_delta_by_step" "$WRAP"
grep -q "sequence_nonflat_score" "$WRAP"
grep -q "boundary_usefulness" "$WRAP"
grep -q "self_route_mass" "$WRAP"
grep -q "useful_transition_mass" "$WRAP"
grep -q "ROUTE_IDENTITY_COLLAPSE" "$WRAP"
grep -q "memory_write_warmup_epochs" "$WRAP"
grep -q "max_candidates_effective" "$WRAP"
grep -q "deploy.*False" "$WRAP"

echo "[validate v4.3] context-controller wrapper markers"
test -f "$CTX"
grep -q "ContextTapeLaneRouterBackbone" "$CTX"
grep -q "read_context_net" "$CTX"
grep -q "route_context_net" "$CTX"
grep -q "boundary_context_net" "$CTX"
grep -q "alive_context_net" "$CTX"
grep -q "write_context_net" "$CTX"
grep -q "context_controller_scale" "$CTX"
grep -q "base.v42.TapeLaneRouterBackbone = ContextTapeLaneRouterBackbone" "$CTX"
grep -q "context_controller_active" "$CTX"
grep -q "deploy.*False" "$CTX"

echo "[validate v4.3] context-controller smoke markers"
test -f "$CTX_SMOKE"
grep -q "CONTEXT_CONTROLLER_SCALE" "$CTX_SMOKE"
grep -q "--synthetic" "$CTX_SMOKE"
grep -q "context_controller" "$CTX_SMOKE"

echo "[validate v4.3] speed markers"
grep -q "make_loaders_fast" "$CTX"
grep -q "evaluate_fast" "$CTX"
grep -q "persistent_workers" "$CTX"
grep -q "prefetch_factor" "$CTX"
grep -q "fast eval computes loss/accuracy" "$CTX"
grep -q "speed cfg" "$SYNC"
grep -q "MAX_TRAIN_BATCHES" "$SYNC"
grep -q "MAX_VAL_BATCHES" "$SYNC"

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
grep -q "run_v4_3_context_controller.sh" "$SYNC"
grep -q "grad_sanity_v4_3_min_heart.sh" "$SYNC"
grep -q -- "--no-save-checkpoints" "$SYNC"
grep -q "RUN_STATUS" "$SYNC"

echo "[validate v4.3] ok"
