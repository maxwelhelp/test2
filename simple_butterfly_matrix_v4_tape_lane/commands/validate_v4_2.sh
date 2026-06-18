#!/usr/bin/env bash
set -euo pipefail

cd "$(git rev-parse --show-toplevel 2>/dev/null || pwd)"

echo "[validate v4.2] python compile"
python -m py_compile simple_butterfly_matrix_v4_tape_lane/tape_lane_transport_v4_2.py

echo "[validate v4.2] required controls"
grep -q "structured_init_strength" simple_butterfly_matrix_v4_tape_lane/tape_lane_transport_v4_2.py
grep -q "read_prior_mode" simple_butterfly_matrix_v4_tape_lane/tape_lane_transport_v4_2.py
grep -q "route_prior_mode" simple_butterfly_matrix_v4_tape_lane/tape_lane_transport_v4_2.py
grep -q "context_primitive_scale" simple_butterfly_matrix_v4_tape_lane/tape_lane_transport_v4_2.py
grep -q "memory_write_target" simple_butterfly_matrix_v4_tape_lane/tape_lane_transport_v4_2.py

echo "[validate v4.2] no active old phase-code names"
if grep -R "class_phase_logits" -n simple_butterfly_matrix_v4_tape_lane/tape_lane_transport_v4_2.py; then exit 1; fi
if grep -R "phase_slot_matrix" -n simple_butterfly_matrix_v4_tape_lane/tape_lane_transport_v4_2.py; then exit 1; fi
if grep -R "phase_balance" -n simple_butterfly_matrix_v4_tape_lane/tape_lane_transport_v4_2.py; then exit 1; fi

echo "[validate v4.2] ok"
