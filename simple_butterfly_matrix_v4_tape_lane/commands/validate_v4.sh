#!/usr/bin/env bash
set -euo pipefail

cd "$(git rev-parse --show-toplevel 2>/dev/null || pwd)"

echo "[validate] python compile"
python -m py_compile simple_butterfly_matrix_v4_tape_lane/tape_lane_transport.py

echo "[validate] hidden hard phase-role grep"
# Old role names are allowed only in docs/comments comparing old versions.
# This check fails only for active code patterns that would reintroduce fixed phases.
if grep -R "PHASES\|phase_names\|_primitive_prior(phase)\|if self.phase\|class_phase_logits\|phase_slot_matrix\|phase_balance" -n simple_butterfly_matrix_v4_tape_lane/*.py; then
  echo "[validate] ERROR: old hard phase code found in v4 python files"
  exit 1
fi

echo "[validate] ok"
