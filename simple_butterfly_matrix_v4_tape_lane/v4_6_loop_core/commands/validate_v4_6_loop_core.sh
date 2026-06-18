#!/usr/bin/env bash
set -euo pipefail

cd "$(git rev-parse --show-toplevel 2>/dev/null || pwd)"
CORE_DIR="simple_butterfly_matrix_v4_tape_lane/v4_6_loop_core"
UNIT_MAIN="$CORE_DIR/tape_lane_transport_v4_6_loop_core.py"
REAL_MAIN="$CORE_DIR/tape_lane_transport_v4_6_1_grouped_core.py"

printf '[v4.6.1 validate] py_compile\n'
python -m py_compile \
  "$UNIT_MAIN" \
  "$REAL_MAIN" \
  "$CORE_DIR/modules/__init__.py" \
  "$CORE_DIR/modules/joint_controller.py" \
  "$CORE_DIR/modules/primitive_selector.py" \
  "$CORE_DIR/modules/matrix_memory.py" \
  "$CORE_DIR/modules/losses.py" \
  "$CORE_DIR/modules/step_analyzer.py"

printf '[v4.6.1 validate] --help smoke\n'
python "$REAL_MAIN" --help >/tmp/v4_6_1_grouped_help.txt
grep -q -- '--gumbel-tau-start' /tmp/v4_6_1_grouped_help.txt
grep -q -- '--primitive-rank' /tmp/v4_6_1_grouped_help.txt
grep -q -- '--route-prior-strength' /tmp/v4_6_1_grouped_help.txt

printf '[v4.6.1 validate] import smoke\n'
python - <<'PY'
import torch
from simple_butterfly_matrix_v4_tape_lane.v4_6_loop_core.modules import JointController, ParallelPrimitiveSelector, MatrixMemory
from simple_butterfly_matrix_v4_tape_lane.v4_6_loop_core.tape_lane_transport_v4_6_1_grouped_core import V461GroupedLoopCoreClassifier, parser
args = parser().parse_args(['--device', 'cpu', '--amp', 'off', '--dim', '16', '--steps', '2', '--primitive-rank', '4', '--num-primitives', '18', '--batch-size', '2', '--eval-batch-size', '2'])
model = V461GroupedLoopCoreClassifier(3, args)
wav = torch.randn(2, 1024)
logits, trace = model(wav, tau=1.0)
assert logits.shape == (2, 3)
assert trace['group_weights'].shape[-1] == model.selector.num_groups
print(type(model).__name__, JointController.__name__, ParallelPrimitiveSelector.__name__, MatrixMemory.__name__, 'groups=', model.selector.group_names)
PY

printf '[v4.6.1 validate] banned runtime mechanisms scan\n'
if grep -RInE 'source string replacement|text patching marker|monkey.?patch|PPO|A3C|CouncilCalibrator|external JSON feedback' "$CORE_DIR" --include='*.py'; then
  echo '[v4.6.1 validate] forbidden runtime mechanism marker found' >&2
  exit 2
fi

printf '[v4.6.1 validate] shell syntax\n'
bash -n "$CORE_DIR/commands/validate_v4_6_loop_core.sh"
bash -n "$CORE_DIR/commands/sync_run_5ep_v4_6_loop_core_push_logs.sh"

printf '[v4.6.1 validate] cleanup pycache\n'
find "$CORE_DIR" -type d -name "__pycache__" -prune -exec rm -rf {} +

printf '[v4.6.1 validate] checkpoint guard\n'
if git status --short "$CORE_DIR" 2>/dev/null | grep -E '\.(pt|pth|ckpt|safetensors|pyc)$'; then
  echo '[v4.6.1 validate] forbidden checkpoint/cache-like file is staged or modified under core dir' >&2
  exit 3
fi

if [ "${RUN_SYNTHETIC_SMOKE:-0}" != "0" ]; then
  printf '[v4.6.1 validate] optional unit-only synthetic smoke on old unit main\n'
  python "$UNIT_MAIN" \
    --synthetic-data \
    --device "${DEVICE:-cpu}" \
    --amp "${AMP:-off}" \
    --epochs 1 \
    --train-limit 8 \
    --val-limit 8 \
    --batch-size 4 \
    --eval-batch-size 4 \
    --workers 0 \
    --dim 16 \
    --steps 2 \
    --primitive-rank 4 \
    --max-train-batches 1 \
    --max-val-batches 1 \
    --out-dir /tmp/v4_6_loop_core_synthetic_smoke
fi

printf '[v4.6.1 validate] OK\n'
