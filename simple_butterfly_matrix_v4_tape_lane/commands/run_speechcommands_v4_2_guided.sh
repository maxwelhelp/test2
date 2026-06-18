#!/usr/bin/env bash
set -euo pipefail

cd "$(git rev-parse --show-toplevel 2>/dev/null || pwd)"

python simple_butterfly_matrix_v4_tape_lane/tape_lane_transport_v4_2.py \
  --data-root ../architecture_builder/data/speechcommands \
  --epochs 5 \
  --train-limit 12000 \
  --val-limit 2000 \
  --batch-size 128 \
  --eval-batch-size 256 \
  --device cuda \
  --amp fp16 \
  --dim 96 \
  --evidence-cells 48 \
  --lanes 4 \
  --cells-per-lane 12 \
  --tape-steps 12 \
  --pair-slots 12 \
  --structured-init-strength 0.35 \
  --read-prior-mode structured \
  --read-prior-strength 0.35 \
  --route-prior-mode weak_flow \
  --route-prior-strength 0.20 \
  --boundary-route-strength 0.25 \
  --context-primitive-scale 0.15 \
  --class-lane-prior-mode soft_cover \
  --class-lane-init-strength 0.10 \
  --lane-prior-strength 0.20 \
  --memory-write-target 0.28 \
  --lambda-memory-overwrite 0.004 \
  --workers 4 \
  --pin-memory \
  --log-every 50 \
  --no-save-checkpoints \
  --out-dir simple_butterfly_matrix_v4_tape_lane/runs/speechcommands_5ep_v4_2_guided
