#!/usr/bin/env bash
set -euo pipefail

cd "$(git rev-parse --show-toplevel 2>/dev/null || pwd)"

python simple_butterfly_matrix_v4_tape_lane/tape_lane_transport_v4_2.py \
  --data-root ../architecture_builder/data/speechcommands \
  --epochs 1 \
  --train-limit 512 \
  --val-limit 256 \
  --batch-size 64 \
  --eval-batch-size 128 \
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
  --pin-memory \
  --log-every 5 \
  --no-save-checkpoints \
  --out-dir simple_butterfly_matrix_v4_tape_lane/runs/smoke_v4_2
