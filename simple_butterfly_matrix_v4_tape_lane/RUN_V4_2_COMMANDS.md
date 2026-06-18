# TapeLaneRouter v4.2 run commands

v4.2 file:

```text
simple_butterfly_matrix_v4_tape_lane/tape_lane_transport_v4_2.py
```

Validate:

```bash
cd /home/maxwelhelp/test/sience/experiments/math_search/WORKING_BEST/test2
bash simple_butterfly_matrix_v4_tape_lane/commands/validate_v4_2.sh
```

Smoke:

```bash
cd /home/maxwelhelp/test/sience/experiments/math_search/WORKING_BEST/test2
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
```

5 epoch guided weak-prior run:

```bash
cd /home/maxwelhelp/test/sience/experiments/math_search/WORKING_BEST/test2
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
```

Neutral-prior ablation:

```bash
cd /home/maxwelhelp/test/sience/experiments/math_search/WORKING_BEST/test2
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
  --structured-init-strength 0.0 \
  --read-prior-mode neutral \
  --read-prior-strength 0.0 \
  --route-prior-mode neutral \
  --route-prior-strength 0.0 \
  --boundary-route-strength 0.0 \
  --context-primitive-scale 0.15 \
  --class-lane-prior-mode neutral \
  --class-lane-init-strength 0.0 \
  --lane-prior-strength 0.0 \
  --memory-write-target 0.28 \
  --lambda-memory-overwrite 0.004 \
  --workers 4 \
  --pin-memory \
  --log-every 50 \
  --no-save-checkpoints \
  --out-dir simple_butterfly_matrix_v4_tape_lane/runs/speechcommands_5ep_v4_2_neutral
```

What changed in v4.2:

```text
structured input basis is weak and controlled by --structured-init-strength
head lane prior is weak: default --lane-prior-strength 0.20, init strength 0.10
read/route priors have neutral/structured modes
memory overwrite loss penalizes only memory_gate above target
transform primitive weights are context-aware through context_to_primitive
all choices remain soft/differentiable
```
