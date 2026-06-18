# simple_butterfly_matrix_v4_tape_lane

TapeLaneRouter version.

This is the next simple version after `simple_butterfly_matrix_v3`.

## What changed

v3 kept the useful idea:

```text
slots -> ClassMatrix[C,D] -> ClassPairMatrix[P,D] -> class update -> logits
```

But v3 still used fixed phase roles:

```text
input / extract / compare / suppress / aggregate
class_phase_logits
phase_slot_matrix
phase_balance
```

v4 removes these fixed phase roles.

New structure:

```text
raw input/audio
  -> MatrixEvidence
  -> X[B, lanes, cells_per_lane, D]
  -> tape steps T0..Tn
       read -> transform -> soft lane route -> residual write
       step_alive[t]
       boundary[t]
       route_matrix[t, from_lane, to_lane]
  -> all tape/lane/cell slots
  -> ClassMatrixLaneHead
  -> logits
```

## Rules kept

```text
no hard router
no top-k router
no fixed extract/compare/suppress/aggregate phases
no dynamic tensor shape growth in MVP
class matrices and class-pair repair are kept
```

## Lanes

Default lanes are weak labels only:

```text
detail
state
abstract
memory
```

They are not hard operator assignments. Training can override the soft routes and soft reads.

## Main files

```text
simple_butterfly_matrix_v4_tape_lane/tape_lane_transport.py
simple_butterfly_matrix_v4_tape_lane/commands/run_smoke.sh
simple_butterfly_matrix_v4_tape_lane/commands/run_speechcommands.sh
simple_butterfly_matrix_v4_tape_lane/commands/sync_run_5ep_push_logs.sh
```

## Fast smoke

From repo root:

```bash
bash simple_butterfly_matrix_v4_tape_lane/commands/run_smoke.sh
```

## 5 epoch SpeechCommands test

From repo root:

```bash
bash simple_butterfly_matrix_v4_tape_lane/commands/run_speechcommands.sh
```

## One-script sync + run + push logs

From repo root:

```bash
bash simple_butterfly_matrix_v4_tape_lane/commands/sync_run_5ep_push_logs.sh
```

This script does:

```text
git pull --ff-only
run 5 epochs
write metrics/report/logs
git add only text/csv/json logs
git commit
git push
```

It does **not** push `.pt` checkpoints.

Override defaults:

```bash
DATA_ROOT=../architecture_builder/data/speechcommands \
TRAIN_LIMIT=12000 \
VAL_LIMIT=2000 \
EPOCHS=5 \
AMP=fp16 \
bash simple_butterfly_matrix_v4_tape_lane/commands/sync_run_5ep_push_logs.sh
```

For Tesla P40 use `AMP=fp16`, not bf16.

## Outputs

Each run writes:

```text
metrics.csv
analysis_epoch_XXX.json
final_report.json
REPORT_TO_CHATGPT.txt
train.log  # when using sync_run_5ep_push_logs.sh
```

Important fields in `analysis_epoch_XXX.json`:

```text
step_alive
boundary
route_matrix
route_entropy
read_group_mass
write_gate_by_step_lane
update_norm_by_step_lane
late_input_read_mass
class_lane_mass
lane_mass_mean
class_top_reads
pair_update_norm
```

## What counts as good

Good early signs:

```text
route_matrix is not only identity
boundary is not all 0 or all 1
step_alive separates useful early steps from dormant late steps
late_input_read_mass is not huge
class_lane_mass uses more than one lane
class_top_reads are not only final slots
```

Bad signs:

```text
all classes read one lane
all steps read raw input late
routes stay identity forever
boundaries collapse to all 0/all 1
step_alive is flat for every step
```
