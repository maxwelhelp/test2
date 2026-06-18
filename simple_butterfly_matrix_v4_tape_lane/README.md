# simple_butterfly_matrix_v4_tape_lane

TapeLaneRouter v4.1.

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

v4.1 removes these fixed phase roles and uses:

```text
raw input/audio
  -> MatrixEvidence
  -> structured lane init X[B, lanes, cells_per_lane, D]
       lane0 detail: resized local evidence + local diff/onset-like signal
       lane1 state: learned attention init over evidence
       lane2 abstract: global mean/std summary cells
       lane3 memory: learned memory seed + weak global summary
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

## No stub code

The repo code uses the real project modules:

```text
MatrixEvidence
ChannelButterfly
BlockButterfly
make_loaders
SpeechCommandsBalanced / synthetic loader from the base project
```

Any local sandbox stubs used to sanity-check syntax are not committed and are not part of this folder.

## Separator

The separator is not only a scalar boundary.

```text
separator_t = boundary_t + route_matrix_t
```

Where:

```text
boundary[t] marks soft program segment pressure
route_matrix[t, from_lane, to_lane] moves information between lanes
```

Boundary biases routing but does not hard reset state.

## Lanes

Default lanes are weak labels only:

```text
detail
state
abstract
memory
```

They are not hard operator assignments. Training can override the soft routes and soft reads.

## Transform primitives

The transform bank is now real compute, not a placeholder:

```text
channel
block
low_rank
ctx_matrix
product_gate
diff
gated_contrast
memory_keep
```

Read/write are dataflow. Transform primitives are compute operations.

## Main files

```text
simple_butterfly_matrix_v4_tape_lane/tape_lane_transport.py
simple_butterfly_matrix_v4_tape_lane/commands/validate_v4.sh
simple_butterfly_matrix_v4_tape_lane/commands/run_smoke.sh
simple_butterfly_matrix_v4_tape_lane/commands/run_speechcommands.sh
simple_butterfly_matrix_v4_tape_lane/commands/sync_run_5ep_push_logs.sh
```

## Validate

From repo root:

```bash
bash simple_butterfly_matrix_v4_tape_lane/commands/validate_v4.sh
```

This checks:

```text
python compile
no old active phase-role code in v4 python files
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

## One-script sync + validate + run + push logs

From repo root:

```bash
bash simple_butterfly_matrix_v4_tape_lane/commands/sync_run_5ep_push_logs.sh
```

This script does:

```text
git pull --ff-only
validate v4 code
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
validate.log  # when using sync_run_5ep_push_logs.sh
train.log     # when using sync_run_5ep_push_logs.sh
```

Important fields in `analysis_epoch_XXX.json`:

```text
lane_init_norm
step_alive
boundary
route_matrix
route_entropy
read_group_mass
primitive_names
primitive_weights
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
primitive_weights differ by lane/step
```

Bad signs:

```text
all classes read one lane
all steps read raw input late
routes stay identity forever
boundaries collapse to all 0/all 1
step_alive is flat for every step
primitive_weights are identical everywhere
```
