# v4.3_min_heart Plan

Single authoritative plan for the conservative bridge from v4.2 to ProgramHeart. Do not add more Heart design docs until this version is implemented and tested.

## Scope

`v4.3_min_heart` is:

```text
tape-lane body
+ costed differentiable paths
+ boundary-coupled route economics
+ sequence emergence diagnostics
+ route/boundary/read/memory/head trace
+ deterministic sparse candidate suggestions
+ full 5ep logging/sync script
```

It is **not**:

```text
auto feedback deployment
actor
critic
TapePlannerAttention
macro promotion
train-step counterfactual verify
compound edits
```

Core question:

```text
Do useful separators and sequential tape behavior emerge, or does the model collapse into a flat all-to-all mixer / raw-detail shortcut?
```

## Hard rules

### Feedback vs gradient

Future feedback must be a detached buffer:

```python
logit_effective = learned_param + learned_context_bias + feedback_bias.detach() + actor_bias.detach()
```

For v4.3 there is no feedback deploy and no actor bias. `learned_context_bias` is a normal differentiable base-model context bias trained by task loss. `feedback_bias` is future tested-action state outside the optimizer.

### Window

```text
heart_window = epoch
```

One window means: train one epoch, evaluate, collect trace, write reports, generate suggestions, no deploy.

### Counterfactual screen

If enabled later, v4.3 only allows forward-only cheap screen:

```text
torch.no_grad()
same heldout microbatch
baseline forward
candidate forward with temporary detached bias
restore immediately
no backward
no optimizer step
no deployment
```

### Boundary exploit prevention

Boundary makes cross-lane route cheaper, but boundary itself is paid:

```text
route_offdiag_outside_boundary_cost = mean(offdiag_mass_norm * (1 - boundary))
boundary_budget_cost = mean(boundary)
```

Log:

```text
boundary_mean
boundary_flatness
boundary_peak_count
boundary_usefulness
```

### Normalized offdiag

Use normalized route offdiag for loss:

```python
offdiag_mass_raw[t] = sum_{from,to,from!=to} R_t[from,to]
offdiag_mass_norm[t] = mean_from(sum_to!=from R_t[from,to])
```

Log raw and normalized.

### Detail shortcut

Top reads are report-only. Loss must use differentiable attention mass:

```text
detail_attention_mass = sum class_slot_attention over detail-lane slots
detail_head_shortcut_cost = relu(detail_attention_mass - target)^2
```

### Skip/residual

Base residual already exists:

```python
X_next = X + gated_update
```

Preferred v4.3: no extra skip path, only log residual proxy. If extra skip is added later, it must be gated and paid.

### Memory

Memory read is not penalized. Memory write is mildly paid. Memory overwrite is paid more strongly.

Required metric:

```text
memory_consumer_score = future_memory_read + head_memory_attention + optional_grad_proxy
```

If memory write and consumer score are both high, suggestions should protect or open memory routes, not suppress useful memory.

### Late input

Early input/evidence access is allowed. Late raw input shortcut is paid:

```python
progress = t / max(1, T - 1)
depth_weight = sigmoid((progress - late_input_start) / late_input_tau)
```

Defaults:

```text
late_input_start = 0.45
late_input_tau = 0.12
```

## Sequence emergence requirement

v4.3 must explicitly test whether a sequential program appears.

A sequence means tape positions are not identical copies. Required diagnostics:

```text
sequence_route_delta_by_step
sequence_primitive_delta_by_step
sequence_read_delta_by_step
sequence_trace_delta_by_step
sequence_change_by_step
sequence_nonflat_score
```

Definitions:

```text
route_delta[t]     = distance(R_t, R_{t+1})
primitive_delta[t] = distance(primitive_w[t], primitive_w[t+1])
read_delta[t]      = distance(read_group[t], read_group[t+1])
trace_delta[t]     = distance(update/write/summary[t], update/write/summary[t+1])
sequence_nonflat_score = mean(route_delta + primitive_delta + read_delta + trace_delta)
```

Boundary should align with sequence change:

```text
boundary_usefulness[t] = boundary[t] * (route_delta[t] + primitive_delta[t] + read_delta[t] + trace_delta[t])
```

Candidate rules:

```text
sequence changes but boundary low -> suggest increase boundary
boundary high but sequence change low -> suggest decrease boundary
sequence_nonflat_score too low -> suggest sequence differentiation pressure
```

This is not a hard phase role. It is only a diagnostic/pressure for tape positions to become different when useful.

## Canonical MVP order

```text
MVP 0: v4.3 path economy + sequence trace + suggestions, no deploy
MVP 1: optional forward-only cheap counterfactual screen, no deploy
MVP 2: detached route/boundary FeedbackBiasBank after screen is stable
MVP 3: primitive/variant heart
MVP 4: memory/write heart
MVP 5: head heart
MVP 6: ExperienceMemory + cold-start critic
MVP 7: multi-projection actors
MVP 8: standard TapePlannerAttention
MVP 9: macro/growth promotion
```

## Required path costs

```text
route_offdiag_outside_boundary_cost
boundary_budget_cost
late_input_read_cost
memory_write_cost
memory_overwrite_cost
detail_head_shortcut_cost
skip_cost placeholder
operator_complexity_cost placeholder
```

## Required trace fields

Every epoch must write `trace_feedback_epoch_XXX.json` with:

```text
route.entropy_mean
route.matrix_by_step
route.offdiag_raw
route.offdiag_norm
route.offdiag_inside_boundary
route.offdiag_outside_boundary
route.boundary_by_step
route.boundary_mean
route.boundary_flatness
route.boundary_peak_count
route.boundary_peaks
route.boundary_usefulness
sequence.route_delta_by_step
sequence.primitive_delta_by_step
sequence.read_delta_by_step
sequence.trace_delta_by_step
sequence.sequence_change_by_step
sequence.sequence_nonflat_score
read.late_input_by_step
read.late_input_cost
memory.write_mean
memory.overwrite_score
memory.future_read
memory.head_consumer
memory.consumer_score
head.detail_attention_mass
head.detail_head_shortcut_cost
head.detail_topread_share
operators.primitive_weights
budget.step_alive
```

## Candidate suggestions

No auto-deploy. Max 8 candidates by default, hard max 12. Every candidate has:

```text
source
target_type
location
action
target
delta_scale
reason
evidence_metrics
risk
deploy=false
```

Required diversity report:

```text
by_source
by_target_type
by_step
by_lane
duplicate_rate
```

## Required files

```text
simple_butterfly_matrix_v4_tape_lane/tape_lane_transport_v4_3_min_heart.py
simple_butterfly_matrix_v4_tape_lane/commands/validate_v4_3_min_heart.sh
simple_butterfly_matrix_v4_tape_lane/commands/sync_run_5ep_v4_3_min_heart_push_logs.sh
v4_3_min_heart/README.md
```

## One command test

```bash
DATA_ROOT=../architecture_builder/data/speechcommands \
EPOCHS=5 \
TRAIN_LIMIT=12000 \
VAL_LIMIT=2000 \
AMP=fp16 \
bash simple_butterfly_matrix_v4_tape_lane/commands/sync_run_5ep_v4_3_min_heart_push_logs.sh
```

## Success after 5 epochs

```text
train/val not collapsed
sequence_nonflat_score not near zero
route/read/primitive deltas show tape positions are not identical
boundary peaks align with sequence_change_by_step
route_entropy below uniform
boundary not flat
boundary_peak_count > 0
offdiag_outside_boundary controlled
late input not rising strongly late
detail_attention_mass below target or decreasing
memory write not constant unless memory_consumer_score is high
candidate_suggestions_epoch_005.json exists and all deploy=false
```

## Do not do in v4.3

```text
auto feedback deployment
critic
actor
attention planner
macro promotion
bias baking
train-step counterfactual verify
compound edits
dense feedback bias
multiple workshops applying changes
```

## Self-check

```text
[x] feedback future detach rule
[x] learned_context_bias defined
[x] window = epoch
[x] counterfactual no backward/no optimizer/no train steps
[x] canonical MVP list unified
[x] boundary exploit closed
[x] offdiag normalized
[x] detail loss differentiable
[x] skip ambiguity resolved
[x] memory write/read separated
[x] late input schedule defined
[x] sequence emergence required and logged
[x] deterministic sparse candidates
[x] same-seed v4.2 comparison required in report/compare_to
[x] no actor/critic/attention in v4.3
```
