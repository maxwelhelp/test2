# v4.3_min_heart Plan

This document fixes the critical critiques of the Program Heart design and defines the next implementation target:

```text
v4.3_min_heart = tape-lane body + costed differentiable paths + route/boundary trace + candidate suggestions, without full actor/critic/planner and without auto-deploy.
```

The goal is not to build the full brain yet.

The goal is to make the architecture economically shaped so it cannot collapse into an all-to-all mixer, and to produce enough reliable trace/candidate data for later Heart stages.

---

## 1. Critiques accepted as hard requirements

### 1.1 Gradient loop vs feedback loop conflict

Feedback bias must not be a normal trainable parameter.

When feedback is eventually enabled, it must be applied as a detached offset:

```python
logit_effective = learned_param + feedback_bias.detach()
```

This separates optimizer updates from tested action feedback.

For `v4.3_min_heart`:

```text
no auto-deploy feedback yet;
only trace and candidate suggestions.
```

But the code should be designed so future feedback buffers can be added cleanly.

### 1.2 Define window explicitly

A `heart_window` is the unit at which the Heart wakes up.

For `v4.3_min_heart`:

```text
heart_window = epoch
```

Meaning:

```text
train epoch normally;
evaluate/collect trace on validation microbatches;
write trace_feedback_epoch_XXX.json;
generate candidate_suggestions_epoch_XXX.json;
no deploy.
```

Later versions can use:

```text
heart_window_steps = 250 or 500 train steps
```

But MVP uses epoch windows for simplicity.

### 1.3 CounterfactualScreen protocol

Counterfactual tests must not mix action effect with training effect.

For cheap screen:

```text
forward only;
no backward;
no optimizer step;
paired same microbatch baseline vs candidate;
restore temporary bias immediately.
```

Full verify with N mini-steps is postponed.

For `v4.3_min_heart`:

```text
candidate suggestions are produced;
optional cheap forward-only screen may be implemented;
no train-step verify;
no deploy.
```

### 1.4 Bias baking postponed

Accepted repeated biases may later be baked into base logits after K confirmations.

For this version:

```text
no feedback deploy -> no baking.
```

But the plan records future rule:

```text
if same action accepted K times and remains useful, bake into learned base or persistent prior, then reset transient feedback bias.
```

### 1.5 Context and action embeddings postponed but specified

Critic is too early.

For this version:

```text
write target-local trace fields and action schema;
do not train critic.
```

The stored candidate must include enough fields for future context/action embeddings.

### 1.6 Critic cold-start postponed

Critic/Actor/Planner start only after enough stable tested records exist.

Minimum future conditions:

```text
>= 500 cheap-screen records
>= 100 accepted/rejected non-no_effect records
model has plateaued or window-to-window metrics are stable enough
feedback deploy proved to affect effective probabilities
```

### 1.7 Diversity and stuck detection

Candidate suggestions must report diversity:

```text
candidate_count_by_source
candidate_count_by_target_type
candidate_count_by_step
candidate_count_by_lane
duplicate_rate
rejected_repeat_rate later
```

If diversity collapses, force safe exploration later.

### 1.8 Attention planner postponed

Trace tokens are small, so standard attention is fine later.

No cheap/linear attention needed for MVP.

For this version:

```text
no TapePlannerAttention;
standard attention can be added later if rule/counterfactual loop works.
```

---

## 2. Main architectural problem for v4.3

If every path is cheap, the model can collapse into:

```text
all-to-all read;
uniform route;
flat boundary;
head shortcut through detail slots;
memory as junk storage;
skip/residual dominance;
operator complexity growth without real gain.
```

Therefore `v4.3_min_heart` must add explicit path economics.

Not hard masks.

All paths remain soft/differentiable, but expensive when they are not justified.

---

## 3. Core design of v4.3_min_heart

Keep from v4.2:

```text
tape-lane body;
structured input init but weak/ablatable;
ClassMatrixLaneHead;
route_matrix[t,from,to];
boundary[t];
step_alive[t];
soft primitive mix;
soft read/write;
SafeBlockButterfly fixed runner.
```

Add:

```text
costed differentiable paths;
boundary-coupled route cost;
skip/residual gate with cost;
route/boundary trace feedback;
candidate suggestions without auto-deploy;
full logging and report export;
sync run script to commit/push logs, not .pt.
```

Do not add yet:

```text
full ProgramHeart;
FeedbackBias auto-deploy;
Critic;
Actor;
TapePlannerAttention;
macro promotion;
full verify with training steps.
```

---

## 4. Boundary must affect route economics

Boundary should not just be logged or weakly bias routes.

Boundary should make cross-lane transitions cheaper when a segment boundary is active.

### 4.1 Route offdiag cost

For each step:

```text
offdiag_route_mass[t] = sum_{from != to} R_t[from,to]
```

Boundary gate:

```text
boundary[t] in [0,1]
```

Cost:

```text
route_offdiag_outside_boundary_cost = offdiag_route_mass[t] * (1 - boundary[t])
```

Meaning:

```text
cross-lane transport is expensive inside a segment;
cross-lane transport becomes cheaper at a boundary.
```

### 4.2 Inside-boundary usage metric

Log:

```text
route_offdiag_inside_boundary = offdiag_route_mass[t] * boundary[t]
route_offdiag_outside_boundary = offdiag_route_mass[t] * (1 - boundary[t])
```

Success:

```text
outside-boundary offdiag decreases;
inside-boundary offdiag is used when boundary peaks.
```

### 4.3 Boundary usefulness

Boundary is useful if it corresponds to actual change.

Log approximate:

```text
trace_delta[t] = distance(trace_summary[t], trace_summary[t+1])
route_delta[t] = distance(R_t, R_{t+1})
primitive_delta[t] = distance(primitive_w[t], primitive_w[t+1])
boundary_usefulness[t] = boundary[t] * (trace_delta + route_delta + primitive_delta)
```

Candidate suggestions:

```text
if boundary high and usefulness low -> suggest decrease boundary
if boundary low and trace/route/primitive delta high -> suggest increase boundary
```

---

## 5. Required path costs

All costs must be optional flags with nonzero defaults for v4.3 run.

### 5.1 route_offdiag_outside_boundary_cost

Purpose:

```text
prevent uniform all-to-all route everywhere;
encourage cross-lane moves mainly at learned separators.
```

Loss term:

```text
lambda_route_offdiag_outside_boundary * mean(offdiag_mass * (1 - boundary))
```

### 5.2 late_input_read_cost

Purpose:

```text
prevent late steps from shortcutting raw input/evidence instead of using tape state.
```

Loss:

```text
depth_weight[t] * read_mass[t,lane,input]
```

Early steps pay little, late steps pay more.

### 5.3 memory_write_cost

Purpose:

```text
allow memory write but avoid constant writing every step.
```

Loss:

```text
memory_write_mass or memory write gate average
```

This is a mild budget.

### 5.4 memory_overwrite_cost

Purpose:

```text
penalize strong overwrite / churn, not all memory use.
```

Loss:

```text
relu(memory_write_gate - memory_write_target)^2
```

Keep read from memory cheap or neutral.

### 5.5 skip_cost

Add skip/residual gate, but make it paid.

Why:

```text
skip is useful for stability, but if free it can dominate and bypass program assembly.
```

Implement:

```text
skip_gate[t,lane] = sigmoid(skip_logit[t,lane])
X_next = X + step_update + skip_gate * skip_update/residual_path
```

Loss:

```text
lambda_skip_cost * mean(skip_gate)
```

MVP option:

```text
if skip path already implicit via residual, log residual dominance and add cost proxy.
```

### 5.6 detail_head_shortcut_cost

Purpose:

```text
prevent head from solving task by reading only detail/raw slots.
```

Log:

```text
detail_topread_share = fraction/mass of class top reads from detail lane
```

Loss:

```text
lambda_detail_head_shortcut * relu(detail_topread_share - target)^2
```

Do not ban detail reads. Penalize dominance.

### 5.7 operator_complexity_cost

Purpose:

```text
prevent rank/depth/variant complexity from growing without benefit.
```

For v4.3, if variants are not implemented yet:

```text
log placeholder as 0;
include plan and CLI field.
```

Later:

```text
rank/depth/radius weights * cost_vector
```

---

## 6. Required logs

Every epoch must export:

```text
analysis_epoch_XXX.json
trace_feedback_epoch_XXX.json
candidate_suggestions_epoch_XXX.json
REPORT_TO_CHATGPT.txt
metrics.csv
```

If sync script is used:

```text
train.log
final_report.json
AGENT_STATUS.md
```

### 6.1 Core metrics

Log:

```text
route_entropy_mean
route_matrix_by_step
boundary_by_step
boundary_peaks
route_offdiag_inside_boundary
route_offdiag_outside_boundary
route_offdiag_outside_boundary_cost
late_input_read_by_step
late_input_read_cost
memory_write_mean
memory_overwrite_score
memory_future_read
skip_gate_mean
skip_cost
detail_topread_share
class_lane_mass
class_top_reads
primitive_weights
step_alive
operator_complexity_cost
```

### 6.2 Trace feedback JSON

`trace_feedback_epoch_XXX.json` should include:

```json
{
  "epoch": 5,
  "window": {"type": "epoch", "index": 5},
  "route": {
    "entropy_mean": 0.0,
    "matrix_by_step": [],
    "offdiag_inside_boundary": [],
    "offdiag_outside_boundary": [],
    "boundary_by_step": [],
    "boundary_usefulness": []
  },
  "read": {
    "late_input_by_step": [],
    "input_read_total": 0.0
  },
  "memory": {
    "write_mean": 0.0,
    "overwrite_score": 0.0,
    "future_read": 0.0,
    "head_consumer": 0.0
  },
  "head": {
    "detail_topread_share": 0.0,
    "class_lane_mass": {},
    "top_reads": []
  },
  "operators": {
    "primitive_weights": [],
    "operator_complexity_cost": 0.0
  },
  "budget": {
    "skip_gate_mean": 0.0,
    "step_alive": [],
    "total_extra_cost": 0.0
  }
}
```

### 6.3 Candidate suggestions JSON

No deploy. Suggestions only.

Candidate schema:

```json
{
  "epoch": 5,
  "window": {"type": "epoch", "index": 5},
  "candidates": [
    {
      "source": "route_boundary_rule",
      "target_type": "boundary",
      "location": {"t": 4},
      "action": "increase",
      "target": "boundary",
      "delta_scale": 0.03,
      "reason": "trace/route/primitive changed strongly but boundary is low",
      "risk": "low",
      "deploy": false
    }
  ],
  "diversity": {
    "by_source": {},
    "by_target_type": {},
    "by_step": {},
    "duplicate_rate": 0.0
  }
}
```

---

## 7. Candidate suggestion rules for v4.3

No auto-deploy.

Generate candidate suggestions from trace.

### 7.1 Boundary suggestions

```text
if boundary[t] high and boundary_usefulness[t] low:
  suggest decrease boundary[t]

if boundary[t] low and trace_delta/route_delta/primitive_delta high:
  suggest increase boundary[t]
```

### 7.2 Route suggestions

```text
if route offdiag outside boundary high:
  suggest increase boundary or decrease specific offdiag route

if memory write high and future memory read high but memory->state route low:
  suggest increase memory->state route

if state lane overloaded and abstract usage low:
  suggest increase state->abstract route at a likely boundary

if route entropy near uniform:
  suggest sharpen largest useful route or increase cost
```

### 7.3 Read suggestions

```text
if late input read high and head detail shortcut high:
  suggest decrease late input read

if early detail read low but input structure indicates local/diff signal:
  suggest increase early detail/input read
```

### 7.4 Memory suggestions

```text
if memory write high and future read low:
  suggest decrease memory write or increase memory protect

if memory future read/head consumer high:
  memory is useful; do not kill memory write, maybe open memory->state route
```

### 7.5 Head suggestions

```text
if detail_topread_share high:
  suggest reduce head detail lane prior/read dominance

if class_lane_mass collapsed to one lane:
  suggest increase class read diversity/lane balance
```

---

## 8. Counterfactual screen protocol for later

Not required for v4.3, but if added it must follow this exact protocol.

Cheap screen only:

```text
select up to 8 candidates;
use same heldout microbatch for baseline and candidate;
forward baseline with no temporary bias;
forward candidate with temporary detached bias;
no backward;
no optimizer step;
restore temporary bias;
record delta;
repeat on second microbatch only for top candidates if needed.
```

Noise:

```text
estimate noise_std from repeated baseline heldout microbatches;
threshold = max(2 * noise_std, relative_min_gain * current_loss)
```

Statuses:

```text
accepted
rejected
uncertain_more_data
no_effect
unsafe
duplicate
conflict
```

`v4.3_min_heart` may only generate suggestions, not deploy them.

---

## 9. Sync run script requirement

Create one command:

```text
simple_butterfly_matrix_v4_tape_lane/commands/sync_run_5ep_v4_3_min_heart_push_logs.sh
```

It must:

```text
git pull --ff-only
validate/compile v4.3
run 5 epochs
tee full output to train.log
save all json/csv/txt/log
avoid committing .pt files
commit + push logs
update AGENT_STATUS.md
```

User command:

```bash
DATA_ROOT=../architecture_builder/data/speechcommands \
EPOCHS=5 \
TRAIN_LIMIT=12000 \
VAL_LIMIT=2000 \
AMP=fp16 \
bash simple_butterfly_matrix_v4_tape_lane/commands/sync_run_5ep_v4_3_min_heart_push_logs.sh
```

Report dir should include timestamp:

```text
simple_butterfly_matrix_v4_tape_lane/agent_reports/v4_3_min_heart_YYYYMMDD_HHMMSS/
```

Do not push checkpoints:

```text
*.pt
*.pth
*.ckpt
*.safetensors
```

---

## 10. Success criteria after 5 epochs

Accuracy:

```text
must not collapse relative to v4.2 smoke/guided trajectory;
5 epoch accuracy may be lower than long v3/v4, but should train normally.
```

Route/boundary:

```text
route entropy below uniform;
boundary not flat;
boundary peaks appear;
route_offdiag_outside_boundary decreases;
route_offdiag_inside_boundary is used around peaks.
```

Input/head:

```text
late_input_read does not grow strongly in late steps;
detail_topread_share is below target / not dominant.
```

Memory:

```text
memory write is not constant every step;
memory has future_read or head consumer if used;
memory read is not killed by memory write costs.
```

Skip/budget:

```text
skip_gate does not dominate;
step_alive not all max;
operator_complexity_cost logged.
```

Candidate suggestions:

```text
candidate_suggestions_epoch_XXX.json exists;
diversity fields populated;
suggestions are plausible and non-duplicate;
no auto-deploy occurred.
```

---

## 11. Implementation checklist

### Code file

Create:

```text
simple_butterfly_matrix_v4_tape_lane/tape_lane_transport_v4_3_min_heart.py
```

Base it on fixed v4.2 runner:

```text
tape_lane_transport_v4_2_fixed.py / tape_lane_transport_v4_2.py
```

But integrate SafeBlockButterfly directly or import fixed runner safely.

### Add CLI flags

```text
--lambda-route-offdiag-outside-boundary
--lambda-detail-head-shortcut
--lambda-skip-cost
--lambda-memory-write-cost
--detail-head-shortcut-target
--boundary-peak-threshold
--heart-window-type epoch
--enable-candidate-suggestions
--enable-counterfactual-screen false by default
```

### Loss additions

```text
route_offdiag_outside_boundary_cost
late_input_read_cost already exists but log by step
memory_write_cost
memory_overwrite_cost
skip_cost
detail_head_shortcut_cost
operator_complexity_cost placeholder if no variants
```

### Log additions

```text
trace_feedback_epoch_XXX.json
candidate_suggestions_epoch_XXX.json
REPORT_TO_CHATGPT.txt extended
metrics.csv columns added
final_report.json
```

### Validation

Create:

```text
commands/validate_v4_3_min_heart.sh
```

Check:

```text
py_compile
no old PHASES/class_phase/phase_slot active code
v4.3 costs exist
candidate suggestion code exists
sync script exists
```

---

## 12. Do not do in v4.3

Do not implement:

```text
actor
critic
TapePlannerAttention
feedback auto-deploy
bias baking
full verify with training steps
macro promotion
many workshops
compound edits
```

Only:

```text
costed paths
trace
suggestions
logs
fast 5ep test
```

---

## 13. Short instruction to the agent

Build `v4.3_min_heart` as a conservative bridge from v4.2 to ProgramHeart.

Do not build the full brain.

Make the body economically meaningful:

```text
all paths soft but not free;
boundary controls cost of cross-lane transport;
input/head/memory/skip shortcuts are allowed but paid;
route/boundary behavior is logged and converted into suggestions.
```

The output of v4.3 should answer:

```text
Are the learned separators meaningful?
Are cross-lane routes used mainly at boundaries?
Is the head avoiding raw-detail shortcut?
Is memory used and later consumed?
What route/boundary actions would the Heart try next?
```

Only after this is stable should we add feedback deploy, critic, actor, or attention planner.
