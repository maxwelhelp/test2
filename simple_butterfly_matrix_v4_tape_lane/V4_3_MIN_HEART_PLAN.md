# v4.3_min_heart Plan

This file is the single authoritative plan for the next conservative bridge from v4.2 to ProgramHeart.

Do not create additional Heart design files until this version is implemented and tested.

`v4.3_min_heart` is **not** the full ProgramHeart.

It is:

```text
tape-lane body
+ costed differentiable paths
+ boundary-coupled route economics
+ route/boundary/read/memory/head trace
+ deterministic sparse candidate suggestions
+ optional cheap forward-only screen later
+ full 5ep logging/sync script
```

It explicitly does **not** include yet:

```text
auto-deploy feedback
actor
critic
TapePlannerAttention
macro promotion
full verify with training steps
compound edits
many workshops
```

The purpose is to answer one core question before building a brain:

```text
Are the learned separators and path economics meaningful, or does the model collapse into all-to-all read/route/head shortcuts?
```

---

## 0. Accepted critiques and mandatory fixes

These are hard requirements, not optional opinions.

### 0.1 Feedback vs gradient loop conflict

When feedback deploy exists in a later version, feedback must be a detached buffer, never an optimizer parameter.

Correct future formula:

```python
logit_effective = learned_param + learned_context_bias + feedback_bias.detach() + actor_bias.detach()
```

For v4.3:

```text
no feedback auto-deploy;
no actor_bias;
only learned_param + learned_context_bias in the model;
only suggestions are emitted.
```

`learned_context_bias` means a normal differentiable context network trained by the task gradient. It is part of the base model.

`feedback_bias` means a future tested-action buffer updated outside optimizer. It is not trainable by gradient.

This distinction must be kept explicit in code comments.

### 0.2 Influence is not all differentiable

Base model choices are differentiable.

Future feedback/actor deployment biases are not optimizer-trained:

```python
feedback_bias = feedback_bias.detach()
actor_bias = actor_bias.detach()
```

Do not write vague comments like `all influence is differentiable` for feedback buffers.

Correct wording:

```text
The model path remains differentiable through learned parameters.
Tested feedback offsets are detached control signals.
```

### 0.3 Heart window definition

For v4.3:

```text
heart_window = epoch
```

A window is:

```text
train one epoch normally;
evaluate validation normally;
collect trace on validation batches;
write trace_feedback_epoch_XXX.json;
generate candidate_suggestions_epoch_XXX.json;
write REPORT_TO_CHATGPT.txt;
no deploy.
```

Later versions may use `heart_window_steps=250/500`, but v4.3 must use epoch windows only.

### 0.4 Counterfactual screen protocol

If implemented in v4.3, it must be forward-only and optional.

Allowed cheap screen:

```text
same heldout microbatch;
no backward;
no optimizer step;
torch.no_grad();
baseline forward;
temporary detached candidate bias forward;
restore immediately;
record delta.
```

Not allowed in v4.3:

```text
run N train mini-steps during action testing;
measure action + training mixed gain;
deploy accepted action.
```

Full verify with training steps is postponed until a future version with full-state rollback.

### 0.5 Boundary exploit must be closed

If route cost is:

```text
offdiag * (1 - boundary)
```

then the model can set `boundary=1` everywhere and get free cross-lane routing.

Therefore v4.3 must include boundary economy:

```text
boundary_budget_cost = mean(boundary)
boundary_flatness metric
boundary_peak_count metric
boundary_usefulness metric
```

Boundary is only good when it aligns with real trace/route/primitive changes.

### 0.6 Normalized offdiag cost

Raw offdiag route mass depends on lane count.

Use normalized cost for loss:

```python
offdiag_mass_raw[t] = sum_{from,to,from!=to} R_t[from,to]
offdiag_mass_norm[t] = mean_from(sum_to!=from R_t[from,to])
```

For `L` lanes, `offdiag_mass_norm` is in `[0,1]`.

Log both raw and normalized.

Use normalized in loss.

### 0.7 Detail shortcut loss must be differentiable

`detail_topread_share` is good for reports but not for loss because top-k is not a stable differentiable training signal.

For loss use:

```text
detail_attention_mass = sum class_slot_attention over detail-lane slots
```

Loss:

```text
detail_head_shortcut_cost = relu(detail_attention_mass - target)^2
```

Keep `detail_topread_share` only for logging/report.

### 0.8 Skip ambiguity

The base residual path already exists:

```python
X_next = X + gated_update
```

Do not accidentally add a second free bypass.

In v4.3 choose one of these:

Option A, preferred MVP:

```text
no extra skip path;
log residual dominance proxy only;
skip_gate_mean = 0;
skip_cost = 0 placeholder.
```

Option B, if extra skip is implemented:

```text
extra_skip = skip_gate * skip_proj(X)
skip_gate is paid by lambda_skip_cost;
base residual remains X + gated_update.
```

Do not create `X + update + free skip_proj(X)`.

### 0.9 Memory write/read distinction

Memory read must not be penalized.

Memory write is allowed but paid mildly.

Memory overwrite/churn is paid more strongly.

Required memory consumer score:

```text
memory_consumer_score = future_memory_read + head_memory_attention + grad_attribution_proxy_optional
```

If memory write is high and consumer score is high, candidate suggestions must not say “kill memory”.

They may suggest:

```text
open memory->state route;
protect memory from overwrite;
reduce excessive overwrite, not memory use.
```

### 0.10 Late input read schedule

Late input cost must not kill early evidence access.

Use a depth weight that is near zero early and grows after the middle of the tape:

```python
progress = t / max(1, T - 1)
depth_weight = sigmoid((progress - late_input_start) / late_input_tau)
```

Suggested defaults:

```text
late_input_start = 0.45
late_input_tau = 0.12
```

Log `late_input_read_by_step`, not just mean.

### 0.11 Candidate suggestions deterministic and sparse

No noisy text spam.

Per epoch:

```text
max_candidates = 8 to 12
all candidates deploy=false
no duplicate candidates
schema must include source, target_type, location, action, delta_scale, reason, evidence_metrics, risk, deploy=false
```

### 0.12 Same-seed comparison required

v4.3 5ep report must state what v4.2 baseline it should be compared against.

Minimum:

```text
compare_to = v4.2_fixed_guided same seed/config if available
```

If not available, report must say:

```text
baseline missing; run v4.2_fixed_guided with same seed/config before judging accuracy.
```

### 0.13 Usage EMA lag rule

Any usage prior or EMA is future work, not v4.3.

When implemented later:

```text
EMA updates after forward/backward of current batch/window;
EMA applies only to next batch/window;
EMA is detached and clamped.
```

Never let current forward use statistics produced later in the same forward.

### 0.14 Critic/action/context embeddings are future, but schemas must be stored

v4.3 must store candidate fields rich enough for future critic.

No critic is trained in v4.3.

---

## 1. Canonical MVP sequence

This list replaces all previous conflicting MVP lists.

### MVP 0: v4.3_min_heart path economy and trace

Current target.

```text
costed differentiable paths;
boundary-coupled route cost;
trace_feedback_epoch_XXX.json;
candidate_suggestions_epoch_XXX.json;
no deploy.
```

### MVP 1: cheap forward-only counterfactual screen

Optional after MVP 0 logs look sane.

```text
paired microbatch baseline vs candidate;
no backward;
no optimizer step;
max 8 candidates;
no deploy.
```

### MVP 2: detached FeedbackBiasBank for route/boundary only

Only after cheap screen is stable.

```text
feedback_route_bias.detach();
feedback_boundary_bias.detach();
max one accepted action per epoch/window;
prove effective route/boundary probabilities changed next window.
```

### MVP 3: primitive/variant heart

Add primitive and rank/depth/radius variant candidates after route/boundary feedback works.

### MVP 4: memory/write heart

Add memory consumer-aware candidates.

### MVP 5: head heart

Add class/pair/head read candidates and head-to-core task hints.

### MVP 6: ExperienceMemory + cold-start critic

Only after stable tested records exist.

Minimum future gate:

```text
>= 500 cheap-screen records;
>= 100 accepted/rejected non-no_effect records;
model/window metrics stable enough;
feedback deploy proved to affect effective probabilities.
```

### MVP 7: multi-projection actors

Small actors per domain, not one giant dense actor.

### MVP 8: standard TapePlannerAttention

Standard attention over compressed trace tokens. No linear/cheap attention needed at this scale.

### MVP 9: macro/growth promotion

Only after critic and tested records are reliable.

---

## 2. Architecture of v4.3_min_heart

Keep v4.2 fixed body:

```text
tape-lane execution;
structured input init but weak/ablatable;
ClassMatrixLaneHead;
route_matrix[t,from,to];
boundary[t];
step_alive[t];
soft primitive mix;
soft read/write;
SafeBlockButterfly or integrated safe block primitive.
```

Add v4.3 mechanics:

```text
costed paths;
boundary route economy;
route/boundary trace;
read/memory/head shortcut trace;
deterministic candidate suggestions;
sync 5ep logs.
```

Do not add:

```text
FeedbackBias auto-deploy;
Actor;
Critic;
TapePlannerAttention;
macro promotion;
full verify with training;
compound edits.
```

---

## 3. Boundary-coupled route economics

### 3.1 Route offdiag metrics

For each `t`:

```python
R = route_matrix[t]  # [L,L], rows sum to 1
I = eye(L)
offdiag = R * (1 - I)
offdiag_mass_raw[t] = offdiag.sum()
offdiag_mass_norm[t] = offdiag.sum(dim=-1).mean()
```

Use `offdiag_mass_norm` for loss.

Log both.

### 3.2 Boundary-gated offdiag cost

```python
inside = offdiag_mass_norm[t] * boundary[t]
outside = offdiag_mass_norm[t] * (1 - boundary[t])
route_offdiag_outside_boundary_cost = outside.mean()
```

Meaning:

```text
cross-lane routing is expensive inside a segment;
cross-lane routing is cheaper at a learned boundary;
boundary itself is not free.
```

### 3.3 Boundary budget and exploit prevention

Add:

```python
boundary_budget_cost = boundary.mean()
```

Suggested loss:

```text
lambda_boundary_budget * mean(boundary)
```

Log:

```text
boundary_mean
boundary_flatness = std(boundary)
boundary_peak_count = count(boundary > boundary_peak_threshold)
boundary_by_step
boundary_peaks
```

Success:

```text
not boundary=1 everywhere;
not boundary=0 everywhere;
peaks appear at meaningful program changes.
```

### 3.4 Boundary usefulness

Compute approximate deltas:

```text
trace_delta[t] = distance(trace_summary[t], trace_summary[t+1])
route_delta[t] = distance(R_t, R_{t+1})
primitive_delta[t] = distance(primitive_w[t], primitive_w[t+1])
```

Use:

```text
boundary_usefulness[t] = boundary[t] * (trace_delta + route_delta + primitive_delta)
```

For logging and suggestions only.

Do not use a strong hard target loss for boundary usefulness in v4.3.

---

## 4. Required path costs

All costs must be CLI-controllable and logged.

### 4.1 route_offdiag_outside_boundary_cost

```text
lambda_route_offdiag_outside_boundary * mean(offdiag_mass_norm * (1 - boundary))
```

Purpose:

```text
prevent uniform all-to-all route everywhere;
encourage cross-lane moves mainly at learned separators.
```

### 4.2 boundary_budget_cost

```text
lambda_boundary_budget * mean(boundary)
```

Purpose:

```text
prevent boundary exploit: boundary high everywhere.
```

### 4.3 late_input_read_cost

Use depth schedule:

```python
progress = t / max(1, T - 1)
depth_weight = sigmoid((progress - late_input_start) / late_input_tau)
late_input_read_cost = mean_t_l(depth_weight[t] * read_mass[t,l,input])
```

Purpose:

```text
allow early evidence;
make late raw-input shortcut expensive.
```

### 4.4 memory_write_cost

Mild write budget:

```text
lambda_memory_write_cost * mean(memory_write_gate_or_mass)
```

Purpose:

```text
avoid constant memory writing every step.
```

Do not penalize memory read.

### 4.5 memory_overwrite_cost

Stronger overwrite/churn cost:

```text
relu(memory_write_gate - memory_write_target)^2
```

Purpose:

```text
penalize excessive overwrite, not useful memory.
```

### 4.6 skip_cost / residual dominance

Preferred v4.3 minimal choice:

```text
no extra skip path;
log residual_dominance_proxy;
skip_gate_mean = 0;
skip_cost = 0 placeholder.
```

If extra skip is implemented:

```text
extra_skip = skip_gate * skip_proj(X);
lambda_skip_cost * mean(skip_gate);
base residual still X + gated_update.
```

### 4.7 detail_head_shortcut_cost

Differentiable loss:

```text
detail_attention_mass = sum class_slot_attention over detail-lane slots
detail_head_shortcut_cost = relu(detail_attention_mass - target)^2
```

Report-only metric:

```text
detail_topread_share = fraction/mass of class top reads from detail lane
```

### 4.8 operator_complexity_cost

If rank/depth/radius variants do not exist yet:

```text
operator_complexity_cost = 0 placeholder;
log field exists;
CLI flag exists;
report says variants not implemented.
```

Later:

```text
sum variant_weight * variant_cost
```

---

## 5. Structured input/head priors in v4.3

v4.3 may keep weak structured input and head priors from v4.2, but they must be ablatable and logged.

### 5.1 Input structure

Input structure answers:

```text
what exists in the data?
```

Examples for audio:

```text
local/detail energy;
diff/onset proxy;
global summary;
noise/smoothness proxy;
low-rank/compressibility proxy.
```

In v4.3, input structure may affect:

```text
initial lane state only;
optional learned_context_bias trained by gradient.
```

It must not become hidden hard phase roles.

### 5.2 Head structure

Head structure answers:

```text
what does the task need?
```

Use generic indexed tokens, not hardcoded class names:

```text
HEAD_CONFUSION_PAIR(class_i, class_j)
LOW_MARGIN_CLASS(class_i)
SHARED_SLOT_COLLAPSE
CLASS_UNIQUE_NEED(class_i)
PAIR_REPAIR_ACTIVE(pair_id)
OVERCONFIDENT_CLASS(class_i)
```

For v4.3 these are trace/report fields and suggestion inputs only.

No head actor/critic.

---

## 6. Candidate suggestions in v4.3

No auto-deploy.

Candidate file is a recommendation list for later human/agent inspection.

### 6.1 Candidate schema

Every candidate must include:

```json
{
  "source": "route_boundary_rule",
  "target_type": "route|boundary|read|memory|head|budget",
  "location": {"t": 4, "lane": null, "from_lane": null, "to_lane": null},
  "action": "increase|decrease|shift|protect|open|close",
  "target": "boundary or route.memory->state or read.input",
  "delta_scale": 0.03,
  "reason": "short readable explanation",
  "evidence_metrics": {},
  "risk": "low|medium|high",
  "deploy": false
}
```

`deploy` must always be false in v4.3.

### 6.2 Diversity fields

```json
{
  "diversity": {
    "by_source": {},
    "by_target_type": {},
    "by_step": {},
    "by_lane": {},
    "duplicate_rate": 0.0
  }
}
```

### 6.3 Max candidates

Default:

```text
max_candidates_per_epoch = 8
hard maximum = 12
```

### 6.4 Candidate rules

Boundary:

```text
if boundary high and usefulness low -> suggest decrease boundary
if boundary low and trace/route/primitive delta high -> suggest increase boundary
```

Route:

```text
if offdiag outside boundary high -> suggest increase boundary at that t or decrease specific offdiag route
if memory write high and future memory read high but memory->state route low -> suggest increase memory->state
if route entropy near uniform -> suggest sharpen useful route or rely on route cost
```

Read:

```text
if late input read high and detail shortcut high -> suggest decrease late input read
```

Memory:

```text
if memory write high and future read/head consumer low -> suggest reduce memory write/overwrite
if memory write high and consumer high -> do not kill memory; suggest protect/open memory->state
```

Head:

```text
if detail_attention_mass/detail_topread_share high -> suggest reduce detail shortcut pressure
if class_lane_mass collapsed -> suggest stronger lane/class read diversity
```

Budget:

```text
if step_alive all high and gain not clear -> suggest stronger alive/complexity budget
if skip/residual proxy dominates -> suggest stronger skip/residual cost or lower extra skip
```

---

## 7. Trace feedback JSON

`trace_feedback_epoch_XXX.json` must include enough for later critic/action/context embeddings, even though critic is not trained.

Required schema:

```json
{
  "epoch": 5,
  "window": {"type": "epoch", "index": 5},
  "compare_to": "v4.2_fixed_guided same seed/config or baseline_missing",
  "route": {
    "entropy_mean": 0.0,
    "matrix_by_step": [],
    "offdiag_raw": [],
    "offdiag_norm": [],
    "offdiag_inside_boundary": [],
    "offdiag_outside_boundary": [],
    "offdiag_outside_boundary_cost": 0.0,
    "boundary_by_step": [],
    "boundary_mean": 0.0,
    "boundary_flatness": 0.0,
    "boundary_peak_count": 0,
    "boundary_peaks": [],
    "boundary_usefulness": []
  },
  "read": {
    "late_input_by_step": [],
    "late_input_cost": 0.0,
    "input_read_total": 0.0
  },
  "memory": {
    "write_mean": 0.0,
    "write_cost": 0.0,
    "overwrite_score": 0.0,
    "future_read": 0.0,
    "head_consumer": 0.0,
    "consumer_score": 0.0
  },
  "head": {
    "detail_attention_mass": 0.0,
    "detail_head_shortcut_cost": 0.0,
    "detail_topread_share": 0.0,
    "class_lane_mass": {},
    "top_reads": []
  },
  "operators": {
    "primitive_weights": [],
    "operator_complexity_cost": 0.0,
    "variants_implemented": false
  },
  "budget": {
    "skip_gate_mean": 0.0,
    "skip_cost": 0.0,
    "residual_dominance_proxy": 0.0,
    "step_alive": [],
    "total_extra_cost": 0.0
  }
}
```

---

## 8. Future context/action embedding schemas

These are not implemented as critic in v4.3, but candidate logs must preserve enough fields.

### 8.1 Context embedding schema

Future fixed-size context vector should be built from:

```text
target-local trace token;
previous step summary;
next step summary;
local segment summary;
input structure summary;
head feedback summary;
budget state;
training progress features.
```

For route action `R_t[from,to]`:

```text
concat(
  trace_token[t, from_lane],
  trace_token[t, to_lane],
  route_row[t, from_lane],
  route_col[t, to_lane],
  boundary[t],
  step_alive[t],
  offdiag_inside/outside metrics,
  memory/head consumer if relevant
)
```

For primitive action:

```text
concat(
  trace_token[t,lane],
  primitive_weights[t,lane],
  primitive_importance optional,
  update_norm[t,lane],
  downstream_consumer_score[t,lane]
)
```

For head action:

```text
concat(
  class_lane_mass[class_i],
  class_top_read_summary[class_i],
  class_margin[class_i],
  confusion_pair_stats[class_i,class_j],
  pair_update_norm
)
```

### 8.2 Action embedding schema

Future action vector must encode:

```text
action_type: increase/decrease/shift/protect/open/close
target_type: route/boundary/read/primitive/variant/write/memory/head/step_alive
step index normalized: t / T
lane/from/to one-hot or embedding
primitive id if any
variant id if any
class/pair id if any
delta_scale
complexity_delta
risk flags
source/workshop id
```

This avoids undefined `action_embedding` later.

---

## 9. Future route basis decomposition

Do not implement route basis decomposition in v4.3 unless trivial.

If implemented later, use concrete lane-index matrices.

For lanes:

```text
0 detail
1 state
2 abstract
3 memory
```

Basis matrices are logits templates `[L,L]`:

```text
identity_basis:
  +1 on i->i

upward_basis:
  detail->state, state->abstract

memory_write_basis:
  state->memory, abstract->memory

memory_read_basis:
  memory->state, memory->abstract

cross_exchange_basis:
  detail<->state, state<->abstract, state<->memory weak symmetric
```

Formula:

```python
route_logits_t = sum_b route_basis_weight[t,b] * route_basis[b] + residual_route_logits[t]
```

But v4.3 may keep raw route logits and just log route behavior.

---

## 10. Optional cheap counterfactual screen for later

Not required for v4.3 initial run.

If added, memory-safe limit:

```text
max_candidates_screened = 8
max_parallel_candidates = min(4, floor(vram_safe_batch / eval_batch_size))
```

If batching K candidates:

```text
K must be capped;
fall back to sequential candidate forwards if OOM risk.
```

Protocol:

```text
torch.no_grad();
same heldout microbatch;
baseline forward;
for candidate or candidate batch: temporary detached bias forward;
restore immediately;
no backward;
no optimizer step.
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

No deploy in v4.3.

---

## 11. Required files for implementation

Create/update code file:

```text
simple_butterfly_matrix_v4_tape_lane/tape_lane_transport_v4_3_min_heart.py
```

Base it on fixed v4.2:

```text
tape_lane_transport_v4_2_fixed.py / tape_lane_transport_v4_2.py
```

Integrate SafeBlockButterfly directly or import fixed runner safely.

Validation script:

```text
simple_butterfly_matrix_v4_tape_lane/commands/validate_v4_3_min_heart.sh
```

Sync run script:

```text
simple_butterfly_matrix_v4_tape_lane/commands/sync_run_5ep_v4_3_min_heart_push_logs.sh
```

---

## 12. CLI/env requirements

### 12.1 Sync script env

`sync_run_5ep_v4_3_min_heart_push_logs.sh` must accept:

```text
DATA_ROOT
EPOCHS
TRAIN_LIMIT
VAL_LIMIT
AMP
OUT_DIR
BATCH_SIZE
EVAL_BATCH_SIZE
WORKERS
SEED
```

Do not hardcode these.

### 12.2 CLI flags for code

Required new flags:

```text
--lambda-route-offdiag-outside-boundary
--lambda-boundary-budget
--lambda-detail-head-shortcut
--lambda-skip-cost
--lambda-memory-write-cost
--detail-head-shortcut-target
--boundary-peak-threshold
--late-input-start
--late-input-tau
--max-candidates-per-epoch
--heart-window-type epoch
--enable-candidate-suggestions
--enable-counterfactual-screen false by default
--compare-to
```

Metrics columns must include all cost fields.

---

## 13. Validation requirements

`validate_v4_3_min_heart.sh` must check:

```text
py_compile v4.3 file;
v4.3 script imports/runs --help;
sync script calls tape_lane_transport_v4_3_min_heart.py;
no active PHASES/class_phase_logits/phase_slot_matrix/phase_balance code;
trace_feedback writer exists;
candidate_suggestions writer exists;
metrics/report include all cost fields;
detail loss uses attention mass, not top-k;
offdiag loss uses normalized mass;
boundary budget exists;
sync script does not commit .pt/.pth/.ckpt/.safetensors.
```

---

## 14. Sync run requirement

The one user command must be:

```bash
DATA_ROOT=../architecture_builder/data/speechcommands \
EPOCHS=5 \
TRAIN_LIMIT=12000 \
VAL_LIMIT=2000 \
AMP=fp16 \
bash simple_butterfly_matrix_v4_tape_lane/commands/sync_run_5ep_v4_3_min_heart_push_logs.sh
```

The script must:

```text
git pull --ff-only;
validate/compile;
run 5 epochs;
tee stdout/stderr to train.log;
save json/csv/txt/log;
not push checkpoints;
commit + push logs;
update AGENT_STATUS.md.
```

Report dir:

```text
simple_butterfly_matrix_v4_tape_lane/agent_reports/v4_3_min_heart_YYYYMMDD_HHMMSS/
```

---

## 15. Required logs

Each run must produce:

```text
metrics.csv
analysis_epoch_XXX.json
trace_feedback_epoch_XXX.json
candidate_suggestions_epoch_XXX.json
REPORT_TO_CHATGPT.txt
final_report.json
train.log
AGENT_STATUS.md
```

No checkpoints in committed logs:

```text
*.pt
*.pth
*.ckpt
*.safetensors
```

---

## 16. Success criteria after 5 epochs

Accuracy:

```text
train/val not collapsed;
accuracy not clearly worse than same-seed v4.2 fixed guided baseline trajectory.
```

Route/boundary:

```text
route_entropy below uniform;
boundary not flat;
boundary_peak_count > 0;
offdiag_outside_boundary decreases or is controlled;
offdiag_inside_boundary is used near boundary peaks;
boundary exploit not present: boundary_mean not near 1 everywhere.
```

Input/head:

```text
late_input_by_step not rising strongly in late steps;
detail_attention_mass below target or decreasing;
detail_topread_share not dominant.
```

Memory:

```text
memory write not constant every step;
if memory write high, memory_consumer_score also high;
memory read not killed.
```

Skip/budget:

```text
skip/residual proxy not dominating;
step_alive not all max;
operator_complexity_cost logged.
```

Suggestions:

```text
candidate_suggestions_epoch_005.json exists;
max 8-12 candidates;
diversity fields populated;
suggestions are deterministic, plausible, non-duplicate;
all deploy=false.
```

---

## 17. Do not do in v4.3

Do not implement:

```text
auto feedback deployment;
critic;
actor;
attention planner;
macro promotion;
bias baking;
train-step counterfactual verify;
compound edits;
dense feedback bias;
multiple workshops applying changes.
```

Only implement:

```text
costed paths;
trace;
deterministic sparse suggestions;
logs;
5ep sync test.
```

---

## 18. Self-check for this plan

This section is included to prevent missing critique fixes.

Required critique fixes included:

```text
[x] feedback vs gradient conflict: future feedback is detach buffer
[x] learned_context_bias defined as differentiable base-model context bias
[x] feedback/actor bias not described as differentiable
[x] window explicitly defined as epoch for v4.3
[x] counterfactual protocol is forward-only, no train steps
[x] canonical MVP list unified
[x] boundary exploit closed with boundary budget/flatness/peak metrics
[x] offdiag cost normalized and raw+norm logged
[x] detail shortcut loss uses differentiable attention mass, topread only report
[x] skip ambiguity resolved with preferred no-extra-skip MVP or paid extra skip
[x] memory write/read distinction and consumer score specified
[x] late input depth schedule specified
[x] deterministic sparse candidate schema and max candidates specified
[x] same-seed v4.2 baseline comparison required
[x] context embedding future schema specified
[x] action embedding future schema specified
[x] usage EMA lag rule specified
[x] critic/actor/attention postponed
[x] standard attention later, no linear attention needed
[x] batched counterfactual memory cap specified for later
[x] concrete route basis templates specified for later
[x] validation checks include all critical code requirements
[x] sync script env flags specified
[x] no new design files required before implementation
```

If an implementation violates any checked item, it is not v4.3_min_heart.

---

## 19. Short instruction to the agent

Build `v4.3_min_heart` as a minimal, conservative bridge.

Do not build the full brain.

Make the body economically meaningful:

```text
all paths remain soft;
no hard masks;
but all shortcuts are priced;
boundary makes cross-lane routing cheaper but boundary itself is priced;
head/detail/input/memory/skip shortcuts are allowed but cannot dominate for free.
```

The output of v4.3 should answer:

```text
Are separators meaningful?
Are cross-lane routes used mainly at boundaries?
Does boundary exploit happen?
Does the head avoid raw detail shortcut?
Is memory used and later consumed?
Are path costs shaping the program without collapsing accuracy?
What route/boundary/read/memory/head actions would the future Heart test?
```

Only after those answers are good should the project move to feedback deploy, critic, actor, or attention planner.
