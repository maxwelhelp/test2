# Program Heart Architecture Plan

Goal: redesign the next tape-lane architecture so it is built from the beginning for intelligent self-editing, feedback, actor/critic learning, and interpretable program assembly.

This document is not a logging plan. Logs are only the raw trace.

The real goal is a closed control loop:

```text
what happened
  -> why it helped/hurt
  -> what action to try
  -> test the action
  -> send feedback to the exact assembly location
  -> the location chooses better next time
```

Core principle:

```text
every selectable element must be addressable;
every addressable element must produce trace;
every traceable element must receive feedback;
feedback must return to the exact choice location, not just to the whole model.
```

---

## 1. Base architecture to support the heart

Use the tape-lane model as the execution body:

```text
Input structure encoder
  -> structured evidence/lane init
  -> program tape of T positions
  -> each position has read -> transform -> write
  -> each position has separator_t = boundary_t + route_matrix_t
  -> ClassMatrixLaneHead / task head
```

Execution state:

```text
X[t, lane, cell, dim]
```

Tape position `t` is not a semantic layer. It is a place where a separator and a transform program can operate.

Every tape position exposes soft choices:

```text
read source selection
read cell selection
primitive / operator selection
operator variant selection: rank/depth/radius/etc.
compose/compare/gate selection
route matrix R_t[from_lane,to_lane]
boundary_t
write target/gate
step_alive_t
head read/class-pair choices
```

All choices must stay soft/differentiable in the main model.

---

## 2. What can change

The system should be able to change any architecture choice that is represented by logits/gates.

### 2.1 Read choices

```text
read input/evidence
read detail lane
read state lane
read abstract/global lane
read memory/control lane
read specific cells
read local/global summaries
```

Possible actions:

```text
increase/decrease read from source group
shift read from input to state
shift read from state to memory
revive suppressed input/detail read when needed
close late input shortcut
```

### 2.2 Transform choices

```text
channel_butterfly
block_butterfly
low_rank
ctx_matrix
product_gate
diff/smooth/wavelet-like
compare/gated_contrast
memory_keep / memory_match
future macros
```

Possible actions:

```text
increase primitive p at t,lane
decrease primitive p
revive suppressed primitive
shift low_rank -> compare
shift product_gate -> residual/keep
choose smaller/larger rank
choose smaller/larger butterfly depth
choose local radius
```

### 2.3 Separator / route choices

```text
route_matrix[t, from_lane, to_lane]
boundary[t]
route entropy
step segment split/merge
```

Possible actions:

```text
increase state->abstract
increase state->memory
increase memory->state
decrease useless self-loop
increase boundary where trace changes
decrease boundary where no real segment difference exists
open dormant step
close useless step
```

### 2.4 Write choices

```text
write to detail/state/abstract/memory lanes
write gate
memory overwrite/keep/protect
global write
residual strength
```

Possible actions:

```text
increase write to state
reduce memory overwrite
protect memory cell/lane
open memory write only if future consumer exists
reduce global write if no head/consumer usage
```

### 2.5 Head choices

```text
class_lane_logits
class slot attention
class_pair_logits
pair_state / pair repair
class unique read
confusion contrast
```

Possible actions:

```text
increase class-pair contrast
increase class-unique lane read
suppress shared slot/lane read
increase memory/global read for confused class
calibrate overconfident class
```

---

## 3. Program Heart modules

The heart is not one magic neural head. It is a control loop made of several modules.

```text
TraceCollector
ProjectionBank
LocalAnalyzers / Workshops
CandidateActionGenerator
ActionCoordinator
CounterfactualScreen
FeedbackBiasBank
ExperienceMemory
Critic
Actor / Planner
Safety / Rollback
```

### 3.1 TraceCollector

Collects compact trace from every addressable choice location.

For each `t,lane` store:

```text
state summary before/after
read mass
primitive weights
operator variant weights
route row/column
boundary
write gate
step_alive
update_norm
head usage
future consumer usage
gradient attribution
loss/head/confusion feedback
```

Important: do not store huge tensors by default.

Use projected summaries:

```text
X[t,lane,cells,D] -> mean/std/max + learned projection -> d_small 16/32
```

Trace token:

```text
trace_token[t,lane] = compressed state + dataflow fields + operator fields + head fields + gradient fields
```

### 3.2 ProjectionBank

Different projections answer different questions. Do not use only one view.

#### Dataflow projection

Question:

```text
where does information flow?
```

Inputs:

```text
read_group_mass
route_matrix
write_gate
consumer_score
late_input_read
```

Outputs/action ideas:

```text
open/close memory read
close input shortcut
increase state->abstract
increase memory->state
```

#### Operator projection

Question:

```text
which operations are useful in which context?
```

Inputs:

```text
primitive_weights
primitive gradients
update_norm
rank/depth/variant usage
input/head structure hints
```

Outputs/action ideas:

```text
increase compare
revive diff
shift low_rank rank
reduce product_gate
increase/decrease butterfly depth
```

#### Boundary/segment projection

Question:

```text
where are real program boundaries?
```

Inputs:

```text
boundary
trace distance before/after
route pattern change
primitive pattern change
head read from segment
```

Outputs/action ideas:

```text
increase boundary
merge segment by decreasing boundary
split segment by increasing boundary
```

#### Memory projection

Question:

```text
is memory useful or junk storage?
```

Inputs:

```text
memory write
future memory read
memory->state route
head usage
memory gradient
memory overwrite score
```

Outputs/action ideas:

```text
reduce memory overwrite
increase memory read later
protect memory cell
open memory compare
```

#### Head/task projection

Question:

```text
what does the task/head need from the program?
```

Inputs:

```text
confusion matrix
class margin
class_lane_mass
class_top_reads
class_read_div
pair_update_norm
head gradients
```

Outputs/action ideas:

```text
class-pair contrast
unique class read
suppress shared slot
increase compare for confused pair
```

### 3.3 Local analyzers / workshops

Each workshop proposes actions for its own domain.

```text
ReadWorkshop
PrimitiveWorkshop
VariantWorkshop
RouteBoundaryWorkshop
WriteWorkshop
MemoryWorkshop
HeadWorkshop
BudgetWorkshop
```

Workshops do not directly apply actions. They only propose candidates.

### 3.4 CandidateActionGenerator

Takes trace/projections and produces sparse candidate actions.

Candidate sources:

```text
gradient/saliency
consumer map problems
head/confusion problems
memory problems
boundary/segment problems
revive suppressed components
safe exploration
experience memory retrieval
```

Candidate schema:

```json
{
  "id": "run_epoch_window_candidate",
  "source": "gradient|consumer|head|memory|boundary|revive|explore|memory_retrieval",
  "target_type": "read|primitive|variant|route|boundary|write|memory|head|step_alive",
  "location": {"t": 6, "lane": "state", "cell": null},
  "action": "increase|decrease|shift|revive|protect|open|close|split|merge",
  "target": "route.memory->state or primitive.compare or read.input",
  "delta_scale": 0.03,
  "context_embedding": "small vector saved separately",
  "expected_effect": "reduce yes/no confusion by increasing compare",
  "complexity_delta": 0.004,
  "risk_flags": ["same_softmax_group", "memory_overwrite"],
  "requires": ["memory lane not dead"],
  "conflicts_with": ["decrease same route"]
}
```

### 3.5 ActionCoordinator

Receives candidates from all workshops and selects safe non-conflicting candidates.

MVP:

```text
max 1 action per window
max 1 workshop touched
scale <= 0.03
```

Later:

```text
max 2-3 actions per window
compound actions allowed only if components were tested separately
```

Conflict examples:

```text
increase memory write while decreasing all future memory reads
activate macro while suppressing required primitive
increase boundary and decrease step_alive of same step
increase input read while shortcut penalty is already high
```

### 3.6 CounterfactualScreen

Tests candidates before deployment.

Cheap screen:

```text
baseline heldout microbatch loss
apply temporary action bias
candidate heldout microbatch loss
revert
```

Verify for top candidates:

```text
save full training state
apply candidate
run N mini-steps or validation window
measure heldout / subproblem gain
rollback or keep
```

Rollback must restore:

```text
model weights
optimizer state
AMP scaler
scheduler state
RNG
running stats if any
feedback bias if needed
```

### 3.7 FeedbackBiasBank

Returns accepted/rejected information to exact choice locations.

Bias tensors:

```text
feedback_read_bias[t,lane,source]
feedback_primitive_bias[t,lane,primitive]
feedback_variant_bias[t,lane,primitive,variant]
feedback_route_bias[t,from_lane,to_lane]
feedback_boundary_bias[t]
feedback_write_bias[t,lane]
feedback_step_alive_bias[t]
feedback_head_bias[class/lane/pair]
```

Rules:

```text
small scale: 0.01-0.05
clamp: e.g. [-0.10,0.10]
decay each window: e.g. 0.95
context-gated: stronger only when current context matches tested context
accepted -> positive bias
rejected -> negative bias or cooldown
uncertain -> no deployment, maybe more probing
```

### 3.8 ExperienceMemory

Stores tested actions, not just logs.

Record:

```json
{
  "context_embedding": "...",
  "action_embedding": "...",
  "action": "increase route memory->state at T6",
  "status": "accepted|rejected|uncertain",
  "heldout_gain": 0.012,
  "noise_std": 0.004,
  "z_score": 3.0,
  "complexity_delta": 0.004,
  "global_val_impact": 0.003,
  "subproblem_gain": {"yes_no_margin": 0.018},
  "epoch": 7,
  "run_id": "..."
}
```

Memory types:

```text
short-term EMA memory for current run
episode JSONL memory for tested actions
critic training set
```

### 3.9 Critic

Critic predicts whether an action will help in a context.

Input:

```text
context_embedding + action_embedding + budget/complexity features
```

Output:

```text
predicted_gain
uncertainty
risk
```

Cold start:

```text
ridge / Bayesian linear critic first
MLP ensemble only after enough records
```

Scoring:

```text
probe/UCB score  = mean_gain + beta*uncertainty - risk - complexity
deploy/LCB score = mean_gain - beta*uncertainty - risk - complexity
```

UCB is for exploration/probing only. LCB is for live deployment.

### 3.10 Actor / Planner

Actor does not directly output dense tensors everywhere at first.

It proposes or biases sparse actions.

Start with:

```text
rule-based actor + critic ranking
```

Later:

```text
TapePlannerAttention over trace tokens
```

Planner tokens:

```text
step tokens
lane tokens
route tokens
primitive tokens
head feedback tokens
memory tokens
budget tokens
```

Planner output:

```text
small bias suggestions for read/primitive/variant/route/boundary/write/head
```

Important:

```text
planner output must be sparse or top-k masked;
planner must obey coordinator/budget;
planner live deployment uses LCB/trust region;
planner exploration uses UCB only in counterfactual tests.
```

---

## 4. Multi-projection actor/critic idea

The user wants many small projections trying different alternatives.

Implement as several small actors/critics, not one giant model.

```text
DataflowActor/Critic
PrimitiveActor/Critic
BoundaryActor/Critic
MemoryActor/Critic
HeadActor/Critic
BudgetCritic
```

Each actor proposes actions in its domain.
Each critic predicts gain/risk for its domain.
The global coordinator chooses among them.

Benefits:

```text
small models
clear responsibility
less dense noise
better interpretability
can train on domain-specific records
```

Diversity:

```text
actors must propose diverse, not identical, candidates
use action similarity penalty
use coverage by target_type/workshop/lane/step
keep exploration quota
```

Candidate mixture per window:

```text
40% best predicted / gradient supported
20% high uncertainty UCB probes
15% suppressed component revival
15% head/memory/task problem candidates
10% safe random low-cost exploration
```

Deployment remains conservative:

```text
only 1 action first;
then 2 actions if individual and pair tests are clean;
no dense all-workshop bias.
```

---

## 5. How elements learn from errors

There are three learning loops.

### 5.1 Gradient loop

Main differentiable model learns from task loss.

```text
loss -> gradients -> read/primitive/route/boundary/write/head parameters
```

This is always on.

### 5.2 Feedback bias loop

Tested edits modify choice bias.

```text
accepted action -> positive feedback bias
rejected action -> negative feedback bias/cooldown
```

This is fast, local, and interpretable.

### 5.3 Critic/actor loop

Critic learns from tested records.
Actor/planner learns which actions to propose.

```text
(context, action, real_gain) -> critic
critic -> better candidate ranking
actor -> better proposals
```

This is slower but generalizes across contexts/runs.

---

## 6. What information must be available to the heart

### 6.1 From input

```text
local/detail structure
onset/diff/wavelet-like structure
global/summary structure
noise/smoothness
low-rank/compressibility proxy
```

Use as weak context, not hard phase roles.

### 6.2 From head

```text
confusion matrix
class margins
class read overlap
class lane mass
class top reads
pair repair strength
per-class loss/accuracy
```

Head feedback says what the task needs.

### 6.3 From gradients

```text
grad * weight for each choice
logit gradients
update gradients
head gradients
suppressed component blind spots
```

### 6.4 From downstream consumers

```text
future reads from written lane/cell
head reads from slots produced by step/segment
route usage after write
memory read after memory write
```

### 6.5 From compute/budget

```text
step_alive cost
rank/depth cost
memory write cost
route entropy
operator complexity
wall-clock overhead
```

---

## 7. Utility and acceptance

Actions are accepted only if they are useful beyond noise.

Utility:

```text
utility = heldout_gain
        + subproblem_gain
        + functional_gain
        - complexity_cost
        - instability_penalty
        - duplicate_penalty
        - compute_cost
```

Dynamic threshold:

```text
noise_std = recent heldout microbatch std
threshold = max(k * noise_std, relative_min_gain * current_loss)
```

Accept:

```text
utility > threshold
and global val does not worsen
and complexity budget ok
and no safety violation
```

Reject:

```text
utility < -threshold
or global val worsens
or instability/overwrite risk too high
```

Otherwise:

```text
uncertain -> store but do not deploy
```

---

## 8. MVP sequence

Do not implement the full brain at once.

### MVP 0: address all choices

Make sure all selectable choices have named addresses:

```text
t/lane/source
 t/lane/primitive
 t/lane/variant
 t/from/to route
 t boundary
 t/lane write
 class/lane head
 class/pair head
```

### MVP 1: TraceCollector + ProjectionBank reports

No model-changing actions yet.

Produce:

```text
trace_feedback_epoch_XXX.json
candidate_suggestions_epoch_XXX.json
```

### MVP 2: Route/Boundary Heart

Start with the separator system because it is the core new idea.

Analyze:

```text
route importance
route consumer score
boundary usefulness
late input shortcut
```

Generate actions:

```text
increase/decrease route entries
increase/decrease boundary
open/close step_alive lightly
```

Screen on heldout microbatch. Do not deploy at first.

### MVP 3: FeedbackBiasBank for route/boundary only

Deploy max one accepted route/boundary edit per epoch/window.

Bias:

```text
feedback_route_bias
feedback_boundary_bias
```

### MVP 4: Primitive/Variant Heart

Add primitive and variant edits:

```text
primitive_bias[t,lane,p]
rank/depth/radius variant_bias
revive suppressed primitives
```

### MVP 5: Memory/Write Heart

Add:

```text
memory write usefulness
memory future consumer
memory overwrite protection
write target bias
```

### MVP 6: Head Heart

Add:

```text
class pair contrast edits
class unique lane read
suppress shared slot/lane
confidence calibration
```

### MVP 7: ExperienceMemory + cold-start critic

Store records and train ridge/Bayesian critic.

### MVP 8: Multi-projection actors

Add small per-workshop actors.

```text
DataflowActor
PrimitiveActor
BoundaryActor
MemoryActor
HeadActor
```

### MVP 9: TapePlannerAttention

Add global attention over trace tokens only after rule/counterfactual loop works.

### MVP 10: macro/growth

Only after critic is reliable:

```text
promote repeated useful action patterns into macro operators/head macros
```

---

## 9. Safety rules

Do not:

```text
use UCB for live deployment
apply dense actor bias everywhere
change many workshops at once in MVP
accept train-gradient-only edits
promote macros before counterfactual memory exists
let feedback use future information in same forward pass
rollback only model weights
let input/head priors become hidden hard roles
```

Always:

```text
small sparse actions
heldout screen
rollback capability
context-gated feedback
clamped bias
decay old bias
compute overhead report
```

---

## 10. What success looks like

Minimum success:

```text
trace shows clear responsibility of choices;
route/boundary actions can be screened without crash;
feedback bias changes exact route/boundary locations;
no training instability;
reports explain accepted/rejected actions.
```

Strong success:

```text
accepted route/boundary/primitive edits improve heldout;
critic predicted gain correlates with real gain;
late input shortcuts are controlled;
memory write is useful and later consumed;
class confusion actions improve margins;
step_alive activates only useful steps;
operator variants choose cheaper/smarter ranks/depths.
```

Research success:

```text
the system discovers reusable action patterns;
promotes macros safely;
builds stronger architectures with less manual priors;
can explain why it moved information, chose primitive, set boundary, or changed head read.
```

---

## 11. Short summary

The heart of the system is:

```text
observable choices
+ compressed trace projections
+ local candidate generators
+ global coordinator
+ counterfactual testing
+ feedback bias to exact locations
+ critic/actor learning over tested experience
```

The main model remains differentiable.
The heart adds understanding, testing, and targeted self-editing.

Final formula:

```text
trainable architecture
+ observable trace
+ tested actions
+ local feedback
+ learned critic/actor
= self-improving matrix program assembler
```
