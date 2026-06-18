# v4.6 FULL LOCKED SPEC — Differentiable Program Assembly Loop

This document is the canonical v4.6 plan for agents. It expands `README.md`, `ARCHITECTURE_PLAN.md`, and `AGENT_TASKS.md`.

Do not implement a different architecture without updating this document first.

---

## 1. Short summary

v4.6 is not another JSON-bias experiment.

v4.6 is a closed differentiable program assembly loop:

```text
context[t,L,d]
  -> JointController -> shared latent z[t,L,d]
  -> parallel signed primitive experts
  -> differentiable sparse selection via gumbel_softmax
  -> coupled route/boundary/fanout/write/memory
  -> task loss
  -> backprop credit through every decision
```

Main principle:

```text
if the decision is inside forward() and differentiable, credit = backprop
```

Offline Council can remain an analyzer/report generator after training, but it must not drive training through JSON feedback in v4.6.0.

---

## 2. What went wrong before

Previous v4.x attempts produced useful signals, but they did not close the loop.

### 2.1 Broken Council loop

Old shape:

```text
Council offline
  -> writes JSON biases
  -> model uses or compensates them
  -> Council does not get real gradient or reliable credit
```

Problem:

```text
Council never learns whether its feedback helped.
The model can compensate constant bias in ~several updates.
JSON bias becomes external noise instead of internal credit.
```

v4.6 decision:

```text
No CouncilCalibrator in training loop.
No JSON feedback as training signal.
Backprop is the credit assignment.
```

### 2.2 Independent controllers

Old shape:

```text
boundary controller
transition controller
fanout controller
primitive controller
memory lane
```

Problem: decisions are coupled in a program, but networks were independent. This allows contradictions:

```text
boundary = ON
route = identity
fanout = all-soft
primitive = unrelated
memory = write junk
```

v4.6 decision:

```text
one JointController produces one shared latent z[t]
all heads branch from the same z[t]
```

### 2.3 Boundary exploit

Observed failure:

```text
boundary_mean near 1
boundary_peak_count = all steps
BOUNDARY_EXPLOIT
```

Meaning:

```text
boundary stopped being a stage separator.
it became a permanent unlock for cross-lane mixing.
```

v4.6 must report and penalize this.

### 2.4 Memory was not real memory

Old memory lane was often just one of four lanes. That is route storage, not matrix memory.

Real memory requires:

```text
W_write: state/z -> memory
W_read:  memory -> state/z
learnable forget lambda
optional address/key later
```

### 2.5 Primitive choice was too soft or collapsed

Observed failure:

```text
one primitive dominates many steps
or weights remain too soft/uniform
```

v4.6 uses:

```text
low-rank experts
signed outputs
gumbel_softmax selection
temperature annealing
primitive diversity logs
```

---

## 3. Key conceptual decisions

### 3.1 Actor-critic is not needed in v4.6.0

Actor-critic is useful when choices are hard discrete and gradients do not pass, or when long-horizon policy planning is needed.

v4.6 uses gumbel_softmax, a differentiable relaxation:

```text
primitive_scores -> gumbel_softmax -> primitive mixture -> loss -> gradient
```

So actor-critic would add:

```text
extra value network
PPO/A3C instability
more hyperparameters
harder debugging
```

Actor-critic is reserved for later:

```text
v4.7 or v4.8, only if hard planning / lookahead is introduced
```

### 3.2 Projection tree is not v4.6.0

A deep projection tree has exponential path count:

```text
K experts, depth D => K^D paths
K=8, D=3 => 512 paths
```

Multiple sequential gumbel choices can be unstable. Also, a pure tree loses causal memory unless carefully designed.

v4.6 uses time-unrolled steps:

```text
step 0 writes memory
step 1 reads memory and transforms
step 2 refines
```

This already creates hierarchy, but with causal memory and clear logs.

### 3.3 Steps are not bad by themselves

The current steps are bad when:

```text
all steps do the same thing
boundary is ON everywhere
route/read deltas are tiny
primitive selection does not specialize
memory is not causally used
```

But steps are necessary for causal memory:

```text
M[t] depends on M[t-1]
```

So do not remove steps before memory semantics work.

Correct interpretation:

```text
bad: repeated identical soft mixer steps
 good: causal program steps with changing read/primitive/route/write/memory
```

### 3.4 Parallel scan is future optimization, not MVP semantics

If memory update is affine:

```text
M[t] = A[t] * M[t-1] + B[t] * x[t]
```

then it can be optimized with prefix/parallel scan later.

But v4.6.0 should use a simple unrolled loop first for correctness, logging, and debugging.

Parallel scan target:

```text
v4.6.2+ or later, after memory update is proven and mostly affine
```

---

## 4. Full target architecture diagram

```text
┌──────────────────────────────────────────────────────────────────────┐
│                      v4.6 DIFFERENTIABLE LOOP                         │
└──────────────────────────────────────────────────────────────────────┘

Input / feature extractor / backbone state
        │
        ▼
┌──────────────────────────────────────┐
│ context[t, L, d]                     │
│ L = lanes:                           │
│   0 detail                           │
│   1 state                            │
│   2 abstract                         │
│   3 memory interface                 │
└───────────────┬──────────────────────┘
                │
                ▼
┌──────────────────────────────────────────────────────────────────────┐
│ JointController                                                       │
│                                                                      │
│ context[t,L,d] -> shared latent z[t,L,d] or z[t,d]                   │
│                                                                      │
│ from same z:                                                          │
│   boundary_head                                                       │
│   route/transition_head                                               │
│   fanout/write_head                                                   │
│   primitive_score_head                                                │
│   primitive_sign_head                                                 │
│   compose_mode_head                                                   │
│   memory_read_write_head                                              │
└───────────────┬──────────────────────────────────────────────────────┘
                │ coupled decisions for step t
                ▼
┌──────────────────────────────────────────────────────────────────────┐
│ ParallelPrimitiveSelector                                             │
│                                                                      │
│ experts computed in parallel:                                         │
│   E0 identity/channel                                                 │
│   E1 low_rank                                                         │
│   E2 diff/shift                                                       │
│   E3 smooth/local                                                     │
│   E4 contrast                                                         │
│   E5 context_matrix                                                   │
│   E6 product_gate                                                     │
│   E7 memory_read/keep                                                 │
│                                                                      │
│ score_k = scorer(z, expert_context)                                  │
│ w_k = gumbel_softmax(score, tau)                                      │
│ sign_k = tanh(sign_head(z)_k or sign_logit_k)                         │
│ y = sum_k w_k * sign_k * E_k(x)                                       │
└───────────────┬──────────────────────────────────────────────────────┘
                │ signed primitive update
                ▼
┌──────────────────────────────────────────────────────────────────────┐
│ Compose + Route + Boundary + Write                                    │
│                                                                      │
│ compose modes:                                                        │
│   replace                                                             │
│   add                                                                 │
│   subtract                                                            │
│   gated_add                                                           │
│                                                                      │
│ route: R[t,L,L]                                                       │
│ boundary: sparse stage marker                                         │
│ write: lane write gates                                               │
└───────────────┬──────────────────────────────────────────────────────┘
                │
                ▼
┌──────────────────────────────────────────────────────────────────────┐
│ MatrixMemory                                                          │
│                                                                      │
│ M[t] = λ * M[t-1] + (1-λ) * W_write(z[t])                             │
│ read[t] = W_read(M[t])                                                │
│                                                                      │
│ optional later: address/key/query                                     │
└───────────────┬──────────────────────────────────────────────────────┘
                │
                ▼
┌──────────────────────────────────────────────────────────────────────┐
│ Task output / classifier head                                         │
│                                                                      │
│ CE/MSE/task loss                                                      │
│ + soft program regularizers                                           │
└───────────────┬──────────────────────────────────────────────────────┘
                │
                ▼
           BACKPROP CREDIT
                │
                └──────────────► JointController
                                  primitive experts
                                  signs
                                  route/boundary/write
                                  memory W_read/W_write/forget
```

---

## 5. Step-t diagram

```text
Step t
──────

context[t,L,d]
      │
      ▼
┌─────────────────────────────┐
│ JointController             │
│ shared z[t]                 │
└────────────┬────────────────┘
             │
             ▼

             z[t]
              │
              ├──────────────► boundary score
              ├──────────────► route/fanout score
              ├──────────────► write gate
              ├──────────────► memory gate
              ├──────────────► primitive scores
              └──────────────► primitive signs

Parallel experts, one forward pass:

┌────────────┐ ┌────────────┐ ┌────────────┐ ┌────────────┐ ┌────────────┐
│ E0 identity│ │ E1 lowrank │ │ E2 diff    │ │ E3 ctx_mat │ │ E4 product │
│ + W0 x     │ │ + V1U1 x   │ │ -/ + diff  │ │ ctx-gated  │ │ gated comp │
└─────┬──────┘ └─────┬──────┘ └─────┬──────┘ └─────┬──────┘ └─────┬──────┘
      │              │              │              │              │
      └──────────────┴──────────────┴──────────────┴──────────────┘
                              │
                              ▼
┌──────────────────────────────────────────────────────────────────────┐
│ signed sparse mixture                                                 │
│                                                                      │
│ scores = primitive_score_head(z)                                      │
│ w = gumbel_softmax(scores, tau)                                       │
│ sign = tanh(primitive_sign_head(z))                                   │
│ y = sum_k w_k * sign_k * E_k(x)                                       │
└──────────────────────────────┬───────────────────────────────────────┘
                               │
                               ▼
┌──────────────────────────────────────────────────────────────────────┐
│ compose y into lane state                                             │
│ route to lanes                                                        │
│ optionally write/read memory                                          │
│ produce next context                                                  │
└──────────────────────────────┬───────────────────────────────────────┘
                               │
                               ▼
                            loss later
                               │
                               ▼
                           backprop
```

---

## 6. Tensor contracts

Default symbols:

```text
B = batch size
T = program steps / tape steps
L = lanes, default 4
D = model dimension
K = primitive experts
R = low rank, default 32..48
M = memory slots, default 1 for MVP or 4 later
C = classes
```

Core tensors:

```text
context:              [B, T, L, D] or generated per step [B,L,D]
z:                    [B, L, D] or [B,D]
primitive_scores:     [B, L, K]
primitive_weights:    [B, L, K]
primitive_signs:      [B, L, K]
expert_outputs:       [B, L, K, D]
signed_update:        [B, L, D]
route:                [B, L, L]
boundary:             [B] or [B,1]
write_gate:           [B, L]
memory_state:         [B, M, D] or [B,D] for MVP
memory_read:          [B, D]
classifier_logits:    [B, C]
```

Important: if using per-step logs, save CPU-detached summaries, not giant tensors.

---

## 7. JointController detailed contract

### Inputs

```text
lane_state[t]:       [B,L,D]
memory_read[t]:      [B,D] or [B,L,D] broadcast
step_embedding[t]:   [D]
optional evidence:   [B,D]
```

Construct context:

```text
ctx[t,L,D] = lane_state + lane_emb + step_emb + memory_read_projection + optional evidence
```

### Outputs

```text
z:                   [B,L,D]
boundary_logit:      [B]
route_logits:        [B,L,L]
write_logits:        [B,L]
fanout_logits:       [B,L]
primitive_scores:    [B,L,K]
primitive_sign_logits:[B,L,K]
compose_logits:      [B,L,n_modes]
memory_write_logits: [B,L] or [B]
memory_read_gate:    [B] or [B,L]
```

### Rule

All outputs must come from the same latent `z` or a shared trunk. No separate unrelated controller networks.

---

## 8. ParallelPrimitiveSelector detailed contract

### Expert types for MVP

Minimum viable list:

```text
0 identity:      y = x
1 channel:       y = channel_mlp(x)
2 low_rank:      y = V(Ux)
3 diff_prev:     y = x - prev_lane_or_prev_step
4 smooth:        y = local/smoothing/channel mix
5 contrast:      y = x - mean/channel baseline
6 ctx_matrix:    y = context-gated linear/low-rank
7 product_gate:  y = gate(x,z) * low_rank2(low_rank1(x))
8 memory_read:   y = W_mem(memory_read)
```

For v4.6.0, keep it small if needed:

```text
identity, channel, low_rank, diff_prev, contrast, ctx_matrix, product_gate, memory_read
```

### Low-rank form

```text
h_k = x @ U_k      where U_k: [D,R]
y_k = h_k @ V_k    where V_k: [R,D]
```

For batched experts:

```python
h = einsum('bld,kdr->blkr', x, U)
y = einsum('blkr,krd->blkd', h, V)
```

### Why low rank

Full `[D,D]` per expert is too free. Experts can all converge to similar dense transforms.

Low-rank bottleneck forces different structural roles.

Recommended:

```text
D=128 -> R=24..32
D=256 -> R=32..48
D=512 -> R=48..64, only after smoke
```

### Scoring

Two acceptable MVP scoring variants:

Variant A:

```text
scores = primitive_score_head(z)
```

Variant B:

```text
scores_k = dot(score_proj(z), expert_summary_k)
```

Use A first for simplicity.

### Selection

Training:

```text
w = F.gumbel_softmax(scores, tau=tau, hard=False)
```

Evaluation/report:

```text
top1 = argmax(w)
```

Do not use hard=True first. It can make training less stable.

### Signs

```text
sign = tanh(sign_logits)
```

Optional coefficient:

```text
coef = sigmoid(coef_logits)
out = sum(w * coef * sign * y_k)
```

MVP can use sign only.

---

## 9. Compose modes

Action per step/lane includes compose mode.

MVP modes:

```text
replace:   new = y
add:       new = x + y
subtract:  new = x - y
 gated_add:new = x + gate*y
```

Differentiable mixture:

```text
mode_w = softmax(compose_logits)
new = sum_m mode_w[m] * compose_m(x,y)
```

If too complex for v4.6.0, start with gated_add only:

```text
new = x + sigmoid(write_gate) * y
```

But keep compose mode in design so agents do not forget it.

---

## 10. Route and boundary

### Route

```text
route_logits: [B,L,L]
route = softmax(route_logits, dim=-1)
```

Interpretation:

```text
source lane -> target lane
```

Route application:

```text
routed[target] = sum_source route[source,target] * update[source]
```

### Boundary

Boundary means stage transition. It should not be always ON.

MVP:

```text
boundary = sigmoid(boundary_logit)
```

but with strong reporting and mild budget loss.

v4.6.1:

```text
soft-topk boundary K=2..4
```

### Boundary losses

MVP:

```text
boundary_count = sum_t boundary[t]
budget_loss = relu(boundary_count - max_peaks)^2 + relu(min_peaks - boundary_count)^2
flat_loss optional = penalty if boundary std too low and mean high
```

Do not make boundary budget too strong at epoch 1. It can kill boundary before specialization.

### Boundary failure flags

```text
BOUNDARY_EXPLOIT if mean > 0.80 and peak_count high
BOUNDARY_DEAD    if mean < 0.05 or peak_count = 0
BOUNDARY_FLAT    if std very low and count not valid
```

---

## 11. MatrixMemory detailed contract

### MVP memory state

Use one vector memory first:

```text
mem: [B,D]
```

Update:

```text
forget = sigmoid(forget_logit)
write = W_write(memory_write_input)
mem_next = forget * mem + (1 - forget) * write
read = W_read(mem_next)
```

### Memory input

Good options:

```text
memory_write_input = pooled z[t]
```

or:

```text
memory_write_input = weighted sum of lane updates using memory_write_gate
```

### Later memory slots

After MVP:

```text
mem: [B,M,D]
query = W_q(z)
key = W_k(mem)
attn = softmax(query*key)
read = sum(attn * W_v(mem))
```

Do not do this in v4.6.0 unless MVP is stable.

### Memory usefulness

Do not call memory useful just because memory write is high.

Need evidence:

```text
memory_write_norm nonzero
memory_read_norm nonzero
memory_read_influence nonzero
ablate memory => loss gets worse
```

MVP proxy:

```text
memory_read_influence_proxy = norm(head_with_mem - head_without_mem) or norm(state_with_read - state_without_read)
```

v4.6.1:

```text
causal memory ablation per step
```

---

## 12. Gumbel temperature schedule

Mandatory.

Default:

```text
tau_start = 1.0
tau_min = 0.2
tau_decay = 0.92 or 0.95
```

Function:

```python
def tau_for_epoch(epoch):
    return max(tau_min, tau_start * (tau_decay ** max(0, epoch-1)))
```

Fast smoke alternative:

```text
tau = max(0.2, 1.0 - 0.08 * epoch)
```

Report every epoch:

```text
gumbel_tau
primitive_entropy
primitive_top1_share
```

If tau is too high:

```text
primitive weights remain diffuse
no specialization
```

If tau is too low too early:

```text
premature collapse
one primitive wins everywhere
```

---

## 13. Loss design

### Primary

```text
task_loss = CE or MSE
```

### Minimal auxiliary losses

```text
boundary_budget_loss
route_entropy_band or route_allowed_loss
primitive_entropy/diversity_loss
program_cost
memory_write_cost very small
logit_norm small
```

### Do not over-regularize

Too many costs can make the model optimize the report instead of the task.

First smoke priority:

```text
see natural behavior
then add penalties based on observed collapse
```

### Cost dominance report

Always log:

```text
aux_loss_total / task_loss
```

Flag:

```text
COST_DOMINANCE if aux/task > threshold, e.g. 0.5 early
```

---

## 14. StepAnalyzer contract

StepAnalyzer is not a trainer. It only logs.

Required summaries:

```text
boundary_by_step
boundary_mean
boundary_std
boundary_peak_count
route_matrix_mean_by_step
route_entropy_by_step
self_route_mass
useful_transition_mass
disallowed_route_mass
primitive_weights_by_step_lane
primitive_top1_by_step_lane
primitive_entropy_by_step_lane
primitive_sign_by_step_lane
primitive_sign_negative_share
compose_mode_by_step_lane
memory_forget
memory_write_norm_by_step
memory_read_norm_by_step
memory_read_influence_proxy
head_lane_attention or topread if present
cost_breakdown
```

Required files:

```text
metrics.csv
trace_epoch_XXX.json
REPORT_TO_CHATGPT.txt
final_report.json
train.log
AGENT_STATUS.md
```

---

## 15. Report interpretation

### Good signs

```text
boundary has 2-4 peaks, not all steps
primitive top1 differs by step/lane
negative signs appear in correction-like experts
route entropy below uniform but not identity collapse
useful_transition_mass grows
memory read influence grows
CE improves without aux dominance
```

### Bad signs

```text
BOUNDARY_EXPLOIT
BOUNDARY_DEAD
ROUTE_UNIFORM
ROUTE_IDENTITY_COLLAPSE
PRIMITIVE_COLLAPSE
PRIMITIVE_UNIFORM
MEMORY_DEAD
MEMORY_JUNK
DETAIL_SHORTCUT
COST_DOMINANCE
```

### Do not declare success from accuracy alone

A model can improve accuracy through shortcuts:

```text
detail top-read shortcut
boundary all-on
memory as slow residual
one primitive everywhere
```

v4.6 success means structural specialization.

---

## 16. Current steps: what is wrong and how v4.6 fixes it

### Problem: same behavior at every step

Symptoms:

```text
route_delta small
read_delta small
primitive_delta small
boundary all-on or flat
```

Fix:

```text
joint z[t] includes step embedding
primitive selector uses gumbel annealing
boundary budget forces sparse stage markers
StepAnalyzer reports deltas
```

### Problem: no causal memory

Symptoms:

```text
memory_write high but no later read influence
```

Fix:

```text
MatrixMemory W_write/W_read/forget
read path must affect state/output
memory ablation later
```

### Problem: soft all-to-all route

Symptoms:

```text
route entropy near log(L)
disallowed route high
```

Fix:

```text
route loss mild
boundary-coupled route
useful transition metrics
```

### Problem: primitive mixture does not specialize

Symptoms:

```text
primitive entropy high forever
or same top1 everywhere
```

Fix:

```text
gumbel tau annealing
low-rank experts
sign usage
primitive diversity loss after first logs
```

---

## 17. Implementation folder rules

All new v4.6 code should live under:

```text
simple_butterfly_matrix_v4_tape_lane/v4_6_loop_core/
```

Do not alter these files during v4.6.0 implementation:

```text
tape_lane_transport_v4_3_min_heart.py
tape_lane_transport_v4_4_*.py
v4_5_projection_feedback files
```

No text patching.

No monkey patching.

Use normal imports and classes.

---

## 18. Implementation order for agents

### Phase A: module skeleton

Create:

```text
modules/__init__.py
modules/joint_controller.py
modules/primitive_selector.py
modules/matrix_memory.py
modules/losses.py
modules/step_analyzer.py
```

Add standalone import smoke.

### Phase B: core model

Create:

```text
tape_lane_transport_v4_6_loop_core.py
```

Minimum model:

```text
feature extractor from old loader or simple frontend
lane state init
T step loop
JointController
PrimitiveSelector
MatrixMemory
classifier head
```

### Phase C: train loop

Add:

```text
--epochs
--train-limit
--val-limit
--batch-size
--dim
--steps
--primitive-rank
--gumbel-tau-start
--gumbel-tau-min
--gumbel-tau-decay
--out-dir
```

### Phase D: reports

Add mandatory reports:

```text
metrics.csv
trace_epoch_XXX.json
REPORT_TO_CHATGPT.txt
final_report.json
```

### Phase E: commands

Create:

```text
commands/validate_v4_6_loop_core.sh
commands/sync_run_5ep_v4_6_loop_core_push_logs.sh
```

---

## 19. First smoke command

Target command:

```bash
EPOCHS=5 \
TRAIN_LIMIT=12000 \
VAL_LIMIT=2000 \
BATCH_SIZE=128 \
EVAL_BATCH_SIZE=256 \
AMP=fp16 \
bash simple_butterfly_matrix_v4_tape_lane/v4_6_loop_core/commands/sync_run_5ep_v4_6_loop_core_push_logs.sh
```

If GPU memory allows:

```bash
BATCH_SIZE=192 EVAL_BATCH_SIZE=512
```

But default should be safe.

---

## 20. First smoke success criteria

Accuracy is secondary.

Required structural checks:

```text
boundary_peak_count not 0 and not T
boundary_mean not > 0.8
route_entropy below uniform but self_route_mass not > 0.9
primitive_entropy decreasing over epochs
primitive top1 not same for all steps/lanes
negative sign share > 0.05 after some training
memory_read_influence_proxy > 0
aux/task ratio reasonable
```

If these fail, do not add actor-critic or tree. Fix the failing mechanism.

---

## 21. Escalation rules

### If boundary all-on

Do:

```text
increase boundary budget
add flatness penalty
v4.6.1 soft-topk boundary
```

Do not:

```text
add Council JSON
add actor-critic
```

### If primitive collapse

Do:

```text
check tau schedule
increase tau early or reduce decay speed
add diversity by lane/stage
lower rank if experts too dense
```

### If primitive uniform

Do:

```text
anneal tau faster
add mild entropy target
increase score head capacity
```

### If memory dead

Do:

```text
ensure memory read enters state/output
reduce memory write cost
increase memory gate path
```

### If memory junk

Do:

```text
add causal read influence report
add write only if future-read loss later
```

### If route all-to-all

Do:

```text
increase route allowed/disallowed penalty mildly
make boundary actually control cross-lane transitions
```

### If route identity collapse

Do:

```text
reduce offdiag penalty
increase useful transition prior near boundary
```

---

## 22. Later versions

### v4.6.1

```text
soft-topk boundary
primitive diversity by lane/stage
memory causal ablation
detail topread loss
```

### v4.6.2

```text
connect program_assembly_curriculum_v2_structured
transfer only controller/primitive priors
never load full builder checkpoint into task model
```

### v4.7

```text
harder decisions
possible actor-critic only if hard planning is introduced
multi-step lookahead
```

### v4.8

```text
two-level primitive tree
program compiler
parallel scan optimization if memory update is affine
```

---

## 23. Final architecture mantra

```text
No broken feedback loop.
No independent contradictory controllers.
No JSON as training credit.
No actor-critic before hard decisions.
No projection tree before simple loop works.
No full-rank primitive experts in MVP.
No success claim from accuracy alone.
```

v4.6 must prove:

```text
boundary specializes
primitive experts specialize
signs are used
route is structured
memory is causally read
backprop trains the whole assembly loop
```
