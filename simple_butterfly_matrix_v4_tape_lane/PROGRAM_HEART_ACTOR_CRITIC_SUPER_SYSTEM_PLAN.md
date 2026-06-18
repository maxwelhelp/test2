# Program Heart / Actor-Critic / Super-System Plan

This file describes the long-term intelligent architecture around the tape-lane matrix program.

It is not the immediate runtime task. The immediate task is still v4.3 canonical main stabilization.

Repository:

```text
maxwelhelp/test2
```

Main project folder:

```text
simple_butterfly_matrix_v4_tape_lane/
```

Related plans:

```text
simple_butterfly_matrix_v4_tape_lane/CURRENT_ROADMAP_V4_3_TO_V4_5.md
simple_butterfly_matrix_v4_tape_lane/V4_3_MIN_HEART_PLAN.md
v4_3_min_heart/README.md
v4_4_min_learning_heart/README.md
```

---

## 0. Where we are now

Current stage:

```text
v4.3 canonical main stabilization
```

Current priority:

```text
Make tape_lane_transport_v4_3_min_heart.py run smoke and 5ep cleanly as the only standard runtime entrypoint.
```

Known recent blocker:

```text
NameError: _candidate_key is not defined in generate_candidate_suggestions()
```

Do not start Actor/Critic until this is fixed and a v4.3 canonical smoke run reaches `run_status=0`.

---

## 1. The intended super-system

The final system has three learning levels.

### Level 1 — Main differentiable matrix program

This is the actual model:

```text
tape/lane/cell state
read -> transform/primitive -> write -> route/boundary -> head
```

It learns by normal gradient descent.

Current components:

```text
read_group logits
route logits
boundary logits
primitive weights
write gates
step_alive
memory lane
class head
```

Future components:

```text
context controllers
matrix memory banks
primitive/route/boundary credit memories
```

### Level 2 — Program Heart

This observes the main model and proposes safe sparse changes.

Pipeline:

```text
TraceCollector
-> Program Quality Score
-> CandidateGenerator
-> ForwardOnlyScreen
-> ActionHistory
-> SimpleBandit
-> FeedbackBiasBank later
```

At first it must be observer-only and no-deploy.

### Level 3 — Meta-Heart

This observes the Program Heart itself.

It answers:

```text
Is Heart useful?
Is candidate diversity collapsing?
Is critic calibrated?
Are projections redundant?
Should exploration increase?
Should delta_scale/window/beta change?
```

Meta-Heart comes much later.

---

## 2. Program Heart stages

### Stage H0 — v4.3 observer and trace

Status: currently being stabilized.

Goal:

```text
produce reliable trace_feedback_epoch_XXX.json
produce candidate_suggestions_epoch_XXX.json
no deploy
no feedback_bias
no actor
no critic
```

Required output fields:

```text
sequence_nonflat_score
read_delta_by_step
route_delta_by_step
boundary_usefulness
self_route_mass
useful_transition_mass
collapse_flags
memory_consumer_proxy
detail_attention_mass
candidate duplicate metrics
```

### Stage H1 — v4.4 minimal learning heart

Implement only after v4.3 smoke and 5ep pass.

Add:

```text
Program Quality Score, PQS
action_history.jsonl
forward-only screen
simple bandit
no deploy
```

PQS should include:

```text
+ route specialization
+ useful transition mass
+ boundary usefulness
+ memory consumer proxy
+ lane diversity
+ sequence nonflatness
- route uniformity
- memory junk
- detail shortcut
- complexity / useless write
```

Forward-only screen protocol:

```text
1. freeze model weights
2. choose small heldout microbatch
3. compute baseline loss/PQS
4. apply temporary candidate bias in no-grad / reversible context
5. forward only, no optimizer step
6. compute delta_loss and delta_PQS
7. store result in action_history.jsonl
8. remove temporary bias
```

No training during screen. No optimizer mutation. No Adam state mutation.

Simple bandit:

```text
rank candidate sources by historical gain
use epsilon-greedy/UCB-style exploration
but still no deploy by default
```

### Stage H2 — FeedbackBiasBank

Only after action_history has useful records.

Purpose:

```text
store accepted tiny bias shifts for specific choice logits
```

Formula:

```python
effective_logits = base_logits + learned_context_bias + feedback_bias.detach()
```

Rules:

```text
feedback_bias is detached
feedback_bias is clamped
feedback_bias decays
feedback_bias is context-gated
only one safe action per window at first
```

Allowed first targets:

```text
route edge bias
boundary step bias
primitive prior bias
read group bias
```

Disallowed initially:

```text
dense tensor edits
multiple simultaneous edits
weight surgery
optimizer state mutation
```

### Stage H3 — Critic

Only after enough screened action records exist.

Input:

```text
context_embedding
action_embedding
action_history summary
trace summary
PQS components
```

Output:

```text
predicted delta_loss
predicted delta_PQS
uncertainty
LCB/UCB score
```

Start with:

```text
ridge / Bayesian linear / small MLP only after enough data
```

Do not train critic on 20 noisy records.

Cold start rule:

```text
< 100 records: rule + bandit only
100-3000 records: ridge/Bayesian linear
> 3000 records: small MLP critic allowed
```

### Stage H4 — Actor

Only after critic is useful.

Actor does not output dense changes.

Actor outputs sparse candidate programs:

```json
[
  {
    "target_type": "route",
    "location": {"t": 6, "from": "memory", "to": "state"},
    "action": "increase",
    "delta_scale": 0.03
  }
]
```

Max actions:

```text
start: 1 action/window
later: 2 actions/window
```

Actor candidate sources:

```text
rule proposals
gradient proposals
collapse flag proposals
exploration proposals
dormant primitive proposals
critic-ranked proposals
```

Actor must preserve diversity:

```text
penalize duplicate candidates
penalize same workshop repeated too often
force exploration of untouched locations
```

### Stage H5 — EditorLoop

Only after H1-H4 are stable.

EditorLoop can make reversible structural edits:

```text
patch candidate
forward screen
optional short verify
accept/reject
rollback full state if reject
```

Rollback must include:

```text
model weights
optimizer state
scheduler state
AMP scaler state
RNG state
any running stats
```

### Stage H6 — Meta-Heart

Monitors Heart.

Trace fields:

```text
candidate_diversity_score
action_repeat_rate
critic_calibration
projection_correlation
workshop_coverage
heart_compute_cost
heart_roi
```

Meta-actions:

```text
increase exploration
reduce delta_scale
change active workshops
change window frequency
change UCB/LCB beta
force diversity
pause bad critic
```

---

## 3. Context controllers

Context controllers are part of Level 1, not Actor/Critic.

They make the main model more context-aware.

Correct pattern:

```python
final_logits = base_logits + alpha * controller_delta
```

Wrong pattern:

```python
final_logits = controller_delta
```

Controllers needed:

```text
primitive_context_controller
boundary_controller
route_controller
read_group_controller
write_gate_controller
halt/alive_controller
```

Safety:

```text
small alpha
bounded alpha
zero-init or near-zero-init controller output
old priors remain as base
```

Suggested learnable alpha:

```python
alpha = 0.01 + 0.09 * sigmoid(alpha_raw)
```

Context controllers should be tested after stable v4.3 canonical main.

---

## 4. Matrix memory

Matrix memory is Level 1 memory support and later Level 2 context support.

Do not implement before stable v4.3 and v4.4 Heart records.

Memory banks:

```text
working_matrix_memory
primitive_credit_memory
route_edge_memory
boundary_step_memory
editor/action_memory
```

### Working matrix memory

```text
slot_memory: [B, M, D], M=8 or 16
fast_U: [B, D, r], r=16
fast_V: [B, r, D]
```

Read:

```text
slot_read = softmax(q @ slots) @ slots
fast_read = (q @ U) @ V
memory_read = slot_read + fast_read
```

Write:

```text
U = decay * U + gate * outer(k, a)
V = decay * V + gate * outer(a, v)
slots = decay * slots + gated_write
```

### Primitive credit memory

```text
primitive_memory: [B, P, C], C=16 or 32
primitive_logits += primitive_memory_bias
```

### Route edge memory

```text
route_memory: [B, L, L, C]
L=4, so 16 edges
route_logits += route_memory_bias
```

### Boundary memory

```text
boundary_memory: [B, T, C]
boundary_logits += boundary_memory_bias
```

Expected effect:

```text
not immediate 10x speedup
+10-20% overhead at first
better program quality and less random all-to-all
possible later speedup if T/cells can be reduced
```

---

## 5. Strict execution order

Do not skip.

```text
1. v4.3 canonical main smoke status=0
2. v4.3 canonical main 5ep status=0
3. analyze logs
4. context-controller experiment
5. v4.4 minimal learning heart: PQS + action_history + forward-only screen + bandit, no deploy
6. v4.5 matrix memory
7. FeedbackBiasBank
8. Critic
9. Actor
10. EditorLoop
11. Meta-Heart
```

Current position:

```text
We are still at step 1.
```

No Actor/Critic coding until step 5 produces action records.

---

## 6. What the agent should do now

Immediate agent task:

```text
Fix v4.3 canonical main so smoke passes with run_status=0.
```

Files to inspect first:

```text
simple_butterfly_matrix_v4_tape_lane/tape_lane_transport_v4_3_min_heart.py
simple_butterfly_matrix_v4_tape_lane/commands/sync_run_5ep_v4_3_min_heart_push_logs.sh
simple_butterfly_matrix_v4_tape_lane/commands/validate_v4_3_min_heart.sh
simple_butterfly_matrix_v4_tape_lane/commands/grad_sanity_v4_3_min_heart.sh
simple_butterfly_matrix_v4_tape_lane/CURRENT_ROADMAP_V4_3_TO_V4_5.md
simple_butterfly_matrix_v4_tape_lane/PROGRAM_HEART_ACTOR_CRITIC_SUPER_SYSTEM_PLAN.md
```

Required before push:

```bash
python -m py_compile simple_butterfly_matrix_v4_tape_lane/tape_lane_transport_v4_3_min_heart.py
bash simple_butterfly_matrix_v4_tape_lane/commands/validate_v4_3_min_heart.sh
bash simple_butterfly_matrix_v4_tape_lane/commands/grad_sanity_v4_3_min_heart.sh
```

Required smoke:

```bash
DATA_ROOT=../architecture_builder/data/speechcommands \
EPOCHS=1 \
TRAIN_LIMIT=12000 \
VAL_LIMIT=2000 \
MAX_TRAIN_BATCHES=10 \
MAX_VAL_BATCHES=3 \
BATCH_SIZE=128 \
EVAL_BATCH_SIZE=256 \
WORKERS=4 \
AMP=fp16 \
bash simple_butterfly_matrix_v4_tape_lane/commands/sync_run_5ep_v4_3_min_heart_push_logs.sh
```

Only if smoke passes:

```bash
git add simple_butterfly_matrix_v4_tape_lane/
git commit -m "Stabilize v4.3 canonical main smoke"
git push
```
