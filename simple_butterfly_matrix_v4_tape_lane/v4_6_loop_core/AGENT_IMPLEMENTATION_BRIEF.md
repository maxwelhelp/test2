# v4.6 Agent Implementation Brief

This is the practical task list for the coding agent. Use this file together with:

```text
V4_6_FULL_LOCKED_SPEC.md
ARCHITECTURE_PLAN.md
AGENT_TASKS.md
README.md
```

The locked spec is the source of truth. If this brief and the locked spec conflict, follow `V4_6_FULL_LOCKED_SPEC.md` and then update this brief.

---

## 0. Mission

Build v4.6 as a new modular system under:

```text
simple_butterfly_matrix_v4_tape_lane/v4_6_loop_core/
```

Do not modify the old v4.3/v4.4/v4.5 files unless explicitly asked later.

The goal is not just higher accuracy. The goal is a closed matrix program assembly loop where decisions are made, executed, measured, and improved by gradient.

---

## 1. The closed-loop invariant

Always compare your code to this loop:

```text
Controller decides
    ↓
Backbone executes decision and writes/reads Memory
    ↓
Task loss + structural losses measure the effect
    ↓
Backprop sends credit to Controller / primitives / route / memory
    ↓
Controller makes better decisions next update
```

If any part is detached, converted to JSON, or only logged but not used in loss/gradient, then it is not part of the closed learning loop.

---

## 2. Four participants and their questions

### 2.1 Joint Controller asks Backbone

Question:

```text
What coupled decision should I make at step t?
```

It must answer jointly:

```text
boundary
transition / route
fanout / write
primitive scores
primitive signs
compose mode
memory read/write gates
```

Rule:

```text
These decisions must come from one shared latent z[t].
Do not implement independent unrelated controllers.
```

### 2.2 Backbone asks Memory Lane

Question:

```text
What do I remember from previous steps?
```

Memory answers through real matrix read:

```text
read[t] = W_read(M[t])
```

Backbone writes through real matrix write:

```text
M[t] = lambda * M[t-1] + (1 - lambda) * W_write(z[t])
```

Rule:

```text
Memory is not just route lane #4.
It must have W_read, W_write, and learnable forget.
```

### 2.3 StepAnalyzer asks Backbone

Question:

```text
What happened after the decisions at step t?
```

It logs:

```text
activations
decisions[t]
boundary/route/primitive/sign/memory stats
delta proxies
loss contribution proxies
collapse flags
```

Rule:

```text
StepAnalyzer logs; it does not train by itself.
Training credit is backprop.
```

### 2.4 Council v3 role

In v4.6.0, Council v3 is not an offline JSON trainer.

Allowed role:

```text
offline analyzer after run
report reader
suggestion generator
```

Not allowed in v4.6.0:

```text
Council writes JSON bias during training
Council replaces gradient credit
CouncilCalibrator drives controller through detached files
```

If a later version wants Council v3 to calibrate aux losses, it must do it as an in-graph differentiable module or as a carefully measured next-version experiment. Do not add it to MVP without evidence.

---

## 3. The questions the agent must ask at every implementation step

Before adding any module, answer these:

1. Does this module receive gradient from task loss or a clear differentiable aux loss?
2. Does it affect the actual forward execution, or only the logs?
3. Does it preserve the closed loop?
4. Does it create a shortcut: all-on boundary, all-to-all route, pure identity route, one primitive everywhere, junk memory?
5. Are its tensor shapes documented and tested?
6. Is it matrix/vectorized and efficient enough for P40?
7. Does the report prove specialization, not just accuracy?
8. Can another agent debug it without reading the entire old codebase?

If the answer is unclear, stop and add a smoke metric or validation check.

---

## 4. Required modular implementation

Create these files:

```text
v4_6_loop_core/
  modules/
    __init__.py
    joint_controller.py
    primitive_selector.py
    matrix_memory.py
    losses.py
    step_analyzer.py
  tape_lane_transport_v4_6_loop_core.py
  commands/
    validate_v4_6_loop_core.sh
    sync_run_5ep_v4_6_loop_core_push_logs.sh
```

No text patching. No monkey patching. No source string replacement.

---

## 5. Module responsibilities

### 5.1 `joint_controller.py`

Implement `JointController`.

Input:

```text
lane_state: [B,L,D]
memory_read: [B,D]
step index / step embedding
optional evidence/context
```

Output:

```text
z: [B,L,D] or [B,D]
boundary_logit: [B]
route_logits: [B,L,L]
write_logits: [B,L]
primitive_scores: [B,L,K]
primitive_sign_logits: [B,L,K]
compose_logits: [B,L,n_modes]
memory_write_logits: [B,L] or [B]
memory_read_gate: [B] or [B,L]
```

Critical rule:

```text
All heads branch from the same shared trunk/latent.
```

### 5.2 `primitive_selector.py`

Implement `ParallelPrimitiveSelector`.

Must include:

```text
low-rank experts, rank default 32 or 48
signed outputs with tanh(sign)
gumbel_softmax selection
temperature input tau
logs of primitive weights/signs/top1/entropy
```

Do not use full-rank `[D,D]` experts in MVP.

### 5.3 `matrix_memory.py`

Implement `MatrixMemory`.

Minimum:

```text
W_write: Linear(D,D,bias=False)
W_read: Linear(D,D,bias=False)
forget_logit: Parameter
mem_state: [B,D]
```

Forward:

```text
forget = sigmoid(forget_logit)
write_vec = W_write(write_input)
mem_next = forget * mem_prev + (1 - forget) * write_vec
read_vec = W_read(mem_next)
```

Report:

```text
forget
write_norm
read_norm
read_influence_proxy
```

### 5.4 `losses.py`

Implement small, clear losses:

```text
task_loss
boundary_budget_loss
route_regularization
primitive_entropy_or_diversity_loss
program_cost
sign_balance_loss optional
memory_write_cost very small
cost_dominance metric
```

First run should not over-regularize. Prefer logs over strong penalties until collapse is observed.

### 5.5 `step_analyzer.py`

Implement report generation only.

Output files must include:

```text
metrics.csv
trace_epoch_XXX.json
REPORT_TO_CHATGPT.txt
final_report.json
train.log
AGENT_STATUS.md
```

Reports must show:

```text
boundary_validity
route_specialization
primitive_specialization
primitive_sign_usage
memory_read_influence
cost_dominance
collapse_flags
```

---

## 6. Efficient matrix rules

This project must stay matrix/vector oriented.

Required:

```text
batch operations, no Python loops over batch
experts computed in parallel
route applied with einsum or batched matmul
low-rank primitives instead of full dense expert matrices
small per-epoch JSON summaries, not huge tensors
AMP fp16 support
```

Acceptable loop:

```text
loop over T steps
```

Reason: memory is causal and depends on previous memory state.

Not acceptable:

```text
loop over batch samples
loop over every primitive with separate Python calls if it can be stacked
saving giant activation tensors to git
```

Parallel scan is future optimization only after memory update is proven affine and useful.

---

## 7. Gumbel temperature schedule

Mandatory.

Training loop must compute:

```python
tau = max(tau_min, tau_start * (tau_decay ** max(0, epoch - 1)))
```

Default:

```text
tau_start = 1.0
tau_min = 0.2
tau_decay = 0.92 or 0.95
```

Report `gumbel_tau` every epoch.

If primitives do not specialize, inspect tau before adding new architecture.

---

## 8. Boundary rule

Boundary must answer:

```text
Where does a stage transition happen?
```

Boundary must not become:

```text
always ON unlock for all routes
```

Required flags:

```text
BOUNDARY_EXPLOIT: mean too high / peak_count too high
BOUNDARY_DEAD: mean too low / no peaks
BOUNDARY_FLAT: no contrast
```

MVP uses sigmoid + budget loss.

If boundary still collapses after first logs, implement soft-topk boundary in v4.6.1.

---

## 9. Route rule

Route must not be all-to-all and must not be pure identity.

Track:

```text
route_entropy
self_route_mass
useful_transition_mass
disallowed_route_mass
```

Useful transitions:

```text
detail -> state
state -> abstract
memory -> state
abstract -> head/output path
```

If route identity collapse appears, reduce offdiag penalty and add useful-transition prior near boundary.

---

## 10. Primitive rule

Primitive selection must become specialized.

Track:

```text
primitive_entropy
primitive_top1_by_step_lane
primitive_top1_share
primitive_sign_negative_share
primitive_sign_by_step
```

Failures:

```text
PRIMITIVE_UNIFORM: no expert selected clearly
PRIMITIVE_COLLAPSE: same expert everywhere
SIGN_DEAD: signs near zero or all same direction
```

Do not increase model complexity before checking tau/rank/loss balance.

---

## 11. Memory rule

Memory is useful only if it is read later and affects output/loss.

Track:

```text
memory_write_norm
memory_read_norm
memory_read_influence_proxy
memory_forget
```

Failures:

```text
MEMORY_DEAD: no write and no read
MEMORY_JUNK: write high but read influence low
```

Do not call memory successful from write mass alone.

---

## 12. Build order for the agent

### Step 1: skeleton

Create module files and import smoke.

### Step 2: implement modules individually

Each module must have shape comments and a tiny self-test under `if __name__ == "__main__"` or a validation script path.

### Step 3: assemble `tape_lane_transport_v4_6_loop_core.py`

Use existing data loader style if convenient, but keep v4.6 architecture independent.

### Step 4: add train loop

Must support:

```text
--epochs
--train-limit
--val-limit
--batch-size
--eval-batch-size
--dim
--steps
--primitive-rank
--gumbel-tau-start
--gumbel-tau-min
--gumbel-tau-decay
--amp
--out-dir
```

### Step 5: add validation

`validate_v4_6_loop_core.sh` checks:

```text
py_compile
--help smoke
imports
no actor critic strings
no text patching markers
no checkpoints staged
```

### Step 6: add sync run

`sync_run_5ep_v4_6_loop_core_push_logs.sh` must:

```text
git pull --rebase --autostash
run validation
run 5 epochs
save logs
commit logs + source
refuse checkpoints
push branch
```

### Step 7: inspect logs and fix

After first run, agent must read `REPORT_TO_CHATGPT.txt`, `final_report.json`, `metrics.csv`, and `trace_epoch_XXX.json`, then write `NEXT_FIXES.md` with P0/P1/P2.

---

## 13. Required first command

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

---

## 14. Agent final report format

After implementation, agent must update:

```text
v4_6_loop_core/IMPLEMENTATION_STATUS.md
```

Required sections:

```text
What was implemented
What was intentionally not implemented
How closed loop is preserved
How memory is real memory
Where gradients flow
How to run smoke
Known risks
Next fixes after first logs
```

---

## 15. The main rule

Every commit must preserve this sentence:

```text
The controller decision affects execution, execution affects loss, loss gradient updates the controller decision path.
```

If this is not true, the implementation is not v4.6.
