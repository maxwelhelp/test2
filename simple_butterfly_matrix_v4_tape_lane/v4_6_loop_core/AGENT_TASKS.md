# v4.6 Loop Core Agent Tasks

This document is the contract for agents. Do not implement extra mechanisms before the listed MVP is working.

## Non-negotiable rules

1. Do not modify v4.3/v4.4/v4.5 main files while implementing v4.6.
2. Do not use text-patching or source string replacement.
3. Do not add actor-critic.
4. Do not add PPO/A3C.
5. Do not add projection trees.
6. Do not add offline JSON feedback as a training mechanism.
7. Do not push checkpoints to git.
8. Keep reports readable and compact.

## Files to create in implementation phase

```text
simple_butterfly_matrix_v4_tape_lane/v4_6_loop_core/
  tape_lane_transport_v4_6_loop_core.py
  modules/
    __init__.py
    joint_controller.py
    primitive_selector.py
    matrix_memory.py
    step_analyzer.py
    losses.py
  commands/
    validate_v4_6_loop_core.sh
    sync_run_5ep_v4_6_loop_core_push_logs.sh
```

## Task 1: JointController

Implement:

```text
context[t,L,d] -> z[t,L,d]
```

Heads:

```text
boundary_head
route_head / transition_head
fanout_head
primitive_score_head
primitive_sign_head
memory_read_write_head
```

Required logs:

```text
controller_z_norm
boundary_logits
route_logits
primitive_scores
primitive_sign_logits
memory_gate_logits
```

## Task 2: ParallelPrimitiveSelector

Implement low-rank experts:

```text
Y_k = V_k(W_k x)
rank = default 32 or 48
```

Implement signed gumbel mixture:

```text
w = gumbel_softmax(score, tau)
sign = tanh(sign_logit or sign_head)
out = sum(w * sign * Y)
```

Required logs:

```text
primitive_weights_by_step
primitive_top1_by_step
primitive_entropy
primitive_sign_mean
primitive_sign_by_step
gumbel_tau
```

## Task 3: MatrixMemory

Implement:

```text
M[t] = forget * M[t-1] + (1 - forget) * W_write(z[t])
read[t] = W_read(M[t])
```

Required logs:

```text
memory_forget
memory_write_norm
memory_read_norm
memory_read_influence_proxy
```

Do not call memory useful unless there is evidence that read affects output/loss.

## Task 4: Losses

Implement minimal losses:

```text
task_loss
boundary_budget_loss
route_entropy_or_allowed_loss
primitive_entropy_or_diversity_loss
program_cost
sign_balance_loss optional
memory_write_cost very small
```

Do not over-regularize first run. Need to see natural behavior.

## Task 5: StepAnalyzer

It is not a credit collector. It only logs what happened.

Required output files:

```text
metrics.csv
trace_epoch_XXX.json
REPORT_TO_CHATGPT.txt
final_report.json
train.log
AGENT_STATUS.md
```

Required report flags:

```text
BOUNDARY_EXPLOIT
BOUNDARY_DEAD
ROUTE_UNIFORM
ROUTE_IDENTITY_COLLAPSE
PRIMITIVE_COLLAPSE
MEMORY_DEAD
MEMORY_JUNK
DETAIL_SHORTCUT
COST_DOMINANCE
```

## Task 6: Validate script

Must check:

```text
python -m py_compile all .py files
--help smoke
imports work
no text patching markers
no actor critic strings
no checkpoint staging
```

## Task 7: Sync script

Must:

```text
git pull --rebase --autostash
run validate
run 5 epoch smoke
commit only logs + source files
refuse .pt/.pth/.ckpt/.safetensors
push branch
```

## First run command target

```bash
EPOCHS=5 \
TRAIN_LIMIT=12000 \
VAL_LIMIT=2000 \
AMP=fp16 \
bash simple_butterfly_matrix_v4_tape_lane/v4_6_loop_core/commands/sync_run_5ep_v4_6_loop_core_push_logs.sh
```

## What counts as success

Do not judge only accuracy.

Success needs:

```text
boundary sparse, not all-on
primitive top1 changes by step/lane
some signed corrections appear
route not all-to-all
route not identity-only
memory read influence nonzero
no cost dominates CE
```

## What to do after first logs

If boundary still all-on:

```text
add soft-topk boundary in v4.6.1
```

If primitives collapse:

```text
increase gumbel annealing pressure
add primitive diversity by lane/stage
reduce rank if experts are too dense
```

If memory is dead:

```text
reduce memory write cost
increase memory read path into output
add causal memory ablation test
```
