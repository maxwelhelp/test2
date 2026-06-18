# v4.6 Loop Core

Standalone experimental architecture plan for a differentiable program assembly loop.

This folder is documentation-first. Do not patch the existing v4.3/v4.4/v4.5 files from here. New code should live under this folder or in a clearly named wrapper until the design is validated.

## Goal

Build a closed differentiable loop:

```text
context[t,L,d]
  -> JointController -> shared latent z[t]
  -> ParallelPrimitiveSelector with signed low-rank experts
  -> coupled route/boundary/fanout/write
  -> MatrixMemory W_read/W_write/forget
  -> task loss
  -> backprop credit through everything
```

The training credit is ordinary backprop. Offline JSON Council feedback is not part of the training loop.

## What v4.6 is

- Joint controller for coupled decisions.
- Parallel primitive projections with gumbel-softmax selection.
- Signed primitive coefficients, so operations can add or subtract/correct.
- Low-rank primitive experts, default rank 32-48 for d around 256.
- Matrix memory with real W_write, W_read and learnable forget.
- StepAnalyzer logs for debugging and specialization analysis.
- Boundary budget/validity losses.
- Temperature annealing for gumbel-softmax.

## What v4.6 is not

- No actor-critic.
- No PPO/A3C.
- No offline JSON feedback loop.
- No text patching.
- No deep projection tree yet.
- No full program compiler yet.

Actor-critic is reserved for a later version if decisions become hard discrete and multi-step planning needs long-horizon credit. With gumbel-softmax, the primitive choice is differentiable, so actor-critic is unnecessary now.

## Why not remove steps

Steps are not the enemy. They are the causal unroll for memory:

```text
M[t] depends on M[t-1]
```

If all steps are removed and everything is parallel with no recurrence, memory loses causal meaning. Later we can optimize linear memory updates with prefix/parallel scan, but first the step semantics must work.

## Planned module layout

```text
v4_6_loop_core/
  README.md
  ARCHITECTURE_PLAN.md
  AGENT_TASKS.md
  tape_lane_transport_v4_6_loop_core.py      # later
  modules/
    __init__.py                              # later
    joint_controller.py                      # later
    primitive_selector.py                    # later
    matrix_memory.py                         # later
    step_analyzer.py                         # later
    losses.py                                # later
  commands/
    validate_v4_6_loop_core.sh               # later
    sync_run_5ep_v4_6_loop_core_push_logs.sh # later
```

## First implementation target: v4.6.0

Minimal code only:

1. `JointController`
2. `ParallelPrimitiveSelector`
3. `MatrixMemory`
4. `StepAnalyzer`
5. task training loop with gumbel temperature annealing
6. reports with specialization metrics

Do not add actor-critic, projection trees, or Council JSON training before v4.6.0 logs prove the base loop works.
