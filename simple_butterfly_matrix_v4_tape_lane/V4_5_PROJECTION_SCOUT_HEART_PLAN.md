# v4.5 Projection Scout / Imagination Heart Plan

Purpose: give weak/discrete/rare choices a useful learning signal before they naturally receive enough gradient in the real model.

This is the central bridge between:

- ordinary backprop through the main model;
- rule-based candidate suggestions;
- future actor/critic/meta-heart.

It is not a replacement for read/primitive/route/write/boundary controllers. It is a cheap counterfactual imagination layer that helps those controllers decide what to try.

## Core problem

Some choices get weak or delayed gradient:

- boundary location: if no boundary is placed, it may not learn where a boundary would have helped;
- transition type: fork/join/memory_recall may never get used enough to learn;
- fanout: top1/top2/top3/all can collapse to a safe average;
- memory write: credit arrives only later when memory is read;
- primitive choice: some primitives are dormant and receive little signal;
- head read: class head may shortcut detail slots before abstract/memory become useful.

The model needs a way to ask:

> In a small projected world, what would happen if I placed this transition here?

## Projection Scout idea

At selected windows, build small projected simulations from current detached trace.

Main state:

```text
X[t, lane, cell, D]
```

Projected scout state:

```text
Z[t, lane, d_small], d_small = 16 or 32
```

Projection:

```python
Z = P_down(X_summary)          # detached or partly detached
```

Candidate actions:

```text
boundary at t
transition_type at t
fanout mode at t
route basis shift at t
primitive prior shift at t,lane
memory write/recall at t
head read prior shift
```

Scout simulates in low dimension:

```python
Z_candidate = small_executor(Z, action)
score = scout_reward(Z_candidate, target_signals)
```

Then sends a learning signal back as an auxiliary target:

```python
controller_logits should increase actions with good scout score
controller_logits should decrease actions with bad scout score
```

This is not full actor/critic yet. It is cheap, local, and supervised by projected counterfactuals.

## Global Alternative Projection Council

The scout should not only test one local edit at a time. The stronger mechanism is a global council that looks at the whole tape once, then builds several cheap alternative programs in projected space.

One global read:

```text
trace_feedback + metrics + head summary + memory summary + route/primitive/read/boundary summaries
        ↓
global_context_encoder
        ↓
K alternative projected programs
```

Each alternative is a small matrix program over `Z[t,lane,d_small]`, not a full D=96/128 execution.

Examples:

```text
Alternative A: early fork + memory_commit + later memory_recall
Alternative B: local detail refinement + late head_prepare
Alternative C: state->abstract split at t=4 + fanout=1/2
Alternative D: no new boundary, just route specialization + primitive shift
Alternative E: dormant primitive wake-up test
```

The council does not need every sub-scout to always propose an alternative. Each scout can abstain.

```text
boundary_scout: proposes A/B, or abstains
transition_scout: proposes transition type schedule, or abstains
primitive_scout: proposes primitive shifts, or abstains
memory_scout: proposes write/recall, or abstains
head_scout: proposes anti-shortcut read shift, or abstains
exploration_scout: occasionally wakes dormant components
```

The council then chooses a small diverse pool:

```text
K_total = 4..8 alternatives
K_verify = 1..2 top alternatives
```

## Alternative diversity rule

The alternatives must be meaningfully different. Otherwise all scouts collapse to the same safe proposal.

Represent each alternative by an action vector:

```text
boundary target over T
transition type over T
fanout over T
route basis weights
primitive bias over T,L,P
memory write/recall targets
head read target
```

Diversity penalty:

```text
similarity(alt_i, alt_j) too high -> reduce score
```

At least one alternative per window should come from an underexplored area if available:

```text
dormant primitive
unused transition type
untried boundary region
low-coverage lane
memory recall not recently tested
head non-detail read
```

## Wake-up / dormant component policy

Some components will not propose alternatives because they are weak, dormant, or currently suppressed. That is normal.

Rules:

```text
if component has low usage for N windows:
    occasionally force one projected alternative
    keep action small
    mark as exploration
    do not auto-deploy without verification

if component repeatedly loses:
    cooldown

if component wins projected score and later verified score:
    increase its proposal quota
```

This creates a controlled reserve of mini-models/scouts rather than a chaotic swarm.

## Council controller

The council itself is a small controller/MLP over global context and scout features.

Inputs:

```text
program quality metrics
collapse flags
class margin/confusion
route entropy / allowed mass
primitive entropy / dominant primitive
boundary dead/exploit status
memory write/read alignment
head shortcut score
recent action history
coverage map
```

Outputs:

```text
which scouts are allowed to propose
how many alternatives to sample
which alternatives enter the diverse pool
whether to run verification
which weak component gets wake-up budget
```

At first, this controller can be rule-based. Later it becomes trainable from verification records.

## Global council flow

```text
1. Read whole trace once.
2. Build shared global context embedding.
3. Build K candidate projected programs.
4. Simulate each in low dimension.
5. Score by task proxy + PQS + diversity - complexity.
6. Select top diverse alternatives.
7. Write scout_targets_epoch_N.json.
8. Optionally verify top-1/top-2 with real forward-only microbatch.
9. Store accepted/rejected/uncertain records.
```

## What the scout sees

Every scout candidate gets context:

```text
current step t
lane summaries
read summaries
primitive weights
route matrix
boundary value
memory write/read proxy
head attention summary
class margin/confusion summary
sequence delta before/after
collapse flags
```

Global council also sees whole-tape context:

```text
boundary_by_step
sequence_change_by_step
route_by_step
primitive_by_step_lane
read_group_by_step
memory write/read by step
head lane/cell attention
recent action history
coverage map
```

It must be causal for forward choices:

- during real forward, a controller only sees current/past context;
- after epoch/window, scout can inspect trace and propose/diffuse feedback for future windows.

## Rewards / scores

Scout score is not only task loss. Use a Program Quality Score plus task proxy.

```text
PQS =
  + route_specialization
  + useful_transition_mass
  + boundary_usefulness
  + primitive_diversity
  + memory_future_alignment
  + head_nonshortcut_score
  - route_uniform
  - boundary_dead_or_exploit
  - memory_junk
  - detail_shortcut
  - complexity_cost
```

Candidate gain:

```text
gain = delta_task_proxy + lambda_pqs * delta_PQS - lambda_cost * action_cost - lambda_similarity * duplicate_penalty
```

## Minimal MVP stages

### MVP 1: Boundary Scout

Reason: boundary currently alternates between exploit and dead/flat.

Scout tests projected boundary placements:

```text
top candidate t from sequence_change_by_step
low boundary near high route/primitive/read delta
boundary near memory write -> future read
boundary near transition-type change
```

Output:

```json
{
  "action": "increase_boundary",
  "t": 5,
  "predicted_gain": 0.013,
  "reason": "high sequence delta + route change + low current boundary",
  "deploy": false
}
```

Training signal:

```python
boundary_score[t] += scout_target_bias[t]
```

Initially no auto-deploy. Only auxiliary loss:

```python
loss += lambda_scout * BCEWithLogits(boundary_logits, scout_boundary_target)
```

### MVP 2: Transition/Fanout Scout

Scout tests:

```text
identity
split/detail->state
state->abstract
state->memory
memory->state
fork
join
head_prepare
fanout 1/2/3/all
```

Output target:

```python
transition_type_logits[t] should match scout_transition_target[t]
fanout_logits[t] should match scout_fanout_target[t]
```

### MVP 3: Primitive Scout

For weak primitives, test small projected substitutions:

```text
replace gated_contrast with diff at t,lane
increase memory_keep before memory write
increase low_rank in abstract lane
increase ctx_matrix near head_prepare
```

Output:

```python
primitive_controller target bias per (t,lane,primitive)
```

### MVP 4: Memory Scout

Tests if memory write is useful:

```text
write high at t, future read low -> junk
write high at t, future read/head consumer high -> useful
recall memory before class confusion -> test memory->state route
```

Output:

```python
memory_write_target[t]
memory_recall_target[t]
route_memory_to_state_target[t]
```

### MVP 5: Global Alternative Council

This is the user's central mechanism.

One pass over whole trace produces 4-8 alternative projected programs.

Output:

```json
{
  "epoch": 7,
  "global_context": {...},
  "alternatives": [
    {
      "id": "alt_boundary_fork_memory",
      "source_scouts": ["boundary", "transition", "memory"],
      "boundary_target": [0,0,1,0,0,1,0,0,1,0,0,0],
      "transition_targets": {"2":"fork", "5":"split", "8":"memory_recall"},
      "fanout_targets": {"2":"two", "5":"one", "8":"one"},
      "predicted_gain": 0.018,
      "diversity_score": 0.74,
      "risk": "medium",
      "deploy": false
    }
  ],
  "selected_targets": {...}
}
```

No runtime mutation first. Report-only.

### MVP 6: Real Microbatch Verification

For top-1 or top-2 council alternatives only:

- copy current model state;
- apply temporary bias/action;
- run forward-only or tiny no-update eval on heldout microbatch;
- compare paired baseline vs candidate;
- log accepted/rejected;
- do not train during verification.

This creates experience records for future critic.

## Critical implementation rules

1. Scout must not update main optimizer directly.
2. Scout produces detached targets/biases, not trainable hidden state mixed with Adam unless intended.
3. Counterfactual screen must be forward-only at first.
4. Max candidates per window: 8-12.
5. Verify only top-1/top-2.
6. Store action history: tested, accepted, rejected, uncertain, cooldown.
7. Do not enable actor/critic until at least a few hundred verified records exist.
8. Alternatives must be diverse; do not allow 8 copies of the same action.
9. Scouts can abstain; only exploration scout can force wake-up.
10. Wake-up alternatives are small and must be marked exploration.

## Why this helps gradient

Normal gradient:

```text
loss -> only choices used in actual forward get strong signal
```

Projection Scout:

```text
trace -> low-dim candidate simulation -> target for controller logits
```

Global Alternative Council:

```text
trace -> several whole-program alternatives -> diverse target pool -> weak controllers receive useful exploratory signal
```

So weak choices receive signal before they are frequently used.

It does not fake final proof. It only gives a cheap prior. Real model still decides via training and verification.

## Where it connects in v4.x

Current v4.4 bridge:

```text
read_controller
primitive_controller
boundary budget
transition_type_controller
fanout_controller
route_controller
write_controller
```

Projection Scout sits above them:

```text
trace_feedback_epoch_N.json
        ↓
ProjectionScout / GlobalAlternativeCouncil
        ↓
scout_targets_epoch_N.json
        ↓
aux losses / temporary bias next run
```

## First code target

Create:

```text
projection_scout_v1.py
```

and optional runner:

```text
commands/run_projection_scout_on_latest_report.sh
```

It should read latest report folder and write:

```text
projection_scout_epoch_XXX.json
```

No runtime mutation first. Report-only.

Second code target:

```text
v4.5 runtime: load scout_targets and add tiny auxiliary losses
```

## Expected effect

Boundary should stop being either all-high or all-flat.

Desired pattern:

```text
boundary_peaks: 2..4
transition_type: interpretable by step
fanout: not always all-soft
route_entropy: lower than uniform but not pure identity
memory: write has future read/head consumer
primitive: less uniform, more per-step specialization
```

## Do not do yet

- full actor;
- full critic;
- permanent auto-deploy;
- large matrix memory;
- hard top-k without soft warmup;
- uncontrolled multi-action patches.

Do this after scout produces stable positive verified records.
