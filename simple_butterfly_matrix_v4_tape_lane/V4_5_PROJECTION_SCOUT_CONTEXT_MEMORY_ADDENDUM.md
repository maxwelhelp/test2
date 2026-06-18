# v4.5 Projection Scout Context + Memory Addendum

This addendum extends `V4_5_PROJECTION_SCOUT_HEART_PLAN.md`.

The key correction: projected alternatives must not only simulate from the current context. They must also model the **target context** that the alternative wants to create.

A projected alternative is not:

```text
current trace -> random small edit -> score
```

It is:

```text
current trace
  -> desired target program/context
  -> projected transition path from current to target
  -> score
  -> controller targets/biases/memory update
```

## 1. Current context vs target context

Every alternative must contain both:

```text
current_context:
  what the program currently is doing

target_context:
  what the alternative wants the program to become
```

Examples:

```json
{
  "id": "alt_fork_memory_recall",
  "current_context": {
    "boundary_state": "dead_flat",
    "route_state": "self_heavy",
    "memory_state": "write_high_future_read_low",
    "head_state": "detail_topread_high"
  },
  "target_context": {
    "boundary_peaks": [2, 6, 9],
    "transition_schedule": {
      "2": "fork_detail_to_state_memory",
      "6": "state_to_abstract",
      "9": "memory_to_state"
    },
    "fanout_schedule": {
      "2": "two",
      "6": "one",
      "9": "one"
    },
    "expected_route_shape": "early_detail_to_memory_then_late_memory_to_state",
    "expected_head_shape": "less_detail_topread_more_state_memory"
  }
}
```

The scout must evaluate whether the target context is reachable and useful, not just whether a single edit looks good.

## 2. Target-context encoder

Add a small encoder:

```text
TargetContextEncoder(current_trace, alternative_spec) -> z_target
```

Inputs:

```text
current route/read/primitive/boundary/head/memory summaries
proposed boundary target
proposed transition schedule
proposed fanout schedule
proposed primitive shifts
proposed memory write/recall plan
proposed head read shift
```

Output:

```text
z_target: small vector describing desired program state
```

Then the projected executor evaluates:

```text
z_current -> projected_program_action -> z_predicted
score = match(z_predicted, z_target) + PQS_gain + task_proxy - cost
```

This prevents fake projections that look good only from the current context but do not actually move the program toward the desired state.

## 3. Alternative as a mini program, not one edit

Each alternative should be a small structured program:

```json
{
  "id": "alt_name",
  "boundary_target": [0,0,1,0,0,1,0,0,1,0,0,0],
  "transition_targets": {"2":"fork", "5":"split", "8":"memory_recall"},
  "fanout_targets": {"2":"two", "5":"one", "8":"one"},
  "route_biases": [...],
  "primitive_biases": [...],
  "memory_targets": {...},
  "head_targets": {...},
  "target_context": {...},
  "predicted_gain": 0.0,
  "program_quality_score": 0.0,
  "diversity_score": 0.0,
  "risk": "low|medium|high",
  "deploy": false
}
```

This is the form that later can be converted into controller targets.

## 4. Where feedback goes

Scout feedback should not be vague. It must be routed to exact controllers.

```text
boundary target        -> boundary planner / boundary logits
transition target      -> transition_type_controller logits
fanout target          -> fanout_controller logits
route target           -> route_controller + route_basis weights
primitive target       -> primitive_controller logits per t,lane,primitive
memory target          -> memory_write / memory_recall controller later
head target            -> head read controller later
council selection gain -> council memory / alternative ranking
```

Signal types:

```text
auxiliary target loss:
  small supervised loss on controller logits

detached scout bias:
  effective_logits = base + controller_delta + scout_bias.detach()

experience memory update:
  store predicted vs observed effect
```

Start with report-only, then auxiliary target, then tiny detached bias, then verification.

## 5. Scout memory / experience memory

The scout needs memory. Without it, it will repeat the same alternatives and never learn which combinations work.

Create:

```text
scout_experience_memory.jsonl
```

Each record:

```json
{
  "run_id": "...",
  "epoch": 7,
  "alternative_id": "alt_fork_memory_recall",
  "current_context_hash": "...",
  "current_context_summary": {...},
  "target_context_summary": {...},
  "action_vector": {...},
  "predicted_gain": 0.018,
  "predicted_pqs_gain": 0.031,
  "diversity_score": 0.74,
  "risk": "medium",
  "verification_status": "not_verified|accepted|rejected|uncertain",
  "observed_gain": null,
  "observed_pqs_gain": null,
  "cooldown": 0,
  "notes": "..."
}
```

The council should query memory before proposing:

```text
similar context + same action failed recently -> penalty / cooldown
similar context + same action helped -> stronger prior
similar context + different action helped -> suggest related alternative
underexplored context/action -> exploration bonus
```

## 6. Embedding of relationships

The scout should form embeddings for program relationships, not only scalar metrics.

Embeddings:

```text
step_embed[t]
lane_embed[lane]
primitive_embed[p]
transition_type_embed[k]
fanout_embed[f]
route_edge_embed[from,to]
memory_event_embed[write,recall,overwrite]
head_read_embed[lane/cell]
```

Alternative embedding:

```text
alt_embed = pool(
  boundary target embeddings,
  transition schedule embeddings,
  fanout embeddings,
  primitive bias embeddings,
  route edge embeddings,
  memory event embeddings,
  head target embeddings
)
```

Context-action compatibility:

```text
compat = dot(context_embed, alt_embed)
```

This allows the system to learn relations like:

```text
high detail shortcut + memory underuse -> memory_recall/head_non_detail alternative
high route uniform + boundary dead -> split/fanout alternative
primitive collapse to gated_contrast -> diff/ctx_matrix wake-up alternative
write high + future_read low -> reduce write or add recall route
```

## 7. Alternative generation policy

Every window:

```text
1. Build global current_context.
2. Retrieve similar memory records.
3. Generate 4-8 alternatives:
   - best known successful pattern
   - correction for current collapse
   - one memory-based alternative
   - one primitive diversification alternative
   - one head anti-shortcut alternative
   - one exploration/wake-up alternative if needed
4. Score alternatives in projected space.
5. Apply diversity penalty.
6. Select top diverse targets.
7. Write report-only JSON first.
```

Scouts can abstain:

```text
if no evidence -> abstain
if repeated failure -> cooldown
if component dormant too long -> exploration scout may wake it
```

## 8. Learning from success/failure

After real verification or after next run metrics:

```text
predicted_gain vs observed_gain
predicted_pqs_gain vs observed_pqs_gain
```

Update memory:

```text
accepted: increase prior for similar context/action
rejected: cooldown exact action in similar context
uncertain: keep but do not strengthen
```

Later, train a small council critic:

```text
(context_embed, alt_embed, action_history) -> predicted_gain
```

Do not enable full actor until there are enough accepted/rejected records.

## 9. Important safety / stability rules

```text
No permanent auto-deploy at first.
No many simultaneous strong edits.
No hard top-k without soft warmup.
No scout bias bigger than controller scale.
No memory update from unverified hallucinated gains.
No repeated same alternative without diversity/cooldown.
```

## 10. First implementation target

`projection_scout_v1.py` should output:

```json
{
  "epoch": 7,
  "mode": "report_only",
  "current_context": {...},
  "retrieved_memory": [...],
  "alternatives": [...],
  "selected_targets": {
    "boundary": {...},
    "transition": {...},
    "fanout": {...},
    "route": {...},
    "primitive": {...},
    "memory": {...},
    "head": {...}
  },
  "memory_updates": [...],
  "deploy": false
}
```

This must be cheap and deterministic first.

Only after report quality is good should runtime load `selected_targets` as small auxiliary losses.
