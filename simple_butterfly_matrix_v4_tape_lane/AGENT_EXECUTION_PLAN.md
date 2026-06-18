# Agent Execution Plan

This is the operational plan for `simple_butterfly_matrix_v4_tape_lane`.
It is the working document for the canonical runtime, not a summary.

Related references:

- [CURRENT_ROADMAP_V4_3_TO_V4_5.md](/home/maxwelhelp/test/sience/experiments/math_search/WORKING_BEST/test2/simple_butterfly_matrix_v4_tape_lane/CURRENT_ROADMAP_V4_3_TO_V4_5.md)
- [V4_3_MIN_HEART_PLAN.md](/home/maxwelhelp/test/sience/experiments/math_search/WORKING_BEST/test2/simple_butterfly_matrix_v4_tape_lane/V4_3_MIN_HEART_PLAN.md)
- [AGENT_STATUS.md](/home/maxwelhelp/test/sience/experiments/math_search/WORKING_BEST/test2/simple_butterfly_matrix_v4_tape_lane/AGENT_STATUS.md)

## Scope

Work only in the tape-lane project:

- `simple_butterfly_matrix_v4_tape_lane/`
- `simple_butterfly_matrix/`
- `v4_3_min_heart/`
- `v4_4_min_learning_heart/`

Do not pull the standard runtime into other projects, datasets, or checkpoint trees.
Do not make the runtime depend on the canonicalizer path.

## Current Source Of Truth

The canonical runtime is:

`simple_butterfly_matrix_v4_tape_lane/tape_lane_transport_v4_3_min_heart.py`

Standard sync must execute that file directly.
The runtime canonicalizer is deprecated and must stay out of the normal launch path.

## Main Objective

Make the model sequence-aware without turning it into a generic all-to-all mixer.
The program should become staged, selective, and measurable:

- routing should specialize
- boundary should actually separate stages
- reads should stop collapsing into the detail shortcut
- memory should be causal when it is introduced
- controllers should be bounded deltas on top of stable base logits

## Order Of Work

1. Keep canonical main stable.
2. Strengthen route/boundary specialization.
3. Add local controllers as small deltas, not replacements.
4. Add memory after controller signals are useful.
5. Move on to v4.4 learning heart only after the canonical main remains stable.

## Runtime Rules

- Do not replace base logits with controller outputs.
- Use `base + alpha * delta` only.
- Keep `alpha` small and initialize controller deltas near zero.
- Do not add Actor, Critic, EditorLoop deploy, or auto-deploy in the canonical runtime.
- Do not require `canonicalize_v4_3_main.py` in standard sync.
- Do not let observer metrics masquerade as causal metrics.

## Controller Pattern

Use this shape everywhere a controller is added:

```python
effective_logits = base_logits + alpha * controller_delta
```

Examples:

```python
route_logits_eff = route_logits[t] + alpha_route * route_controller(ctx)
boundary_logits_eff = boundary_logit[t] + alpha_boundary * boundary_controller(ctx)
read_logits_eff = read_group_logits[t] + alpha_read * read_controller(ctx)
write_gate_eff = write_gate_logit[t, lane] + alpha_write * write_controller(ctx)
primitive_logits_eff = primitive_logits + alpha_primitive * primitive_controller(ctx)
```

Practical rule:

- if the controller output is too strong, shrink `alpha`
- if the controller is ignored, improve the controller input or the bias term
- if the controller destabilizes training, keep the base path and only weaken the delta

## Route And Boundary

These are the first places to fix because they shape the whole program.

Desired behavior:

- route should choose meaningful transitions, not uniform mixing
- boundary should mark actual stage changes
- route should not send everything to everything
- boundary should not collapse to a flat low-value field

What to watch:

- `route_entropy`
- `self_route_mass`
- `useful_transition_mass`
- `allowed_route_mass`
- `disallowed_route_mass`
- `boundary_mean`
- `boundary_soft_peak_count`
- `boundary_peak_count`
- `boundary_flatness`
- `sequence_nonflat_score`

If route entropy stays near uniform, increase route specialization pressure.
If boundary stays flat, increase boundary contrast or reweight the boundary loss.

## Read Shortcut Control

The detail lane must not dominate the whole model.

Watch:

- `detail_attention_mass`
- `detail_topread_share`
- `detail_head_shortcut_cost`
- `boundary_usefulness_proxy`

Interpretation:

- `detail_topread_share` close to `1.0` means the model is still taking the easiest path too often
- a lower `detail_topread_share` is good only if `val_acc` does not collapse
- if detail attention stays high while route and boundary are flat, the program is still shallow

## Memory

Memory is allowed only after route/boundary/controller behavior is useful.

Start with small, explicit structures:

- `memory_slots: [B, M, D]`
- `M=8`
- `rank=16`

Use memory for:

- delayed future-read credit
- primitive bias
- route/boundary bias

Do not use memory as a substitute for routing.
Do not call proxy metrics causal memory unless there is real `write -> future_read` evidence.

Relevant metrics:

- `memory_write`
- `memory_consumer_proxy`
- `future_read`
- `memory_slot_entropy`
- `memory_change_norm`

## What Good Looks Like

The direction is good when these move together:

- `val_acc`
- `sequence_nonflat_score`
- `route_allowed_mass`
- `boundary_soft_peak_count`
- `boundary_usefulness_proxy`
- `memory_consumer_proxy`

And these remain controlled:

- `route_entropy`
- `route_disallowed_mass`
- `detail_topread_share`
- `collapse_flags`

## Concrete Decision Examples

Use the metrics to change the model, not intuition alone.

Examples:

```text
if route_entropy is still close to uniform:
    increase route specialization pressure

if boundary_peak_count stays at 0:
    raise boundary contrast or increase boundary loss weight

if detail_topread_share stays near 1.0:
    lower the detail shortcut target and raise the detail penalty

if memory_write is high but future_read is low:
    penalize the memory write path or reduce memory gate strength

if sequence_nonflat_score rises and val_acc holds:
    keep the change

if val_acc drops hard after a change:
    back off the newest penalty before adding more structure
```

## Validation Flow

For any change in the scoped runtime:

1. edit only scoped files
2. run `python -m py_compile simple_butterfly_matrix_v4_tape_lane/tape_lane_transport_v4_3_min_heart.py`
3. run `bash simple_butterfly_matrix_v4_tape_lane/commands/validate_v4_3_min_heart.sh`
4. run `bash simple_butterfly_matrix_v4_tape_lane/commands/grad_sanity_v4_3_min_heart.sh`
5. run a smoke or speed smoke
6. inspect the report, not only the final accuracy
7. commit only the intended files
8. push to GitHub

## Smoke Criteria

Smoke is useful only if it proves the canonical main still works.

The run is considered acceptable when the log includes:

```text
epoch 001/1 train=... val=...
[v4.3 sync] done ... status=0
```

If speed profiling is enabled, capture the speed lines too.

## Implementation Boundaries

Keep these separate:

- canonical main runtime
- deprecated canonicalizer
- context-controller wrapper
- separate experiments for v4.4 learning heart

Standard sync uses the canonical main only.

## GitHub Workflow

For the canonical main and its command scripts:

1. make a small change
2. verify the direct runtime
3. run the validation scripts
4. run smoke
5. commit
6. push
7. only then move to the next layer

## Next Step After Stability

After stable smoke and 5-epoch runs are back in place, move to:

`simple_butterfly_matrix_v4_tape_lane/v4_4_min_learning_heart/`

The next phase should add:

- PQS
- action_history
- forward-only screen
- simple bandit
- no deploy

Only then should more aggressive controller or memory restructuring be introduced.
