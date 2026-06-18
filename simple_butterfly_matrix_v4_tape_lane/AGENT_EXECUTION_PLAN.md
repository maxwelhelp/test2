# Agent Execution Plan

This is the short operational plan for `simple_butterfly_matrix_v4_tape_lane`.
It is the practical companion to:

- [CURRENT_ROADMAP_V4_3_TO_V4_5.md](/home/maxwelhelp/test/sience/experiments/math_search/WORKING_BEST/test2/simple_butterfly_matrix_v4_tape_lane/CURRENT_ROADMAP_V4_3_TO_V4_5.md)
- [V4_3_MIN_HEART_PLAN.md](/home/maxwelhelp/test/sience/experiments/math_search/WORKING_BEST/test2/simple_butterfly_matrix_v4_tape_lane/V4_3_MIN_HEART_PLAN.md)

## Goal

Turn the tape-lane model into a sequence-aware program without breaking the working main entrypoint.

The current runtime source of truth is:

`simple_butterfly_matrix_v4_tape_lane/tape_lane_transport_v4_3_min_heart.py`

## Order Of Work

1. Keep canonical main stable.
2. Specialize route and boundary so the tape becomes staged, not flat.
3. Add local controllers as bounded deltas, not replacements.
4. Add causal memory, only after controllers work.
5. Move to v4.4 learning heart after the runtime is stable.

## What To Fix First

- Route must stop behaving like all-to-all mixing.
- Boundary must become a real stage separator.
- Detail shortcut must not dominate head reads.
- Memory must be measured causally, not only by proxy.

## Rules

- Never replace a working base logit with a controller output.
- Use `base + alpha * delta`, with small `alpha` and near-zero init.
- Do not add deploy, actor, or critic in the canonical runtime yet.
- Do not use runtime canonicalizer scripts as a required step.

## Controller Pattern

Use this shape:

```python
effective_logits = base_logits + alpha * controller_delta
```

Examples:

- `route_logits[t] + alpha_route * route_controller(ctx)`
- `boundary_logit[t] + alpha_boundary * boundary_controller(ctx)`
- `read_group_logits[t] + alpha_read * read_controller(ctx)`
- `write_gate_logit[t] + alpha_write * write_controller(ctx)`
- `primitive_logits + alpha_primitive * primitive_controller(ctx)`

## Memory Pattern

Add memory only when the controller layer is useful.

Start with:

- `memory_slots: [B, M, D]`
- `M=8`
- `rank=16`

Use it for:

- delayed future read credit
- primitive bias
- route/boundary bias

Do not use memory as a replacement for routing logic.

## Decision Examples

- If `route_entropy` stays near uniform and `disallowed_route_mass` is high, increase route specialization pressure.
- If `boundary_peak_count == 0` and `boundary_flatness` is near zero, increase boundary peak contrast.
- If `detail_topread_share` stays close to `1.0`, lower the detail shortcut target and increase the detail penalty.
- If `memory_write` is high but `future_read` stays low, penalize the memory write path.
- If `sequence_nonflat_score` rises while `val_acc` holds, keep the change.
- If `val_acc` drops sharply, reduce the new penalties before adding more structure.

## GitHub Workflow

For each change:

1. edit only the scoped files
2. run `py_compile`
3. run `validate_v4_3_min_heart.sh`
4. run `grad_sanity_v4_3_min_heart.sh`
5. run a smoke or speed smoke
6. commit
7. push to GitHub

## Current Success Signal

The model is heading the right way when all of these improve together:

- `val_acc`
- `sequence_nonflat_score`
- `route_allowed_mass`
- `boundary_soft_peak_count`
- `boundary_usefulness_proxy`
- `memory_consumer_proxy`

And these stay controlled:

- `route_entropy`
- `route_disallowed_mass`
- `detail_topread_share`
- `collapse_flags`
