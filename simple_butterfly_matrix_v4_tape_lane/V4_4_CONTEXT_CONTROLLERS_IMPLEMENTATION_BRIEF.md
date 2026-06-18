# v4.4 Context Controllers Implementation Brief

Current priority: implement context/local controllers only after v4.3 canonical main smoke and 5ep are stable.

Do not replace v4.3. Create a new main file:

```text
simple_butterfly_matrix_v4_tape_lane/tape_lane_transport_v4_4_context_controllers.py
```

Base it on v4.3 canonical main and keep v4.3 unchanged.

## Core rule

Controllers are additive deltas:

```python
effective_logits = base_logits + alpha * controller_delta
```

Never replace base logits with controller output.

Use bounded alpha:

```python
alpha = alpha_min + (alpha_max - alpha_min) * sigmoid(alpha_raw)
```

Recommended defaults:

```text
alpha_min = 0.01
alpha_max = 0.10
```

Controller outputs should be near-zero at initialization.

## Controllers to add first

1. primitive_controller: most important; adds context-dependent bias to primitive logits inside transform units.
2. read_controller: adds bias to read_group_logits[t], output shape [B, L, L+1].
3. route_controller: adds bias to route_logits[t], output shape [B, L, L]. Route becomes per-sample.
4. boundary_controller: adds bias to boundary_logit[t], output shape [B].
5. write_controller: adds bias to write_gate_logit[t,lane], output shape [B, L].

Do not add alive_controller first. Adaptive depth is riskier and should be added after read/route/write/boundary/primitive are stable.

## Context inputs

Use only causal current-forward context:

```text
lane_state_summary
read_packet_summary
update_summary
evidence_summary
memory_lane_summary
step_embedding
boundary value for current step
route summary from current step if already computed
```

Do not use downstream/future trace inside the same forward pass.

## Required logs

Add to report/trace:

```text
controller_alpha_read
controller_alpha_route
controller_alpha_boundary
controller_alpha_write
controller_alpha_primitive
controller_delta_norm_read
controller_delta_norm_route
controller_delta_norm_boundary
controller_delta_norm_write
controller_delta_norm_primitive
route_per_sample_enabled
```

Keep all v4.3 metrics:

```text
sequence_nonflat_score
read_delta_by_step
route_delta_by_step
boundary_usefulness
self_route_mass
useful_transition_mass
memory_consumer_proxy
collapse_flags
candidate_suggestions deploy=false
```

## Route einsum change

Base v42 route is [L,L]. v4.4 route should be [B,L,L].

Use:

```python
routed = torch.einsum("bft,bfad->btad", route, update)
```

not:

```python
routed = torch.einsum("ft,bfad->btad", route, update)
```

## Causal memory diagnostics, not matrix memory yet

Add diagnostics before matrix memory:

```text
memory_write_without_future_read
memory_same_step_shortcut
memory_future_alignment
memory_causal_proxy_by_step
```

Do not punish memory write globally. Penalize only junk memory:

```text
write[t] high + future_read[t+1..] low + head_consumer low
```

## Explicitly not part of v4.4 controllers

Do not add:

```text
Actor
Critic
EditorLoop deploy
FeedbackBias auto-deploy
Matrix memory
TapePlannerAttention
Meta-Heart
```

## Success criteria

Compare v4.4 to v4.3 with same seed/config.

Good signs:

```text
val_acc drop <= 2-3 percent absolute
route_entropy < 1.20 or lower than v4.3
allowed_route_mass > 0.70 if allowed-route metric is implemented
disallowed_route_mass < 0.30 if implemented
boundary_peak_count 2..4
boundary_flatness > 0.03
detail_topread_share < 0.75
memory_future_alignment improves over v4.3
sequence_nonflat_score is nonzero and preferably higher than v4.3
candidate_suggestions remain deploy=false
```

## Execution order

1. Stabilize v4.3 canonical main smoke and 5ep.
2. Implement v4.4 context controllers as new file.
3. Smoke v4.4.
4. Run 5ep A/B comparison v4.3 vs v4.4.
5. Only after that: v4.4 learning heart, then matrix memory.
