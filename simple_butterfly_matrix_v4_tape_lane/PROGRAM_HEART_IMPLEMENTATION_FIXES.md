# Program Heart Implementation Fixes

This document patches important implementation risks in the Program Heart plans.

The key concern:

```text
The heart must not become just a pretty logger or a second unstable optimizer.
```

The first real test is:

```text
Can a tested feedback_bias survive and influence the next window, or does normal gradient training immediately erase it?
```

If feedback bias does not survive / does not measurably affect effective logits, higher actor-critic stages are pointless.

---

## 1. FeedbackBias must be a detached buffer

Problem:

```text
learned logits and feedback bias affect the same effective logits.
optimizer updates learned logits every batch.
If feedback is written into learned parameters, Adam can erase or distort it.
```

Rule:

```python
logit_effective = learned_param + feedback_bias.detach()
```

`feedback_bias` must be a non-optimizer state buffer, not a normal trainable parameter.

Use:

```python
self.register_buffer("feedback_route_bias", torch.zeros(T, L, L))
self.register_buffer("feedback_boundary_bias", torch.zeros(T))
self.register_buffer("feedback_primitive_bias", torch.zeros(T, L, P))
```

Then:

```python
route_logits = self.route_logits[t] + context_route_bias + self.feedback_route_bias[t].detach()
route = softmax(route_logits, dim=-1)
```

For MVP 3, implement only:

```text
feedback_route_bias
feedback_boundary_bias
```

Do not add all feedback tensors at once.

---

## 2. Track effective logit impact

Every epoch/window, report whether feedback bias is actually visible.

Add metrics:

```text
feedback_route_bias_norm
feedback_boundary_bias_norm
feedback_route_bias_max_abs
feedback_boundary_bias_max_abs
learned_route_logit_norm
learned_boundary_logit_norm
feedback_to_learned_ratio
route_effective_delta_after_feedback
```

Example:

```python
feedback_to_learned_ratio = feedback_route_bias.norm() / (route_logits_param.norm() + 1e-8)
```

Also log softmax effect:

```text
route_prob_before_feedback
route_prob_after_feedback
prob_delta
```

Goal:

```text
prove accepted feedback changes actual R_t probabilities in the next window.
```

If probability delta is near zero, feedback scale is too small or competing logits are too strong.

---

## 3. Decay vs accumulation

Problem:

```text
bias *= 0.95 each window can erase useful repeated actions.
```

Correct rule:

```text
all bias decays slowly;
repeated accepted action reinforces toward clamp;
rejected action cools down or pushes opposite;
uncertain does not deploy.
```

Implementation:

```python
# at start/end of window
feedback_bias *= decay

# accepted action
feedback_bias[address] = clamp(
    feedback_bias[address] + eta_accept * action_delta,
    -max_bias,
    max_bias,
)

# rejected action
feedback_bias[address] = clamp(
    feedback_bias[address] - eta_reject * action_delta,
    -max_bias,
    max_bias,
)
```

Recommended MVP values:

```text
max_bias = 0.10
eta_accept = 1.0
eta_reject = 0.5
decay = 0.98 per epoch/window, not 0.95 initially
action_delta = 0.02 to 0.03
```

If the same action is repeatedly accepted, it should accumulate toward the clamp.

---

## 4. Separate memory statuses

Do not collapse all weak results into `uncertain`.

Use statuses:

```text
accepted
rejected
uncertain_more_data
no_effect
unsafe
duplicate
conflict
```

Meaning:

```text
accepted:
  gain > threshold and safe

rejected:
  negative effect < -threshold

uncertain_more_data:
  variance too high / not enough microbatches
  can be retested later

no_effect:
  abs(delta) below noise and enough samples
  cooldown lightly; not worth priority

unsafe:
  instability, nonfinite, memory overwrite, global val damage
  strong cooldown

duplicate:
  same as recently tested accepted/rejected candidate

conflict:
  violates coordinator constraints
```

This matters because:

```text
uncertain_more_data should be probed again;
no_effect should not waste repeated budget.
```

---

## 5. CounterfactualScreen budget

Problem:

```text
full verify is expensive: save full state -> train N steps -> measure -> rollback.
```

MVP budget:

```text
cheap screen: up to 8 candidates per epoch/window
verify: top-1 candidate only
no multi-step verify until cheap screen proves useful
```

Stage budgets:

```text
MVP 2:
  cheap screen only, no deployment

MVP 3:
  cheap screen up to 8
  deploy at most 1 route/boundary feedback action
  no full verify yet unless cheap screen gain is strong

MVP 4+:
  verify top-1 or top-2 only
```

Cheap screen formula:

```text
for candidate in candidates:
  measure baseline heldout microbatch loss
  apply temporary detached bias
  measure candidate loss
  revert temporary bias
  record delta
```

Use multiple heldout microbatches only for top candidates.

---

## 6. Full-state rollback rule

When full verify is enabled, rollback must restore complete training state.

Save:

```text
model.state_dict()
optimizer.state_dict()
AMP scaler state
scheduler state if any
RNG state: torch CPU, torch CUDA, Python random, numpy if used
feedback bias buffers
running stats if any
```

Do not restore only model weights.

Bad rollback silently corrupts Adam momentum and makes results hard to interpret.

---

## 7. Context embedding schema

Problem:

```text
`context_embedding = small vector` is too vague.
If it is only global mean trace, different situations become indistinguishable.
```

For MVP 7 critic, context embedding must be target-local + neighborhood + task/head.

Recommended schema:

```text
context_embedding(action) = concat(
  target_trace_token[t,lane],
  prev_trace_summary[t-1],
  next_trace_summary[t+1],
  local_segment_summary,
  input_structure_token_summary,
  head_feedback_summary,
  budget_state,
  training_progress_features
)
```

For route action `R_t[from,to]`:

```text
target_trace_token = concat(
  trace_token[t, from_lane],
  trace_token[t, to_lane],
  route_row[t, from_lane],
  route_col[t, to_lane],
  boundary[t],
  step_alive[t]
)
```

For primitive action:

```text
target_trace_token = concat(
  trace_token[t,lane],
  primitive_weights[t,lane],
  primitive_grad_importance[t,lane],
  update_norm[t,lane],
  downstream_consumer_score[t,lane]
)
```

For head action:

```text
target_trace_token = concat(
  class_lane_mass[class],
  class_top_read_summary[class],
  class_margin[class],
  confusion_pair_stats,
  pair_update_norm
)
```

Keep final size fixed, e.g. 128 or 256 dims.

Use typed embeddings:

```text
action_type_embed
target_type_embed
lane_embed
primitive_embed
route_edge_embed
class/pair_embed
```

---

## 8. Bias interference monitoring

Problem:

Multiple active biases in the same softmax group can fight each other.

Example:

```text
increase state->abstract and state->memory simultaneously
```

Both live in route softmax row `state -> *`, so they compete.

Report per softmax group:

```text
active_bias_count
positive_bias_sum
negative_bias_sum
bias_l2
bias_entropy
largest_bias_entry
conflicting_sign_count
```

For route:

```text
route_bias_group[t, from_lane, :]
```

For primitives:

```text
primitive_bias_group[t,lane,:]
```

Coordinator must block dense/conflicting edits in MVP:

```text
max one positive bias per softmax group
or require explicit shift action: decrease A + increase B
```

---

## 9. Priority implementation sequence

Use this strict order.

### Step 0: Address and effective-logit plumbing

Add detached feedback buffers for route/boundary only.

No actions yet.

Report effective logit/prob changes if manual feedback is injected.

### Step 1: TraceCollector for route/boundary

Collect:

```text
route matrix
boundary
route entropy
step_alive
read/write group mass
future consumer score approximation
head lane usage
```

### Step 2: Candidate suggestions, no deploy

Generate route/boundary candidates:

```text
increase memory->state
increase state->abstract
decrease useless self-loop
increase/decrease boundary
```

Write `candidate_suggestions_epoch_XXX.json`.

### Step 3: Cheap CounterfactualScreen only

Screen top candidates on heldout microbatch.

Write:

```text
counterfactual_results_epoch_XXX.jsonl
```

Do not deploy yet.

### Step 4: FeedbackBiasBank deploy one action

Deploy at most one accepted route/boundary action.

Then answer the key question:

```text
Does detached feedback_bias survive and change next-window route/boundary probabilities?
```

If no, fix scale/clamp/context before adding primitive/head heart.

### Step 5: Primitive/variant heart

Only after Step 4 works.

### Step 6: Critic

Only after enough tested action records.

### Step 7: Attention planner

Only after critic/screen loop works.

---

## 10. First experiment to prove the heart is real

Run two versions:

```text
A: route/boundary feedback disabled
B: route/boundary feedback enabled, max 1 accepted action per epoch
```

Compare:

```text
val_acc
val_loss
route_prob_delta after feedback
feedback_to_learned_ratio
number accepted/rejected/no_effect
late_input_read_mass
memory route usage
boundary usefulness
```

The most important diagnostic is not accuracy first.

It is:

```text
accepted feedback caused measurable effective route/boundary probability change in the next epoch/window.
```

If this is false, the heart loop is not connected.

---

## 11. Short summary

Critical fixes:

```text
feedback_bias is detached buffer, not optimizer parameter
cheap screen has hard candidate budget
full verify is top-1/top-2 only
context embedding is target-local + neighborhood + head/input + budget
repeated accepted action accumulates despite decay
uncertain_more_data != no_effect
monitor softmax-group bias interference
prove feedback affects effective probabilities before building critic/attention
```

This should be implemented before higher-level ProgramHeart modules.
