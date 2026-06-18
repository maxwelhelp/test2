# v4.4_min_learning_heart

Purpose: first safe learning layer above `v4.3_min_heart`.

v4.3 is observer:

```text
model trains -> Heart observes -> JSON reports/suggestions
```

v4.4 becomes action-learning shadow mode:

```text
trace -> PQS -> candidate suggestions -> forward-only screen -> action_history -> simple bandit update
```

No feedback deploy yet.

## Scope

Add:

```text
Program Quality Score (PQS) as report/reward, not train loss
action_history.jsonl
forward-only candidate screen
simple bandit candidate ranking
cooldown/no-repeat logic
```

Do not add yet:

```text
feedback_bias deploy
critic neural net
actor
planner attention
multiplicative bias
PQS as direct train loss
```

## PQS

PQS is intrinsic program-quality reward.

```text
PQS =
  + route_specialization
  + boundary_quality
  + memory_efficiency
  + lane_diversity
  + sequence_quality
  - shortcut_penalty
  - complexity_penalty
```

PQS is used for:

```text
candidate ranking
screen result scoring
future critic target
reports
```

PQS is not added to the training loss in v4.4.

## action_history.jsonl

Each screened action writes one record:

```json
{
  "epoch": 5,
  "location": "t6.route.memory->state",
  "action": "increase",
  "delta_scale": 0.03,
  "baseline_loss": 1.42,
  "candidate_loss": 1.40,
  "delta_loss": 0.02,
  "baseline_pqs": 0.31,
  "candidate_pqs": 0.34,
  "delta_pqs": 0.03,
  "status": "accepted|rejected|no_effect|uncertain|unsafe",
  "cooldown": 0,
  "source": "route_boundary_rule"
}
```

## Forward-only screen

Protocol:

```text
torch.no_grad()
same heldout microbatch
baseline forward
candidate temporary detached bias forward
restore immediately
no backward
no optimizer step
no deploy
```

Hard limits:

```text
max screened candidates per epoch: 8
max parallel candidates: 4 or lower if VRAM risk
```

## Bandit

Simple table, no neural critic yet.

Key examples:

```text
route.increase.memory_to_state
boundary.increase.sequence_change
head.decrease.detail_shortcut
memory.decrease.overwrite
```

Stored stats:

```json
{
  "trials": 12,
  "accepted": 5,
  "mean_delta_loss": 0.006,
  "mean_delta_pqs": 0.018,
  "uncertainty": 0.12,
  "cooldown": 0
}
```

Screen/explore score:

```text
UCB = mean_gain + beta * uncertainty - risk - complexity
```

Deploy score is not used in v4.4 because deploy is disabled.

## Candidate selection

Order:

```text
rules generate candidates
history removes cooldown/duplicates
bandit ranks candidates
forward-only screen tests candidates
action_history updates
bandit_state.json updates
```

## Safety

```text
all candidates deploy=false
no model logits changed permanently
no feedback_bias buffers active
no train-step verify
no optimizer rollback needed
```

## Success criteria

After 5 epochs:

```text
action_history.jsonl exists
bandit_state.json exists
screened candidates have delta_loss and delta_pqs
PQS is logged per epoch
bandit does not repeat rejected actions endlessly
no training collapse vs v4.3
```

## Future v4.5

Only after v4.4 has stable screened records:

```text
detached FeedbackBiasBank for route/boundary
max 1 accepted LCB-safe action per epoch
bias cooldown
bias rollback
```
