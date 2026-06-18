# v4.6.1 Grouped Loop Core Implementation Status

## What is now implemented

- New real-data entrypoint: `tape_lane_transport_v4_6_1_grouped_core.py`.
- Sync script now uses the v4.6.1 grouped entrypoint.
- Synthetic data and synthetic fallback are disabled in evidence runs. Synthetic is kept only as validate/unit smoke through the old unit main.
- Route prior default is `0.0`; no hard/manual route prior is active by default.
- `JointController` now receives structured context stats: input density, variance, norm, memory norm, previous update norm, step progress, lane state norm, lane state variance.
- `JointController` also receives previous choice context from learned group/primitive embeddings.
- `ParallelPrimitiveSelector` is now hierarchical: group choice -> primitive choice inside group.
- Primitive groups are logical action categories: keep, channel, correction, aggregation, memory, composition.
- Report-only credit ablation is added: ablate each group and primitive on a small validation batch and record delta CE. This does not train the model and does not write JSON feedback into the training path.
- Failed run artifacts are not committed automatically; only status can be updated.

## Closed-loop invariant

The controller decision affects execution, execution affects loss, loss gradient updates the controller decision path.

The new group/primitive selector is still differentiable:

```text
group_scores -> group softmax
primitive_scores -> primitive softmax inside group
final_weight[group, primitive] = group_weight[group] * primitive_weight[primitive | group]
```

The final signed update is executed in the forward pass, produces logits, and is trained by normal backprop.

## Evidence rules

Evidence run:

```text
SYNTHETIC_DATA=0
ALLOW_SYNTHETIC_FALLBACK=0
ROUTE_PRIOR_STRENGTH=0.0
NUM_PRIMITIVES=18
```

Synthetic runs are allowed only for unit smoke:

```bash
RUN_SYNTHETIC_SMOKE=1 DEVICE=cpu AMP=off bash simple_butterfly_matrix_v4_tape_lane/v4_6_loop_core/commands/validate_v4_6_loop_core.sh
```

## Main command

```bash
EPOCHS=5 \
TRAIN_LIMIT=12000 \
VAL_LIMIT=2000 \
BATCH_SIZE=128 \
EVAL_BATCH_SIZE=256 \
AMP=fp16 \
DATA_ROOT=../architecture_builder/data/speechcommands \
bash simple_butterfly_matrix_v4_tape_lane/v4_6_loop_core/commands/sync_run_5ep_v4_6_loop_core_push_logs.sh
```

## What to inspect after run

- `REPORT_TO_CHATGPT.txt`
- `final_report.json`
- `trace_epoch_XXX.json`
- `credit_ablation_epoch_XXX.json`
- `metrics.csv`

Important fields:

```text
boundary_mean / peak_count
group weights by step/lane
primitive weights by step/lane
primitive sign negative share
memory read influence
credit ablation per group
credit ablation per primitive
route entropy/self/useful transition mass
cost dominance
collapse flags
```

## Known risks

- Group selector may collapse to one group; inspect group weights and group credit ablation.
- Credit ablation is report-only and more expensive than normal eval.
- Context stats are weak navigation context, not proof of causal usefulness by themselves.
- If memory credit is negative or near zero, memory path should be reduced or ablated in v4.6.2.
