# v4.6 Loop Core Implementation Status

## What was implemented

- Modular runtime under `simple_butterfly_matrix_v4_tape_lane/v4_6_loop_core/`.
- `JointController`: one shared latent `z[t,L,D]` drives boundary, route, write/fanout, primitive scores, primitive signs, compose mode, and memory gates.
- `ParallelPrimitiveSelector`: parallel low-rank/matrix primitive bank with signed `gumbel_softmax` mixture.
- `MatrixMemory`: real `W_write`, `W_read`, and learnable forget λ.
- `losses.py`: mild differentiable structural losses for boundary, route, primitive specialization, signs, program cost, memory cost, and logit norm.
- `StepAnalyzer`: compact report-only summaries, collapse flags, `REPORT_TO_CHATGPT.txt`, `trace_epoch_XXX.json`, `final_report.json`, `AGENT_STATUS.md`.
- Main train/eval file: `tape_lane_transport_v4_6_loop_core.py`.
- Command scripts: validation and 5-epoch sync run with checkpoint guard.

## What was intentionally not implemented

- No offline training feedback through JSON.
- No Council calibrator in the training path.
- No policy-gradient/value training.
- No projection tree before the simple differentiable loop proves itself.
- No full-rank primitive expert bank in the MVP.
- No checkpoints are saved or staged by the sync script.

## How closed loop is preserved

The controller decision affects execution, execution affects loss, loss gradient updates the controller decision path.

The controller outputs directly change primitive selection, signs, route, boundary, write gates, compose mode, and memory write/read gates inside `forward()`. The task loss is computed from the logits produced after those decisions are executed, so ordinary backprop updates the controller, primitive experts, route heads, sign heads, memory matrices, and classifier path.

## How memory is real memory

Memory is not just lane #4. `MatrixMemory` keeps a causal vector state:

```text
M[t] = λ · M[t-1] + (1-λ) · W_write(z[t])
read[t] = W_read(M[t])
```

The read vector enters the next controller context, the lane state update path, and the output head, and the report logs read/write norms plus read influence proxy.

## Where gradients flow

Gradients flow through:

- controller trunk and all decision heads;
- Gumbel-softmax relaxed primitive weights;
- signed primitive coefficients via `tanh(sign_logits)`;
- low-rank primitive matrices and channel/product/context experts;
- route softmax and boundary sigmoid;
- compose/write gates;
- `W_write`, `W_read`, and forget λ;
- output memory path and classifier.

## How to run smoke

Validation:

```bash
bash simple_butterfly_matrix_v4_tape_lane/v4_6_loop_core/commands/validate_v4_6_loop_core.sh
```

Full intended 5-epoch run:

```bash
EPOCHS=5 \
TRAIN_LIMIT=12000 \
VAL_LIMIT=2000 \
BATCH_SIZE=128 \
EVAL_BATCH_SIZE=256 \
AMP=fp16 \
bash simple_butterfly_matrix_v4_tape_lane/v4_6_loop_core/commands/sync_run_5ep_v4_6_loop_core_push_logs.sh
```

No-data local smoke:

```bash
RUN_SYNTHETIC_SMOKE=1 DEVICE=cpu AMP=off bash simple_butterfly_matrix_v4_tape_lane/v4_6_loop_core/commands/validate_v4_6_loop_core.sh
```

## Known risks

- Boundary can still collapse; first logs decide whether v4.6.1 needs soft-topk boundary.
- Primitive selection may be too uniform if tau decays too slowly, or collapse if tau decays too fast.
- Memory can become junk if write norm grows without read influence.
- The Conv1D frontend is intentionally simple; the point of first smoke is structural behavior, not maximum SpeechCommands accuracy.

## Next fixes after first logs

- If `BOUNDARY_EXPLOIT` or `BOUNDARY_DEAD`: tune boundary budget, then add soft-topk in v4.6.1.
- If `PRIMITIVE_UNIFORM`: anneal tau faster or increase score head capacity.
- If `PRIMITIVE_COLLAPSE`: increase early tau, lower primitive rank, add diversity by step/lane.
- If `MEMORY_DEAD`: increase memory read path into state/output and lower memory write cost.
- If `MEMORY_JUNK`: add causal memory ablation and reduce blind write pressure.
- If `ROUTE_UNIFORM`: mildly increase allowed/disallowed route pressure.
- If `ROUTE_IDENTITY_COLLAPSE`: reduce identity pressure and boost useful boundary-coupled transitions.
