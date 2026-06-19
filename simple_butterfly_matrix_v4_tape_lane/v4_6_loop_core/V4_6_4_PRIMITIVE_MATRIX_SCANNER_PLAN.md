# v4.6.4 Primitive Matrix Scanner / Action-Matrix Program Plan

<!--
AGENT_NAVIGATION: line ranges below are generated for quick reading.
If this file is edited, regenerate the line ranges.
Fast navigation commands:
  grep -n "^## " V4_6_4_PRIMITIVE_MATRIX_SCANNER_PLAN.md
  sed -n 'START,ENDp' V4_6_4_PRIMITIVE_MATRIX_SCANNER_PLAN.md
-->

## Agent quick navigation

| Lines | Section | What to read here |
|---:|---|---|
| 44-71 | Purpose and terminology lock | fixed naming: layer/slot/action matrix |
| 73-106 | Coverage checklist from user notes | audit of all transferred ideas |
| 108-133 | Why v4.6.4 is needed | what v4.6.3 still lacks |
| 135-154 | Core idea | one sentence target |
| 156-188 | Full architecture diagram | forward architecture |
| 190-235 | Explicit closed-loop diagram | credit loop back into modules |
| 237-275 | Sequential layer specialization | listen/diversity/specialization losses |
| 277-305 | PrimitiveMatrix | 5x5/10x10 topology and layout |
| 307-341 | WindowScanner | 3x3 scanning over operation space |
| 343-362 | ProjectionScanner | separate readable projections |
| 364-395 | Three gradient paths from projections | A/B/C/D learning routes |
| 397-419 | Top-K low-rank simulation | cheap thinking before acting |
| 421-452 | ActionMatrix inside each layer | cell instruction/execution |
| 454-490 | Bypass / replace / disable | repair and collapse protections |
| 492-510 | Edge operation and gate separation | edge/write/phase/output gates |
| 512-543 | Output count / branching | variable outputs with fixed slots |
| 545-573 | Input encoder and no-conv result | Conv/no-conv lesson and matrix frontend |
| 575-590 | Simulation quality loss | predicted_gain vs real_gain |
| 592-622 | Losses and protections | base/later losses |
| 624-646 | Topology and diversity losses | near/far primitive constraints |
| 648-670 | Credit and ablation | credit levels and interpretation |
| 672-711 | Reports required | run artifacts and report sections |
| 713-762 | Implementation plan | staged v4.6.4-a..h |
| 764-781 | What not to do immediately | avoid overbuilding/hardcoding |
| 783-816 | Evaluation plan | comparisons and metrics |
| 818-840 | Current conclusion | locked direction |

---

## Purpose and terminology lock

This document locks the next architecture direction after v4.6.3.

Terminology:
```text
step -> layer
lane -> slot
primitive selector -> layer operation selector
edge -> source->target cell connection
route -> edge_gate
trace -> program trace
```

Meaning:
```text
Layer = one sequential program stage.
Slot = one state/output position inside a layer.
Operation = what happens inside a layer.
ActionMatrix = matrix of operations inside one layer.
PrimitiveMatrix = global topological library of available actions.
Scanner = module that scans primitive/action space before execution.
Proposal = candidate action suggested by scanner.
Simulation = cheap low-rank preview of a proposal.
Executor = full operation applied after selection.
```

---

## Coverage checklist from user notes

```text
[x] PrimitiveMatrix 5x5 first, later 10x10
[x] PrimitiveMatrix topology instead of flat primitive list
[x] WindowScanner 3x3 first, later 5x5
[x] ProjectionScanner with proj_context/proj_before/proj_candidate/proj_memory/proj_head
[x] Top-K low-rank simulation before full execution
[x] controller sees simulated outcomes before choosing
[x] ActionMatrix per layer: each cell = operation/action
[x] source->target edge context before primitive choice
[x] separate edge_gate/write_gate/phase_gate/output_gate
[x] route is not collapsed into write_gate
[x] replace / bypass / disable cells
[x] bypass-all collapse protection
[x] edge_op scalar/sign per connection first
[x] topology loss + diversity loss
[x] scanner proposal / sim-quality metric
[x] delayed credit penalty from validation ablation
[x] credit on layer/cell/primitive/branch/bypass/replace/output/memory
[x] layer listens to previous layer explicitly
[x] layer diversity / anti-copy loss
[x] layer specialization credit
[x] roles are weak priors only, not hard expand/merge masks
[x] variable output count via fixed max slots + soft gates
[x] final output always one output_state
[x] dynamic output tape across layers/slots
[x] structured input encoder: Conv now, later matrix DCT/frame frontend
[x] no-conv ablation result captured: raw pooling failed
[x] reports include action matrices, heatmaps, active slots, split/merge, credit
[x] staged implementation, not all-at-once
```

---

## Why v4.6.4 is needed

v4.6.3 fixed the main logical issue:
```text
old: primitive[source] then route source -> target
new: primitive[source,target] chosen from edge context
```

But v4.6.3 still lacks:
```text
structured primitive space
local primitive neighborhoods
cheap proposal simulation
explicit replace/bypass repair
ActionMatrix report per layer
layer listening / layer specialization
branch/output count selection
sim-quality training path
```

New goal:
```text
scan structured primitive space -> simulate Top-K candidates -> choose operation cell -> execute fully -> learn whether simulation predicted useful behavior
```

---

## Core idea

v4.6.4 turns the layer into a differentiable ActionMatrix program:
```text
Layer[t] = ActionMatrix[t, source_slot, target_slot]

Each cell chooses:
  action group
  primitive/action
  operation rank
  compose mode
  sign
  edge operation scalar/sign
  bypass / replace / disable
  output write
```

Primitive choices come from a topological PrimitiveMatrix scanned by windows and low-rank proposal simulations.

---

## Full architecture diagram

```text
REAL INPUT
  ↓
Structured / learned input encoder
  ↓
state_grid[layer=0, slots, D]
  ↓
┌──────────────────────────── LAYER t ────────────────────────────┐
│ input: state_grid[t], memory[t], previous layer output/action     │
│                                                                  │
│ PrimitiveMatrix / Action Space                                   │
│   5x5 first, later 10x10                                         │
│   embeddings + topology coords + sim projections                  │
│      ↓                                                           │
│ Window Projection Scanner                                        │
│   scans 3x3 primitive/action neighborhoods                        │
│      ↓                                                           │
│ Top-K Low-Rank Candidate Simulator                               │
│   simulates K proposals with rank 16/32                           │
│      ↓                                                           │
│ ActionMatrix Controller                                          │
│   chooses group/primitive/rank/compose/sign/edge_op/bypass/output │
│      ↓                                                           │
│ Executor                                                         │
│   runs full operation, writes state_next, memory, output tape      │
└──────────────────────────────────────────────────────────────────┘
  ↓ repeated for layers
Output Tape -> final classifier -> task loss
```

---

## Explicit closed-loop diagram

```text
PrimitiveMatrix topology
  ↓
primitive/category embeddings
  ↓
WindowScanner 3x3
  ↓
proposal_scores[cell, group/primitive/replace/bypass]
  ↓
ProjectionScanner: proj_context, proj_before, proj_candidate, proj_memory, proj_head
  ↓
Top-K candidates per ActionMatrix cell
  ↓
Low-Rank Simulator rank 16/32
  ↓
sim_results[K] + predicted_gain[K]
  ↓
ActionMatrix Controller
  ↓
gumbel/soft choice per cell
  ↓
Executor full operation
  ↓
state_next + memory_next + output_tape
  ↓
task_ce + structural losses
  ↓
Credit Collector / ablation
  ↓
real_gain[cell, primitive, layer, branch, output]
  ├─ sim_quality_loss -> Simulator learns consequences
  ├─ proposal_loss -> Scanner learns proposals
  ├─ topology_loss -> PrimitiveMatrix learns topology
  ├─ credit_penalty -> Controller avoids bad action mass
  └─ layer_credit -> Layers specialize instead of copying
```

Default credit schedule:
```text
epoch N: train normally + measure ablation credit on validation subset
epoch N+1: softly penalize suspicious action mass
```

---

## Sequential layer specialization

Later layers must listen to previous layers.

Layer input:
```text
layer_input[t] = concat([
  state_grid[t],
  memory[t],
  prev_layer_output[t-1],
  prev_layer_action_embed[t-1],
  prev_layer_slot_activity[t-1],
  prev_layer_output_writes[t-1],
])
```

layer_listen_loss:
```text
pred_current_action[t] = LayerListenProbe(prev_layer_action_embed[t-1], prev_layer_output[t-1])
layer_listen_loss = CE_or_MSE(pred_current_action[t], stopgrad(action_summary[t]))
```

diversity_between_layers_loss:
```text
similarity = cosine(action_summary[t], action_summary[t-1])
diversity_loss = relu(similarity - max_allowed_similarity)^2
```

specialization_credit:
```text
ablate layer[t]
if delta_CE <= 0: layer[t] suspicious
ablate layer[t] with bypass from t-1 to t+1
if no loss increase: layer[t] is not specialized
```

Roles are weak priors only. No hard expand/merge alternation.

---

## PrimitiveMatrix

Start:
```text
PrimitiveMatrix 5x5 = 25 cells
window = 3x3
top_k = 4
sim_rank = 16 or 32
```

Later:
```text
PrimitiveMatrix 10x10 = 100 cells
window = 3x3 or 5x5
top_k = 4..8
```

Seed 5x5 layout:
```text
identity      gated_keep   diff        contrast    smooth
low_rank      channel      ctx_matrix  product     gated_add
merge         split        route       edge_gate   write_gate
memory_read   memory_write forget      recall      memory_gate
output_write  output_mix   bypass      replace     disable
```

This is a seed topology, not a hard taxonomy.

---

## WindowScanner

WindowScanner scans operation space, not audio.

For cell `(r,c)`:
```text
center_embed = E[r,c]
neighbor_embeds = E[r-1:r+2, c-1:c+2]
window_context = concat(center, mean(neighbors), max(neighbors), topology_position)
```

Scan input:
```text
window_context
layer_context
source_slot_state
target_slot_state
source-target delta
memory_stats
previous_action_embed
optional head/input stats
```

Outputs:
```text
proposal_group_score
proposal_primitive_score
replace_score
bypass_score
disable_score
edge_op_score
output_score
```

---

## ProjectionScanner

Scanner should use cheap projections before full execution.

```text
proj_context      = W_context(current state)
proj_before       = W_before(previous action choice)
proj_candidate[k] = W_candidate(candidate primitive/action)
proj_memory       = W_memory(memory stats)
proj_head         = W_head(optional head/input stats)
```

Candidate score:
```text
score[k] = f(proj_context, proj_before, proj_candidate[k], proj_memory, proj_head)
```

Separate projections make reports readable and gradients separable.

---

## Three gradient paths from projections

Path A: simulation quality.
```text
sim_result[k] = low_rank_sim(candidate[k], state_cell, memory)
predicted_gain[k] = sim_quality_head(sim_result[k])
real_gain[k] = stopgrad(delayed_credit[cell, candidate[k]])
loss_A = MSE(predicted_gain[k], real_gain[k])
```

Path B: task loss through choice weights.
```text
choice_logits[k] = choice_head(proj_context, proj_before, proj_candidate[k], sim_result[k])
choice_weights = gumbel_softmax(choice_logits)
full_result = executor(choice_weights, full_primitives, state)
loss_B = task_ce(full_result)
```

Path C: topology and primitive embedding losses.
```text
topology_loss -> primitive embeddings
diversity_loss -> primitive embeddings/projection heads
usage_balance_loss -> action-group usage and primitive usage
```

Path D: delayed credit penalty.
```text
if real_gain[cell, candidate] < 0:
  credit_penalty += action_mass[cell, candidate] * abs(real_gain)
```

---

## Top-K low-rank simulation

Controller should think before acting.

```text
state -> scanner proposes K candidates
      -> simulate all K cheaply
      -> controller sees predicted outcomes
      -> choose and execute full operation
      -> train scanner/simulator with real credit
```

Pseudo-code:
```python
proposals = scanner(state_grid, primitive_matrix, memory)
topk = proposals.topk(k=4)
sim_results = [low_rank_simulator(c, state_cell, memory) for c in topk]
choice_logits = controller(context, sim_results, proposal_scores)
choice_weights = gumbel_softmax(choice_logits)
full_update = executor(choice_weights, full_primitives, state_cell, memory)
```

---

## ActionMatrix inside each layer

Each layer owns:
```text
ActionMatrix[layer, source_slot, target_slot]
```

Each cell chooses:
```text
group, primitive, rank_mode, compose_mode, sign, edge_op,
bypass_gate, replace_gate, disable_gate, output_gate
```

Execution:
```text
candidate_update = PrimitiveExecutor(primitive, source, target, memory)
edge_update = edge_op * sign * candidate_update
cell_output = (1 - bypass) * edge_update + bypass * passthrough
state_next[target] += edge_gate * write_gate * phase_gate * cell_output
```

Reports must show:
```text
top action per cell
top primitive per cell
bypass cells
replace cells
disabled cells
output cells
```

---

## Bypass / replace / disable

Bypass:
```text
cell_out = (1 - bypass_gate) * primitive_out + bypass_gate * passthrough
```

Risk:
```text
bypass everywhere -> program disappears
```

Protection:
```text
bypass_budget_loss
bypass_all_collapse_penalty
minimum_active_operation_mass
```

Replace:
```text
replace_weight[cell, candidate]
```

Disable:
```text
disable_gate[cell]
```

Protection:
```text
disable_budget
min_useful_cells_per_layer
credit penalty if disabling hurts
```

---

## Edge operation and gate separation

Keep gates separated:
```text
edge_gate[i,j]   = is there a source i -> target j connection?
write_gate[i,j]  = how much should this connection write?
phase_gate[i,j]  = is this a structural transition?
output_gate[t,j] = should this layer/slot write to output tape?
```

Do not collapse route into write_gate.

Edge op starts simple:
```text
edge_op = scalar strength + sign
edge_strength = tanh(edge_param + edge_prior)
```

---

## Output count / branching

Use fixed max slots + soft gates.

```text
max_slots = 8 or 16
slot_alive[layer, slot]
split_count[layer, slot] = softmax([0 child, 1 child, 2 child])
child_gate[layer, parent, child]
merge_gate[layer, child, collector]
```

Roles are weak priors only:
```text
role_embed[layer] = learned vector
role_prior_logits[layer] = small additive bias
```

Forbidden:
```text
hard mask: layer 0 must expand
hard mask: layer 1 must merge
hard mask: no two expand-like layers in a row
```

Final output:
```text
output_state = weighted_merge(all active output writes)
logits = classifier(output_state)
```

---

## Input encoder and no-conv result

v4.6.3 no-conv ablation showed raw pooling + linear projection collapses:
```text
best_acc near random
BOUNDARY_DEAD / BOUNDARY_FLAT / PRIMITIVE_COLLAPSE
memory no_memory delta near zero
output_gate almost off
```

Interpretation:
```text
edge/action program needs meaningful structured input.
Conv1D was a strong feature extractor.
No-conv pooling destroyed useful audio structure.
```

Policy:
```text
Allowed now: Conv frontend for baseline continuity.
Required future ablation: structured matrix frontend:
  frame/unfold
  window matrix
  DCT/FFT-like matrix
  log-energy / spectral bands / delta / onset
  projection to state_grid
```

---

## Simulation quality loss

```text
predicted_gain[cell, candidate]
real_gain[cell, candidate]
sim_quality_loss = MSE(predicted_gain, stopgrad(real_gain))
```

Report:
```text
sim_pred_vs_real_corr
```

If correlation is low, scanner is hallucinating. If it improves, controller is learning to think before acting.

---

## Losses and protections

Required base:
```text
task_ce
edge_gate_cost
write_cost
output_gate_cost
bypass_budget_loss
disable_budget_loss
min_active_cells_loss
primitive_usage_balance
primitive_topology_loss
primitive_diversity_loss
edge_sign_balance_loss
layer_listen_loss
diversity_between_layers_loss
specialization_credit_loss
credit_bad_cell_loss
credit_bad_primitive_loss
credit_bad_edge_loss
```

Later:
```text
sim_quality_loss
scanner_proposal_accuracy_loss
predicted_gain_calibration_loss
```

---

## Topology and diversity losses

Goals:
```text
nearby primitive cells should be related
nearby primitive cells must not collapse into identical behavior
far cells should remain separable
```

Losses:
```text
topology_near_loss
diversity_loss
far_separation_loss
```

Simple form:
```text
cos_dist(center, neighbor) <= margin_near
cos_dist(center, far) >= margin_far
```

---

## Credit and ablation

Credit levels:
```text
layer credit: disable whole layer
cell credit: disable ActionMatrix[layer,i,j]
primitive credit: disable primitive globally/category
branch credit: disable child branch
bypass credit: force bypass / force non-bypass
replace credit: disable replacement candidate
output credit: disable output write layer/slot
memory credit: no memory / no write / no read
scanner credit: disable scanner proposals
simulation credit: disable simulated outcomes
```

Interpretation:
```text
delta CE > 0 -> disabling hurt -> useful
delta CE < 0 -> disabling helped -> suspicious
```

---

## Reports required

Every run:
```text
LATEST_RUN_REPORT.md
REPORT_TO_CHATGPT.txt
final_report.json
metrics.csv
trace_epoch_XXX.json
credit_ablation_epoch_XXX.json
```

v4.6.4 sections:
```text
ActionMatrix top action per layer/cell
PrimitiveMatrix usage/bypass/replace heatmaps
active slots per layer
split count per layer
merge graph per layer
output writes per layer/slot
layer_listen_score
layer_action_similarity[t,t-1]
layer_specialization_credit
useful/suspicious cells and branches
sim_pred_vs_real_corr
scanner proposal vs real credit
```

ASCII report example:
```text
Layer 02 ActionMatrix top actions

        target0   target1   target2   target3
src0    BYPASS    DIFF      LOW_RANK  OUTPUT
src1    MERGE     KEEP      GATED_ADD MEMORY_W
src2    SPLIT     CHANNEL   DISABLE   PRODUCT
src3    MEMORY_R  RECALL    MERGE     KEEP
```

---

## Implementation plan

Do not implement all at once.

```text
v4.6.4-a: documentation + stable naming

v4.6.4-b: PrimitiveMatrix only
  modules/primitive_matrix.py
  PrimitiveMatrix5x5
  topology coordinates
  neighbor window extraction
  usage/topology/diversity metrics

v4.6.4-c: WindowScanner proposals
  WindowProjectionScanner
  proposal_scores[group/primitive]
  replace/bypass/disable scores

v4.6.4-d: Layer listening and specialization metrics
  prev_layer_output/action_embed
  layer_listen_score
  layer_action_similarity
  weak layer_listen_loss
  layer ablation credit

v4.6.4-e: Top-K low-rank simulation
  top_k proposals
  rank-16/32 simulator
  controller sees simulated outcomes

v4.6.4-f: ActionMatrix executor
  explicit cell_action table:
  group/primitive/rank/compose/sign/edge_op/bypass/replace/disable/output

v4.6.4-g: Branch / output count
  slot_alive
  split_count 0/1/2
  child_gate
  merge_gate
  final collector

v4.6.4-h: Simulation quality learning
  predicted_gain
  real_gain from delayed credit
  sim_quality_loss
  sim_pred_vs_real_corr
```

---

## What not to do immediately

Avoid initially:
```text
10x10 primitive matrix
hard top-k during training
same-batch credit penalty
full edge MLP per connection
strong topology loss
strong bypass penalty
hard manual layer roles
hard expand/merge alternation constraints
claiming no-conv/raw-input success without evidence
```

Start with soft, small, report-heavy version.

---

## Evaluation plan

Compare:
```text
v4.6.3 edge program with Conv frontend
v4.6.3 no-conv ablation
v4.6.4 primitive matrix scanner
v4.6.4 without scanner
v4.6.4 without simulation
v4.6.4 without layer listening
v4.6.4 without bypass/replace
v4.6.4 without memory
v4.6.4 with structured matrix frontend
```

Metrics:
```text
val_acc
frontend dependence
primitive collapse flags
route/edge uniformity
active cells per layer
active slots per layer
layer_listen_score
layer_action_similarity
layer_specialization_credit
useful/suspicious cell/branch credit
memory ablation delta
output ablation delta
sim_pred_vs_real_corr
scanner proposal vs real credit correlation
```

---

## Current conclusion

The next version should become a primitive-space scanner and ActionMatrix program.

Key change:
```text
Controller stops choosing from a flat primitive list.
Controller scans a topological primitive matrix,
sees local primitive neighborhoods,
simulates Top-K candidates in low rank,
and writes an explicit ActionMatrix for the layer.
```

Specialization rule:
```text
A layer must listen to the previous layer,
use the previous layer action trace,
and learn a different useful role instead of copying the same ActionMatrix.
```

This is the direct path toward an interpretable differentiable program builder.
