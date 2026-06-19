# v4.6.4 Primitive Matrix Scanner / Action-Matrix Program Plan

## Purpose

This document locks the next architecture direction after v4.6.3.

The main correction is terminological and architectural:

```text
Old wording in code/conversation:
- step
- lane
- primitive selector
- edge program

Clear wording for the next version:
- layer = one sequential program step
- slot = a state/output position inside a layer
- layer operation = an action placed in a matrix cell
- action matrix = matrix of operations inside a layer
- primitive matrix = topological space of available primitive actions
- scanner = low-rank module that scans primitive/action space before execution
- proposal = candidate action suggested by scanner
- simulation = cheap low-rank preview of what a proposal would do
- executor = full operation actually applied after selection
```

So when we say `layer`, it means the same idea that older code called `step`: a sequential stage that receives previous state and produces the next state. The difference is that v4.6.4 will make the contents of each layer explicit as an action matrix.

---

## Why v4.6.4 is needed

v4.6.3 fixed the biggest logical issue:

```text
old: primitive[source] then route source -> target
new: primitive[source,target] chosen from edge context
```

That made operation choice aware of where the result will be written.

But v4.6.3 is still not a full program builder. It still chooses from a mostly flat primitive list per edge. It does not yet have a structured primitive space, local primitive neighborhoods, cheap proposal simulation, explicit replace/bypass repair, or a clear action matrix per layer.

The new goal:

```text
Controller should not just choose a primitive.
Controller should scan a structured space of primitives,
simulate top-K candidates cheaply,
choose an operation cell,
execute it fully,
and learn whether the simulation predicted useful behavior.
```

This follows the user-provided design note: controller as a world model / simulator that sees K low-rank alternatives before executing one, plus primitive matrix topology and window scanning. See the uploaded note for the original idea and vocabulary. fileciteturn116file0

---

## Core idea in one sentence

v4.6.4 should turn the layer from a soft edge router into a differentiable action-matrix program:

```text
Layer[t]
  = ActionMatrix[t, source_slot, target_slot]
  where each cell chooses:
    action group
    primitive
    operation rank
    compose mode
    sign
    edge operation
    bypass/replace/disable
    output write
```

The primitive choices come from a topological PrimitiveMatrix that is scanned by windows and low-rank proposal simulations.

---

## Full architecture diagram

```text
                         REAL INPUT
                             │
                             ▼
                Structured / learned input encoder
           (for audio: conv OR later matrix DCT/frames)
                             │
                             ▼
                      state_grid[slots, D]
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────────┐
│                         LAYER t                                     │
│                                                                     │
│  Previous layer state + memory + previous action choices             │
│                             │                                       │
│                             ▼                                       │
│  ┌───────────────────────────────────────────────────────────────┐  │
│  │                 PrimitiveMatrix / Action Space                 │  │
│  │                                                               │  │
│  │  5x5 first, later 10x10                                      │  │
│  │                                                               │  │
│  │  [local/diff] [smooth] [shift] [gate] [product]               │  │
│  │  [low_rank]   [channel] [merge] [split] [route]               │  │
│  │  [memory_read][memory_write][forget][recall][output]          │  │
│  │                                                               │  │
│  │  Each cell has:                                               │  │
│  │    primitive embedding                                        │  │
│  │    category embedding                                         │  │
│  │    topology coordinates                                       │  │
│  │    low-rank simulation projection                             │  │
│  │    optional edge-op prior                                     │  │
│  └───────────────────────────────────────────────────────────────┘  │
│                             │                                       │
│                             ▼                                       │
│  ┌───────────────────────────────────────────────────────────────┐  │
│  │                    Window Projection Scanner                   │  │
│  │                                                               │  │
│  │  For every primitive-matrix cell:                              │  │
│  │    scan 3x3 neighborhood                                      │  │
│  │    project center + neighbors                                 │  │
│  │    combine with current state/memory/head/context              │  │
│  │    produce proposal scores                                    │  │
│  │                                                               │  │
│  │  Output:                                                      │  │
│  │    proposal_scores[layer_cell, action_group]                  │  │
│  │    proposal_scores[layer_cell, primitive]                     │  │
│  │    replace_score                                              │  │
│  │    bypass_score                                               │  │
│  │    edge_op_score/sign                                         │  │
│  └───────────────────────────────────────────────────────────────┘  │
│                             │                                       │
│                             ▼                                       │
│  ┌───────────────────────────────────────────────────────────────┐  │
│  │                 Top-K Low-Rank Candidate Simulator             │  │
│  │                                                               │  │
│  │  For each action-matrix cell:                                 │  │
│  │    take top_k proposals                                       │  │
│  │    simulate each with rank 16/32                              │  │
│  │    predict cheap delta / quality                              │  │
│  │                                                               │  │
│  │  Controller sees:                                             │  │
│  │    current context                                            │  │
│  │    previous choice                                            │  │
│  │    all K simulated outcomes                                   │  │
│  │    memory stats                                               │  │
│  │    optional head/input stats                                  │  │
│  └───────────────────────────────────────────────────────────────┘  │
│                             │                                       │
│                             ▼                                       │
│  ┌───────────────────────────────────────────────────────────────┐  │
│  │                    ActionMatrix Controller                     │  │
│  │                                                               │  │
│  │  Chooses per matrix cell:                                     │  │
│  │    action group                                               │  │
│  │    primitive                                                  │  │
│  │    rank mode                                                  │  │
│  │    compose mode                                               │  │
│  │    sign                                                       │  │
│  │    edge_op scalar/sign                                        │  │
│  │    bypass / replace / disable                                 │  │
│  │    output write                                               │  │
│  │                                                               │  │
│  │  Uses soft/gumbel weights during training.                     │  │
│  └───────────────────────────────────────────────────────────────┘  │
│                             │                                       │
│                             ▼                                       │
│  ┌───────────────────────────────────────────────────────────────┐  │
│  │                         Executor                               │  │
│  │                                                               │  │
│  │  Runs full chosen operations:                                 │  │
│  │    message[source,target]                                     │  │
│  │    state_next[target]                                         │  │
│  │    memory read/write                                          │  │
│  │    output tape write                                          │  │
│  │                                                               │  │
│  │  Also supports:                                               │  │
│  │    bypass: skip bad cell, pass signal forward                 │  │
│  │    replace: use a better primitive than current cell          │  │
│  │    disable: no write, but keep trace for credit               │  │
│  └───────────────────────────────────────────────────────────────┘  │
│                             │                                       │
│                             ▼                                       │
│                       state_grid[t+1]                              │
└─────────────────────────────────────────────────────────────────────┘
                             │
                  repeated for layers t=0..T-1
                             │
                             ▼
                        Output Tape
         sum over selected layer/slot output writes
                             │
                             ▼
                       final classifier
                             │
                             ▼
                          task loss
                             │
                             ▼
                 Credit / ablation / sim-quality
                             │
                             ▼
        scanner, simulator, controller, executor all get gradients
```

---

## Terminology lock

To avoid confusion in code and reports:

| Old name | New name | Meaning |
|---|---|---|
| step | layer | Sequential program stage |
| lane | slot | State/output position inside a layer |
| primitive selector | layer operation selector | Chooses operation in an action-matrix cell |
| edge | cell connection / source->target | Directed slot connection inside a layer |
| route | edge_gate | Whether source->target connection is active |
| write_gate | write amount | How much the active connection writes |
| boundary | phase/structure gate | Whether the operation marks a structural transition |
| output_gate | output write gate | Whether layer/slot writes to output tape |
| trace | program trace | Full record of selected operations, gates, credit |

Important naming rule:

```text
Layer = sequential stage.
Operation = what happens inside a layer.
ActionMatrix = grid of operations inside a layer.
PrimitiveMatrix = global/topological library of possible operations.
```

---

## PrimitiveMatrix

### Size

Start small:

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

### Why not 10x10 immediately

10x10 has too many degrees of freedom for the first implementation. It will make it hard to know whether improvements come from topology, scanner, bypass, simulation, or just parameter count. 5x5 is enough to test the idea.

### Example 5x5 layout

```text
┌──────────────┬──────────────┬──────────────┬──────────────┬──────────────┐
│ identity     │ gated_keep   │ diff         │ contrast     │ smooth       │
├──────────────┼──────────────┼──────────────┼──────────────┼──────────────┤
│ low_rank     │ channel      │ ctx_matrix   │ product      │ gated_add    │
├──────────────┼──────────────┼──────────────┼──────────────┼──────────────┤
│ merge        │ split        │ route        │ edge_gate    │ write_gate   │
├──────────────┼──────────────┼──────────────┼──────────────┼──────────────┤
│ memory_read  │ memory_write │ forget       │ recall       │ memory_gate  │
├──────────────┼──────────────┼──────────────┼──────────────┼──────────────┤
│ output_write │ output_mix   │ bypass       │ replace      │ disable      │
└──────────────┴──────────────┴──────────────┴──────────────┴──────────────┘
```

This is not a hard final taxonomy. It is a seed topology. The embeddings and credit should adjust usage over training.

### Topology meaning

Close cells should be semantically related but not identical:

```text
identity / gated_keep / diff / smooth are local transformations.
low_rank / channel / ctx_matrix are learned transformations.
merge / split / route / write_gate are structural operations.
memory_read / memory_write / forget / recall are state operations.
output_write / bypass / replace / disable are program-control operations.
```

---

## WindowScanner

The WindowScanner scans the PrimitiveMatrix like a small convolution over operation space, not over audio.

For each primitive-matrix cell `(r,c)`:

```text
center_embed = E[r,c]
neighbor_embeds = E[r-1:r+2, c-1:c+2]
window_context = concat(center, mean(neighbors), max(neighbors), topology_position)
```

Then it combines that with program state:

```text
scan_input = [
  window_context,
  layer_context,
  source_slot_state,
  target_slot_state,
  source-target delta,
  memory_stats,
  previous_action_embed,
  optional head/input stats,
]
```

Outputs:

```text
proposal_group_score[cell, group]
proposal_primitive_score[cell, primitive]
replace_score[cell]
bypass_score[cell]
disable_score[cell]
edge_op_score[cell]
output_score[cell]
```

---

## ProjectionScanner

The scanner should not fully execute all primitives. It should use cheap projections.

Separate projections are important. Do not concatenate everything into one black box immediately.

```text
proj_context  = W_context(current state)
proj_before   = W_before(previous action choice)
proj_candidate[k] = W_candidate(candidate primitive/action)
proj_memory   = W_memory(memory stats)
proj_head     = W_head(optional head/attention stats)
```

Candidate score:

```text
score[k] = f(
  proj_context,
  proj_before,
  proj_candidate[k],
  proj_memory,
  proj_head
)
```

This makes reports more understandable: we can see whether the score came from state, previous choice, candidate identity, memory, or head/input context.

---

## Top-K low-rank simulation

The controller should think before acting.

Current weak behavior:

```text
state -> choose one primitive -> execute -> learn after the fact
```

New behavior:

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

sim_results = []
for candidate in topk:
    sim = low_rank_simulator(candidate, state_cell, memory)  # rank 16/32
    sim_results.append(sim)

choice_logits = controller(context, sim_results, proposal_scores)
choice_weights = gumbel_softmax(choice_logits)

full_update = executor(choice_weights, full_primitives, state_cell, memory)
```

### Why this is efficient

K low-rank simulations can be similar in cost to one full operation, but the controller sees K possible futures. It receives more information per training step.

---

## ActionMatrix inside each layer

Each layer owns an ActionMatrix:

```text
ActionMatrix[layer, source_slot, target_slot]
```

Each cell chooses an instruction:

```text
cell_action = {
  group,
  primitive,
  rank_mode,
  compose_mode,
  sign,
  edge_op,
  bypass_gate,
  replace_gate,
  disable_gate,
  output_gate,
}
```

Execution:

```text
candidate_update = PrimitiveExecutor(primitive, source, target, memory)
edge_update = edge_op * sign * candidate_update
cell_output =
  (1 - bypass) * edge_update
  + bypass * bypass_value

state_next[target] += edge_gate * write_gate * boundary * cell_output
```

### Not every cell must be active

All cells are soft during training:

```text
active_mass = edge_gate * write_gate * boundary * (1 - disable_gate)
```

But reports should show the effective hard interpretation:

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

### Bypass

Bypass means the cell does not transform, but it passes signal forward if there is a valid outgoing connection.

```text
A -> [B] -> C
B bypassed: A -> C signal continues
```

Differentiable form:

```text
cell_out = (1 - bypass_gate) * primitive_out + bypass_gate * passthrough
```

Risk: model bypasses everything.

Protection:

```text
bypass_budget_loss
bypass_all_collapse_penalty
minimum_active_operation_mass
```

### Replace

Replace means the scanner proposes a better primitive for this cell than the current/topological cell.

```text
replace_weight[cell, candidate]
```

This allows the program to repair bad cells.

### Disable

Disable means the cell becomes no-op. It should be allowed but budgeted.

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

## Edge operation

Edge op should be simple first.

Per source->target connection:

```text
edge_op = scalar strength + sign
```

Example:

```text
source A -> target B: +0.73
source B -> target C: -0.21
source C -> target D: bypass
```

Do not start with a full edge MLP for every edge. That will hide meaning and explode parameters.

Start with:

```text
edge_strength = tanh(edge_param + edge_prior)
edge_sign_balance_loss
edge_strength_cost
```

---

## Output count / branching

The user's branching idea should be implemented with fixed max slots and soft gates, not dynamic tensor shapes.

Concept:

```text
layer 0:
  one input slot may produce 2 active output slots

layer 1:
  each active slot may produce children or merge

final:
  all active slots are collected into one output_state
```

Implementation:

```text
max_slots = 8 or 16
slot_alive[layer, slot]
split_count[layer, slot] = softmax([0 child, 1 child, 2 child])
child_gate[layer, parent, child]
merge_gate[layer, child, collector]
```

### Alternating layer roles

Use soft role embeddings, not hard rules:

```text
layer 0: expand-biased
layer 1: merge-biased
layer 2: expand-biased
layer 3: merge/output-biased
```

The controller can violate these roles if CE loss demands it.

### Final output

Always produce one final output:

```text
output_state = weighted_merge(all active output writes)
logits = classifier(output_state)
```

---

## Simulation quality loss

Not for the first implementation if it makes debugging too hard, but it is the next important stage.

The simulator predicts quality:

```text
predicted_gain[cell, candidate]
```

After execution/ablation, real credit gives:

```text
real_gain[cell, candidate]
```

Loss:

```text
sim_quality_loss = MSE(predicted_gain, stopgrad(real_gain))
```

Report metric:

```text
sim_pred_vs_real_corr
```

If correlation is low, scanner is hallucinating. If correlation improves, controller is learning to think before acting.

---

## Losses and protections

Required in v4.6.4 base:

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

There are two different goals:

1. nearby primitive cells should have related meaning;
2. nearby primitive cells must not collapse into identical behavior.

So use both:

```text
topology_near_loss:
  neighbors should be moderately close

diversity_loss:
  all primitives should not be identical

far_separation_loss:
  far cells should not be too close
```

Simple form:

```text
cos_dist(center, neighbor) <= margin_near
cos_dist(center, far) >= margin_far
```

Use small weights first. Too strong topology loss can force fake structure and hurt task learning.

---

## Credit and ablation

Credit should be measured at several levels:

```text
layer credit:
  disable whole layer

cell credit:
  disable ActionMatrix[layer,i,j]

primitive credit:
  disable primitive globally or per category

branch credit:
  disable child branch

bypass credit:
  force bypass / force non-bypass

replace credit:
  disable replacement candidate

output credit:
  disable output write layer/slot

memory credit:
  no memory / no write / no read
```

Interpretation:

```text
delta CE > 0:
  disabling hurt -> component useful

delta CE < 0:
  disabling helped -> component suspicious
```

Delayed credit penalty:

```text
epoch N:
  measure credit on validation subset

epoch N+1:
  softly penalize action mass for suspicious components
```

Do not use same-batch hard credit as the first version; it is noisy and can destabilize training.

---

## Reports required

Every run should produce:

```text
LATEST_RUN_REPORT.md
REPORT_TO_CHATGPT.txt
final_report.json
metrics.csv
trace_epoch_XXX.json
credit_ablation_epoch_XXX.json
```

Additional v4.6.4 report sections:

```text
ActionMatrix top action per layer/cell
PrimitiveMatrix usage heatmap
PrimitiveMatrix bypass heatmap
PrimitiveMatrix replace heatmap
active slots per layer
split count per layer
merge graph per layer
output writes per layer/slot
useful/suspicious cells
useful/suspicious branches
sim_pred_vs_real_corr if enabled
```

ASCII report example:

```text
Layer 02 ActionMatrix top actions

        target0   target1   target2   target3
src0    BYPASS    DIFF      LOW_RANK  OUTPUT
src1    MERGE     KEEP      GATED_ADD MEMORY_W
src2    SPLIT     CHANNEL   DISABLE   PRODUCT
src3    MEMORY_R  RECALL    MERGE     KEEP

PrimitiveMatrix usage heatmap

identity  gated_keep diff      contrast smooth
low_rank  channel    ctx       product  gated_add
merge     split      route     edge     write
mem_read  mem_write  forget    recall   mem_gate
output    bypass     replace   disable  noop
```

---

## Implementation plan

### v4.6.4-a: documentation + stable naming

Files:

```text
V4_6_4_PRIMITIVE_MATRIX_SCANNER_PLAN.md
```

No runtime behavior changes.

### v4.6.4-b: PrimitiveMatrix only

Add:

```text
modules/primitive_matrix.py
PrimitiveMatrix5x5
primitive embeddings
topology coordinates
neighbor window extraction
usage/topology/diversity metrics
```

Keep execution close to v4.6.3.

Goal:

```text
prove primitive topology does not break training
```

### v4.6.4-c: WindowScanner proposals

Add:

```text
WindowProjectionScanner
proposal_scores[group]
proposal_scores[primitive]
replace/bypass/disable scores
```

Still no simulation quality loss.

Goal:

```text
controller receives structured candidate proposals
```

### v4.6.4-d: Top-K low-rank simulation

Add:

```text
top_k proposals per action cell
rank-16/32 simulator bank
controller sees simulated outcomes
```

Goal:

```text
controller chooses from predicted consequences, not just logits
```

### v4.6.4-e: ActionMatrix executor

Add explicit action matrix:

```text
ActionMatrix[layer, source_slot, target_slot]
cell_action = group/primitive/rank/compose/sign/edge_op/bypass/replace/disable/output
```

Goal:

```text
make each layer reportable as a matrix of actions
```

### v4.6.4-f: Branch / output count

Add:

```text
slot_alive
split_count 0/1/2
child_gate
merge_gate
final collector
```

Goal:

```text
model can vary number of intermediate outputs while final output stays one
```

### v4.6.4-g: Simulation quality learning

Add:

```text
predicted_gain
real_gain from delayed credit
sim_quality_loss
sim_pred_vs_real_corr report
```

Goal:

```text
scanner learns to propose actions that really help
```

---

## What not to do immediately

Do not implement all at once.

Avoid initially:

```text
10x10 primitive matrix
hard top-k during training
same-batch credit penalty
full edge MLP per connection
strong topology loss
strong bypass penalty
hard manual layer roles
```

Start with soft, small, report-heavy version.

---

## Evaluation plan

Compare at least:

```text
v4.6.3 edge program with conv frontend
v4.6.3 no-conv ablation
v4.6.4 primitive matrix scanner
v4.6.4 without scanner
v4.6.4 without simulation
v4.6.4 without bypass/replace
v4.6.4 without memory
```

Important metrics:

```text
val_acc
frontend dependence
primitive collapse flags
route/edge uniformity
active cells per layer
active slots per layer
useful/suspicious cell credit
useful/suspicious branch credit
memory ablation delta
output ablation delta
sim_pred_vs_real_corr
```

---

## Current conclusion

The next version should not be just another edge controller. It should become a primitive-space scanner and action-matrix program.

The key change:

```text
Controller stops choosing from a flat primitive list.
Controller scans a topological primitive matrix,
sees local primitive neighborhoods,
simulates top-K candidates in low rank,
and then writes an explicit action matrix for the layer.
```

That is the direct path toward an interpretable differentiable program builder.
