# v4.6.4 Primitive Matrix Scanner / Action-Matrix Program Plan

<!--
Agent navigation.
Use:
  grep -n '^## ' V4_6_4_PRIMITIVE_MATRIX_SCANNER_PLAN.md
  sed -n 'START,ENDp' V4_6_4_PRIMITIVE_MATRIX_SCANNER_PLAN.md
-->

## Agent quick navigation

| Lines | Section | Purpose |
|---:|---|---|
| 46-79 | Purpose / short lock | what this plan targets |
| 81-126 | Terminology lock | layer/slot/action naming |
| 128-164 | Coverage checklist | all transferred ideas |
| 166-196 | Why v4.6.4 is needed | what v4.6.3 lacks |
| 198-262 | Full architecture diagram | forward data path |
| 264-318 | Closed-loop learning diagram | credit loop |
| 320-361 | PrimitiveMatrix | 5x5/10x10 action topology |
| 363-412 | WindowScanner and ProjectionScanner | scanner design |
| 414-451 | Top-K low-rank simulation | cheap thinking before acting |
| 453-493 | Projection gradient paths | how scanner/simulator learn |
| 495-543 | ActionMatrix execution | cell instruction execution |
| 545-574 | Gate separation | separate gates |
| 576-622 | Skip / replace / disable | repair and protections |
| 624-666 | Branching and variable output count | soft branch/merge slots |
| 668-722 | Sequential layer specialization | layers listen and specialize |
| 724-773 | Input encoder honesty | Conv/no-conv and matrix frontend |
| 775-858 | Honest input curriculum | Teacher/Audit/Deploy |
| 860-935 | Honesty audits | honesty tests |
| 937-1034 | Universal insertion into neural network layers | standalone and plug-in modes |
| 1036-1098 | Transformer wrapper modes | after/before/replace attention |
| 1100-1136 | Losses and protections | loss list |
| 1138-1187 | Credit and ablation | credit levels |
| 1189-1243 | Reports required | artifacts and report fields |
| 1245-1317 | Implementation plan | staged v4.6.4-a..k |
| 1319-1339 | What not to do | avoid shortcuts |
| 1341-1391 | Evaluation plan | comparisons and metrics |
| 1393-1426 | Acceptance rules | success criteria |
| 1428-1458 | Current conclusion | locked direction |

---

## Purpose / short lock

This is the final architecture plan for the next version after v4.6.3.

The plan must cover two targets at once:

```text
1. current standalone experiments:
   audio / SpeechCommands / other real datasets

2. future plug-in mode:
   insert the same mechanism inside existing neural networks
   before Attention, after Attention, or instead of Attention / block operation
```

The core must be universal:

```text
PrimitiveMatrix -> Scanner -> Top-K Simulator -> ActionMatrix Controller -> Executor -> Credit
```

Only wrappers change:

```text
audio wrapper
structured matrix frontend wrapper
Transformer after-attention wrapper
Transformer before-attention wrapper
Transformer replacement wrapper
CNN / vision wrapper
MLP / block wrapper
```

---

## Terminology lock

Use these names in code, reports, and commits.

```text
old step        -> layer
old lane        -> slot
old route       -> edge_gate
old primitive selector -> layer operation selector
old trace       -> program trace
```

Definitions:

```text
Layer:
  one sequential program stage. Layer[t] receives state[t] and produces state[t+1].

Slot:
  one state/output position inside a layer.

PrimitiveMatrix:
  global topological library of possible operations/actions.

ActionMatrix:
  matrix of operations inside one layer:
  ActionMatrix[layer, source_slot, target_slot]

Scanner:
  cheap module that scans primitive/action space before full execution.

Proposal:
  candidate action suggested by scanner.

Simulation:
  low-rank preview of what a proposal would do.

Executor:
  full operation applied after controller selection.

Skip:
  pass signal through instead of transforming.
  This replaces the confusing name "bypass" in new code.
```

---

## Coverage checklist

Everything below must be preserved in implementation.

```text
[x] PrimitiveMatrix 5x5 first, later 10x10
[x] primitive topology instead of flat primitive list
[x] WindowScanner 3x3 first, later 5x5
[x] separate projections: context / previous action / candidate / memory / head-input
[x] Top-K low-rank simulation before full execution
[x] ActionMatrix per layer, each cell is an operation
[x] source->target context before primitive choice
[x] separate edge_gate / write_gate / phase_gate / output_gate
[x] route is not collapsed into write_gate
[x] replace / skip / disable cells with collapse protection
[x] edge_op scalar/sign per connection first
[x] topology loss and diversity loss
[x] delayed credit penalty from validation ablation
[x] credit on layer/cell/primitive/branch/skip/replace/output/memory/scanner/simulator
[x] layers listen to previous layer explicitly
[x] layer diversity and anti-copy specialization credit
[x] weak role priors only, no hard expand/merge masks
[x] variable output count via fixed max slots and soft gates
[x] final output always one output_state
[x] dynamic output tape across layers/slots
[x] Conv allowed only as temporary scaffold, not deploy proof
[x] no-conv result recorded: raw pooling failed
[x] honest input curriculum: Teacher -> Audit -> Deploy
[x] allowed hints are meta-process only and decay to zero
[x] answer/leakage hints are banned
[x] honesty audit: no_conv / no_hints / deploy / shuffled / random / transfer
[x] universal plug-in mode: standalone / after Attention / before Attention / replace Attention
[x] layer insertion tests: identity / random / frozen / baseline comparisons
[x] staged implementation, not all at once
```

---

## Why v4.6.4 is needed

v4.6.3 fixed the first logical bug:

```text
old:
  primitive[source] selected first
  route[source,target] decided later

new:
  primitive[source,target] selected from edge context
```

But v4.6.3 still lacks the real program-builder pieces:

```text
structured primitive/action space
local neighborhoods in primitive space
cheap proposal simulation before acting
explicit ActionMatrix per layer
replace / skip / disable repair
variable number of intermediate outputs
layer listening / specialization
sim-quality training
honesty curriculum and deploy audits
universal plug-in layer wrappers
```

So v4.6.4 must not be "one more edge router". It must be an ActionMatrix program builder.

---

## Full architecture diagram

```text
REAL INPUT
  |
  v
Input mode
  - raw / basic structured input in deploy
  - optional scaffold only during Teacher/Audit
  |
  v
state_grid[layer=0, slots, D]
  |
  v
+-------------------------------------------------------------------+
|                            Layer t                                |
|                                                                   |
| Inputs:                                                           |
|   state_grid[t]                                                   |
|   memory[t]                                                       |
|   previous layer output                                           |
|   previous layer action trace                                     |
|   optional allowed process hints                                  |
|                                                                   |
| PrimitiveMatrix 5x5 / 10x10                                       |
|   operation embeddings + topology coordinates                     |
|        |                                                          |
|        v                                                          |
| WindowScanner                                                     |
|   scans 3x3 neighborhood in operation space                       |
|        |                                                          |
|        v                                                          |
| ProjectionScanner                                                 |
|   proj_context / proj_before / proj_candidate / proj_memory       |
|        |                                                          |
|        v                                                          |
| Top-K Low-Rank Simulator                                          |
|   previews K candidate operations                                 |
|        |                                                          |
|        v                                                          |
| ActionMatrix Controller                                           |
|   chooses per source->target cell:                                |
|   group / primitive / rank / compose / sign / edge_op             |
|   edge_gate / write_gate / phase_gate / skip / replace / output   |
|        |                                                          |
|        v                                                          |
| Executor                                                          |
|   runs full operation                                             |
|   writes state_next + memory_next + output_tape                   |
+-------------------------------------------------------------------+
  |
  v
repeat layers
  |
  v
Output tape across layers/slots
  |
  v
final output_state
  |
  v
task head / loss
```

---

## Closed-loop learning diagram

The architecture is not finished unless the feedback loop is explicit.

```text
PrimitiveMatrix topology
  -> primitive/category embeddings
  -> WindowScanner
  -> proposal_scores
  -> ProjectionScanner
  -> Top-K candidates
  -> Low-Rank Simulator
  -> sim_results[K] + predicted_gain[K]
  -> ActionMatrix Controller
  -> soft/gumbel choice
  -> Executor full operation
  -> state_next + memory_next + output_tape
  -> task_ce + structural losses
  -> Credit Collector / ablation
  -> real_gain[layer, cell, primitive, branch, output, memory]
```

Credit returns through several paths:

```text
sim_quality_loss:
  Simulator learns to predict consequences.

proposal_loss:
  Scanner learns to propose useful candidates.

topology_loss:
  PrimitiveMatrix learns meaningful topology.

credit_penalty:
  Controller avoids bad action mass.

layer_credit:
  Layers specialize instead of copying each other.
```

Default credit schedule:

```text
epoch N:
  train normally
  measure ablation credit on validation subset

epoch N+1:
  softly penalize suspicious action mass
```

Do not start with same-batch hard credit. It is noisy and unstable.

---

## PrimitiveMatrix

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

Seed 5x5 layout:

```text
identity      gated_keep   diff        contrast    smooth
low_rank      channel      ctx_matrix  product     gated_add
merge         split        route       edge_gate   write_gate
memory_read   memory_write forget      recall      memory_gate
output_write  output_mix   skip        replace     disable
```

This is a seed topology, not a hard taxonomy. Training, usage, and credit may move behavior around.

Topology meaning:

```text
row 0: local/state-preserving transformations
row 1: learned transformations and products
row 2: structural routing/splitting/writing
row 3: memory operations
row 4: output/program-control operations
```

---

## WindowScanner and ProjectionScanner

WindowScanner scans operation space, not audio/image/time.

For PrimitiveMatrix cell `(r,c)`:

```text
center_embed = E[r,c]
neighbors = E[r-1:r+2, c-1:c+2]
window_context = concat(center, mean(neighbors), max(neighbors), topology_position)
```

The scanner also sees current program context:

```text
source_slot_state
target_slot_state
source-target delta
source-target product
layer context
memory stats
previous action embed
optional allowed process hints
```

ProjectionScanner keeps learning readable:

```text
proj_context      = W_context(current state/context)
proj_before       = W_before(previous action choice)
proj_candidate[k] = W_candidate(candidate primitive/action)
proj_memory       = W_memory(memory stats)
proj_head_input   = W_head(allowed head/input meta-hints)
```

Candidate score:

```text
score[k] = f(
  proj_context,
  proj_before,
  proj_candidate[k],
  proj_memory,
  proj_head_input
)
```

Do not replace this with one black-box concatenation first. Separate projections make reports and credit interpretable.

---

## Top-K low-rank simulation

Controller should think before acting.

Weak behavior:

```text
state -> choose primitive -> execute -> learn after the fact
```

v4.6.4 behavior:

```text
state -> scanner proposes K candidates
      -> simulate K cheaply
      -> controller sees predicted outcomes
      -> controller chooses
      -> executor runs full operation
      -> credit trains scanner/simulator/controller
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

---

## Projection gradient paths

There are four learning paths. All must be represented in reports eventually.

Path A: simulation quality.

```text
sim_result[k] = low_rank_sim(candidate[k], state_cell, memory)
predicted_gain[k] = sim_quality_head(sim_result[k])
real_gain[k] = stopgrad(delayed_credit[cell, candidate[k]])
loss_A = MSE(predicted_gain[k], real_gain[k])
```

Gradient goes into simulator, sim_quality_head, candidate projections, and primitive embeddings.

Path B: task loss through soft/gumbel choice.

```text
choice_weights = gumbel_softmax(choice_logits)
full_result = executor(choice_weights, full_primitives, state)
loss_B = task_ce(full_result)
```

Gradient goes through executor -> choice weights -> proposal scores -> scanner projections.

Path C: topology/diversity losses.

```text
topology_loss -> primitive embeddings
diversity_loss -> primitive embeddings/projection heads
usage_balance_loss -> action group and primitive usage
```

Path D: delayed credit penalty.

```text
if real_gain[cell, candidate] < 0:
  credit_penalty += action_mass[cell, candidate] * abs(real_gain)
```

---

## ActionMatrix execution

Each layer owns:

```text
ActionMatrix[layer, source_slot, target_slot]
```

Each cell chooses:

```text
group
primitive/action
rank_mode
compose_mode
sign
edge_op scalar/sign
edge_gate
write_gate
phase_gate
skip_gate
replace_gate
disable_gate
output_gate
```

Execution:

```text
candidate_update = PrimitiveExecutor(primitive, source, target, memory)
edge_update = edge_op * sign * candidate_update
cell_output = (1 - skip_gate) * edge_update + skip_gate * passthrough
active_mass = edge_gate * write_gate * phase_gate * (1 - disable_gate)
state_next[target] += active_mass * cell_output
```

Reports must show an effective hard interpretation:

```text
top action per cell
top primitive per cell
active cells
skip cells
replace cells
disabled cells
output cells
```

---

## Gate separation

Keep the gates separate because they answer different questions:

```text
edge_gate[i,j]:
  is source i -> target j active at all?

write_gate[i,j]:
  how much should the active edge write?

phase_gate[i,j]:
  is this operation a structural transition / boundary?

output_gate[t,j]:
  should layer t slot j write to output tape?

skip_gate[i,j]:
  should this cell pass signal without transformation?

disable_gate[i,j]:
  should this cell do no write?

replace_gate[i,j,k]:
  should scanner replace current/topological cell by candidate k?
```

Never collapse `edge_gate` into `write_gate`. It destroys analysis and credit.

---

## Skip / replace / disable

Skip replaces the earlier ambiguous word "bypass".

Skip:

```text
cell_out = (1 - skip_gate) * primitive_out + skip_gate * passthrough
```

Risk:

```text
skip everywhere -> program disappears
```

Protections:

```text
skip_budget_loss
skip_all_collapse_penalty
minimum_active_operation_mass
```

Replace:

```text
replace_weight[cell, candidate]
```

This lets scanner repair bad cells.

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

## Branching and variable output count

Use fixed max slots and soft gates, not dynamic tensor shapes.

```text
max_slots = 8 or 16
slot_alive[layer, slot]
split_count[layer, slot] = softmax([0 child, 1 child, 2 children])
child_gate[layer, parent, child]
merge_gate[layer, child, collector]
```

The model may learn:

```text
one input slot -> two child slots
two child slots -> one collector
collector -> memory/output/correction slots
```

Roles are weak priors only:

```text
role_embed[layer] = learned vector
role_prior_logits[layer] = small additive bias
```

Forbidden:

```text
hard rule: layer 0 must expand
hard rule: layer 1 must merge
hard rule: no two expand-like layers in a row
```

Final output always has one state:

```text
output_state = weighted_merge(all active output writes)
logits = classifier(output_state)
```

---

## Sequential layer specialization

Layer `t` must explicitly listen to layer `t-1`.

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
pred_current_action[t] =
  LayerListenProbe(prev_layer_action_embed[t-1], prev_layer_output[t-1])

layer_listen_loss =
  CE_or_MSE(pred_current_action[t], stopgrad(action_summary[t]))
```

diversity_between_layers_loss:

```text
similarity = cosine(action_summary[t], action_summary[t-1])
diversity_loss = relu(similarity - max_allowed_similarity)^2
```

specialization_credit:

```text
ablate layer[t]
if delta_CE <= 0:
  layer[t] suspicious

ablate layer[t] and allow layer[t-1] to skip into layer[t+1]
if loss does not increase:
  layer[t] is not specialized
```

Reports:

```text
layer_listen_score
layer_action_similarity[t,t-1]
layer_specialization_credit
```

---

## Input encoder honesty

Current fact from v4.6.3:

```text
normal Conv frontend: learned and produced useful edge/memory/output credit
no-conv raw pooling + linear frontend: collapsed near random
```

Interpretation:

```text
edge/action program needs meaningful structured input.
Conv1D was a strong feature extractor.
Raw pooling destroyed useful audio structure.
```

Policy:

```text
Conv is allowed only as baseline/scaffold continuity.
Conv performance is not Deploy proof.
```

Future structured matrix frontend:

```text
audio:
  frame/unfold
  window matrix
  DCT/FFT-like matrix
  log-energy / spectral bands / delta / onset
  projection to state_grid

vision:
  patches / simple patch projection
  optional non-answer local change maps

text:
  token embeddings / subword tokens
```

Rule:

```text
basic reading language is allowed
answer-like summary is forbidden
```

---

## Honest input curriculum

Conv and hints are scaffolding, not the building.

A successful architecture must survive Deploy mode without scaffold/hints.

Input levels:

```text
Level 0 raw:
  waveform / pixels / token ids

Level 1 basic features:
  mel-spectrogram, image patches, subword tokens
  allowed as reading language

Level 2 local scaffold:
  Conv/local filters/local patterns
  allowed only temporarily if removed

Level 3 answer-like summary:
  predicted class, softmax probabilities, class boundary, class-specific important positions
  forbidden
```

Allowed hints are meta-process only:

```text
gradient_norm per slot
attention/focus entropy
stopgrad confidence margin
activation novelty
input density / local change map
soft usage-history priors
hint confidence
```

Forbidden hints:

```text
predicted_class
softmax_output
which_slot_to_use
which_primitive_to_use
which_action_to_choose
class-dependent boundary/location
```

Teacher / Audit / Deploy:

```text
Teacher 0%-33%:
  raw/basic + scaffold + allowed hints
  task_ce + scaffold_loss + sim_quality_loss

Audit 33%-66%:
  scaffold decays or is removed
  hints decay
  honesty audit every N epochs

Deploy 66%-100%:
  raw/basic only
  no scaffold
  no external hints
```

Scheduler:

```python
class ScaffoldScheduler:
    def __init__(self, total_epochs):
        self.total = total_epochs

    def conv_weight(self, epoch):
        return max(0.0, 1.0 - epoch / (self.total * 0.33))

    def hint_weight(self, epoch):
        return max(0.0, 1.0 - epoch / (self.total * 0.66))

    def is_deploy(self, epoch):
        return epoch > self.total * 0.66
```

---

## Honesty audits

Audit modes:

```text
full:
  current normal mode

no_conv:
  remove Conv/local scaffold

no_hints:
  remove all head/grad/history hints

deploy:
  raw/basic input only, no scaffold, no hints

shuffle_hints:
  give hints from another sample/batch

random_hints:
  replace hints with random noise

transfer:
  train task A with hints, evaluate task B without hints
```

Pseudo-code:

```python
def honesty_audit(model, val_loader):
    results = {}
    results["full"] = evaluate(model, val_loader, mode="full")
    results["no_conv"] = evaluate(model, val_loader, mode="no_conv")
    results["no_hints"] = evaluate(model, val_loader, mode="no_hints")
    results["deploy"] = evaluate(model, val_loader, mode="deploy")
    results["shuffle"] = evaluate(model, val_loader, mode="shuffle_hints")
    results["random"] = evaluate(model, val_loader, mode="random_hints")

    honesty_score = results["deploy"].acc / max(1e-8, results["full"].acc)
    return results, honesty_score
```

Interpretation:

```text
honesty_score > 0.90:
  excellent

0.80 <= honesty_score <= 0.90:
  acceptable but scaffold-dependent

honesty_score < 0.80:
  not honest yet
```

Shortcut signs:

```text
conv_ablation_delta > 15%:
  too dependent on scaffold

hint_ablation_delta > 20%:
  too dependent on hints

shuffle_hints no effect:
  hints ignored/useless

random_hints improves:
  hint path may leak or regularize by accident

primitive collapse in deploy:
  builder cannot organize without scaffold
```

---

## Universal insertion into neural network layers

v4.6.4 must be usable as a plug-in layer, not only as an audio classifier.

Three insertion positions:

```text
A. Replace Attention / block operation:
   x -> LayerNorm -> PrimitiveMatrixScanner -> FF -> next layer

B. Before Attention:
   x -> LayerNorm -> PrimitiveMatrixScanner -> Attention -> FF

C. After Attention:
   x -> Attention -> PrimitiveMatrixScanner -> FF
```

Recommended path:

```text
1. after Attention:
   safest add-on test

2. before Attention:
   tests whether mechanism prepares better input

3. replace Attention:
   real replacement test
```

Plug-in API:

```python
class PrimitiveMatrixScannerLayer(nn.Module):
    def forward(
        self,
        x_prev,
        next_context=None,
        process_hint=None,
        mode="deploy",
    ):
        proposals = self.scanner(x_prev, process_hint, next_context)
        sim_results = self.simulate_topk(proposals, x_prev)
        action_matrix = self.controller(x_prev, sim_results)
        x_out, trace = self.executor(action_matrix, x_prev)
        return x_out, trace
```

Allowed from previous layer:

```text
x_prev
activation statistics
slot norm / variance / density
activation novelty
stopgrad gradient norm per slot
```

Allowed from next layer:

```text
expected_norm
expected activation scale
expected shape/channel count
```

Forbidden from next layer:

```text
next layer weights as direct readable shortcut
target class information
exact desired activation
```

Allowed from task head:

```text
ordinary loss gradient
stopgrad confidence margin
entropy / uncertainty scalar
```

Forbidden from task head:

```text
predicted class
softmax probability vector
which slot / primitive / action to choose
```

Rule:

```text
Head teaches through gradient.
Head must not tell the mechanism the answer or operation.
```

---

## Transformer wrapper modes

```python
class TransformerLayerWithMechanism(nn.Module):
    def __init__(self, d_model, mode="after"):
        super().__init__()
        self.mode = mode
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.attn = MultiHeadAttention(d_model)
        self.ff = FeedForward(d_model)
        self.mechanism = PrimitiveMatrixScannerLayer(d_model)

    def forward(self, x):
        if self.mode == "replace":
            x = x + self.mechanism(self.norm1(x))[0]
            x = x + self.ff(self.norm2(x))

        elif self.mode == "before":
            x = x + self.mechanism(self.norm1(x))[0]
            x = x + self.attn(self.norm1(x))
            x = x + self.ff(self.norm2(x))

        elif self.mode == "after":
            x = x + self.attn(self.norm1(x))
            x = x + self.mechanism(self.norm1(x))[0]
            x = x + self.ff(self.norm2(x))

        return x
```

Layer honesty tests:

```text
with_mechanism
identity_mechanism
random_mechanism
frozen_mechanism
attention_baseline if applicable
```

Honest improvement:

```text
with_mechanism > identity
with_mechanism > random
with_mechanism > frozen
```

Replacement test:

```text
mechanism_replace_attention >= attention_baseline
or close accuracy with better cost/interpretability
```

Add-on test:

```text
attention + mechanism > attention only
```

---

## Losses and protections

Base losses:

```text
task_ce
edge_gate_cost
write_cost
output_gate_cost
skip_budget_loss
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
honesty_penalty
```

Later losses:

```text
sim_quality_loss
scanner_proposal_accuracy_loss
predicted_gain_calibration_loss
plug_in_layer_honesty_loss
```

Do not start with strong weights. Keep all new regularizers weak and report-heavy first.

---

## Credit and ablation

Credit levels:

```text
layer:
  disable whole layer

cell:
  disable ActionMatrix[layer, source, target]

primitive:
  disable primitive globally or category

branch:
  disable child branch

skip:
  force skip / force non-skip

replace:
  disable replacement candidate

output:
  disable output write layer/slot

memory:
  no memory / no write / no read

scanner:
  disable scanner proposals

simulation:
  disable simulated outcomes

honesty:
  no_conv / no_hints / deploy / shuffled / random
```

Interpretation:

```text
delta_CE > 0:
  disabling hurt, component useful

delta_CE < 0:
  disabling helped, component suspicious
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
PrimitiveMatrix usage heatmap
PrimitiveMatrix skip heatmap
PrimitiveMatrix replace heatmap
active slots per layer
split count per layer
merge graph per layer
output writes per layer/slot
layer_listen_score
layer_action_similarity[t,t-1]
layer_specialization_credit
useful/suspicious cells
useful/suspicious branches
sim_pred_vs_real_corr
scanner proposal vs real credit
honesty_score
full/no_conv/no_hints/deploy/shuffled/random accuracies
conv_ablation_delta
hint_ablation_delta
deploy primitive diversity
deploy action-matrix readability
insertion_mode for plug-in runs
mechanism_delta vs identity/random/frozen/baseline
action_matrix_vs_attention_similarity
```

ASCII report example:

```text
Layer 02 ActionMatrix top actions

        target0   target1   target2   target3
src0    SKIP      DIFF      LOW_RANK  OUTPUT
src1    MERGE     KEEP      GATED_ADD MEMORY_W
src2    SPLIT     CHANNEL   DISABLE   PRODUCT
src3    MEMORY_R  RECALL    MERGE     KEEP
```

---

## Implementation plan

Do not implement all at once.

```text
v4.6.4-a:
  documentation + stable naming

v4.6.4-b:
  PrimitiveMatrix5x5 only
  topology coordinates
  neighbor window extraction
  usage/topology/diversity metrics

v4.6.4-c:
  WindowScanner proposals
  proposal_scores[group/primitive]
  replace/skip/disable scores

v4.6.4-d:
  layer listening and specialization metrics
  prev_layer_output/action_embed
  layer_listen_score
  layer_action_similarity
  weak layer_listen_loss
  layer ablation credit

v4.6.4-e:
  Top-K low-rank simulation
  rank-16/32 simulator
  controller sees simulated outcomes

v4.6.4-f:
  ActionMatrix executor
  explicit cell_action table
  group/primitive/rank/compose/sign/edge_op/skip/replace/disable/output

v4.6.4-g:
  branch/output count
  slot_alive
  split_count 0/1/2
  child_gate
  merge_gate
  final collector

v4.6.4-h:
  simulation quality learning
  predicted_gain
  real_gain from delayed credit
  sim_quality_loss
  sim_pred_vs_real_corr

v4.6.4-i:
  honest scaffold curriculum and audit
  Teacher/Audit/Deploy scheduler
  no_conv/no_hints/deploy/shuffle/random audits
  honesty_score report
  deploy checkpoint guard

v4.6.4-j:
  universal plug-in wrappers
  after-attention wrapper
  before-attention wrapper
  attention replacement wrapper
  identity/random/frozen/baseline tests

v4.6.4-k:
  structured matrix frontend
  audio frame/window/DCT/log-energy/delta/onset
  compare Conv scaffold vs matrix frontend vs deploy
```

---

## What not to do

Avoid initially:

```text
10x10 PrimitiveMatrix
hard top-k during training
same-batch credit penalty
full edge MLP per connection
strong topology loss
strong skip penalty
hard manual layer roles
hard expand/merge alternation constraints
answer-like hints
direct operation-choice hints
reading next layer weights as shortcut
calling Teacher scaffold performance Deploy success
claiming raw-input success without evidence
```

---

## Evaluation plan

Compare:

```text
v4.6.3 edge + Conv
v4.6.3 no-conv
v4.6.4 PrimitiveMatrix only
v4.6.4 with scanner
v4.6.4 with simulation
v4.6.4 full
v4.6.4 without scanner
v4.6.4 without simulation
v4.6.4 without layer listening
v4.6.4 without skip/replace
v4.6.4 without memory
v4.6.4 structured matrix frontend
v4.6.4 Teacher vs Audit vs Deploy
v4.6.4 shuffled/random hints
v4.6.4 Transformer after-attention plug-in
v4.6.4 Transformer before-attention plug-in
v4.6.4 Transformer attention replacement
```

Metrics:

```text
val_acc
frontend dependence
collapse flags
edge uniformity
active cells
active slots
layer_listen_score
layer_action_similarity
layer_specialization_credit
memory ablation delta
output ablation delta
sim_pred_vs_real_corr
scanner-vs-real correlation
honesty_score
conv_ablation_delta
hint_ablation_delta
deploy primitive diversity
deploy ActionMatrix readability
mechanism_delta vs identity/random/frozen
action_matrix_vs_attention_similarity
speed / params / memory
```

---

## Acceptance rules

Do not call the architecture successful unless these checks are passed.

Standalone mode:

```text
full accuracy improves or matches v4.6.3
ActionMatrix readable
no primitive collapse
non-trivial layer specialization
useful cell/branch/output credit
```

Honest deploy mode:

```text
honesty_score >= 0.90 preferred
honesty_score >= 0.80 acceptable only with clear remaining dependency
Deploy keeps primitive diversity
Deploy keeps readable ActionMatrix
Deploy does not trigger boundary dead / output dead / primitive collapse
```

Plug-in mode:

```text
after-attention: attention + mechanism > attention only
before-attention: input preparation improves downstream attention
replacement: mechanism >= attention baseline or close with better cost/interpretability
with_mechanism > identity/random/frozen mechanism
```

---

## Current conclusion

Locked direction:

```text
Controller stops choosing from a flat primitive list.
Controller scans a topological PrimitiveMatrix.
Controller sees local primitive neighborhoods.
Controller simulates Top-K candidates in low rank.
Controller writes explicit ActionMatrix layers.
```

Layer rule:

```text
A layer must listen to previous layer action trace and learn a different useful role.
```

Honesty rule:

```text
Conv and hints are scaffolding, not the building.
Deploy must survive without scaffold/hints.
```

Universal rule:

```text
The core must work both as a standalone model and as an insertable neural-network layer.
Only wrappers change.
```
