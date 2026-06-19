# v4.6.4 Primitive Matrix Scanner / Action-Matrix Program Plan

<!--
Agent navigation.
Use:
  grep -n '^## ' V4_6_4_PRIMITIVE_MATRIX_SCANNER_PLAN.md
  sed -n 'START,ENDp' V4_6_4_PRIMITIVE_MATRIX_SCANNER_PLAN.md

Navigation is regenerated only after all content edits.
-->

## Agent quick navigation

| Lines | Section | Purpose |
|---:|---|---|
| 51-87 | Purpose / short lock | what this plan targets |
| 89-138 | Terminology lock | layer/slot/action naming |
| 140-185 | Coverage checklist | all transferred ideas |
| 187-493 | Critical failure modes and required fixes | mandatory engineering fixes |
| 495-554 | Minimal vertical slice first | first empirical proof target |
| 556-616 | Full architecture diagram | forward data path |
| 618-658 | Closed-loop learning diagram | credit loop |
| 660-698 | PrimitiveMatrix | 5x5/10x10 action topology |
| 700-756 | HybridScanner | grid+semantic+usage+random scanner |
| 758-782 | ProjectionScanner | separate readable projections |
| 784-832 | Top-K low-rank simulation and mandatory influence | cheap thinking before acting |
| 834-873 | Projection gradient paths | how scanner/simulator learn |
| 875-925 | ActionMatrix execution | cell instruction execution |
| 927-951 | Gate separation | separate gates |
| 953-993 | Branching and variable output count | soft branch/merge slots |
| 995-1054 | Sequential layer specialization | layers listen and specialize |
| 1056-1107 | Input encoder honesty | Conv/no-conv and matrix frontend |
| 1109-1190 | Honest input curriculum | Teacher/Audit/Deploy |
| 1192-1234 | Mode-specific credit | teacher/audit/deploy credit buffers |
| 1236-1309 | Honesty audits | honesty tests |
| 1311-1393 | Universal insertion into neural network layers | standalone and plug-in modes |
| 1395-1438 | TokenSlotAdapter for attention replacement | token-slot mapping and incremental decode |
| 1440-1487 | Transformer wrapper modes | after/before/replace attention |
| 1489-1514 | Loss staging | start with few losses |
| 1516-1546 | Losses and protections | loss list |
| 1548-1597 | Credit and ablation | credit levels |
| 1599-1666 | Reports required | artifacts and report fields |
| 1668-1731 | Implementation plan | staged implementation |
| 1733-1756 | What not to do | avoid shortcuts |
| 1758-1815 | Evaluation plan | comparisons and metrics |
| 1817-1862 | Acceptance rules | success criteria |
| 1864-1897 | Current conclusion | locked direction |

---

## Purpose / short lock

This file is the corrected plan for the next architecture after v4.6.3.

It covers two targets:

```text
1. standalone experiments now:
   audio / SpeechCommands / synthetic program tests / other real datasets

2. future plug-in layer mode:
   insert the same mechanism into existing networks:
   after Attention, before Attention, or as Attention/block replacement
```

Core invariant:

```text
PrimitiveMatrix -> HybridScanner -> Top-K Simulator -> ActionMatrix Controller -> Executor -> Credit
```

The core should stay reusable. Wrappers/adapters may be non-trivial and are allowed to differ by domain:

```text
audio wrapper
structured matrix frontend wrapper
Transformer token-slot adapter
CNN / vision adapter
MLP / block adapter
```

Important correction:

```text
"only wrappers change" is true only for simple add-on modes.
Attention replacement requires a real TokenSlotAdapter and incremental slot update.
```

## Terminology lock

Use these names in code, reports, and commits.

```text
old step                  -> layer
old lane                  -> slot
old edge/route             -> source->target cell connection / edge_gate
old primitive selector     -> layer operation selector
old trace                  -> program trace
old bypass                 -> skip
```

Definitions:

```text
Layer:
  one sequential program stage. Layer[t] receives state[t] and produces state[t+1].

Slot:
  state/output position inside a layer.

PrimitiveMatrix:
  global topological library of possible operations/actions.

ActionMatrix:
  matrix of operations inside one layer:
  ActionMatrix[layer, source_slot, target_slot]

HybridScanner:
  scanner that combines local grid window, semantic top-k, usage/credit top-k, and random exploration.

Proposal:
  candidate action suggested by scanner.

Simulation:
  low-rank preview of what a proposal would do.

Executor:
  full operation applied after controller selection.

Skip:
  pass signal through instead of transforming.

Disable:
  no write from this cell.

Replace:
  candidate/topology replacement mechanism. It is not a second runtime primitive selector.
```

## Coverage checklist

Everything below must be preserved in implementation.

```text
[x] PrimitiveMatrix 5x5 first, later 10x10
[x] primitive topology instead of flat primitive list
[x] functional initialization for primitive embeddings
[x] HybridScanner, not pure fixed 3x3 WindowScanner
[x] local grid candidates + semantic top-k + usage top-k + random exploration
[x] semantic health metrics: semantic collapse, semantic_grid_mismatch, global_rescue_rate
[x] separate projections: context / previous action / candidate / memory / head-input
[x] Top-K low-rank simulation before full execution
[x] simulator must affect choice_logits, not sit as decorative auxiliary head
[x] logit component normalization and sim/context influence health metrics
[x] budgeted hierarchical credit, not combinatorial full ablation
[x] random cell/primitive credit budget independent of layer-level result
[x] credit EMA, credit age, credit staleness metrics
[x] ActionMatrix per layer, each cell is an operation
[x] source->target context before primitive choice
[x] separate edge_gate / write_gate / phase_gate / output_gate
[x] cell_mode = softmax([transform, skip, disable])
[x] replace is candidate/topology selection, not another no-op gate
[x] edge_op scalar/sign per connection first
[x] delayed credit penalty from validation ablation
[x] mode-specific credit buffers: teacher / audit / deploy
[x] deploy credit starts only after honesty floor is reached
[x] layers listen to previous layer explicitly
[x] hard_delete_layer_eval uses fixed identity/simple adapter, not learned ablation adapter
[x] layer diversity and anti-copy specialization credit
[x] weak role priors only, no hard expand/merge masks
[x] variable output count via fixed max slots and soft gates
[x] final output always one output_state
[x] dynamic output tape across layers/slots
[x] Conv allowed only as temporary scaffold, not deploy proof
[x] no-conv result recorded: raw pooling failed
[x] Teacher -> Audit -> Deploy honesty curriculum
[x] allowed hints are meta-process only and decay to zero
[x] answer/leakage hints are banned
[x] honesty audit: no_conv / no_hints / deploy / shuffled / random / transfer
[x] universal plug-in mode exists, but attention replacement requires TokenSlotAdapter
[x] incremental slot update is required for autoregressive replacement speed
[x] first code target is a minimal vertical slice, not the whole huge plan
[x] start with only few losses, add more one by one
[x] staged implementation, not all at once
```

## Critical failure modes and required fixes

These fixes are mandatory before coding scanner/simulator.

### 1. Pure grid WindowScanner is not enough

Problem:

```text
center_embed = E[r,c]
neighbors = E[r-1:r+2, c-1:c+2]
```

If scanner only sees physical grid neighbors, embedding semantics can learn one topology while the fixed grid still scans another. The grid becomes a permanent wrong neighborhood.

Fix: use HybridScanner.

```text
candidate_set(cell) =
  local_grid_window_3x3
  ∪ semantic_topk_by_embedding
  ∪ usage_topk_by_credit/history
  ∪ random_explore_small
```

Start:

```text
local_grid_k = 9
semantic_k = 4
usage_k = 2
random_k = 1
```

The grid is a weak inductive prior, not a prison.

### 2. Semantic top-k can collapse

Risk: if primitive embeddings collapse or drift randomly, semantic_topk becomes random.

Protections:

```text
functional embedding initialization
primitive_usage_balance
near/far topology metrics
semantic_neighbor_entropy
embedding_covariance_rank
semantic_collapse_flag
```

Define:

```text
semantic_grid_mismatch =
  fraction of semantic_topk neighbors that are not in local_grid_window

global_rescue_rate =
  among selected semantic/usage/random candidates not present in the grid window,
  fraction with positive real_gain or positive task contribution

semantic_candidate_quality =
  mean real_gain for semantic_topk candidates
```

### 3. Simulator can become decorative

Problem: `sim_quality_loss` can train a prediction head that the controller ignores.

Fix: predicted gain must enter the actual choice path.

```text
choice_logits =
  LN(context_logits)
  + alpha * LN(predicted_gain_logits)
  + beta  * LN(sim_result_logits)
  + gamma * LN(proposal_logits)
```

Use normalization so context logits cannot simply dominate by scale.

Health metrics:

```text
predicted_gain_choice_corr
choice_without_sim_delta
sim_disabled_delta
sim_vs_context_logit_norm_ratio
sim_vs_context_grad_norm_ratio
sim_pred_vs_real_corr
```

Optional weak alignment loss:

```text
choice_gain_alignment_loss =
  KL(choice_weights || softmax(stopgrad(predicted_gain) / tau_gain))
```

Do not make it too strong. Controller may need to disagree with simulator.

### 4. Full ablation is too expensive

Do not ablate all layers/cells/primitives/branches/outputs every epoch.

Use budgeted hierarchical credit:

```text
credit_budget_per_epoch = 32 or 64 components
credit_batch = 16 or 32 samples
credit_every = 1..3 epochs
```

Candidate selection:

```text
top active_mass
top uncertainty / entropy
top negative predicted_gain
top previous suspicious credit
random exploration
```

Avoid Simpson's paradox: even if a layer looks neutral, still reserve cell/primitive random budget.

```text
credit_budget:
  40% suspicious from hierarchy
  30% high activity / high uncertainty
  20% random cells/primitives independent of layer result
  10% outputs/memory/scanner/simulator spot checks
```

Store credit as EMA:

```text
credit_ema = decay * old + (1 - decay) * new
credit_age tracked
old credit decays in penalty weight
```

Report:

```text
credit_budget_used
credit_components_tested
credit_staleness_mean
credit_age_max
credit_mode
credit_random_fraction
```

### 5. skip / disable / replace must not overlap

Do not use independent skip_gate, disable_gate, replace_gate as three no-op mechanisms.

Use mutually exclusive cell mode:

```text
cell_mode = softmax([transform, skip, disable])
```

Replace is separate and means candidate/topology selection:

```text
replace_distribution = softmax(candidate_replace_logits)
candidate = replace_mixture(candidates)
```

Execution:

```text
transformed = executor(candidate)
cell_output =
  mode_transform * transformed
  + mode_skip * passthrough
  + mode_disable * 0
```

Report:

```text
transform_mass
skip_mass
disable_mass
replace_entropy
replace_usage
skip_all_flag
disable_all_flag
```

### 6. Attention replacement needs TokenSlotAdapter

Attention operates over variable token positions. ActionMatrix operates over fixed slots. Replacement is not just a wrapper.

Required modules:

```text
TokenToSlotAdapter:
  tokens -> fixed slots

PrimitiveMatrixScannerCore:
  slots -> slots

SlotToTokenAdapter:
  slots -> token updates
```

For causal language modeling:

```text
no future leakage
causal token-to-slot routing
prefix/chunk causal masks
incremental slot update
```

Incremental update is required:

```text
slots_t = update(slots_{t-1}, new_token_t)
```

Do not recompute `slots = f(all_tokens)` every generated token, or inference becomes O(n²) without attention kernels.

### 7. Layer specialization ablation must be physical

Do not test specialization only by forcing internal skip gates.

Use hard delete evaluation:

```text
hard_delete_layer_eval:
  remove layer[t] from computation
  state[t] -> fixed external adapter -> layer[t+1]
```

External adapter must be simple and fixed:

```text
identity if shape matches
fixed linear projection if dimensions differ
not trained separately for each ablation
```

Report separately:

```text
external_delete_delta
internal_skip_mass
layer_transform_mass
layer_output_credit
layer_action_similarity
```

### 8. Credit must match curriculum mode

Teacher credit with hints does not necessarily transfer to Deploy without hints.

Maintain:

```text
credit_teacher
credit_audit
credit_deploy
```

Rules:

```text
Deploy simulator target uses credit_deploy.
Teacher credit is scaffold-only and decays or is flushed at phase transitions.
Audit credit can bridge teacher to deploy.
```

Deploy credit write gate:

```text
if honesty_score < honesty_floor:
  run deploy audit for monitoring only
  do not write into credit_deploy training buffer
else:
  write deploy credit into credit_deploy
```

Default:

```text
honesty_floor = 0.30 or task-specific minimum above random
```

### 9. Do not build ten stages before testing

First code target is a minimal vertical slice:

```text
1 layer
4 slots
PrimitiveMatrix 5x5
HybridScanner
Top-K simulator
mandatory sim influence
budgeted credit
small synthetic task with known program
```

If simulator is decorative there, fix it there. Do not wait until transformer plug-in.

## Minimal vertical slice first

Before full v4.6.4, implement a small proof slice.

Goal:

```text
prove scanner/simulator/controller can discover a known program,
not merely solve task through hidden dense shortcuts
```

Scale:

```text
layers = 1 or 2
slots = 4
PrimitiveMatrix = 5x5
top_k = 4
sim_rank = 8 or 16
credit_budget_per_epoch = 32
batch small enough for frequent diagnostics
```

Synthetic tasks with known answer:

```text
Task A: source slot 0 -> diff -> target slot 1 -> output
Task B: source slot 0 -> split into two branches -> merge -> output
Task C: memory_read/write required
Task D: skip is useful for one cell but harmful if global
Task E: semantic rescue needed: correct primitive is outside grid window but semantic_topk can find it
```

Metrics required in the first run:

```text
program_recovery_rate
top_action_matches_known_program
choice_without_sim_delta
sim_disabled_delta
predicted_gain_choice_corr
sim_pred_vs_real_corr
semantic_grid_mismatch
global_rescue_rate
credit_staleness_mean
transform/skip/disable mass
```

Acceptance for vertical slice:

```text
known program recovered or close
sim_disabled_delta > small positive threshold
choice_without_sim_delta > small positive threshold
global_rescue_rate > 0 on semantic rescue task
no skip-all collapse
credit budget stays bounded
```

Only after this slice passes should real audio or transformer plug-ins be expanded.

## Full architecture diagram

```text
REAL INPUT
  |
  v
Input mode
  - raw/basic structured input in Deploy
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
| HybridScanner                                                     |
|   local grid + semantic top-k + usage top-k + random explore      |
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
|   uses context + predicted_gain + sim_result + proposal logits    |
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

## Closed-loop learning diagram

```text
PrimitiveMatrix topology
  -> primitive/category embeddings
  -> HybridScanner
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
  -> Budgeted Credit Collector / ablation
  -> real_gain[layer, cell, primitive, branch, output, memory]
```

Credit returns through:

```text
sim_quality_loss:
  Simulator learns to predict consequences.

proposal_loss / task path:
  Scanner learns to propose useful candidates.

topology_loss:
  PrimitiveMatrix embeddings remain meaningful.

credit_penalty:
  Controller avoids suspicious action mass.

layer_credit:
  Layers specialize instead of copying each other.

honesty_credit:
  Deploy/no-hints behavior is protected from Teacher shortcut learning.
```

## PrimitiveMatrix

Start:

```text
PrimitiveMatrix 5x5 = 25 cells
top_k = 4
sim_rank = 8/16 for vertical slice, 16/32 for real task
```

Later:

```text
PrimitiveMatrix 10x10 = 100 cells
only after 5x5 scanner/simulator passes diagnostics
```

Seed 5x5 layout:

```text
identity      gated_keep   diff        contrast    smooth
low_rank      channel      ctx_matrix  product     gated_add
merge         split        route       edge_gate   write_gate
memory_read   memory_write forget      recall      memory_gate
output_write  output_mix   skip        replace     disable
```

Functional descriptor initialization:

```text
family: local / learned / routing / memory / output / control
arity: unary / binary / memory
effect: preserve / transform / merge / split / write / read / skip / disable
cost: cheap / medium / expensive
rank_capable: yes/no
sign_capable: yes/no
```

Initialize primitive embeddings from descriptor vectors plus small noise. Do not start from pure random embeddings.

## HybridScanner

HybridScanner replaces pure WindowScanner.

Candidate set:

```text
local_grid_candidates:
  fixed 3x3 grid neighborhood around primitive cell

semantic_candidates:
  top-k nearest primitives by embedding similarity

usage_candidates:
  top-k by useful credit / historical usage in similar context

random_candidates:
  small exploration budget
```

Default:

```text
local_grid_k = 9
semantic_k = 4
usage_k = 2
random_k = 1
```

Scanner inputs:

```text
source_slot_state
target_slot_state
source-target delta
source-target product
layer context
memory stats
previous action embed
optional allowed process hints
candidate functional descriptor
candidate embedding
candidate source type: grid/semantic/usage/random
```

Reports:

```text
grid_candidate_usage
semantic_candidate_usage
usage_candidate_usage
random_candidate_usage
semantic_grid_mismatch
global_rescue_rate
semantic_candidate_quality
semantic_collapse_flag
```

## ProjectionScanner

Keep projections separate.

```text
proj_context      = W_context(current state/context)
proj_before       = W_before(previous action choice)
proj_candidate[k] = W_candidate(candidate primitive/action)
proj_memory       = W_memory(memory stats)
proj_head_input   = W_head(allowed meta-hints only)
```

Candidate score:

```text
proposal_score[k] = f(
  proj_context,
  proj_before,
  proj_candidate[k],
  proj_memory,
  proj_head_input
)
```

Do not replace this with one black-box concatenation first. Separate projections make reports and credit interpretable.

## Top-K low-rank simulation and mandatory influence

Controller should think before acting.

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
    sim = low_rank_simulator(candidate, state_cell, memory)
    sim_results.append(sim)

predicted_gain = sim_quality_head(sim_results)

choice_logits = (
    LN(context_logits)
    + alpha * LN(predicted_gain_logits)
    + beta * LN(sim_result_logits)
    + gamma * LN(proposal_logits)
)

choice_weights = gumbel_softmax(choice_logits)
full_update = executor(choice_weights, full_primitives, state_cell, memory)
```

Simulator health checks are mandatory:

```text
choice_without_sim_delta
sim_disabled_delta
predicted_gain_choice_corr
sim_vs_context_logit_norm_ratio
sim_vs_context_grad_norm_ratio
sim_pred_vs_real_corr
```

If disabling simulator does not hurt, simulator is decorative and the run is not accepted.

## Projection gradient paths

Path A: simulation quality.

```text
sim_result[k] = low_rank_sim(candidate[k], state_cell, memory)
predicted_gain[k] = sim_quality_head(sim_result[k])
real_gain[k] = stopgrad(mode_specific_credit[cell, candidate[k]])
loss_A = MSE(predicted_gain[k], real_gain[k])
```

Path B: task loss through soft/gumbel choice.

```text
choice_weights = gumbel_softmax(choice_logits)
full_result = executor(choice_weights, full_primitives, state)
loss_B = task_ce(full_result)
```

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

Path E: choice-gain alignment, weak.

```text
choice_gain_alignment_loss =
  KL(choice_weights || softmax(stopgrad(predicted_gain) / tau_gain))
```

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
cell_mode = transform / skip / disable
replace_distribution over candidates
output_gate
```

Execution:

```text
candidate = replace_mixture(candidates)
transformed = PrimitiveExecutor(candidate, source, target, memory)
edge_update = edge_op * sign * transformed

cell_output =
  mode_transform * edge_update
  + mode_skip * passthrough
  + mode_disable * 0

active_mass = edge_gate * write_gate * phase_gate
state_next[target] += active_mass * cell_output
```

Reports:

```text
top action per cell
top primitive per cell
transform/skip/disable mass
replace entropy
active cells
output cells
```

## Gate separation

Keep gates separate.

```text
edge_gate[i,j]:
  is source i -> target j active at all?

write_gate[i,j]:
  how much should the active edge write?

phase_gate[i,j]:
  is this operation a structural transition / boundary?

output_gate[t,j]:
  should layer t slot j write to output tape?

cell_mode[i,j]:
  transform / skip / disable

replace_distribution[i,j,k]:
  which candidate action replaces the topological/default action?
```

Never collapse `edge_gate` into `write_gate`.

## Branching and variable output count

Use fixed max slots and soft gates.

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

Hard delete specialization evaluation:

```text
hard_delete_layer_eval:
  remove layer[t] from graph
  state[t] -> fixed adapter -> layer[t+1]
```

Fixed adapter:

```text
identity if shape matches
fixed non-trained linear projection if needed
not separately trained per ablation
```

Reports:

```text
external_delete_delta
internal_skip_mass
layer_transform_mass
layer_output_credit
layer_action_similarity
layer_listen_score
```

## Input encoder honesty

Current v4.6.3 fact:

```text
normal Conv frontend:
  useful edge/memory/output credit appeared

no-conv raw pooling + linear frontend:
  collapsed near random
  boundary dead / primitive collapse / output off
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

## Mode-specific credit

Credit must match the mode in which the program will run.

Buffers:

```text
credit_teacher
credit_audit
credit_deploy
```

Rules:

```text
Teacher credit:
  allowed for scaffold learning only
  decays or flushes at phase transition

Audit credit:
  bridge between teacher and deploy

Deploy credit:
  main target for sim_quality and final controller penalty
```

Deploy credit write gate:

```text
if honesty_score < honesty_floor:
  run deploy audit for monitoring only
  do not write into credit_deploy training buffer
else:
  write into credit_deploy
```

Default:

```text
honesty_floor = max(random_acc * 1.5, task_specific_minimum)
```

Never train final simulator only on Teacher/hinted credit.

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

## Universal insertion into neural network layers

v4.6.4 core must be usable as a plug-in layer, but plug-in work starts only after standalone vertical slice passes.

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
   real replacement test, requires TokenSlotAdapter
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

Forbidden:

```text
next layer weights as direct readable shortcut
target class information
exact desired activation
predicted class
softmax probability vector
which slot / primitive / action to choose
```

Rule:

```text
Head teaches through gradient.
Head must not tell the mechanism the answer or operation.
```

## TokenSlotAdapter for attention replacement

Attention replacement is not just a wrapper.

Required:

```text
TokenToSlotAdapter:
  token activations [B, T, D] -> slots [B, S, D]

PrimitiveMatrixScannerCore:
  slots [B, S, D] -> updated slots [B, S, D]

SlotToTokenAdapter:
  updated slots -> token updates [B, T, D]
```

For causal/autoregressive mode:

```text
no future leakage
causal token-to-slot routing
prefix/chunk causal masks
incremental slot update
```

Incremental slot update:

```text
slots_t = update(slots_{t-1}, new_token_t)
token_update_t = slot_to_token(slots_t, token_t)
```

Do not recompute slots from all tokens at every generated token.

Reports:

```text
token_slot_reconstruction_error
slot_usage_by_position
causal_leakage_check
incremental_decode_cost
action_matrix_vs_attention_similarity
```

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

## Loss staging

Do not start with all losses.

Vertical slice losses:

```text
task_ce
sim_quality_loss
one collapse protection loss:
  min_transform_mass or skip_all_penalty
```

Then add one by one:

```text
choice_gain_alignment_loss
credit_penalty
primitive_usage_balance
topology/diversity losses
layer_listen_loss
layer_diversity_loss
honesty_penalty
```

Every added loss must have before/after metrics showing it helped or at least did not break core behavior.

## Losses and protections

Base/later loss list:

```text
task_ce
edge_gate_cost
write_cost
output_gate_cost
skip_budget_loss
disable_budget_loss
min_active_cells_loss
min_transform_mass
primitive_usage_balance
primitive_topology_loss
primitive_diversity_loss
semantic_collapse_loss
edge_sign_balance_loss
sim_quality_loss
choice_gain_alignment_loss
credit_bad_cell_loss
credit_bad_primitive_loss
credit_bad_edge_loss
layer_listen_loss
diversity_between_layers_loss
specialization_credit_loss
honesty_penalty
plug_in_layer_honesty_loss
```

Weights must start small. Reports must show raw loss values and weighted contributions.

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

mode:
  force transform / force skip / force disable

replace:
  disable replacement candidate

output:
  disable output write layer/slot

memory:
  no memory / no write / no read

scanner:
  disable scanner proposal source: grid/semantic/usage/random

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

Credit is budgeted, hierarchical, EMA-smoothed, and mode-specific.

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
predicted_gain_choice_corr
choice_without_sim_delta
sim_disabled_delta
sim_vs_context_logit_norm_ratio
sim_vs_context_grad_norm_ratio
scanner proposal vs real credit
semantic_grid_mismatch
global_rescue_rate
semantic_candidate_quality
semantic_collapse_flag
credit_budget_used
credit_components_tested
credit_staleness_mean
honesty_score
full/no_conv/no_hints/deploy/shuffled/random accuracies
conv_ablation_delta
hint_ablation_delta
deploy primitive diversity
deploy ActionMatrix readability
insertion_mode for plug-in runs
mechanism_delta vs identity/random/frozen/baseline
action_matrix_vs_attention_similarity
token_slot_reconstruction_error
causal_leakage_check
incremental_decode_cost
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

## Implementation plan

Do not implement all at once. Use empirical gates.

```text
v4.6.4-a:
  corrected document + stable naming + agent navigation

v4.6.4-b:
  minimal vertical slice on synthetic known-program tasks
  1-2 layers, 4 slots, PrimitiveMatrix5x5
  HybridScanner + mandatory sim influence + budgeted credit

v4.6.4-c:
  PrimitiveMatrix5x5 infrastructure
  functional embedding initialization
  topology/semantic health metrics

v4.6.4-d:
  HybridScanner production version
  grid + semantic + usage + random candidates
  global_rescue_rate report

v4.6.4-e:
  budgeted hierarchical credit infrastructure
  credit EMA / age / staleness
  random cell budget independent of layer-level result

v4.6.4-f:
  ActionMatrix executor
  cell_mode=[transform, skip, disable]
  replace as candidate/topology selection

v4.6.4-g:
  layer listening and hard-delete specialization evaluation
  fixed delete adapter
  layer_listen_score / layer_action_similarity

v4.6.4-h:
  branching/output count
  slot_alive, split_count, child_gate, merge_gate, final collector

v4.6.4-i:
  honest scaffold curriculum and mode-specific credit
  Teacher/Audit/Deploy scheduler
  no_conv/no_hints/deploy/shuffle/random audits
  deploy credit write gate

v4.6.4-j:
  structured matrix frontend
  audio frame/window/DCT/log-energy/delta/onset
  compare Conv scaffold vs matrix frontend vs deploy

v4.6.5-a:
  universal plug-in wrappers only after standalone acceptance
  after-attention wrapper first
  before-attention second
  attention replacement last

v4.6.5-b:
  TokenSlotAdapter + incremental slot update for attention replacement
```

Transformer replacement is intentionally pushed after standalone proof. Do not start there.

## What not to do

Avoid initially:

```text
10x10 PrimitiveMatrix
pure fixed WindowScanner as only candidate source
hard top-k during training
same-batch credit penalty
full combinatorial ablation
full edge MLP per connection
strong topology loss
strong skip penalty
hard manual layer roles
hard expand/merge alternation constraints
answer-like hints
direct operation-choice hints
reading next layer weights as shortcut
learned external adapter in hard-delete layer ablation
using Teacher credit as Deploy sim-quality target
calling Teacher scaffold performance Deploy success
claiming raw-input success without evidence
starting with transformer replacement before standalone proof
```

## Evaluation plan

Compare:

```text
v4.6.3 edge + Conv
v4.6.3 no-conv
v4.6.4 minimal vertical slice synthetic
v4.6.4 PrimitiveMatrix only
v4.6.4 with HybridScanner
v4.6.4 with simulation
v4.6.4 full standalone
v4.6.4 without scanner
v4.6.4 without simulation
v4.6.4 without layer listening
v4.6.4 without skip/replace
v4.6.4 without memory
v4.6.4 structured matrix frontend
v4.6.4 Teacher vs Audit vs Deploy
v4.6.4 shuffled/random hints
v4.6.5 Transformer after-attention plug-in
v4.6.5 Transformer before-attention plug-in
v4.6.5 Transformer attention replacement
```

Metrics:

```text
val_acc
program_recovery_rate on synthetic
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
predicted_gain_choice_corr
choice_without_sim_delta
sim_disabled_delta
scanner-vs-real correlation
semantic_grid_mismatch
global_rescue_rate
credit_staleness_mean
honesty_score
conv_ablation_delta
hint_ablation_delta
deploy primitive diversity
deploy ActionMatrix readability
mechanism_delta vs identity/random/frozen
action_matrix_vs_attention_similarity
speed / params / memory
incremental_decode_cost
```

## Acceptance rules

Do not call the architecture successful unless these checks pass.

Minimal vertical slice:

```text
known program recovery is positive
sim_disabled_delta > threshold
choice_without_sim_delta > threshold
global_rescue_rate > 0 on semantic rescue task
no skip-all collapse
credit budget bounded
```

Standalone mode:

```text
full accuracy improves or matches v4.6.3
ActionMatrix readable
no primitive collapse
non-trivial layer specialization
useful cell/branch/output credit
simulator is not decorative
HybridScanner uses non-grid candidates meaningfully
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
causal replacement passes no-future-leakage and incremental update checks
```

## Current conclusion

Locked direction:

```text
Controller stops choosing from a flat primitive list.
Controller scans a topological PrimitiveMatrix.
Controller uses HybridScanner so topology is not trapped by fixed grid.
Controller simulates Top-K candidates in low rank.
Simulator must affect the actual choice.
Controller writes explicit ActionMatrix layers.
Credit is budgeted, mode-specific, and not combinatorial.
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
The core should become insertable into neural networks,
but attention replacement requires TokenSlotAdapter and incremental slot update.
Do standalone proof first.
```
