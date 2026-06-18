# Program Heart: Action Influence, Attention, Fast Projections, Structured I/O

This addendum explains the missing practical mechanics around `PROGRAM_HEART_ARCHITECTURE_PLAN.md`:

1. how an action changes priors / bases / logits;
2. how the system measures whether this change helped;
3. MVP rule-based heart vs attention planner;
4. how to speed analysis and compute;
5. how structured input and structured head feedback should connect to automatic priors.

---

## 1. What an action actually changes

An action is not a text recommendation. It must compile into a small numerical bias or gate update at a precise address.

Examples:

```text
increase route memory->state at T6
  -> feedback_route_bias[6, memory, state] += 0.03

revive diff at T1 detail lane
  -> feedback_primitive_bias[1, detail, diff] += 0.03

reduce late input shortcut at T9
  -> feedback_read_bias[9, *, input] -= 0.03

increase yes/no class contrast
  -> feedback_head_pair_bias[yes_no_pair] += 0.03

open dormant step T10
  -> feedback_step_alive_bias[10] += 0.02

prefer smaller low_rank rank at T5 abstract lane
  -> feedback_variant_bias[5, abstract, low_rank, rank16] += 0.03
     feedback_variant_bias[5, abstract, low_rank, rank64] -= 0.02
```

So actions can affect:

```text
read priors
primitive priors
operator variant priors: rank/depth/radius/basis
route matrix priors
boundary priors
write priors
memory protect/overwrite priors
head class/pair priors
step_alive / length priors
```

These are not hard edits. They are small soft biases added to logits.

---

## 2. How action influence is applied

Every choice should have the form:

```python
choice_logits = base_logits + learned_context_bias + feedback_bias + optional_actor_bias
choice = softmax(choice_logits)  # or sigmoid for gates
```

For gates:

```python
gate_logit = base_gate_logit + context_gate_bias + feedback_gate_bias
gate = sigmoid(gate_logit)
```

For route:

```python
route_logits_t = base_route_logits[t] + context_route_bias[t] + feedback_route_bias[t]
R_t = softmax(route_logits_t, dim=-1)
```

For primitive:

```python
primitive_logits = base + lane_bias + context_bias + feedback_primitive_bias[t,lane]
primitive_w = softmax(primitive_logits)
```

For variants:

```python
rank_w = softmax(base_rank_logits + context_rank_bias + feedback_rank_bias)
y = sum_r rank_w[r] * low_rank_r(x)
```

All influence remains differentiable.

---

## 3. How the system knows if the change helped

There are four levels of evidence.

### 3.1 Immediate gradient evidence

For any soft choice:

```text
importance ~= abs(choice_value * grad(choice_logit or choice_value))
```

This says whether a selected element mattered locally.

Weakness:

```text
suppressed components have weak gradients;
compound actions may not be visible;
train gradient can reward shortcuts.
```

### 3.2 Downstream consumer evidence

Ask:

```text
if this step/lane wrote something, did anyone later read it?
did the head use it?
did gradient flow through it?
```

Example:

```text
memory write high, future memory read low, head memory usage low
  -> memory write probably wasteful
```

### 3.3 Counterfactual evidence

Temporarily apply an action and measure heldout microbatch loss:

```text
baseline_loss
candidate_loss
gain = baseline_loss - candidate_loss
```

This detects useful revived components and avoids blind trust in gradient.

### 3.4 Long-run verified evidence

For selected candidates:

```text
save full state
apply action
run N small train steps or validation window
measure heldout + subproblem metrics
rollback or keep
```

This becomes the strongest memory record.

---

## 4. What learns from the action result

### 4.1 The main model

Still learns normally by gradient.

```text
CE/task loss -> weights/logits update
```

### 4.2 FeedbackBiasBank

Learns fast local behavior:

```text
accepted action -> positive local bias
rejected action -> negative local bias or cooldown
uncertain -> no deployment / more probes
```

### 4.3 Critic

Learns prediction:

```text
(context_embedding, action_embedding) -> expected_gain, uncertainty, risk
```

### 4.4 Actor / Planner

Learns to propose actions that critic and heldout tests support.

It should not output dense changes everywhere in MVP.

It should output sparse, typed candidate actions.

---

## 5. MVP vs attention planner

### 5.1 Rule-based MVP heart

The first MVP does not need attention.

It uses:

```text
trace metrics
rules
candidate generation
counterfactual screen
feedback bias
```

Example rule:

```text
if memory_write high and future_memory_read low:
  propose decrease memory_write

if memory_write high and future_memory_read high but memory->state route low:
  propose increase memory->state route

if late_input_read high and class/lane specialization weak:
  propose decrease late input read

if confusion yes/no high and compare primitive low:
  propose increase compare in likely useful steps
```

Benefits:

```text
simple
interpretable
cheap
debuggable
clear credit assignment
```

Weakness:

```text
rules miss subtle interactions
hard to combine many weak signs
manual thresholds
```

### 5.2 Attention planner added to MVP

A small attention planner can read all trace tokens and propose actions.

Tokens:

```text
step/lane trace tokens
route tokens
primitive tokens
memory tokens
head feedback tokens
input structure tokens
budget tokens
```

Output:

```text
candidate action logits or small biases
```

Benefits:

```text
sees global context
sees interactions between far steps
can coordinate multiple weak signals
can learn reusable patterns
```

Risks:

```text
can become dense second optimizer
can exploit critic mistakes
harder to debug
can ignore safety unless constrained
```

### 5.3 Attention instead of MVP

Do not start with attention replacing MVP.

If attention is first, failures are hard to debug:

```text
bad trace?
bad action schema?
bad attention?
bad critic?
bad feedback bias?
```

Correct order:

```text
rule MVP -> counterfactual memory -> critic -> attention planner
```

Attention should improve the candidate generator/coordinator, not bypass safety.

### 5.4 Attention plus MVP

Best final form:

```text
rule candidates + attention candidates + critic ranking + coordinator + counterfactual screen
```

The rule engine gives safety and obvious fixes.
The attention planner gives global pattern discovery.
The critic decides what is worth testing/deploying.

---

## 6. Multi-projection actors/critics

Instead of one giant actor/critic, use several small specialist actor/critic pairs.

```text
DataflowActor/Critic
PrimitiveActor/Critic
VariantActor/Critic
RouteBoundaryActor/Critic
MemoryActor/Critic
HeadActor/Critic
BudgetCritic
```

Each sees a compact projection:

```text
Dataflow: read/write/route/consumer
Primitive: primitive/variant/update/grad
Boundary: boundary/trace-change/segment-use
Memory: memory-write/read/protect/head-use
Head: confusion/margin/class-lane/pair
Budget: complexity/cost/stability
```

Each proposes diverse candidate actions.

Coordinator chooses:

```text
not conflicting
low risk
high LCB deploy score
within budget
not duplicate
```

Diversity requirement:

```text
actors must not all propose same action type;
track coverage by action type / step / lane / primitive / route edge;
add similarity penalty between proposals.
```

---

## 7. Structured input connection

Input structure answers:

```text
what exists in the data?
```

For audio:

```text
local detail / onset / diff
frequency band energy
global shape
noise/smoothness
compressibility / low-rank proxy
silence/background
```

Convert input structure to small tokens:

```text
input_tokens = [LOCAL, DIFF, SMOOTH, GLOBAL, LOW_RANK, NOISE, BOUNDARY, ENERGY]
```

Use input tokens in three ways:

### 7.1 Initialization

Weak lane initialization:

```text
detail lane gets local/diff view
state lane gets learned evidence read
abstract lane gets pooled/global view
memory lane gets learned seed + weak summary
```

### 7.2 Context priors

Input structure can add weak bias:

```text
LOCAL/DIFF high -> primitive diff/smooth/local gets +small bias early/detail lane
LOW_RANK high -> low_rank variants get +small bias
NOISE high -> gate/filter primitive gets +small bias
GLOBAL high -> state->abstract / abstract read gets +small bias
```

Bias is small and clamped.

### 7.3 Candidate actions

Input structure suggests candidates, not hard decisions:

```text
onset high but diff primitive unused -> propose revive diff
noise high but gate unused -> propose increase filter/gate
low_rank proxy high but low_rank unused -> propose increase low_rank small rank
```

---

## 8. Structured head connection

Head structure answers:

```text
what does the task need?
```

Head can provide:

```text
class confusion matrix
class margin
class read overlap
class lane mass
class top read slots
pair repair strength
per-class loss
confidence calibration
```

Convert to head tokens:

```text
HEAD_CONFUSION_PAIR(yes,no)
LOW_MARGIN_CLASS(stop)
SHARED_SLOT_COLLAPSE
CLASS_UNIQUE_NEED
PAIR_REPAIR_ACTIVE
OVERCONFIDENT_CLASS
```

Use head tokens in three ways:

### 8.1 Head-local actions

```text
increase class-pair contrast
increase class unique lane read
suppress shared lane read
increase pair_state usage
```

### 8.2 Core feedback actions

Head can request core changes:

```text
yes/no confused and compare low -> propose compare primitive in relevant state/abstract steps
class reads only abstract and misses detail -> propose detail->state route or detail read
memory-heavy class has no memory consumer -> propose memory->state route
```

### 8.3 Automatic task prior

Head structure becomes weak task prior:

```text
classification with many confusions -> compare/gated_contrast bias
low margins -> class unique slot/read diversity bias
shared read collapse -> slot/class read diversity pressure
```

Again: weak, clamped, not hardcoded.

---

## 9. Automatic priors from input/head/trace

There are several prior layers.

### 9.1 Start priors

Before training:

```text
structured input init
weak class lane prior
weak read/route prior
```

These are small and ablatable.

### 9.2 Context priors

During forward:

```text
input/head/trace context -> small dynamic bias
```

Example:

```text
current sample has high noise -> gate/filter bias
current head margin low -> compare/unique-read bias
```

### 9.3 Usage priors

From EMA of what worked in the same run:

```text
usage_ema -> weak self-organizing prior
```

Must be lagged/detached/clamped.

### 9.4 Tested feedback priors

From counterfactual accepted/rejected actions:

```text
accepted/rejected memory -> feedback_bias
```

### 9.5 Critic priors

From learned experience:

```text
critic predicts useful action -> candidate ranking / actor distillation
```

Only LCB-safe actions are deployed.

---

## 10. Fast computation and tree/projection optimization

The exact tape execution is sequential:

```text
X_{t+1} depends on X_t
```

But the heart analysis does not need full heavy exact state everywhere.

Use a two-plane system:

### 10.1 Exact execution plane

```text
D=96 or larger
full cells/lane
sequential tape
used for real output/loss
```

### 10.2 Projection/analysis plane

```text
d_small=16/32
pooled cells
compressed trace tokens
parallel across steps/lanes/tokens
used for planning/actions
```

This plane can be much cheaper and mostly parallel.

### 10.3 Tree analysis

Build a dataflow tree/graph from trace:

```text
nodes: step/lane/primitive/route/head
edges: read/write/route/consumer/head-read
features: mass, grad, update_norm, usage
```

Analyze with cheap message passing:

```text
future consumer score backward
importance score forward/backward
memory usefulness
segment boundary usefulness
```

This is not the main neural forward. It is a cheap graph over trace statistics.

### 10.4 Batched candidate screens

Instead of testing candidates one by one, batch them.

For K candidate actions:

```text
replicate val microbatch K times or vectorize bias dimension
apply K temporary biases
run forward in one/few batched calls
compare candidate losses
```

This can make counterfactual screening much faster.

### 10.5 Cache trace projections

Cache per epoch/window:

```text
input structure tokens
head structure tokens
trace tokens
consumer map
candidate embeddings
```

Do not recompute heavy things repeatedly.

### 10.6 Approximate planning is okay

Planner/critic does not need exact full forward.

It only needs to rank candidates.

Final acceptance still uses exact heldout screen.

---

## 11. Basis/rank/depth auto-choice

For each operator family, expose variants through soft gates.

### Low-rank

```text
rank choices: 8,16,32,64
```

```python
y = sum_r softmax(rank_logits)[r] * low_rank_r(x)
```

Cost:

```text
rank64 more expensive than rank8
```

Action examples:

```text
increase rank16 at T5 abstract
reduce rank64 if no gain
revive small-rank low_rank where low_rank proxy high
```

### Butterfly depth

```text
depth choices: 1,2,3,4
```

Action examples:

```text
increase depth in state lane if block mixing useful
reduce depth if update not consumed
```

### Local/wavelet/radius

```text
radius choices: 1,2,4,8
```

Action examples:

```text
increase local radius if head needs wider context
reduce radius if noise/overmixing
```

### Route basis

Route matrix can have bases:

```text
identity
upward/state->abstract
memory write
memory read
cross-lane exchange
```

Represent:

```text
R_t_logits = sum_b route_basis_weight[t,b] * route_basis[b] + residual_route_logits[t]
```

Actions can change route basis weights instead of raw entries.

This is more structured and faster to analyze.

---

## 12. MVP recommendations

### MVP A: rule heart without attention

Implement:

```text
TraceCollector
ProjectionBank basic metrics
CandidateActionGenerator rules
CounterfactualScreen no deployment
```

Start with route/boundary only.

### MVP B: feedback bias deployment

Deploy one accepted route/boundary edit.

### MVP C: primitive/variant heart

Add primitive and low_rank rank actions.

### MVP D: structured input/head action hints

Generate candidates from input/head tokens.

### MVP E: cold-start critic

Train ridge critic on tested actions.

### MVP F: attention planner

Add small attention planner after rule/counterfactual loop works.

Attention complements MVP by proposing better candidates and seeing global interactions. It should not replace heldout screen or safety.

---

## 13. Short answer

Actions influence architecture by changing small soft feedback biases at exact addresses.

The system learns from actions by:

```text
gradient
+ downstream consumer map
+ counterfactual heldout tests
+ feedback bias
+ critic/actor memory
```

MVP gives interpretable rule-based actions and safe testing.

Attention gives global coordination and pattern discovery, but should come after MVP so failures are debuggable.

Input structure tells the system what exists in the data.
Head structure tells the system what the task needs.
Trace tells what the program actually did.
Critic tells which changes really helped.

Together:

```text
input/head/trace/gradient -> projections -> candidates -> screen -> feedback -> better choices
```
