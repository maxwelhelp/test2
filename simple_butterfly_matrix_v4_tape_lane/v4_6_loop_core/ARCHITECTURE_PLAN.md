# v4.6 Loop Core Architecture Plan

## 0. Why this exists

Previous versions showed useful signals but not a clean program assembly loop:

- Council produced offline JSON biases, but did not learn from whether they helped.
- Scout/controller decisions were weak or compensated by the model.
- Boundary repeatedly collapsed into ON-everywhere.
- Primitive choice was not specialized enough.
- Memory behaved like a normal lane, not as causal read/write memory.
- Text-patching made the system fragile.

v4.6 fixes the loop by making the decision process differentiable and inside the forward pass.

## 1. Core principle

If decisions are differentiable, credit is backprop.

Offline Council JSON is only needed when the loop is broken. In v4.6, the scorer/controller lives inside the model, so loss gradients train the decision maker directly.

```text
decision[t] -> execution -> task loss -> gradient -> better decision[t]
```

Council can remain as an offline analyzer after a run, but it must not drive training via JSON biases in v4.6.0.

## 2. Target dataflow

```text
Input / backbone features
        |
        v
context[t,L,d]
        |
        v
JointController(context[t]) -> shared z[t]
        |---------------------------------------------|
        |                                             |
        v                                             v
coupled heads                                  ParallelPrimitiveSelector
boundary / transition / fanout / write         signed low-rank experts
memory read/write                              gumbel-softmax selection
        |                                             |
        |------------------- signed update -----------|
                              |
                              v
Route R[t,L,L] + MatrixMemory M[t]
                              |
                              v
next state / task head
                              |
                              v
task loss + regularizers
                              |
                              v
backprop through all decisions
```

## 3. JointController

### Problem it solves

Old design had independent controllers:

```text
boundary_logit
transition_net
fanout_net
primitive_unit
```

This allows contradictory decisions, for example:

```text
boundary=1, transition=identity, fanout=all_soft, primitive=random
```

### v4.6 design

One shared latent per step:

```text
z[t] = JointController(context[t,L,d])
```

Then multiple heads from the same z:

```text
boundary_head(z)
transition_head(z)
fanout_head(z)
primitive_score_head(z)
primitive_sign_head(z)
memory_read_write_head(z)
```

The shared latent makes boundary, route, fanout, primitive and memory a coupled decision.

## 4. ParallelPrimitiveSelector

### Purpose

At each step/lane, compute many candidate primitive projections in parallel, then select a sparse signed mixture.

### Primitive experts

Start with K experts:

```text
identity
channel_mlp
low_rank
shift_or_diff
smooth_or_local
contrast
ctx_matrix
product_gate
memory_read_or_keep
```

Exact list can be reduced for MVP.

### Low-rank expert rule

Use low-rank matrices, not full d x d matrices.

For d=256:

```text
rank = 32..48
```

Do not use full rank in MVP. Full matrices let all experts collapse into similar dense transforms.

### Signed output

Each primitive output has a sign/coefficient:

```text
sign_k = tanh(sign_logit_k) or tanh(sign_head(z)_k)
out = sum_k w_k * sign_k * Y_k
```

This is critical. Without sign, the model can only add/strengthen. With sign, it can subtract, compensate and build correction/difference behavior.

### Gumbel softmax

Use differentiable sparse selection:

```text
w = gumbel_softmax(scores, tau, hard=False)
```

Do not use actor-critic. Gumbel-softmax keeps gradients through the choice.

### Temperature annealing

Mandatory:

```text
tau = max(tau_min, tau_start * tau_decay ** epoch)
```

Recommended for first smoke:

```text
tau_start = 1.0
tau_min = 0.2
tau_decay = 0.92 or 0.95
```

Without annealing, weights stay too soft and primitives do not specialize.

## 5. MatrixMemory

### Problem it solves

Old memory was just another lane in route matrix. That is not real memory.

### MVP memory

```text
M[t] = lambda * M[t-1] + (1 - lambda) * W_write(z[t])
read[t] = W_read(M[t])
lambda = sigmoid(forget_logit)
```

Minimum module:

```python
class MatrixMemory(nn.Module):
    def __init__(self, d):
        self.W_write = nn.Linear(d, d, bias=False)
        self.W_read = nn.Linear(d, d, bias=False)
        self.forget_logit = nn.Parameter(torch.tensor(0.0))
```

### Causal requirement

Memory is useful only if:

```text
write[t] -> read[t+k] -> output/loss
```

Reports must include memory influence, not just memory mass.

## 6. Boundary

Boundary must not be free sigmoid ON everywhere.

v4.6.0 MVP:

```text
boundary_score[t] = boundary_head(z[t])
boundary_gate = sigmoid(boundary_score[t])
boundary_budget_loss keeps 2-4 peaks
flat/all-on/all-off flags
```

v4.6.1 upgrade:

```text
soft-topk boundary K=2..4
```

Boundary success requires:

```text
not all-on
not all-off
peaks align with route/read/primitive/update changes
```

## 7. Step semantics

Each step is a program action tuple:

```text
Action[t,lane] =
  read_source
  primitive mixture
  sign/coefficient
  compose_mode
  route_target/fanout
  write_target
  memory_action
  boundary
```

MVP compose modes:

```text
replace
add
subtract
gated_add
```

Do not implement deep projection trees in v4.6.0. Product composition can be represented by `product_gate` expert first.

## 8. Why not actor-critic now

Actor-critic is useful when choices are hard discrete and gradients do not pass, or when planning needs long-horizon policy gradients.

v4.6 uses differentiable Gumbel-softmax, so gradients already train selection.

Actor-critic would add:

- a second value network
- PPO/A3C instability
- more debug cost
- unclear benefit for MVP

Actor-critic is reserved for v4.7 if hard decisions or multi-step lookahead are required.

## 9. Why not projection tree now

A tree has K^depth paths. With K=8 and depth=3, that is 512 paths. Multiple sequential Gumbel choices can be unstable.

Current steps + memory already create hierarchy over time:

```text
step 0 writes memory
step 1 reads memory and transforms
step 2 refines
```

This is a causal unrolled tree. Keep it first.

## 10. Parallel scan note

If memory update is affine:

```text
M[t] = A[t] * M[t-1] + B[t] * x[t]
```

then in the future it can be optimized with parallel/prefix scan.

But v4.6.0 should use a simple Python/PyTorch step loop first for correctness and logging. Do not hide bugs behind scan kernels before proving semantics.

## 11. Required logs

Reports must include:

```text
val_acc / task loss
boundary_mean
boundary_peak_count
boundary_flatness
boundary_validity_flags
route_entropy
self_route_mass
useful_transition_mass
disallowed_route_mass
primitive_entropy
primitive_top1_share
primitive_sign_mean
primitive_sign_by_step
gumbel_tau
memory_forget
memory_write_norm
memory_read_norm
memory_read_influence
detail_topread_share
program_stage_summary
```

## 12. Success criteria for first 5-epoch smoke

Not necessarily high accuracy. Need structural sanity:

```text
boundary not all-on
boundary not all-off
route not uniform
route not pure identity
primitive top1 changes by step/lane
some signs are negative/correction-like
memory read influence nonzero
extra costs do not dominate CE
```

## 13. Implementation order

### v4.6.0

- Build modules as normal Python classes.
- Use one main training file.
- No text patching.
- No actor-critic.
- No tree.
- No JSON feedback.
- Add strong reports.

### v4.6.1

- Soft-topk boundary.
- Primitive diversity by lane/stage.
- Memory causal ablation.
- Detail topread loss.

### v4.7

- Harder decisions.
- Possible actor-critic if hard discrete planning is introduced.

### v4.8

- Two-level primitive tree or program compiler if v4.6 proves the loop.
