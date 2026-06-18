# Current Roadmap: v4.3 Canonical → Context Controllers → Learning Heart → Matrix Memory

This is the current source-of-truth roadmap for the `simple_butterfly_matrix_v4_tape_lane` project.

Repository: `maxwelhelp/test2`

Working folder:

```text
/home/maxwelhelp/test/sience/experiments/math_search/WORKING_BEST/test2
```

Project scope for the agent:

```text
simple_butterfly_matrix_v4_tape_lane/
simple_butterfly_matrix/
v4_3_min_heart/
v4_4_min_learning_heart/
```

Do **not** modify unrelated projects, datasets, or checkpoints.
Do **not** commit `.pt`, `.pth`, `.ckpt`, `.safetensors`.

---

## 0. Current position

We are currently at:

```text
Stage: v4.3 canonical main stabilization
Goal: make tape_lane_transport_v4_3_min_heart.py the only runtime source of truth
Status: not stable yet — standard smoke currently fails in candidate_suggestions
```

Latest observed failure:

```text
NameError: name '_candidate_key' is not defined
```

Failure location:

```text
simple_butterfly_matrix_v4_tape_lane/tape_lane_transport_v4_3_min_heart.py
generate_candidate_suggestions()
```

Root cause:

```text
generate_candidate_suggestions() calls _candidate_key(c), but _candidate_key is missing in main.
```

Important architectural correction:

```text
The runtime must not rely on canonicalize_v4_3_main.py.
The runtime must not rely on run_v4_3_min_heart_audit_fixed.sh.
The runtime must not rely on run_v4_3_context_controller.sh.
```

The canonical runtime entrypoint must be:

```text
simple_butterfly_matrix_v4_tape_lane/tape_lane_transport_v4_3_min_heart.py
```

---

## 1. Immediate P0 tasks before any new architecture

### P0.1 Fix `_candidate_key` in main

Add this helper before `generate_candidate_suggestions()`:

```python
def _candidate_key(c):
    """Stable duplicate key for candidate suggestions."""
    loc = c.get("location", {})
    if isinstance(loc, dict):
        loc_key = tuple(sorted((str(k), str(v)) for k, v in loc.items()))
    else:
        loc_key = str(loc)
    return (
        str(c.get("source", "")),
        str(c.get("target_type", "")),
        str(c.get("action", "")),
        str(c.get("target", "")),
        loc_key,
    )
```

Then verify:

```bash
python -m py_compile simple_butterfly_matrix_v4_tape_lane/tape_lane_transport_v4_3_min_heart.py
grep -n "def _candidate_key" simple_butterfly_matrix_v4_tape_lane/tape_lane_transport_v4_3_min_heart.py
```

### P0.2 Remove runtime canonicalizer dependency

`sync_run_5ep_v4_3_min_heart_push_logs.sh` should run main directly:

```bash
python simple_butterfly_matrix_v4_tape_lane/tape_lane_transport_v4_3_min_heart.py ...
```

It should not run these as runtime patchers:

```text
canonicalize_v4_3_main.py
run_v4_3_min_heart_audit_fixed.sh
run_v4_3_context_controller.sh
```

`canonicalize_v4_3_main.py` may remain in the repo only as deprecated migration/debug script, not as runtime dependency.

### P0.3 Validate canonical fixes exist directly in main

The main file must directly contain:

```text
sequence_nonflat_score = route_delta + primitive_delta + update/trace_delta + read_delta
read_delta_by_step
boundary_usefulness includes read_delta
self_route_mass
useful_transition_mass
ROUTE_IDENTITY_COLLAPSE
memory_consumer_proxy
update_collapse_proxy
collapse_flags
```

### P0.4 Validate and grad sanity must not depend on canonicalizer

`validate_v4_3_min_heart.sh` should check main directly.

`grad_sanity_v4_3_min_heart.sh` should import main directly and not patch it.

Required grad sanity checks:

```text
full        -> route/boundary/read/write/head
route_only  -> route + boundary
read_only   -> read
memory_only -> write
detail_only -> head
```

For `detail_only`, do not rely only on hinge `detail_head_shortcut_cost`, because it can be exactly zero on a tiny smoke batch. Use a direct head-attention proxy such as:

```python
(attn ** 2).mean()
```

### P0.5 Smoke must pass before any feature work

Smoke command:

```bash
DATA_ROOT=../architecture_builder/data/speechcommands \
EPOCHS=1 \
TRAIN_LIMIT=12000 \
VAL_LIMIT=2000 \
MAX_TRAIN_BATCHES=10 \
MAX_VAL_BATCHES=3 \
BATCH_SIZE=128 \
EVAL_BATCH_SIZE=256 \
WORKERS=4 \
AMP=fp16 \
bash simple_butterfly_matrix_v4_tape_lane/commands/sync_run_5ep_v4_3_min_heart_push_logs.sh
```

Success means:

```text
epoch 001/1 train=... val=...
run_status=0
```

If status is not 0, do not continue to 5ep.

---

## 2. P1 logical cleanup after smoke passes

### P1.1 Candidate suggestions

Candidate suggestions must remain observer-only:

```text
deploy=false
no auto-deploy
no feedback_bias write
```

Duplicate metrics must be real:

```text
attempted_candidates
skipped_duplicates
duplicate_rate = skipped_duplicates / max(1, attempted_candidates)
```

### P1.2 Memory terminology

Current memory consumer metric is a proxy, not causal credit.

Use:

```text
memory_consumer_proxy
```

Do not claim it is real `write -> future_read` causal credit until a real counterfactual or temporal attribution is added.

### P1.3 Update/residual metric

If current metric is really `1 / update_norm`, name it:

```text
update_collapse_proxy
```

Do not call it true residual dominance unless it compares:

```text
update_norm / state_norm
```

### P1.4 Not implemented costs

If skip and operator complexity are not active, report explicitly:

```text
skip_cost: not_implemented
operator_complexity_cost: not_implemented
```

Do not include them in success criteria until implemented.

---

## 3. Stage v4.3 canonical success criteria

After a real 5ep run, inspect:

```text
val_acc
sequence_nonflat_score
boundary_mean
boundary_flatness
boundary_peak_count
boundary_usefulness
route_entropy
route_offdiag_outside_boundary
self_route_mass
useful_transition_mass
memory_consumer_proxy
detail_attention_mass
collapse_flags
candidate duplicate_rate
```

Good signs:

```text
training reaches status=0
val_acc rises above random
sequence_nonflat_score is nonzero and preferably rising
route_entropy is below uniform or decreasing
memory_write has nonzero consumer proxy
candidate_suggestions are non-duplicate and deploy=false
```

Bad signs:

```text
BOUNDARY_DEAD for all epochs
ROUTE_UNIFORM does not improve
ROUTE_IDENTITY_COLLAPSE
MEMORY_JUNK
DETAIL_SHORTCUT dominates
candidate suggestions all identical
```

---

## 4. Context controllers — next, not now

Context controllers are useful, but they must be tested after stable v4.3 canonical main.

Purpose:

```text
old logits + small learned context delta
```

Not:

```text
replace old system with controller
```

Correct formula:

```python
final_logits = base_logits + alpha * controller_delta
```

Controllers needed later:

```text
primitive_context_controller
boundary_controller
route_controller
read_group_controller
write_gate_controller
halt/alive_controller
```

Best safe implementation:

```text
learnable alpha per controller
alpha small and bounded
controller final layer zero-init or near-zero-init
```

Recommended alpha policy:

```python
alpha = 0.01 + 0.09 * sigmoid(alpha_raw)
```

or fixed first experiment:

```text
CONTEXT_CONTROLLER_SCALE=0.05 / 0.15
```

The experimental wrapper exists:

```text
simple_butterfly_matrix_v4_tape_lane/commands/run_v4_3_context_controller.sh
```

But standard canonical v4.3 sync must not use it yet.

---

## 5. Matrix memory plan — v4.5 direction

Matrix memory is a good fit for this project, but do not implement it before stable v4.3 canonical and v4.4 learning heart.

Current memory is simple:

```text
memory lane
read_group can read memory lane
head can attend memory slots
```

This is not yet real matrix fast memory.

Do not replace the grammar with memory. Instead, add memory as a bus that gives context/credit to the existing grammar:

```text
read -> transform/primitive -> write -> route/boundary
```

### 5.1 Typed memory banks

Recommended memory banks:

```text
working_matrix_memory      data/state working memory
primitive_credit_memory    which primitives helped in this context
route_edge_memory          which lane-to-lane edges are useful
boundary_step_memory       where separators are useful
editor/action_memory       offline tested edits, later
```

### 5.2 Working matrix memory MVP

Start with:

```text
slot_memory: [B, M, D], M=8 or 16
fast_U:      [B, D, r], r=16
fast_V:      [B, r, D]
```

Read:

```text
slot_read = softmax(q @ slot_memory) @ slot_memory
fast_read = (q @ U) @ V
memory_read = slot_read + fast_read
```

Write:

```text
U = decay * U + gate * outer(k, a)
V = decay * V + gate * outer(a, v)
slots = decay * slots + gated_write
```

Initial values:

```text
M=8
rank=16
decay=0.97
```

### 5.3 Primitive credit memory

For primitives:

```text
channel
block
low_rank
ctx_matrix
product_gate
diff
gated_contrast
memory_keep
```

Add:

```text
primitive_memory: [B, P, C], C=16 or 32
```

Use as bias:

```text
primitive_logits = base + context_bias + primitive_memory_bias
```

### 5.4 Route and boundary memory

Route memory:

```text
route_memory: [B, L, L, C]
L=4, so only 16 edges
```

Boundary memory:

```text
boundary_memory: [B, T, C]
T=12
```

Use as bias:

```text
route_logits = base_route_logits + boundary_bias + route_memory_bias
boundary_logits = base_boundary_logits + boundary_context_bias + boundary_memory_bias
```

### 5.5 Expected cost

For current setup:

```text
T=12
lanes=4
cells=12
D=96
```

Matrix memory will not immediately give 10x speedup. The first win is better program quality and less all-to-all mixing.

Expected overhead:

```text
slot memory M=8: tiny
fast rank=16: about +5-10% compute
primitive/route/boundary memory: tiny
all together: about +10-20% overhead initially
```

Potential speedup comes later if memory allows reducing:

```text
T=12 -> T=8
cells=12 -> 8
less route mixing
less useless writes
```

---

## 6. v4.4 Learning Heart before Actor/Critic

Do not add Actor/Critic yet.

First add v4.4 minimal learning heart:

```text
PQS: Program Quality Score
action_history.jsonl
forward-only candidate screen
simple bandit
no deploy
```

Path:

```text
v4_4_min_learning_heart/README.md
```

Actor/Critic needs tested action records. Without them, critic learns noise.

---

## 7. Actor/Critic later

Only after action history exists.

Critic:

```text
(context, action, action_history) -> predicted_gain + uncertainty
```

Start with ridge/Bayesian linear, not big MLP.

Actor:

```text
select top-1 or top-2 sparse actions per window
```

Actor must not output dense tensors over everything.

Candidate sources:

```text
rule candidates
gradient candidates
collapse flags
exploration candidates
dormant primitive candidates
critic-ranked candidates later
```

---

## 8. Meta-heart later

Only after v4.4/v4.5/v4.6.

Meta-heart monitors the Heart itself:

```text
candidate diversity
critic calibration
projection collapse
action repeat rate
heart ROI
adaptive window size
adaptive beta/delta_scale
```

Not part of v4.3 stabilization.

---

## 9. Execution order

Strict order:

```text
1. Fix v4.3 canonical main P0 bugs.
2. Smoke v4.3 canonical main until status=0.
3. Run 5ep v4.3 canonical main.
4. Analyze logs.
5. Test context-controller separately.
6. Implement v4.4 learning heart: PQS + action_history + forward-only screen + bandit, no deploy.
7. Implement matrix memory v4.5.
8. Add FeedbackBiasBank / EditorLoop only after tested action records.
9. Add Critic/Actor.
10. Add Meta-heart.
```

Do not skip steps.

---

## 10. Agent command checklist

Before coding:

```bash
git status --short
git pull --rebase
```

After changes:

```bash
python -m py_compile simple_butterfly_matrix_v4_tape_lane/tape_lane_transport_v4_3_min_heart.py
bash simple_butterfly_matrix_v4_tape_lane/commands/validate_v4_3_min_heart.sh
bash simple_butterfly_matrix_v4_tape_lane/commands/grad_sanity_v4_3_min_heart.sh
```

Smoke:

```bash
DATA_ROOT=../architecture_builder/data/speechcommands \
EPOCHS=1 \
TRAIN_LIMIT=12000 \
VAL_LIMIT=2000 \
MAX_TRAIN_BATCHES=10 \
MAX_VAL_BATCHES=3 \
BATCH_SIZE=128 \
EVAL_BATCH_SIZE=256 \
WORKERS=4 \
AMP=fp16 \
bash simple_butterfly_matrix_v4_tape_lane/commands/sync_run_5ep_v4_3_min_heart_push_logs.sh
```

5ep after smoke only:

```bash
DATA_ROOT=../architecture_builder/data/speechcommands \
EPOCHS=5 \
TRAIN_LIMIT=12000 \
VAL_LIMIT=2000 \
BATCH_SIZE=192 \
EVAL_BATCH_SIZE=512 \
WORKERS=6 \
LOG_EVERY=100 \
AMP=fp16 \
bash simple_butterfly_matrix_v4_tape_lane/commands/sync_run_5ep_v4_3_min_heart_push_logs.sh
```

Commit only if smoke reaches `run_status=0`.
