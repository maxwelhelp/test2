# v4.3_min_heart

Root entry folder for the conservative v4.3 bridge.

Main implementation:

```text
simple_butterfly_matrix_v4_tape_lane/tape_lane_transport_v4_3_min_heart.py
```

Authoritative plan:

```text
simple_butterfly_matrix_v4_tape_lane/V4_3_MIN_HEART_PLAN.md
```

Validation:

```bash
bash simple_butterfly_matrix_v4_tape_lane/commands/validate_v4_3_min_heart.sh
```

One-command 5 epoch run + log push:

```bash
DATA_ROOT=../architecture_builder/data/speechcommands \
EPOCHS=5 \
TRAIN_LIMIT=12000 \
VAL_LIMIT=2000 \
AMP=fp16 \
bash simple_butterfly_matrix_v4_tape_lane/commands/sync_run_5ep_v4_3_min_heart_push_logs.sh
```

Scope:

```text
costed paths + trace + deterministic candidate suggestions
no actor
no critic
no planner attention
no feedback auto-deploy
```

Main sequence requirement:

```text
tape positions must not collapse into identical mixers;
logs must show whether route/read/primitive/boundary change along the tape;
sequence_nonflat_score is the first diagnostic.
```
