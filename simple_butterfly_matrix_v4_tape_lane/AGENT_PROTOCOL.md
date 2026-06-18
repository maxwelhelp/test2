# AGENT_PROTOCOL

Цель: агент пишет коротко, только главное. Один ответ = один блок статуса.

## Главное место общения

```text
simple_butterfly_matrix_v4_tape_lane/AGENT_STATUS.md
```

Агент после каждой правки обновляет только этот файл.

## Формат AGENT_STATUS.md

```text
STATUS: OK / NEED_RUN / BLOCKED
VERSION: v4.2
LAST_COMMIT: <sha>

CHANGED:
- <file>: <1 short reason>

FIXED:
- <bug/risk>: <fix>

RUN_NOW:
<one command>

CHECK_LOGS:
- <metric/file 1>
- <metric/file 2>
- <metric/file 3>

BLOCKERS:
- none / <short blocker>

NEXT:
- <one next action>
```

## Hard rules

```text
max 30 lines
no long explanations
no theory
no duplicate text from README
always give one exact command
always say if old scripts still point to old version
```

## Priority words

```text
P0 = blocks run or wrong version
P1 = run works but result may be misleading
P2 = improvement / cleanup
```

## Example

```text
STATUS: NEED_RUN
VERSION: v4.2
LAST_COMMIT: abc123

CHANGED:
- commands/sync_run_5ep_push_logs_v4_2.sh: added validate+run+push logs
- tape_lane_transport_v4_2.py: fixed weak_flow off-diagonal routes

FIXED:
- P0 old sync script risk: v4.2 has own sync script
- P1 weak_flow identity-only: added detail->state/state->abstract/state->memory

RUN_NOW:
bash simple_butterfly_matrix_v4_tape_lane/commands/sync_run_5ep_push_logs_v4_2.sh

CHECK_LOGS:
- REPORT_TO_CHATGPT.txt
- metrics.csv val_acc
- analysis_epoch_005.json route_matrix/primitive_weights

BLOCKERS:
- none

NEXT:
- wait for 5ep logs
```
