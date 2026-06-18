# Latest v4.6.1 Grouped Run Report

- version: `v4.6.1_grouped_core`
- report_dir: `simple_butterfly_matrix_v4_tape_lane/agent_reports/v4_6_1_grouped_real_20260619_013939`
- best_acc: **72.20%** @ epoch 5
- last train/val acc: **69.59% / 72.20%**
- collapse_flags: **NONE**

## Structural summary

- boundary mean/std/peaks: `0.3042` / `0.0626` / `3` peaks=[2, 3, 4]
- route entropy/self/useful/disallowed: `0.3304` / `0.2436` / `0.4738` / `0.2826`
- primitive entropy/top1/negative-sign: `1.7976` / `0.4688` / `0.5184`
- memory forget/write/read/influence: `0.7198` / `6.7231` / `3.6678` / `1.7659`

## Credit ablation interpretation

Positive delta CE means removing the component hurts, so the component was useful on this batch. Negative delta CE means removing it helped, so the component is suspicious/junk on this batch.

### Useful groups
- `channel`: +0.334444
- `memory`: +0.027744
- `correction`: +0.023801
- `keep`: +0.013298
- `composition`: -0.006207
- `aggregation`: -0.006300

### Suspicious groups
- `aggregation`: -0.006300
- `composition`: -0.006207
- `keep`: +0.013298
- `correction`: +0.023801
- `memory`: +0.027744
- `channel`: +0.334444

### Useful primitives
- `channel`: +0.161913
- `low_rank_extra_17`: +0.100657
- `low_rank`: +0.055470
- `lane_mean`: +0.047719
- `contrast`: +0.034526
- `forget_like`: +0.028044
- `ctx_matrix`: +0.024579
- `gated_add`: +0.016469

### Suspicious primitives
- `memory_read`: -0.029415
- `gated_keep`: -0.015190
- `mul_filter`: -0.006306
- `pool_context`: -0.005165
- `diff_prev`: -0.004074
- `memory_write_candidate`: -0.000424
- `product_gate`: +0.000285
- `smooth_lanes`: +0.006909

## Artifacts

- `simple_butterfly_matrix_v4_tape_lane/agent_reports/v4_6_1_grouped_real_20260619_013939/REPORT_TO_CHATGPT.txt`
- `simple_butterfly_matrix_v4_tape_lane/agent_reports/v4_6_1_grouped_real_20260619_013939/final_report.json`
- `simple_butterfly_matrix_v4_tape_lane/agent_reports/v4_6_1_grouped_real_20260619_013939/metrics.csv`
- `simple_butterfly_matrix_v4_tape_lane/agent_reports/v4_6_1_grouped_real_20260619_013939/trace_epoch_005.json`
- `simple_butterfly_matrix_v4_tape_lane/agent_reports/v4_6_1_grouped_real_20260619_013939/credit_ablation_epoch_005.json`
- `simple_butterfly_matrix_v4_tape_lane/agent_reports/v4_6_1_grouped_real_20260619_013939/train.log`

## Current conclusion

The loop is not proven as an intelligent program yet. Need frontend-only/full-loop/no-memory/no-route/step-ablation baselines.
