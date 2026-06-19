# Latest v4.6.3 Edge Program No-Conv Report

- report_dir: `simple_butterfly_matrix_v4_tape_lane/agent_reports/v4_6_3_edge_program_no_conv_20260619_161503`
- version: `v4.6.3_edge_program_no_conv`
- frontend: `no_conv_linear_patch`
- edge_credit_penalty: `1`
- best_acc: **10.90%** @ epoch 3
- last train/val acc: **19.24% / 10.60%**
- collapse_flags: **BOUNDARY_DEAD,BOUNDARY_FLAT,PRIMITIVE_COLLAPSE**

## Structural summary
- step boundary mean/std/peaks: `0.2829` / `0.0051` / `0`
- route entropy/self/useful/disallowed: `0.2798` / `0.2488` / `0.4698` / `0.2814`
- primitive entropy/top1/negative-sign: `1.7246` / `0.8750` / `0.6134`
- edge gate/boundary/write mean: `0.1844` / `0.8391` / `0.1114`
- output_gate_mean: `0.0022`
- memory forget/write/read/influence: `0.7274` / `1.1520` / `0.0435` / `0.0317`

## Useful groups
- `channel`: -0.000040
- `keep`: -0.000757
- `correction`: -0.002211
- `composition`: -0.002525
- `memory`: -0.002548
- `aggregation`: -0.003395

## Suspicious groups
- `aggregation`: -0.003395
- `memory`: -0.002548
- `composition`: -0.002525
- `correction`: -0.002211
- `keep`: -0.000757
- `channel`: -0.000040

## Useful primitives
- `product_gate`: +0.002950
- `ctx_matrix`: +0.000818
- `low_rank`: +0.000088
- `low_rank_extra_17`: -0.000194
- `pool_context`: -0.000251
- `subtract_memory`: -0.000540
- `channel`: -0.000688
- `smooth_lanes`: -0.000814
- `diff_prev`: -0.000823
- `gated_keep`: -0.001192

## Suspicious primitives
- `mul_filter`: -0.003607
- `memory_write_candidate`: -0.002437
- `lane_mean`: -0.002275
- `contrast`: -0.002118
- `forget_like`: -0.002058
- `memory_read`: -0.001601
- `gated_add`: -0.001482
- `identity`: -0.001310
- `gated_keep`: -0.001192
- `diff_prev`: -0.000823

## Useful steps
- `step_00`: +0.000299
- `step_04`: -0.000461
- `step_05`: -0.001401
- `step_03`: -0.001665
- `step_02`: -0.001872
- `step_01`: -0.001975
- `step_07`: -0.002021
- `step_06`: -0.002052

## Suspicious steps
- `step_06`: -0.002052
- `step_07`: -0.002021
- `step_01`: -0.001975
- `step_02`: -0.001872
- `step_03`: -0.001665
- `step_05`: -0.001401
- `step_04`: -0.000461

## Useful edges
- `edge_1_1`: +0.001186
- `edge_2_2`: +0.000282
- `edge_3_0`: +0.000075
- `edge_2_3`: -0.000038
- `edge_0_2`: -0.000424
- `edge_1_2`: -0.000677
- `edge_3_2`: -0.000920
- `edge_1_3`: -0.001076
- `edge_1_0`: -0.001259
- `edge_2_1`: -0.001372

## Suspicious edges
- `edge_3_3`: -0.003097
- `edge_3_1`: -0.003019
- `edge_0_1`: -0.001693
- `edge_0_0`: -0.001583
- `edge_2_0`: -0.001390
- `edge_0_3`: -0.001387
- `edge_2_1`: -0.001372
- `edge_1_0`: -0.001259
- `edge_1_3`: -0.001076
- `edge_3_2`: -0.000920

## Useful outputs
- `out_05_3`: +0.003385
- `out_06_3`: +0.002512
- `out_03_2`: +0.002204
- `out_02_3`: +0.001822
- `out_03_3`: +0.001391
- `out_07_0`: +0.001083
- `out_07_3`: +0.001019
- `out_05_1`: +0.000322
- `out_00_0`: +0.000212
- `out_06_0`: +0.000191

## Suspicious outputs
- `out_04_3`: -0.003171
- `out_07_2`: -0.002496
- `out_04_0`: -0.002274
- `out_02_1`: -0.002243
- `out_00_3`: -0.001868
- `out_01_2`: -0.001728
- `out_06_2`: -0.001724
- `out_01_3`: -0.001661
- `out_04_2`: -0.001533
- `out_00_1`: -0.001408

## Memory ablation
- `no_memory`: -0.000178

## Penalty/cost metrics
- `edge_gate_cost`: 0.025873
- `output_gate_cost`: 0.002654
- `dead_step_loss`: 0.001069
- `edge_credit_penalty`: 0.004797
- `bad_edge_loss`: 0.000084
- `bad_output_loss`: 0.000006

## Current conclusion
This is the no-convolution ablation. Compare against the normal v4.6.3 edge run. If accuracy stays close, the edge program is doing more work; if it collapses, Conv1D was doing most of the early feature extraction.
