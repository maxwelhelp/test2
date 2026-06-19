# Latest v4.6.3 Edge Program Report

- report_dir: `simple_butterfly_matrix_v4_tape_lane/agent_reports/v4_6_3_edge_program_20260619_155949`
- version: `v4.6.3_edge_program`
- edge_credit_penalty: `1`
- best_acc: **71.00%** @ epoch 5
- last train/val acc: **67.07% / 71.00%**
- collapse_flags: **ROUTE_UNIFORM**

## Structural summary
- step boundary mean/std/peaks: `0.3056` / `0.1105` / `3`
- route entropy/self/useful/disallowed: `1.3852` / `0.2500` / `0.2500` / `0.4999`
- primitive entropy/top1/negative-sign: `2.0542` / `0.4375` / `0.4886`
- edge gate/boundary/write mean: `0.4656` / `0.4687` / `0.4514`
- output_gate_mean: `0.5046`
- memory forget/write/read/influence: `0.7253` / `6.6355` / `3.5487` / `3.5605`

## Useful groups
- `channel`: +0.310057
- `memory`: +0.094405
- `keep`: +0.064234
- `composition`: +0.034547
- `correction`: +0.030720
- `aggregation`: +0.007140

## Suspicious groups
- `aggregation`: +0.007140
- `correction`: +0.030720
- `composition`: +0.034547
- `keep`: +0.064234
- `memory`: +0.094405
- `channel`: +0.310057

## Useful primitives
- `low_rank`: +0.135365
- `memory_write_candidate`: +0.113352
- `mul_filter`: +0.105706
- `low_rank_extra_17`: +0.079565
- `product_gate`: +0.075207
- `gated_add`: +0.064844
- `ctx_matrix`: +0.062894
- `pool_context`: +0.054872
- `gated_keep`: +0.046182
- `smooth_lanes`: +0.045637

## Suspicious primitives
- `forget_like`: -0.016386
- `memory_read`: -0.009714
- `identity`: +0.008737
- `contrast`: +0.017201
- `lane_mean`: +0.018278
- `subtract_memory`: +0.027415
- `diff_prev`: +0.042464
- `channel`: +0.044778
- `smooth_lanes`: +0.045637
- `gated_keep`: +0.046182

## Useful steps
- `step_02`: +0.369024
- `step_05`: +0.332795
- `step_03`: +0.324569
- `step_04`: +0.151581
- `step_06`: +0.133013
- `step_01`: +0.101552
- `step_07`: +0.074699
- `step_00`: +0.068658

## Suspicious steps
- `step_00`: +0.068658
- `step_07`: +0.074699
- `step_01`: +0.101552
- `step_06`: +0.133013
- `step_04`: +0.151581
- `step_03`: +0.324569
- `step_05`: +0.332795
- `step_02`: +0.369024

## Useful edges
- `edge_0_2`: +0.103942
- `edge_3_3`: +0.078182
- `edge_3_0`: +0.073396
- `edge_3_2`: +0.069892
- `edge_2_0`: +0.062923
- `edge_2_1`: +0.060143
- `edge_0_1`: +0.059785
- `edge_3_1`: +0.055732
- `edge_2_3`: +0.055179
- `edge_2_2`: +0.054260

## Suspicious edges
- `edge_0_3`: -0.060610
- `edge_1_3`: +0.015169
- `edge_1_2`: +0.033471
- `edge_0_0`: +0.036301
- `edge_1_1`: +0.040548
- `edge_1_0`: +0.050637
- `edge_2_2`: +0.054260
- `edge_2_3`: +0.055179
- `edge_3_1`: +0.055732
- `edge_0_1`: +0.059785

## Useful outputs
- `out_01_2`: +0.106416
- `out_01_3`: +0.100501
- `out_04_3`: +0.093447
- `out_06_3`: +0.080080
- `out_02_2`: +0.076616
- `out_06_2`: +0.073165
- `out_07_0`: +0.072954
- `out_02_0`: +0.071292
- `out_02_1`: +0.070529
- `out_00_3`: +0.070170

## Suspicious outputs
- `out_05_3`: -0.030304
- `out_04_0`: -0.025079
- `out_04_1`: -0.006380
- `out_03_3`: +0.000362
- `out_00_1`: +0.000778
- `out_07_1`: +0.002685
- `out_06_1`: +0.011448
- `out_03_1`: +0.014824
- `out_03_2`: +0.016729
- `out_04_2`: +0.018555

## Memory ablation
- `no_memory`: +0.500922

## Penalty/cost metrics
- `edge_gate_cost`: 0.229766
- `output_gate_cost`: 0.399159
- `dead_step_loss`: 0.000650
- `edge_credit_penalty`: 0.002777
- `bad_edge_loss`: 0.000000
- `bad_output_loss`: 0.000376

## Current conclusion
This is the first edge-conditioned version: primitive/operator choice is now made per source-target edge, with separate edge/write/boundary/output gates. Compare it against v4.6.2 and check whether useful edge/output credit appears without killing accuracy.
