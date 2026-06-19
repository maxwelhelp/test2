# Latest v4.6.3 Edge Program Report

- report_dir: `simple_butterfly_matrix_v4_tape_lane/agent_reports/v4_6_3_edge_program_20260619_143908`
- version: `v4.6.3_edge_program`
- edge_credit_penalty: `1`
- best_acc: **73.90%** @ epoch 5
- last train/val acc: **68.10% / 73.90%**
- collapse_flags: **ROUTE_UNIFORM**

## Structural summary
- step boundary mean/std/peaks: `0.2974` / `0.1558` / `4`
- route entropy/self/useful/disallowed: `1.3858` / `0.2499` / `0.2500` / `0.5001`
- primitive entropy/top1/negative-sign: `2.1253` / `0.6562` / `0.4897`
- edge gate/boundary/write mean: `0.3807` / `0.4171` / `0.3971`
- output_gate_mean: `0.4443`
- memory forget/write/read/influence: `0.7227` / `6.6562` / `4.2076` / `4.2419`

## Useful groups
- `channel`: +0.091284
- `keep`: +0.019217
- `aggregation`: -0.016019
- `correction`: -0.018283
- `composition`: -0.035522
- `memory`: -0.053413

## Suspicious groups
- `memory`: -0.053413
- `composition`: -0.035522
- `correction`: -0.018283
- `aggregation`: -0.016019
- `keep`: +0.019217
- `channel`: +0.091284

## Useful primitives
- `channel`: +0.061504
- `gated_keep`: +0.030646
- `product_gate`: +0.005137
- `smooth_lanes`: +0.002124
- `diff_prev`: -0.026018
- `identity`: -0.027935
- `lane_mean`: -0.033335
- `contrast`: -0.039906
- `ctx_matrix`: -0.042381
- `mul_filter`: -0.044433

## Suspicious primitives
- `pool_context`: -0.091022
- `gated_add`: -0.067836
- `forget_like`: -0.067414
- `subtract_memory`: -0.062428
- `low_rank`: -0.061991
- `low_rank_extra_17`: -0.055373
- `memory_write_candidate`: -0.054962
- `memory_read`: -0.052169
- `mul_filter`: -0.044433
- `ctx_matrix`: -0.042381

## Useful steps
- `step_02`: +0.351122
- `step_05`: +0.210478
- `step_06`: +0.107463
- `step_07`: +0.082977
- `step_04`: +0.036210
- `step_01`: +0.020186
- `step_00`: -0.029939
- `step_03`: -0.050801

## Suspicious steps
- `step_03`: -0.050801
- `step_00`: -0.029939
- `step_01`: +0.020186
- `step_04`: +0.036210
- `step_07`: +0.082977
- `step_06`: +0.107463
- `step_05`: +0.210478
- `step_02`: +0.351122

## Useful edges
- `edge_2_1`: +0.028496
- `edge_3_2`: +0.025472
- `edge_2_2`: +0.011293
- `edge_1_0`: +0.008869
- `edge_3_3`: +0.006931
- `edge_2_0`: -0.003632
- `edge_1_2`: -0.005672
- `edge_0_1`: -0.014676
- `edge_0_3`: -0.020204
- `edge_2_3`: -0.021789

## Suspicious edges
- `edge_1_1`: -0.054907
- `edge_3_1`: -0.049015
- `edge_3_0`: -0.046923
- `edge_0_0`: -0.038586
- `edge_1_3`: -0.032356
- `edge_0_2`: -0.028866
- `edge_2_3`: -0.021789
- `edge_0_3`: -0.020204
- `edge_0_1`: -0.014676
- `edge_1_2`: -0.005672

## Useful outputs
- `out_02_3`: +0.050954
- `out_06_1`: +0.045877
- `out_02_1`: +0.026716
- `out_02_2`: +0.021885
- `out_00_3`: +0.020962
- `out_07_2`: +0.012836
- `out_00_2`: +0.009657
- `out_06_0`: +0.006840
- `out_06_3`: +0.003908
- `out_05_1`: -0.001185

## Suspicious outputs
- `out_04_0`: -0.105156
- `out_04_3`: -0.096797
- `out_04_1`: -0.057366
- `out_00_1`: -0.056835
- `out_03_1`: -0.050761
- `out_05_3`: -0.049062
- `out_03_2`: -0.046997
- `out_07_1`: -0.046526
- `out_05_2`: -0.044034
- `out_07_3`: -0.036882

## Memory ablation
- `no_memory`: +1.039399

## Penalty/cost metrics
- `edge_gate_cost`: 0.199420
- `output_gate_cost`: 0.382286
- `dead_step_loss`: 0.000464
- `edge_credit_penalty`: 0.003396
- `bad_edge_loss`: 0.000157
- `bad_output_loss`: 0.000505

## Current conclusion
This is the first edge-conditioned version: primitive/operator choice is now made per source-target edge, with separate edge/write/boundary/output gates. Compare it against v4.6.2 and check whether useful edge/output credit appears without killing accuracy.
