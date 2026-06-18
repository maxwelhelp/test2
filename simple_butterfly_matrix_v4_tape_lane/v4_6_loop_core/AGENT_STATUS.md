# Agent Status

Current stage: v4.6 loop core MVP
Entrypoint: simple_butterfly_matrix_v4_tape_lane/v4_6_loop_core/tape_lane_transport_v4_6_loop_core.py
Smoke status: pass
Timestamp: 20260619_003038
Report dir: simple_butterfly_matrix_v4_tape_lane/agent_reports/v4_6_loop_core_20260619_003038
Run status: 0

Closed-loop invariant:
- The controller decision affects execution, execution affects loss, loss gradient updates the controller decision path.

Expected artifacts:
- metrics.csv
- trace_epoch_XXX.json
- REPORT_TO_CHATGPT.txt
- final_report.json
- train.log
- run_status.txt

Known remaining issues:
- read simple_butterfly_matrix_v4_tape_lane/agent_reports/v4_6_loop_core_20260619_003038/REPORT_TO_CHATGPT.txt and inspect structural flags

No checkpoints should be committed.

Summary:
- best_acc: 76.15% @ epoch 5
- boundary_mean/peaks: 0.2837/1
- route_entropy/self/useful: 0.4305/0.2470/0.4382
- primitive_entropy/top1/neg_sign: 0.8635/1.0000/0.3394
- memory_write/read/influence: 6.2654/5.5829/3.4423
- collapse_flags: PRIMITIVE_COLLAPSE
