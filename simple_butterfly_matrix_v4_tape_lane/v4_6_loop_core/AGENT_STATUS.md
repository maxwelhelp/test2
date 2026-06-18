# Agent Status

Current stage: v4.6 loop core MVP
Entrypoint: simple_butterfly_matrix_v4_tape_lane/v4_6_loop_core/tape_lane_transport_v4_6_loop_core.py
Smoke status: pass
Timestamp: 20260619_001119
Report dir: simple_butterfly_matrix_v4_tape_lane/agent_reports/v4_6_loop_core_20260619_001119
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
- read simple_butterfly_matrix_v4_tape_lane/agent_reports/v4_6_loop_core_20260619_001119/REPORT_TO_CHATGPT.txt and inspect structural flags

No checkpoints should be committed.

Summary:
- best_acc: 76.65% @ epoch 5
- boundary_mean/peaks: 0.2748/0
- route_entropy/self/useful: 0.6078/0.2384/0.4338
- primitive_entropy/top1/neg_sign: 0.7450/0.7500/0.4250
- memory_write/read/influence: 7.1657/5.2727/3.0486
- collapse_flags: BOUNDARY_DEAD
