# Agent Status

Last run: v4.3_min_heart_audit_fixed
Timestamp: 20260618_095226
Report dir: simple_butterfly_matrix_v4_tape_lane/agent_reports/v4_3_min_heart_20260618_095226
Command: sync_run_5ep_v4_3_min_heart_push_logs.sh
Run status: 0

Expected artifacts:
- grad_sanity.json
- grad_sanity.log
- metrics.csv
- analysis_epoch_XXX.json
- trace_feedback_epoch_XXX.json
- candidate_suggestions_epoch_XXX.json
- REPORT_TO_CHATGPT.txt
- final_report.json
- train.log
- run_status.txt

No checkpoints should be committed.

Summary:
- best_acc: 40.70% @ epoch 5
- boundary_mean: 0.2559
- route_entropy: 1.3720
- detail_attention_mass: 0.2818
- memory_write/consumer: 0.4125/0.4547
- collapse_flags: ROUTE_UNIFORM
