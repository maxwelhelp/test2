# Agent Status

Last run: v4.3_canonical_main
Timestamp: 20260618_113503
Report dir: simple_butterfly_matrix_v4_tape_lane/agent_reports/v4_3_canonical_20260618_113503
Command: sync_run_5ep_v4_3_min_heart_push_logs.sh
Run status: 2

Canonical main:
- canonicalizer: simple_butterfly_matrix_v4_tape_lane/commands/canonicalize_v4_3_main.py
- entrypoint: simple_butterfly_matrix_v4_tape_lane/tape_lane_transport_v4_3_min_heart.py
- wrapper: not used by standard sync
- still observer-only: candidate deploy=false, no editor auto-deploy

Speed config:
- batch_size: 192
- eval_batch_size: 512
- workers: 6
- log_every: 100
- max_train_batches: 0
- max_val_batches: 0

Expected artifacts:
- canonicalize.log
- grad_sanity.json
- grad_sanity.log
- speed_config.txt
- metrics.csv
- analysis_epoch_XXX.json
- trace_feedback_epoch_XXX.json
- candidate_suggestions_epoch_XXX.json
- REPORT_TO_CHATGPT.txt
- final_report.json
- train.log
- run_status.txt

No checkpoints should be committed.
