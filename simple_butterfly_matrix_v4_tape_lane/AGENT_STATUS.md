# Agent Status

Current stage: v4.3 canonical main stabilization
Entrypoint: simple_butterfly_matrix_v4_tape_lane/tape_lane_transport_v4_3_min_heart.py
Runtime wrappers: not used by standard sync
Canonicalizer: deprecated / not used in runtime
Smoke status: pass

Last run: v4.3_canonical_main
Timestamp: 20260618_115926
Report dir: simple_butterfly_matrix_v4_tape_lane/agent_reports/v4_3_canonical_20260618_115926
Command: sync_run_5ep_v4_3_min_heart_push_logs.sh
Run status: 0

Canonical main:
- canonicalizer: deprecated / not used in runtime
- entrypoint: simple_butterfly_matrix_v4_tape_lane/tape_lane_transport_v4_3_min_heart.py
- wrapper: not used by standard sync
- still observer-only: candidate deploy=false, no editor auto-deploy

Speed config:
- batch_size: 128
- eval_batch_size: 256
- workers: 4
- log_every: 100
- max_train_batches: 10
- max_val_batches: 3

Expected artifacts:
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

Known remaining issues:
- no runtime canonicalizer in standard sync; context-controller is intentionally separate

Next step:
- only after stable 5ep, test context-controller separately

No checkpoints should be committed.

Summary:
- best_acc: 14.32% @ epoch 1
- boundary_mean: 0.2236
- route_entropy: 1.3821
- self_route_mass: 0.2904
- useful_transition_mass: 0.2377
- detail_attention_mass: 0.2501
- memory_write/proxy: 0.4126/0.4565
- collapse_flags: BOUNDARY_DEAD,ROUTE_UNIFORM
