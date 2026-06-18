# Agent Status

Current stage: v4.3 canonical main stabilization
Entrypoint: simple_butterfly_matrix_v4_tape_lane/tape_lane_transport_v4_3_min_heart.py
Execution plan: simple_butterfly_matrix_v4_tape_lane/AGENT_EXECUTION_PLAN.md
Runtime wrappers: not used by standard sync
Canonicalizer: deprecated / not used in runtime
Smoke status: pass

Last run: v4.3_canonical_main
Timestamp: 20260618_125911
Report dir: simple_butterfly_matrix_v4_tape_lane/agent_reports/v4_3_canonical_20260618_125911
Command: sync_run_5ep_v4_3_min_heart_push_logs.sh
Run status: 0

Canonical main:
- canonicalizer: deprecated / not used in runtime
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
- profile_speed: 1

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
- best_acc: 52.35% @ epoch 9
- boundary_mean: 0.2725
- route_entropy: 1.3592
- self_route_mass: 0.3355
- useful_transition_mass: 0.2596
- detail_attention_mass: 0.2971
- memory_write/proxy: 0.4211/0.4531
- collapse_flags: BOUNDARY_DEAD,ROUTE_UNIFORM
