# Agent Status

Current stage: v4.3 canonical main speed profiling
Entrypoint: simple_butterfly_matrix_v4_tape_lane/tape_lane_transport_v4_3_min_heart.py
Runtime wrappers: not used by standard sync
Canonicalizer: deprecated / not used in runtime
Smoke status: pass (profile-speed wiring smoke)

Last run: v4.3_canonical_main
Timestamp: 20260618_121407
Report dir: simple_butterfly_matrix_v4_tape_lane/agent_reports/v4_3_canonical_20260618_121407
Command: sync_run_5ep_v4_3_min_heart_push_logs.sh
Run status: 0

Canonical main:
- canonicalizer: deprecated / not used in runtime
- entrypoint: simple_butterfly_matrix_v4_tape_lane/tape_lane_transport_v4_3_min_heart.py
- wrapper: not used by standard sync
- still observer-only: candidate deploy=false, no editor auto-deploy

Speed config:
- batch_size: 64
- eval_batch_size: 128
- workers: 0
- log_every: 100
- max_train_batches: 2
- max_val_batches: 1
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
- full Tesla P40 10ep speed/quality profile still needs user run
- no runtime canonicalizer in standard sync; context-controller is intentionally separate

Next step:
- run 10ep canonical main with PROFILE_SPEED=1 on Tesla P40
- only after stable canonical main, test context-controller separately

No checkpoints should be committed.

Summary:
- profile smoke used TRAIN_LIMIT=512 VAL_LIMIT=256 MAX_TRAIN_BATCHES=2 MAX_VAL_BATCHES=1 on CPU fallback
- best_acc: 10.16% @ epoch 1
- boundary_mean: 0.2229
- route_entropy: 1.3824
- self_route_mass: 0.2889
- useful_transition_mass: 0.2382
- detail_attention_mass: 0.2502
- memory_write/proxy: 0.4133/0.4564
- collapse_flags: BOUNDARY_DEAD,ROUTE_UNIFORM
