# Agent Status

Current stage: v4.6.1 grouped real-data loop core
Entrypoint: simple_butterfly_matrix_v4_tape_lane/v4_6_loop_core/tape_lane_transport_v4_6_1_grouped_core.py
Smoke status: pass
Timestamp: 20260619_011028
Report dir: simple_butterfly_matrix_v4_tape_lane/agent_reports/v4_6_1_grouped_real_20260619_011028
Run status: 0

Evidence rules:
- synthetic data disabled in sync evidence run
- route_prior_strength default 0.0
- grouped primitive selector default 18 primitives

Closed-loop invariant:
- The controller decision affects execution, execution affects loss, loss gradient updates the controller decision path.

Known remaining issues:
- inspect grouped/context/credit reports in simple_butterfly_matrix_v4_tape_lane/agent_reports/v4_6_1_grouped_real_20260619_011028

Summary:
- best_acc: 74.15% @ epoch 5
- boundary_mean/peaks: 0.2790/1
- route_entropy/self/useful: 0.3827/0.2523/0.4304
- primitive_entropy/top1/neg_sign: 1.8288/0.2500/0.5037
- memory_write/read/influence: 6.2460/5.8160/3.3169
- collapse_flags: NONE
