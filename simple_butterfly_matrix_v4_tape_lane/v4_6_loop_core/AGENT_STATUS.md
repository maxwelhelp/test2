# Agent Status

Current stage: v4.6.1 grouped real-data loop core
Entrypoint: simple_butterfly_matrix_v4_tape_lane/v4_6_loop_core/tape_lane_transport_v4_6_1_grouped_core.py
Smoke status: pass
Timestamp: 20260619_012801
Report dir: simple_butterfly_matrix_v4_tape_lane/agent_reports/v4_6_1_grouped_real_20260619_012801
Run status: 0

Evidence rules:
- synthetic data disabled in sync evidence run
- route_prior_strength default 0.0
- grouped primitive selector default 18 primitives

Closed-loop invariant:
- The controller decision affects execution, execution affects loss, loss gradient updates the controller decision path.

Known remaining issues:
- inspect grouped/context/credit reports in simple_butterfly_matrix_v4_tape_lane/agent_reports/v4_6_1_grouped_real_20260619_012801

Summary:
- best_acc: 74.15% @ epoch 5
- boundary_mean/peaks: 0.2679/3
- route_entropy/self/useful: 0.4182/0.2512/0.4402
- primitive_entropy/top1/neg_sign: 1.8621/0.3438/0.4898
- memory_write/read/influence: 6.8362/4.6237/2.5545
- collapse_flags: NONE
