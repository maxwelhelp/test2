# Agent Status

Current stage: v4.6.1 grouped real-data loop core
Entrypoint: simple_butterfly_matrix_v4_tape_lane/v4_6_loop_core/tape_lane_transport_v4_6_1_grouped_core.py
Smoke status: pass
Timestamp: 20260619_012515
Report dir: simple_butterfly_matrix_v4_tape_lane/agent_reports/v4_6_1_grouped_real_20260619_012515
Run status: 0

Evidence rules:
- synthetic data disabled in sync evidence run
- route_prior_strength default 0.0
- grouped primitive selector default 18 primitives

Closed-loop invariant:
- The controller decision affects execution, execution affects loss, loss gradient updates the controller decision path.

Known remaining issues:
- inspect grouped/context/credit reports in simple_butterfly_matrix_v4_tape_lane/agent_reports/v4_6_1_grouped_real_20260619_012515

Summary:
- best_acc: 10.55% @ epoch 1
- boundary_mean/peaks: 0.2341/0
- route_entropy/self/useful: 1.1451/0.2595/0.3393
- primitive_entropy/top1/neg_sign: 2.1162/0.8438/0.3188
- memory_write/read/influence: 6.4629/0.9637/0.0907
- collapse_flags: BOUNDARY_DEAD
