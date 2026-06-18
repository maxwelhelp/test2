# Agent Status

Current stage: v4.6.1 grouped real-data loop core
Entrypoint: simple_butterfly_matrix_v4_tape_lane/v4_6_loop_core/tape_lane_transport_v4_6_1_grouped_core.py
Smoke status: pass
Timestamp: 20260619_013939
Report dir: simple_butterfly_matrix_v4_tape_lane/agent_reports/v4_6_1_grouped_real_20260619_013939
Run status: 0
Latest compact report: simple_butterfly_matrix_v4_tape_lane/v4_6_loop_core/LATEST_RUN_REPORT.md

Evidence rules:
- synthetic data disabled in sync evidence run
- route_prior_strength default 0.0
- grouped primitive selector default 18 primitives

Closed-loop invariant:
- The controller decision affects execution, execution affects loss, loss gradient updates the controller decision path.

Known remaining issues:
- inspect grouped/context/credit reports in simple_butterfly_matrix_v4_tape_lane/agent_reports/v4_6_1_grouped_real_20260619_013939

Summary:
- best_acc: 72.20%
- latest_report: simple_butterfly_matrix_v4_tape_lane/v4_6_loop_core/LATEST_RUN_REPORT.md
- run_summary: simple_butterfly_matrix_v4_tape_lane/agent_reports/v4_6_1_grouped_real_20260619_013939/RUN_SUMMARY.md
- useful_groups: [('channel', 0.3344442844390869), ('memory', 0.027743637561798096), ('correction', 0.023800909519195557)]
- suspicious_groups: [('aggregation', -0.00630033016204834), ('composition', -0.006207168102264404), ('keep', 0.013298451900482178)]
