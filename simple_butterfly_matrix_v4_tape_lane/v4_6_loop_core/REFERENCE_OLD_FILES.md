# v4.6 Reference Old Files Guide

Use this guide before coding v4.6. The goal is to reuse working loader/report ideas from old versions, not to copy old architecture mistakes.

Repository branch for active work:

```text
codex-full-agent-plan
```

Main project folder:

```text
simple_butterfly_matrix_v4_tape_lane/
```

v4.6 folder:

```text
simple_butterfly_matrix_v4_tape_lane/v4_6_loop_core/
```

---

## 1. Mandatory v4.6 documents

Read these first:

```text
simple_butterfly_matrix_v4_tape_lane/v4_6_loop_core/V4_6_FULL_LOCKED_SPEC.md
simple_butterfly_matrix_v4_tape_lane/v4_6_loop_core/AGENT_IMPLEMENTATION_BRIEF.md
simple_butterfly_matrix_v4_tape_lane/v4_6_loop_core/ARCHITECTURE_PLAN.md
simple_butterfly_matrix_v4_tape_lane/v4_6_loop_core/AGENT_TASKS.md
simple_butterfly_matrix_v4_tape_lane/v4_6_loop_core/README.md
```

Rule:

```text
V4_6_FULL_LOCKED_SPEC.md is the source of truth.
```

---

## 2. Old files to study, but not modify

### v4.3 canonical main

```text
simple_butterfly_matrix_v4_tape_lane/tape_lane_transport_v4_3_min_heart.py
```

Study for:

```text
loader style
training loop style
metrics/report style
boundary/route/read/memory diagnostics
AGENT_STATUS/report conventions
```

Do not copy:

```text
independent controller logic
old boundary exploit logic
weak memory-as-lane semantics
placeholder skip/operator costs without clear implementation
```

### v4.3 sync command

```text
simple_butterfly_matrix_v4_tape_lane/commands/sync_run_5ep_v4_3_min_heart_push_logs.sh
```

Study for:

```text
git pull/rebase behavior
run_status handling
log folder layout
no-checkpoint push guard
AGENT_STATUS writing
```

Do not copy blindly. v4.6 should have its own command under:

```text
simple_butterfly_matrix_v4_tape_lane/v4_6_loop_core/commands/
```

### v4.4/v4.5 reports

Look in:

```text
simple_butterfly_matrix_v4_tape_lane/agent_reports/
```

Study for failure patterns:

```text
BOUNDARY_EXPLOIT
boundary all-on
route partially specialized but not clean
primitive collapse / weak specialization
detail_topread shortcut
memory write without causal proof
```

Do not treat high accuracy as success if these failures remain.

### Program assembly curriculum

```text
simple_butterfly_matrix_v4_tape_lane/program_assembly_curriculum_v1.py
simple_butterfly_matrix_v4_tape_lane/program_assembly_curriculum_v2_structured.py
```

Study for:

```text
program assembly vocabulary
read/primitive/route/write/boundary labels
repair/denoise idea
structured JSONL idea
separation between builder pretraining and task training
```

Do not load the whole curriculum checkpoint into v4.6 task model. Later transfer only controller/primitive priors/adapters if metrics justify it.

---

## 3. GitHub links

Use these browser links if needed:

```text
https://github.com/maxwelhelp/test2/tree/codex-full-agent-plan/simple_butterfly_matrix_v4_tape_lane/v4_6_loop_core
https://github.com/maxwelhelp/test2/blob/codex-full-agent-plan/simple_butterfly_matrix_v4_tape_lane/tape_lane_transport_v4_3_min_heart.py
https://github.com/maxwelhelp/test2/blob/codex-full-agent-plan/simple_butterfly_matrix_v4_tape_lane/commands/sync_run_5ep_v4_3_min_heart_push_logs.sh
https://github.com/maxwelhelp/test2/blob/codex-full-agent-plan/simple_butterfly_matrix_v4_tape_lane/program_assembly_curriculum_v2_structured.py
https://github.com/maxwelhelp/test2/tree/codex-full-agent-plan/simple_butterfly_matrix_v4_tape_lane/agent_reports
```

If the branch does not contain a file locally, run:

```bash
git checkout codex-full-agent-plan
git pull --rebase --autostash origin codex-full-agent-plan
```

---

## 4. What to reuse from old code

Reuse:

```text
SpeechCommands/data loader approach if stable
argparse style
AMP/fp16 flags
metrics.csv writing
REPORT_TO_CHATGPT.txt format
final_report.json format
AGENT_STATUS.md format
sync script no-checkpoint policy
```

Rewrite fresh:

```text
controller architecture
primitive selector
memory implementation
boundary/route coupling
step analyzer
losses
```

---

## 5. Core implementation rule

Always verify this sentence:

```text
The controller decision affects execution, execution affects loss, loss gradient updates the controller decision path.
```

If an old file breaks this rule, do not copy that part.

---

## 6. Quick local navigation for agent

From repo root:

```bash
cd simple_butterfly_matrix_v4_tape_lane
ls
ls v4_6_loop_core
sed -n '1,220p' v4_6_loop_core/V4_6_FULL_LOCKED_SPEC.md
sed -n '1,220p' v4_6_loop_core/AGENT_IMPLEMENTATION_BRIEF.md
sed -n '1,220p' tape_lane_transport_v4_3_min_heart.py
```

Then implement only inside:

```text
v4_6_loop_core/
```
