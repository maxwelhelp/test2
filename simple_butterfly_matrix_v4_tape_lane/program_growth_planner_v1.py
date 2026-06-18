#!/usr/bin/env python3
"""Program Growth Planner v1.

Reads a v4.5 report and proposes where to insert the next tape step.

Design:
- start from a tiny 2-step seed when requested;
- do not hard-code first/last as universal roles;
- use weak task-aware priors derived from task mode / head behaviour;
- find weak steps by route/read/primitive/update/boundary/head/memory proxies;
- propose insert_before / insert_after candidates;
- no weight mutation, no checkpoints, no auto-deploy.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List


def read_json(p: Path, default=None):
    try:
        with p.open("r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {} if default is None else default


def write_json(p: Path, obj):
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2, sort_keys=True)
        f.write("\n")


def fnum(x, d=0.0):
    try:
        return float(x)
    except Exception:
        return d


def latest_trace(report_dir: Path):
    xs = sorted(report_dir.glob("trace_feedback_epoch_*.json"))
    if not xs:
        raise FileNotFoundError(f"no trace_feedback_epoch_*.json in {report_dir}")
    def ep(p):
        try: return int(p.stem.split("_")[-1])
        except Exception: return -1
    p = max(xs, key=ep)
    return ep(p), p


def get(d: Dict[str, Any], *ks, default=None):
    cur: Any = d
    for k in ks:
        if isinstance(cur, dict) and k in cur:
            cur = cur[k]
        else:
            return default
    return cur


def as_series(x, T):
    x = x if isinstance(x, list) else []
    out = [fnum(v) for v in x[:T]]
    return out + [0.0] * max(0, T - len(out))


def mean_nested_step(x, T):
    out = []
    for t in range(T):
        vals = []
        raw = x[t] if isinstance(x, list) and t < len(x) else []
        stack = [raw]
        while stack:
            cur = stack.pop()
            if isinstance(cur, list):
                stack.extend(cur)
            else:
                try: vals.append(float(cur))
                except Exception: pass
        out.append(sum(vals) / max(1, len(vals)))
    return out


def infer_task_mode(trace: Dict[str, Any]) -> str:
    # Current SpeechCommands runs are classification. Keep this configurable and weak.
    classes = trace.get("classes") or get(trace, "final_report", "classes", default=[])
    if isinstance(classes, list) and len(classes) >= 2:
        return "classification"
    return "unknown"


def build_step_scores(trace: Dict[str, Any], task_mode: str) -> Dict[str, Any]:
    route = trace.get("route", {}) if isinstance(trace.get("route"), dict) else {}
    seq = trace.get("sequence", {}) if isinstance(trace.get("sequence"), dict) else {}
    read = trace.get("read", {}) if isinstance(trace.get("read"), dict) else {}
    memory = trace.get("memory", {}) if isinstance(trace.get("memory"), dict) else {}
    head = trace.get("head", {}) if isinstance(trace.get("head"), dict) else {}
    rep = trace.get("matrix_report", {}) if isinstance(trace.get("matrix_report"), dict) else {}

    boundary = route.get("boundary_by_step", route.get("boundary", []))
    if not isinstance(boundary, list):
        boundary = []
    T = max(2, len(boundary), len(seq.get("sequence_change_by_step", []) or []), len(read.get("read_delta_by_step", []) or []))

    boundary = as_series(boundary, T)
    seq_change = as_series(seq.get("sequence_change_by_step", []), T)
    route_delta = as_series(route.get("route_delta_by_step", []), T)
    read_delta = as_series(read.get("read_delta_by_step", []), T)
    prim_delta = as_series(seq.get("primitive_delta_by_step", []), T)
    update_delta = as_series(seq.get("update_delta_by_step", []), T)
    useful = as_series(route.get("useful_transition_by_step", []), T)
    memory_write = as_series(memory.get("write_by_step", memory.get("memory_write_by_step", [])), T)
    future_read = as_series(memory.get("future_read_by_step", memory.get("memory_future_read_by_step", [])), T)

    if not any(seq_change):
        seq_change = as_series(get(trace, "sequence", "sequence_change_by_step", default=[]), T)
    if not any(update_delta):
        update_delta = mean_nested_step(get(rep, "update_norm_by_step_lane", default=[]), T)
    if not any(memory_write):
        memory_write = mean_nested_step(get(rep, "write_gate_by_step_lane", default=[]), T)

    detail_topread = fnum(head.get("detail_topread_share", trace.get("detail_topread_share", 0.0)), 0.0)
    last = T - 1
    step_scores = []
    for t in range(T):
        change_need = seq_change[t] + route_delta[t] + read_delta[t] + prim_delta[t] + update_delta[t]
        boundary_bad = max(0.0, change_need - boundary[t])
        route_bad = max(0.0, 0.12 - useful[t])
        memory_bad = max(0.0, memory_write[t] - future_read[t])
        input_side_bad = 0.0
        output_side_bad = 0.0
        if t == 0:
            # Weak input-side prior: first step should produce useful update/read, not necessarily fixed primitive.
            input_side_bad = max(0.0, 0.08 - (read_delta[t] + update_delta[t] + prim_delta[t]))
        if task_mode == "classification" and t == last:
            # Weak output-side prior: classification often benefits from class-ready summary, but not hard-coded.
            output_side_bad = max(0.0, 0.10 - (useful[t] + update_delta[t] + read_delta[t])) + max(0.0, detail_topread - 0.70)
        score = 1.5 * boundary_bad + 1.2 * route_bad + 0.7 * memory_bad + 0.7 * input_side_bad + 0.8 * output_side_bad
        step_scores.append({
            "t": t,
            "weakness_score": round(float(score), 6),
            "signals": {
                "boundary": boundary[t],
                "sequence_change": seq_change[t],
                "route_delta": route_delta[t],
                "read_delta": read_delta[t],
                "primitive_delta": prim_delta[t],
                "update_delta": update_delta[t],
                "useful_transition": useful[t],
                "memory_write": memory_write[t],
                "future_read": future_read[t],
                "detail_topread_share": detail_topread,
                "input_side_bad": input_side_bad,
                "output_side_bad": output_side_bad,
            },
        })
    return {"T": T, "last_step": last, "task_mode": task_mode, "step_scores": step_scores}


def propose_growth(scores: Dict[str, Any]) -> List[Dict[str, Any]]:
    T = int(scores["T"])
    last = int(scores["last_step"])
    task_mode = str(scores.get("task_mode", "unknown"))
    ranked = sorted(scores["step_scores"], key=lambda r: r["weakness_score"], reverse=True)
    proposals: List[Dict[str, Any]] = []

    for r in ranked[: min(5, len(ranked))]:
        t = int(r["t"])
        sig = r["signals"]
        if t == 0 and sig.get("input_side_bad", 0.0) > 0.0:
            proposals.append({
                "action": "insert_after",
                "position": 0,
                "new_T": T + 1,
                "stage_type": "input_evidence_refine",
                "reason": "input-side step has weak read/update/primitive movement; insert local evidence refine after input step",
                "evidence": r,
                "target_effect": {"route": ["detail->state"], "primitive": ["diff", "gated_contrast", "ctx_matrix"], "boundary": "only if after-context changes"},
            })
        elif t == last and task_mode == "classification" and sig.get("output_side_bad", 0.0) > 0.0:
            proposals.append({
                "action": "insert_before",
                "position": last,
                "new_T": T + 1,
                "stage_type": "pre_output_summary_refine",
                "reason": "classification output side looks weak or detail-head shortcut is high; insert refine before output/summary step",
                "evidence": r,
                "target_effect": {"route": ["state->abstract", "memory->state"], "primitive": ["ctx_matrix", "product_gate", "low_rank"], "boundary": "soft sparse peak before output only if useful"},
            })
        elif sig.get("memory_write", 0.0) > sig.get("future_read", 0.0) + 0.05:
            proposals.append({
                "action": "insert_after",
                "position": t,
                "new_T": T + 1,
                "stage_type": "memory_recall_or_cleanup",
                "reason": "memory write appears higher than future read; insert a recall/check step after writer",
                "evidence": r,
                "target_effect": {"route": ["memory->state"], "primitive": ["ctx_matrix", "memory_keep"], "boundary": "memory recall boundary only if contrastive"},
            })
        else:
            side = "after" if t < last else "before"
            proposals.append({
                "action": f"insert_{side}",
                "position": t,
                "new_T": T + 1,
                "stage_type": "local_transform_route_refine",
                "reason": "step has high weakness score; model a new transform/route refine stage on the side with weaker context",
                "evidence": r,
                "target_effect": {"route": ["detail->state", "state->abstract"], "primitive": ["diff", "ctx_matrix", "gated_contrast"], "boundary": "sparse peak only if projected after-context improves"},
            })

    proposals.append({
        "action": "keep_T_strengthen_current_choices",
        "position": last,
        "new_T": T,
        "stage_type": "no_growth_baseline",
        "reason": "growth is optional; first compare to strengthening current read/route/primitive choices without adding exact depth",
        "target_effect": {"route": ["allowed edges only"], "primitive": ["context-selected mixture"], "boundary": "no new boundary required"},
    })
    return proposals


def run(args):
    rd = Path(args.report_dir).resolve()
    epoch, tp = latest_trace(rd)
    trace = read_json(tp, {})
    task_mode = args.task_mode if args.task_mode != "auto" else infer_task_mode(trace)
    scores = build_step_scores(trace, task_mode)
    proposals = propose_growth(scores)
    out = {
        "version": "program_growth_planner_v1_two_step_compatible",
        "mode": "report_only_growth_decision",
        "report_dir": str(rd),
        "epoch": epoch,
        "weak_prior_policy": {
            "seed_steps": "2 by default",
            "first_step": "weak input/evidence prior only, not fixed",
            "last_step": "weak output/summary prior only for classification, not fixed",
            "middle_steps": "created only when weakness/projection says depth is useful",
            "note": "for another task/head, task_mode can change; growth is decided by trace weakness not hard-coded stage names",
        },
        "scores": scores,
        "growth_proposals": proposals,
        "recommended_next_command_hint": "rerun with TAPE_STEPS=new_T from top proposal; do not push checkpoints",
    }
    op = rd / f"program_growth_plan_epoch_{epoch:03d}.json"
    write_json(op, out)
    print(f"[growth_planner] wrote {op}")
    for p in proposals[:5]:
        print(f"  {p['action']} pos={p['position']} new_T={p['new_T']} type={p['stage_type']} reason={p['reason']}")


def parser():
    p = argparse.ArgumentParser()
    p.add_argument("--report-dir", required=True)
    p.add_argument("--task-mode", choices=["auto", "classification", "unknown"], default="auto")
    return p


if __name__ == "__main__":
    run(parser().parse_args())
