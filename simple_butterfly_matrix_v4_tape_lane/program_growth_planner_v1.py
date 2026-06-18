#!/usr/bin/env python3
"""Program Growth Planner v1.

Reads a v4.5 report and proposes where to insert the next tape step.

Design:
- start from a small 3-step program skeleton;
- keep the last step as aggregation/head-preparation for classification;
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
    """Convert step list of nested numbers to one scalar per step."""
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


def build_step_scores(trace: Dict[str, Any]) -> Dict[str, Any]:
    route = trace.get("route", {}) if isinstance(trace.get("route"), dict) else {}
    seq = trace.get("sequence", {}) if isinstance(trace.get("sequence"), dict) else {}
    read = trace.get("read", {}) if isinstance(trace.get("read"), dict) else {}
    memory = trace.get("memory", {}) if isinstance(trace.get("memory"), dict) else {}
    rep = trace.get("matrix_report", {}) if isinstance(trace.get("matrix_report"), dict) else {}

    boundary = route.get("boundary_by_step", route.get("boundary", []))
    if not isinstance(boundary, list):
        boundary = []
    T = max(3, len(boundary), len(seq.get("sequence_change_by_step", []) or []), len(read.get("read_delta_by_step", []) or []))

    boundary = as_series(boundary, T)
    seq_change = as_series(seq.get("sequence_change_by_step", []), T)
    route_delta = as_series(route.get("route_delta_by_step", []), T)
    read_delta = as_series(read.get("read_delta_by_step", []), T)
    prim_delta = as_series(seq.get("primitive_delta_by_step", []), T)
    update_delta = as_series(seq.get("update_delta_by_step", []), T)
    useful = as_series(route.get("useful_transition_by_step", []), T)
    memory_write = as_series(memory.get("write_by_step", memory.get("memory_write_by_step", [])), T)
    future_read = as_series(memory.get("future_read_by_step", memory.get("memory_future_read_by_step", [])), T)

    # If detailed arrays are missing, fall back to matrix report nested arrays.
    if not any(seq_change):
        seq_change = as_series(get(trace, "sequence", "sequence_change_by_step", default=[]), T)
    if not any(update_delta):
        update_delta = mean_nested_step(get(rep, "update_norm_by_step_lane", default=[]), T)
    if not any(memory_write):
        memory_write = mean_nested_step(get(rep, "write_gate_by_step_lane", default=[]), T)

    last = T - 1
    step_scores = []
    for t in range(T):
        # Weakness: high change but no boundary/useful transition, low update/read/primitive movement,
        # memory write without future read, or last aggregator not clearly active.
        change_need = seq_change[t] + route_delta[t] + read_delta[t] + prim_delta[t] + update_delta[t]
        boundary_bad = max(0.0, change_need - boundary[t])
        route_bad = max(0.0, 0.12 - useful[t])
        memory_bad = max(0.0, memory_write[t] - future_read[t])
        agg_bad = 0.0
        if t == last:
            # Last step should aggregate; if it has low change/useful/memory/head prep, mark it weak.
            agg_bad = max(0.0, 0.10 - (useful[t] + update_delta[t] + read_delta[t]))
        score = 1.7 * boundary_bad + 1.2 * route_bad + 0.7 * memory_bad + 1.0 * agg_bad
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
            },
        })
    return {"T": T, "last_step": last, "step_scores": step_scores}


def propose_growth(scores: Dict[str, Any]) -> List[Dict[str, Any]]:
    T = int(scores["T"])
    last = int(scores["last_step"])
    ranked = sorted(scores["step_scores"], key=lambda r: r["weakness_score"], reverse=True)
    proposals: List[Dict[str, Any]] = []

    # Never insert after the aggregator as default. For classification, last stays aggregation/head-prep.
    for r in ranked[: min(4, len(ranked))]:
        t = int(r["t"])
        sig = r["signals"]
        if t == last:
            proposals.append({
                "action": "insert_before",
                "position": last,
                "new_T": T + 1,
                "stage_type": "pre_aggregation_refine",
                "reason": "last aggregation step is weak; insert a refine/head-prepare step before final aggregator instead of moving aggregator",
                "evidence": r,
                "target_effect": {
                    "route": ["state->abstract", "memory->state"],
                    "primitive": ["ctx_matrix", "product_gate", "low_rank"],
                    "boundary": "soft peak before aggregator",
                },
            })
        elif sig.get("memory_write", 0.0) > sig.get("future_read", 0.0) + 0.05:
            proposals.append({
                "action": "insert_after",
                "position": t,
                "new_T": T + 1,
                "stage_type": "memory_recall_or_cleanup",
                "reason": "memory write appears higher than future read; insert a recall/check step after writer",
                "evidence": r,
                "target_effect": {
                    "route": ["memory->state"],
                    "primitive": ["ctx_matrix", "memory_keep"],
                    "boundary": "memory recall boundary only if contrastive",
                },
            })
        else:
            proposals.append({
                "action": "insert_after",
                "position": t,
                "new_T": T + 1,
                "stage_type": "local_transform_route_refine",
                "reason": "step has high weakness score; insert a transform/route refine stage after it",
                "evidence": r,
                "target_effect": {
                    "route": ["detail->state", "state->abstract"],
                    "primitive": ["diff", "ctx_matrix", "gated_contrast"],
                    "boundary": "sparse peak if after-context changes",
                },
            })

    # Always include a conservative baseline: keep T and strengthen final aggregator.
    proposals.append({
        "action": "keep_T_strengthen_last_aggregator",
        "position": last,
        "new_T": T,
        "stage_type": "aggregation_head_prepare",
        "reason": "classification usually benefits from a final aggregation/head-prepare stage; try strengthening before adding depth",
        "target_effect": {
            "route": ["state->abstract", "memory->state", "abstract->abstract"],
            "primitive": ["ctx_matrix", "product_gate"],
            "boundary": "no new boundary required",
        },
    })
    return proposals


def run(args):
    rd = Path(args.report_dir).resolve()
    epoch, tp = latest_trace(rd)
    trace = read_json(tp, {})
    scores = build_step_scores(trace)
    proposals = propose_growth(scores)
    out = {
        "version": "program_growth_planner_v1",
        "mode": "report_only_growth_decision",
        "report_dir": str(rd),
        "epoch": epoch,
        "skeleton_assumption": {
            "first_step": "evidence/local preparation",
            "middle_steps": "transform/route/memory work",
            "last_step": "aggregation/head preparation for classification",
            "note": "for sequence generation this assumption may change; for SpeechCommands classification it is a sane default",
        },
        "scores": scores,
        "growth_proposals": proposals,
        "recommended_next_command_hint": "rerun with TAPE_STEPS=new_T from top proposal; do not push checkpoints",
    }
    op = rd / f"program_growth_plan_epoch_{epoch:03d}.json"
    write_json(op, out)
    print(f"[growth_planner] wrote {op}")
    for p in proposals[:4]:
        print(f"  {p['action']} pos={p['position']} new_T={p['new_T']} type={p['stage_type']} reason={p['reason']}")


def parser():
    p = argparse.ArgumentParser()
    p.add_argument("--report-dir", required=True)
    return p


if __name__ == "__main__":
    run(parser().parse_args())
