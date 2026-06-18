#!/usr/bin/env python3
"""v4.5 Projection Scout / Global Alternative Council MVP.

Report-only central imagination mechanism for TapeLane v4.4 reports.

It reads the latest v4_4_context_controllers report folder, builds a current
program context, generates several diverse small projected alternative programs,
scores them with a Program Quality Score proxy, and writes controller targets.

No model mutation. No auto-deploy. No actor/critic. No weights.

Output:
    projection_scout_epoch_XXX.json
    scout_experience_memory.jsonl

The goal is to give weak controllers a future auxiliary target/bias source:
    boundary / transition / fanout / route / primitive / memory / head
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple


LANES = ["detail", "state", "abstract", "memory"]
PRIMITIVES = [
    "channel",
    "block",
    "low_rank",
    "ctx_matrix",
    "product_gate",
    "diff",
    "gated_contrast",
    "memory_keep",
]
TRANSITIONS = [
    "identity",
    "detail_to_state",
    "state_to_abstract",
    "state_to_memory",
    "memory_to_state",
    "fork_detail_to_state_memory",
    "join_detail_state_to_abstract",
    "head_prepare",
]
FANOUTS = ["one", "two", "three", "all_soft"]


def _read_json(path: Path, default: Any = None) -> Any:
    try:
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return default


def _write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2, sort_keys=True)
        f.write("\n")


def _append_jsonl(path: Path, records: Iterable[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False, sort_keys=True) + "\n")


def _as_float(x: Any, default: float = 0.0) -> float:
    try:
        if x is None:
            return default
        if isinstance(x, bool):
            return float(x)
        return float(x)
    except Exception:
        return default


def _as_list(x: Any) -> List[Any]:
    return x if isinstance(x, list) else []


def _mean(xs: Sequence[float], default: float = 0.0) -> float:
    xs = [float(x) for x in xs if x is not None and math.isfinite(float(x))]
    return sum(xs) / len(xs) if xs else default


def _std(xs: Sequence[float]) -> float:
    m = _mean(xs)
    return math.sqrt(_mean([(float(x) - m) ** 2 for x in xs], 0.0)) if xs else 0.0


def _softmax(xs: Sequence[float], temp: float = 1.0) -> List[float]:
    if not xs:
        return []
    t = max(float(temp), 1e-6)
    mx = max(xs)
    ex = [math.exp((x - mx) / t) for x in xs]
    s = sum(ex) or 1.0
    return [v / s for v in ex]


def _top_indices(xs: Sequence[float], k: int, min_gap: int = 0) -> List[int]:
    order = sorted(range(len(xs)), key=lambda i: xs[i], reverse=True)
    out: List[int] = []
    for i in order:
        if len(out) >= k:
            break
        if min_gap and any(abs(i - j) < min_gap for j in out):
            continue
        out.append(i)
    return sorted(out)


def _hash_obj(obj: Any, n: int = 16) -> str:
    raw = json.dumps(obj, sort_keys=True, ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()[:n]


def find_latest_report(root: Path, prefix: str = "v4_4_context_controllers_") -> Path:
    reports = root / "simple_butterfly_matrix_v4_tape_lane" / "agent_reports"
    candidates = [p for p in reports.glob(prefix + "*") if p.is_dir()]
    if not candidates:
        raise FileNotFoundError(f"No report dirs found under {reports} matching {prefix}*")
    candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return candidates[0]


def find_latest_trace(report_dir: Path) -> Tuple[Optional[int], Optional[Path]]:
    traces = sorted(report_dir.glob("trace_feedback_epoch_*.json"))
    if not traces:
        return None, None
    def ep(p: Path) -> int:
        s = p.stem.split("_")[-1]
        try:
            return int(s)
        except Exception:
            return -1
    p = max(traces, key=ep)
    return ep(p), p


def read_metrics_last(report_dir: Path) -> Dict[str, Any]:
    path = report_dir / "metrics.csv"
    if not path.exists():
        return {}
    rows: List[Dict[str, str]] = []
    try:
        with path.open("r", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
    except Exception:
        return {}
    if not rows:
        return {}
    row = rows[-1]
    return {k: _as_float(v, v) for k, v in row.items()}


def get_nested(d: Dict[str, Any], *keys: str, default: Any = None) -> Any:
    cur: Any = d
    for k in keys:
        if isinstance(cur, dict) and k in cur:
            cur = cur[k]
        else:
            return default
    return cur


def normalize_step_series(xs: Sequence[float], T: int) -> List[float]:
    out = [float(x) for x in list(xs)[:T]]
    if len(out) < T:
        out.extend([0.0] * (T - len(out)))
    return out


def extract_trace_features(trace: Dict[str, Any]) -> Dict[str, Any]:
    # Support multiple report schemas by probing common keys.
    bstep = get_nested(trace, "boundary_by_step", default=None)
    if bstep is None:
        bstep = get_nested(trace, "boundary", "boundary_by_step", default=[])
    seq = get_nested(trace, "sequence_change_by_step", default=None)
    if seq is None:
        seq = get_nested(trace, "sequence", "sequence_change_by_step", default=[])
    read = get_nested(trace, "read_group_by_step", default=None)
    if read is None:
        read = get_nested(trace, "read", "read_group_by_step", default=[])
    prim = get_nested(trace, "primitive_by_step", default=None)
    if prim is None:
        prim = get_nested(trace, "primitive_weights_by_step", default=[])
    route = get_nested(trace, "route_matrix_by_step", default=None)
    if route is None:
        route = get_nested(trace, "route_by_step", default=[])

    T = max(len(_as_list(bstep)), len(_as_list(seq)), len(_as_list(read)), len(_as_list(prim)), len(_as_list(route)), 12)
    boundary = normalize_step_series([_as_float(x) for x in _as_list(bstep)], T)
    seq_change = normalize_step_series([_as_float(x) for x in _as_list(seq)], T)

    return {
        "T": T,
        "boundary_by_step": boundary,
        "sequence_change_by_step": seq_change,
        "read_group_by_step": _as_list(read),
        "primitive_by_step": _as_list(prim),
        "route_by_step": _as_list(route),
    }


def summarize_primitives(prim_by_step: List[Any], T: int) -> Dict[str, Any]:
    # prim can be [T][L][P] or [T][P] or absent.
    per_step: List[Dict[str, float]] = []
    totals = {p: 0.0 for p in PRIMITIVES}
    count = 0
    for t in range(T):
        raw = prim_by_step[t] if t < len(prim_by_step) else []
        vals = [0.0] * len(PRIMITIVES)
        if isinstance(raw, list) and raw:
            if raw and isinstance(raw[0], list):
                # average over lanes
                lanes = raw
                for lane in lanes:
                    if isinstance(lane, list):
                        for i in range(min(len(PRIMITIVES), len(lane))):
                            vals[i] += _as_float(lane[i])
                denom = max(1, sum(1 for lane in lanes if isinstance(lane, list)))
                vals = [v / denom for v in vals]
            else:
                for i in range(min(len(PRIMITIVES), len(raw))):
                    vals[i] = _as_float(raw[i])
        s = sum(vals)
        if s > 0:
            vals = [v / s for v in vals]
        d = {PRIMITIVES[i]: vals[i] for i in range(len(PRIMITIVES))}
        per_step.append(d)
        for p, v in d.items():
            totals[p] += v
        count += 1
    avg = {p: totals[p] / max(1, count) for p in PRIMITIVES}
    dominant = max(avg, key=avg.get) if avg else None
    entropy = -sum(v * math.log(max(v, 1e-9)) for v in avg.values()) if avg else 0.0
    return {"avg": avg, "dominant": dominant, "entropy": entropy, "by_step": per_step}


def route_stats(route_by_step: List[Any]) -> Dict[str, Any]:
    entropies: List[float] = []
    allowed_masses: List[float] = []
    useful_masses: List[float] = []
    self_masses: List[float] = []
    for mat in route_by_step:
        if not isinstance(mat, list) or len(mat) < 4:
            continue
        # assume [from][to]
        rows = mat[:4]
        for r in rows:
            if not isinstance(r, list) or not r:
                continue
            vals = [_as_float(x) for x in r[:4]]
            s = sum(vals) or 1.0
            vals = [v / s for v in vals]
            entropies.append(-sum(v * math.log(max(v, 1e-9)) for v in vals))
        # allowed: self + detail->state + detail->memory + state->abstract + state->memory + memory->state
        def v(i: int, j: int) -> float:
            try:
                row = [_as_float(x) for x in rows[i][:4]]
                s = sum(row) or 1.0
                return row[j] / s
            except Exception:
                return 0.0
        self_mass = sum(v(i, i) for i in range(4)) / 4.0
        useful = (v(0, 1) + v(0, 3) + v(1, 2) + v(1, 3) + v(3, 1)) / 5.0
        allowed = (sum(v(i, i) for i in range(4)) + v(0, 1) + v(0, 3) + v(1, 2) + v(1, 3) + v(3, 1)) / 9.0
        self_masses.append(self_mass)
        useful_masses.append(useful)
        allowed_masses.append(allowed)
    return {
        "route_entropy_mean": _mean(entropies, 0.0),
        "allowed_route_mass_proxy": _mean(allowed_masses, 0.0),
        "self_route_mass_proxy": _mean(self_masses, 0.0),
        "useful_transition_mass_proxy": _mean(useful_masses, 0.0),
    }


def build_current_context(final_report: Dict[str, Any], trace: Dict[str, Any], metrics: Dict[str, Any], report_text: str) -> Dict[str, Any]:
    features = extract_trace_features(trace)
    T = features["T"]
    boundary = features["boundary_by_step"]
    seq = features["sequence_change_by_step"]
    prim_summary = summarize_primitives(features["primitive_by_step"], T)
    rstats = route_stats(features["route_by_step"])

    # Pull known scalar metrics from report/trace/metrics with fallbacks.
    collapse_flags = get_nested(trace, "collapse_flags", default=[])
    if not collapse_flags:
        collapse_flags = get_nested(final_report, "collapse_flags", default=[])
    if isinstance(collapse_flags, str):
        collapse_flags = [collapse_flags]

    val_acc = _as_float(metrics.get("val_acc", metrics.get("val", 0.0)), 0.0)
    if val_acc > 1.0:
        val_acc /= 100.0

    ctx = {
        "T": T,
        "epoch": metrics.get("epoch", None),
        "val_acc": val_acc,
        "boundary_mean": _mean(boundary, 0.0),
        "boundary_std": _std(boundary),
        "boundary_peak_count_proxy": sum(1 for x in boundary if x > 0.55),
        "boundary_state": "dead_flat" if _std(boundary) < 0.015 and _mean(boundary, 0.0) < 0.45 else ("exploit" if _mean(boundary, 0.0) > 0.8 else "active"),
        "sequence_change_by_step": seq,
        "sequence_nonflat_score": _mean(seq, 0.0),
        "sequence_peak_steps": _top_indices(seq, k=min(4, T), min_gap=1),
        "primitive": prim_summary,
        "route": rstats,
        "collapse_flags": collapse_flags,
        "report_text_flags": [line.strip() for line in report_text.splitlines() if "BOUNDARY" in line or "ROUTE" in line or "MEMORY" in line or "DETAIL" in line][:20],
    }
    ctx["context_hash"] = _hash_obj({k: ctx[k] for k in ["boundary_state", "sequence_peak_steps", "primitive", "route", "collapse_flags"]})
    return ctx


def boundary_target_from_steps(T: int, steps: Sequence[int], mass: float = 1.0) -> List[float]:
    out = [0.0] * T
    for s in steps:
        if 0 <= s < T:
            out[s] = mass
    return out


def score_alternative(ctx: Dict[str, Any], alt: Dict[str, Any]) -> Dict[str, float]:
    boundary_target = alt.get("boundary_target", [])
    n_peaks = sum(1 for x in boundary_target if _as_float(x) > 0.5)
    seq = ctx.get("sequence_change_by_step", [])
    peak_steps = [i for i, x in enumerate(boundary_target) if _as_float(x) > 0.5]
    seq_match = _mean([seq[i] if i < len(seq) else 0.0 for i in peak_steps], 0.0)

    transition_targets = alt.get("transition_targets", {}) or {}
    fanout_targets = alt.get("fanout_targets", {}) or {}
    primitive_biases = alt.get("primitive_biases", {}) or {}
    memory_targets = alt.get("memory_targets", {}) or {}
    head_targets = alt.get("head_targets", {}) or {}

    boundary_score = min(1.0, max(0.0, n_peaks / 3.0)) * 0.25 + min(1.0, seq_match * 20.0) * 0.25
    transition_score = min(1.0, len(transition_targets) / 3.0) * 0.18
    fanout_score = min(1.0, len(fanout_targets) / 3.0) * 0.08
    primitive_score = min(1.0, len(primitive_biases) / 4.0) * 0.12
    memory_score = (0.10 if memory_targets else 0.0)
    head_score = (0.08 if head_targets else 0.0)

    route = ctx.get("route", {})
    route_entropy = _as_float(route.get("route_entropy_mean"), 1.386)
    route_specialization_gain = max(0.0, min(0.12, (1.386 - route_entropy) * 0.12))

    # Penalize too many edits and exact duplicates later with diversity.
    complexity = 0.01 * (n_peaks + len(transition_targets) + len(fanout_targets) + len(primitive_biases))
    pqs = boundary_score + transition_score + fanout_score + primitive_score + memory_score + head_score + route_specialization_gain - complexity
    task_proxy = min(0.05, max(0.0, pqs * 0.08))
    risk = "low" if complexity < 0.06 else ("medium" if complexity < 0.11 else "high")
    return {
        "program_quality_score": round(pqs, 6),
        "predicted_gain": round(task_proxy + pqs * 0.03, 6),
        "complexity_cost": round(complexity, 6),
        "risk_numeric": {"low": 0.2, "medium": 0.5, "high": 0.8}[risk],
    }


def action_signature(alt: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "boundary": [i for i, v in enumerate(alt.get("boundary_target", [])) if _as_float(v) > 0.5],
        "transition": alt.get("transition_targets", {}),
        "fanout": alt.get("fanout_targets", {}),
        "primitive": alt.get("primitive_biases", {}),
        "memory": alt.get("memory_targets", {}),
        "head": alt.get("head_targets", {}),
    }


def signature_similarity(a: Dict[str, Any], b: Dict[str, Any]) -> float:
    sa = json.dumps(action_signature(a), sort_keys=True)
    sb = json.dumps(action_signature(b), sort_keys=True)
    set_a = set(sa.replace('{', ' ').replace('}', ' ').replace(',', ' ').split())
    set_b = set(sb.replace('{', ' ').replace('}', ' ').replace(',', ' ').split())
    if not set_a and not set_b:
        return 1.0
    return len(set_a & set_b) / max(1, len(set_a | set_b))


def diversify(alts: List[Dict[str, Any]], keep: int = 8) -> List[Dict[str, Any]]:
    alts = sorted(alts, key=lambda x: x.get("predicted_gain", 0.0), reverse=True)
    selected: List[Dict[str, Any]] = []
    for alt in alts:
        if not selected:
            alt["diversity_score"] = 1.0
            selected.append(alt)
            continue
        sim = max(signature_similarity(alt, s) for s in selected)
        alt["diversity_score"] = round(1.0 - sim, 6)
        alt["predicted_gain"] = round(_as_float(alt.get("predicted_gain")) - 0.02 * sim, 6)
        if sim < 0.82 or len(selected) < 3:
            selected.append(alt)
        if len(selected) >= keep:
            break
    return sorted(selected, key=lambda x: x.get("predicted_gain", 0.0), reverse=True)


def build_alternatives(ctx: Dict[str, Any]) -> List[Dict[str, Any]]:
    T = int(ctx.get("T", 12))
    seq = ctx.get("sequence_change_by_step", [0.0] * T)
    peaks3 = _top_indices(seq, k=min(3, T), min_gap=2)
    if len(peaks3) < 3:
        peaks3 = sorted(set(peaks3 + [max(1, T // 4), max(2, T // 2), max(3, (3 * T) // 4)]))[:3]
    peaks2 = peaks3[:2]
    late = min(T - 1, max(peaks3[-1] + 1, (3 * T) // 4)) if peaks3 else T - 2

    dominant = ctx.get("primitive", {}).get("dominant", "gated_contrast")
    weak_prim = "diff" if dominant == "gated_contrast" else "gated_contrast"

    base_current = {
        "boundary_state": ctx.get("boundary_state"),
        "sequence_peak_steps": ctx.get("sequence_peak_steps"),
        "route": ctx.get("route"),
        "primitive_dominant": dominant,
        "collapse_flags": ctx.get("collapse_flags"),
    }

    alts: List[Dict[str, Any]] = []

    def add(alt: Dict[str, Any]) -> None:
        alt["current_context"] = base_current
        alt["action_vector"] = action_signature(alt)
        scores = score_alternative(ctx, alt)
        alt.update(scores)
        alt["risk"] = "low" if scores["risk_numeric"] <= 0.25 else ("medium" if scores["risk_numeric"] <= 0.55 else "high")
        alt.pop("risk_numeric", None)
        alt["deploy"] = False
        alt["id"] = alt.get("id") or _hash_obj(alt["action_vector"], 10)
        alts.append(alt)

    # A: explicit stage program: fork + split + memory recall.
    add({
        "id": "alt_boundary_fork_memory_recall",
        "source_scouts": ["boundary", "transition", "memory"],
        "boundary_target": boundary_target_from_steps(T, peaks3, 1.0),
        "transition_targets": {str(peaks3[0]): "fork_detail_to_state_memory", str(peaks3[1]): "state_to_abstract", str(peaks3[-1]): "memory_to_state"},
        "fanout_targets": {str(peaks3[0]): "two", str(peaks3[1]): "one", str(peaks3[-1]): "one"},
        "route_biases": {str(peaks3[0]): {"detail->state": 0.25, "detail->memory": 0.25}, str(peaks3[-1]): {"memory->state": 0.30}},
        "primitive_biases": {str(peaks3[0]): {"detail": {"memory_keep": 0.15}}, str(peaks3[1]): {"state": {"ctx_matrix": 0.15}}},
        "memory_targets": {"write_at": peaks3[0], "recall_at": peaks3[-1]},
        "head_targets": {"late_non_detail_read": True},
        "target_context": {"program_shape": "early fork, mid abstract split, late memory recall", "desired_boundary_peaks": peaks3},
        "reason": "Boundary is weak/flat; test a full fork->split->recall projected program around high sequence-change steps.",
    })

    # B: local detail refinement without relying on boundary.
    add({
        "id": "alt_local_detail_refinement_no_boundary",
        "source_scouts": ["primitive", "route"],
        "boundary_target": boundary_target_from_steps(T, [], 0.0),
        "transition_targets": {str(p): "identity" for p in peaks2},
        "fanout_targets": {str(p): "one" for p in peaks2},
        "route_biases": {str(p): {"detail->detail": 0.25, "state->state": 0.20} for p in peaks2},
        "primitive_biases": {str(p): {"detail": {"diff": 0.12, "ctx_matrix": 0.10}} for p in peaks2},
        "memory_targets": {},
        "head_targets": {"reduce_detail_topread": True, "increase_state_abstract_mass": True},
        "target_context": {"program_shape": "local refinement only, no new boundary", "desired_route": "sharper self/detail-state"},
        "reason": "If boundary remains dead, still improve by sharpening local detail refinement and reducing head shortcut.",
    })

    # C: abstract path.
    split = peaks3[1] if len(peaks3) > 1 else max(1, T // 2)
    add({
        "id": "alt_state_to_abstract_head_prepare",
        "source_scouts": ["transition", "head"],
        "boundary_target": boundary_target_from_steps(T, [split, late], 1.0),
        "transition_targets": {str(split): "state_to_abstract", str(late): "head_prepare"},
        "fanout_targets": {str(split): "one", str(late): "two"},
        "route_biases": {str(split): {"state->abstract": 0.35}, str(late): {"abstract->abstract": 0.15, "memory->abstract": 0.10}},
        "primitive_biases": {str(split): {"state": {"low_rank": 0.12, "ctx_matrix": 0.14}}, str(late): {"abstract": {"product_gate": 0.12}}},
        "memory_targets": {},
        "head_targets": {"increase_abstract_read": True},
        "target_context": {"program_shape": "state abstraction then head prepare", "desired_boundary_peaks": [split, late]},
        "reason": "Build an abstract lane path so class head has a non-detail source.",
    })

    # D: memory junk correction.
    mstep = peaks3[0] if peaks3 else max(1, T // 3)
    rstep = peaks3[-1] if peaks3 else max(2, (2 * T) // 3)
    add({
        "id": "alt_memory_write_recall_alignment",
        "source_scouts": ["memory", "route"],
        "boundary_target": boundary_target_from_steps(T, [mstep, rstep], 1.0),
        "transition_targets": {str(mstep): "state_to_memory", str(rstep): "memory_to_state"},
        "fanout_targets": {str(mstep): "one", str(rstep): "one"},
        "route_biases": {str(mstep): {"state->memory": 0.30}, str(rstep): {"memory->state": 0.35}},
        "primitive_biases": {str(mstep): {"state": {"memory_keep": 0.18}}, str(rstep): {"memory": {"ctx_matrix": 0.12}}},
        "memory_targets": {"write_at": mstep, "recall_at": rstep, "avoid_write_without_future_read": True},
        "head_targets": {"increase_memory_consumer": True},
        "target_context": {"program_shape": "causal memory write then recall", "desired_memory_alignment": "write followed by future read/head consumer"},
        "reason": "Test whether high memory write becomes useful if paired with explicit later recall.",
    })

    # E: dormant primitive wake-up.
    wake = peaks3[0] if peaks3 else 0
    add({
        "id": "alt_dormant_primitive_wakeup",
        "source_scouts": ["exploration", "primitive"],
        "boundary_target": boundary_target_from_steps(T, [wake], 0.6),
        "transition_targets": {str(wake): "detail_to_state"},
        "fanout_targets": {str(wake): "two"},
        "route_biases": {str(wake): {"detail->state": 0.20}},
        "primitive_biases": {str(wake): {"detail": {weak_prim: 0.20, "block": 0.08}}},
        "memory_targets": {},
        "head_targets": {},
        "target_context": {"program_shape": "small wake-up test", "woken_primitive": weak_prim},
        "reason": "Controlled exploration: wake a less-used primitive in a high-change region.",
    })

    # F: route specialization only.
    add({
        "id": "alt_route_specialization_allowed_edges",
        "source_scouts": ["route"],
        "boundary_target": boundary_target_from_steps(T, [], 0.0),
        "transition_targets": {},
        "fanout_targets": {str(p): "one" for p in peaks3},
        "route_biases": {str(p): {"detail->state": 0.10, "state->abstract": 0.10, "memory->state": 0.10} for p in peaks3},
        "primitive_biases": {},
        "memory_targets": {},
        "head_targets": {},
        "target_context": {"program_shape": "route specialization without boundary", "desired_route": "allowed_edges_up"},
        "reason": "If boundaries remain unstable, improve allowed route edges directly.",
    })

    return diversify(alts, keep=8)


def build_selected_targets(alternatives: List[Dict[str, Any]], top_k: int = 2) -> Dict[str, Any]:
    selected = alternatives[:top_k]
    boundary: Dict[str, float] = {}
    transition: Dict[str, Dict[str, float]] = {}
    fanout: Dict[str, Dict[str, float]] = {}
    route: Dict[str, Dict[str, float]] = {}
    primitive: Dict[str, Any] = {}
    memory: Dict[str, Any] = {}
    head: Dict[str, Any] = {}

    for rank, alt in enumerate(selected):
        w = 1.0 / (rank + 1)
        for t, v in enumerate(alt.get("boundary_target", [])):
            if _as_float(v) > 0:
                boundary[str(t)] = max(boundary.get(str(t), 0.0), round(_as_float(v) * w, 4))
        for t, typ in (alt.get("transition_targets", {}) or {}).items():
            transition.setdefault(str(t), {})[typ] = max(transition.get(str(t), {}).get(typ, 0.0), round(w, 4))
        for t, typ in (alt.get("fanout_targets", {}) or {}).items():
            fanout.setdefault(str(t), {})[typ] = max(fanout.get(str(t), {}).get(typ, 0.0), round(w, 4))
        for t, edges in (alt.get("route_biases", {}) or {}).items():
            route.setdefault(str(t), {})
            for e, v in edges.items():
                route[str(t)][e] = round(route[str(t)].get(e, 0.0) + _as_float(v) * w, 4)
        for t, lanes in (alt.get("primitive_biases", {}) or {}).items():
            primitive.setdefault(str(t), {})
            if isinstance(lanes, dict):
                for lane, ps in lanes.items():
                    primitive[str(t)].setdefault(lane, {})
                    if isinstance(ps, dict):
                        for p, v in ps.items():
                            primitive[str(t)][lane][p] = round(primitive[str(t)][lane].get(p, 0.0) + _as_float(v) * w, 4)
        memory.update(alt.get("memory_targets", {}) or {})
        head.update(alt.get("head_targets", {}) or {})

    return {
        "top_alternative_ids": [a["id"] for a in selected],
        "boundary": boundary,
        "transition": transition,
        "fanout": fanout,
        "route": route,
        "primitive": primitive,
        "memory": memory,
        "head": head,
        "runtime_feedback_spec": {
            "mode": "report_only_now_aux_loss_later",
            "boundary": "BCE/soft target over boundary logits by step",
            "transition": "CE/soft target over transition_type_controller logits",
            "fanout": "CE/soft target over fanout_controller logits",
            "route": "small detached route edge bias or auxiliary edge target",
            "primitive": "small primitive_controller target bias per step/lane/primitive",
            "memory": "future memory controller target when implemented",
            "head": "head read target when implemented",
            "max_scout_bias": 0.03,
            "deploy": False,
        },
    }


def load_report_text(report_dir: Path) -> str:
    p = report_dir / "REPORT_TO_CHATGPT.txt"
    try:
        return p.read_text(encoding="utf-8", errors="replace")
    except Exception:
        return ""


def run(args: argparse.Namespace) -> Path:
    root = Path(args.repo_root).resolve()
    report_dir = Path(args.report_dir).resolve() if args.report_dir else find_latest_report(root)
    epoch, trace_path = find_latest_trace(report_dir)
    if trace_path is None:
        raise FileNotFoundError(f"No trace_feedback_epoch_*.json in {report_dir}")

    final_report = _read_json(report_dir / "final_report.json", {}) or {}
    trace = _read_json(trace_path, {}) or {}
    metrics = read_metrics_last(report_dir)
    report_text = load_report_text(report_dir)
    ctx = build_current_context(final_report, trace, metrics, report_text)
    if epoch is None:
        epoch = int(_as_float(metrics.get("epoch"), 0))
    ctx["epoch"] = epoch

    alternatives = build_alternatives(ctx)
    selected_targets = build_selected_targets(alternatives, top_k=args.top_k)

    out = {
        "version": "v4.5_projection_scout_v1_report_only",
        "mode": "report_only",
        "deploy": False,
        "report_dir": str(report_dir),
        "epoch": epoch,
        "current_context": ctx,
        "alternatives": alternatives,
        "selected_targets": selected_targets,
        "notes": [
            "This file does not mutate the model.",
            "Use selected_targets later as tiny auxiliary losses or detached scout bias after report quality is validated.",
            "Alternatives include target_context, not only local edits from current context.",
        ],
    }

    out_path = Path(args.output).resolve() if args.output else report_dir / f"projection_scout_epoch_{epoch:03d}.json"
    _write_json(out_path, out)

    mem_path = Path(args.memory).resolve() if args.memory else report_dir / "scout_experience_memory.jsonl"
    records = []
    for alt in alternatives:
        records.append({
            "run_id": report_dir.name,
            "epoch": epoch,
            "alternative_id": alt["id"],
            "current_context_hash": ctx.get("context_hash"),
            "current_context_summary": alt.get("current_context"),
            "target_context_summary": alt.get("target_context"),
            "action_vector": alt.get("action_vector"),
            "predicted_gain": alt.get("predicted_gain"),
            "predicted_pqs_gain": alt.get("program_quality_score"),
            "diversity_score": alt.get("diversity_score"),
            "risk": alt.get("risk"),
            "verification_status": "not_verified",
            "observed_gain": None,
            "observed_pqs_gain": None,
            "cooldown": 0,
            "deploy": False,
        })
    _append_jsonl(mem_path, records)
    print(f"[projection_scout] report_dir={report_dir}")
    print(f"[projection_scout] wrote {out_path}")
    print(f"[projection_scout] appended {len(records)} records to {mem_path}")
    print("[projection_scout] top alternatives:")
    for a in alternatives[: min(5, len(alternatives))]:
        print(f"  {a['id']}: gain={a.get('predicted_gain')} pqs={a.get('program_quality_score')} risk={a.get('risk')} reason={a.get('reason')}")
    return out_path


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="v4.5 Projection Scout / Global Alternative Council report-only MVP")
    p.add_argument("--repo-root", default=".", help="Repository root, default current directory")
    p.add_argument("--report-dir", default=None, help="Specific v4_4 report directory; default latest")
    p.add_argument("--output", default=None, help="Output JSON path; default inside report dir")
    p.add_argument("--memory", default=None, help="Experience memory JSONL path; default inside report dir")
    p.add_argument("--top-k", type=int, default=2, help="How many top alternatives to merge into selected_targets")
    return p


if __name__ == "__main__":
    run(parser().parse_args())
