#!/usr/bin/env python3
"""Idempotently patch tape_lane_transport_v4_3_min_heart.py to canonical v4.3 logic.

This is a migration helper while the file is still changing fast. It edits the main file in-place so the runtime entrypoint is the main Python module, not a wrapper.
"""
from __future__ import annotations

from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[2]
TARGET = ROOT / "simple_butterfly_matrix_v4_tape_lane" / "tape_lane_transport_v4_3_min_heart.py"


def replace_block(text: str, start_pat: str, end_pat: str, new_block: str, label: str) -> str:
    m = re.search(start_pat, text)
    if not m:
        raise SystemExit(f"missing start block: {label}")
    n = re.search(end_pat, text[m.start():])
    if not n:
        raise SystemExit(f"missing end block: {label}")
    end = m.start() + n.start()
    return text[:m.start()] + new_block.rstrip() + "\n\n\n" + text[end:]


def main() -> int:
    text = TARGET.read_text(encoding="utf-8")
    original = text

    helper = r'''
def _route_extra_terms_from_routes(routes: torch.Tensor, lanes: int) -> Dict[str, torch.Tensor]:
    if routes.numel() == 0:
        z = torch.zeros((), device=routes.device if hasattr(routes, "device") else "cpu")
        return {
            "self_route_mass": z,
            "useful_transition_mass": z,
            "self_route_by_step": torch.empty(0, device=z.device),
            "useful_transition_by_step": torch.empty(0, device=z.device),
        }
    eye = torch.eye(routes.shape[-1], dtype=routes.dtype, device=routes.device)
    self_by_step = (routes * eye.view(1, routes.shape[-1], routes.shape[-1])).sum(dim=(-2, -1)) / float(max(1, lanes))
    useful_parts = []
    if lanes >= 2:
        useful_parts.append(routes[:, 0, 1])  # detail -> state
    if lanes >= 3:
        useful_parts.append(routes[:, 1, 2])  # state -> abstract
    if lanes >= 4:
        useful_parts.append(routes[:, 1, 3])  # state -> memory
        useful_parts.append(routes[:, 3, 1])  # memory -> state
    useful_by_step = torch.stack(useful_parts, dim=0).mean(dim=0) if useful_parts else torch.zeros(routes.shape[0], dtype=routes.dtype, device=routes.device)
    return {
        "self_route_mass": self_by_step.mean(),
        "useful_transition_mass": useful_by_step.mean(),
        "self_route_by_step": self_by_step,
        "useful_transition_by_step": useful_by_step,
    }
'''
    if "def _route_extra_terms_from_routes" not in text:
        text = text.replace("\n\ndef late_input_cost_from_read", helper + "\n\ndef late_input_cost_from_read", 1)

    new_sequence = r'''def sequence_terms(baux, args) -> Dict[str, torch.Tensor]:
    routes = baux.routes.float()
    prim = baux.primitive_weights.float()
    upd = baux.update_norms.float()
    read = baux.read_group_mass.float()
    device = baux.slots.device
    if routes.numel() and routes.shape[0] > 1:
        route_delta_mean = (routes[1:] - routes[:-1]).abs().mean(dim=(-2, -1)).mean()
    else:
        route_delta_mean = torch.zeros((), device=device)
    if prim.numel() and prim.shape[0] > 1:
        prim_delta_mean = (prim[1:] - prim[:-1]).abs().mean(dim=(-2, -1)).mean()
    else:
        prim_delta_mean = torch.zeros((), device=device)
    if upd.numel() and upd.shape[1] > 1:
        u = upd.mean(dim=(0, 3))  # [T,L]
        upd_delta_mean = (u[1:] - u[:-1]).abs().mean()
    else:
        upd_delta_mean = torch.zeros((), device=device)
    if read.numel() and read.shape[0] > 1:
        read_delta_mean = (read[1:] - read[:-1]).abs().mean()
    else:
        read_delta_mean = torch.zeros((), device=device)
    route_extra = _route_extra_terms_from_routes(routes, int(args.lanes))
    return {
        "sequence_route_delta_mean": route_delta_mean,
        "sequence_primitive_delta_mean": prim_delta_mean,
        "sequence_update_delta_mean": upd_delta_mean,
        "sequence_read_delta_mean": read_delta_mean,
        "sequence_nonflat_score": route_delta_mean + prim_delta_mean + upd_delta_mean + read_delta_mean,
        "self_route_mass": route_extra["self_route_mass"].to(device),
        "useful_transition_mass": route_extra["useful_transition_mass"].to(device),
    }'''
    text = replace_block(text, r"def sequence_terms\(baux, args\).*?\n", r"def aux_losses_v43\(", new_sequence, "sequence_terms")

    text = text.replace('out["residual_dominance_proxy"] = residual_proxy.detach()', 'out["update_collapse_proxy"] = residual_proxy.detach()\n    out["residual_dominance_proxy"] = residual_proxy.detach()  # deprecated alias')

    text = text.replace('loss = loss + args.lambda_memory_write_cost * losses["memory_write_cost"]\n            loss = loss + args.lambda_memory_overwrite * losses["memory_overwrite_cost"]', 'memory_warmup = max(1, int(getattr(args, "memory_write_warmup_epochs", 2)))\n            memory_scale = min(1.0, float(epoch) / float(memory_warmup))\n            loss = loss + (args.lambda_memory_write_cost * memory_scale) * losses["memory_write_cost"]\n            loss = loss + (args.lambda_memory_overwrite * memory_scale) * losses["memory_overwrite_cost"]')

    new_trace = r'''def build_trace_feedback(rep: Dict, args, epoch: int) -> Dict:
    routes = _tensor_from_list(rep.get("route_matrix", []))
    boundary = _tensor_from_list(rep.get("boundary", []))
    read_group = _tensor_from_list(rep.get("read_group_mass", []))
    primitive = _tensor_from_list(rep.get("primitive_weights", []))
    updates = _tensor_from_list(rep.get("update_norm_by_step_lane", []))
    gates = _tensor_from_list(rep.get("write_gate_by_step_lane", []))
    lanes = int(args.lanes)
    mem_lane = min(lanes - 1, 3)

    if routes.numel():
        eye = torch.eye(routes.shape[-1], dtype=routes.dtype)
        offdiag = routes * (1.0 - eye.view(1, routes.shape[-1], routes.shape[-1]))
        offdiag_raw = offdiag.sum(dim=(-2, -1))
        offdiag_norm = offdiag.sum(dim=-1).mean(dim=-1).clamp(0.0, 1.0)
        route_extra = _route_extra_terms_from_routes(routes, lanes)
        self_route_by_step = route_extra["self_route_by_step"]
        useful_transition_by_step = route_extra["useful_transition_by_step"]
        self_route_mass = float(route_extra["self_route_mass"])
        useful_transition_mass = float(route_extra["useful_transition_mass"])
    else:
        offdiag_raw = torch.empty(0)
        offdiag_norm = torch.empty(0)
        self_route_by_step = torch.empty(0)
        useful_transition_by_step = torch.empty(0)
        self_route_mass = 0.0
        useful_transition_mass = 0.0
    if boundary.numel() and offdiag_norm.numel():
        inside = offdiag_norm * boundary
        outside = offdiag_norm * (1.0 - boundary)
    else:
        inside = torch.empty(0)
        outside = torch.empty(0)

    def step_delta(x: torch.Tensor) -> torch.Tensor:
        if x.numel() == 0 or x.shape[0] <= 1:
            return torch.zeros(max(1, int(args.tape_steps)))[: int(args.tape_steps)]
        d = (x[1:] - x[:-1]).abs()
        while d.dim() > 1:
            d = d.mean(dim=-1)
        return torch.cat([d, torch.zeros(1, dtype=d.dtype)], dim=0)

    route_delta = step_delta(routes)
    primitive_delta = step_delta(primitive)
    read_delta = step_delta(read_group)
    trace_delta = step_delta(updates) if updates.numel() else torch.zeros_like(route_delta)
    common_len = min(len(route_delta), len(primitive_delta), len(trace_delta), len(read_delta), len(boundary)) if boundary.numel() else 0
    if common_len:
        seq_change = route_delta[:common_len] + primitive_delta[:common_len] + trace_delta[:common_len] + read_delta[:common_len]
        usefulness = boundary[:common_len] * seq_change
    else:
        usefulness = torch.empty(0)
        seq_change = torch.empty(0)

    late_by_step = []
    if read_group.numel():
        input_mass_by_step = read_group[:, :, 0].mean(dim=-1)
        progress = torch.arange(input_mass_by_step.numel(), dtype=torch.float32) / float(max(1, input_mass_by_step.numel() - 1))
        depth_weight = torch.sigmoid((progress - float(args.late_input_start)) / max(1e-6, float(args.late_input_tau)))
        late_by_step = (input_mass_by_step * depth_weight).tolist()
        input_total = float(input_mass_by_step.mean())
        memory_future_read = float(read_group[:, :, 1 + mem_lane].mean()) if read_group.shape[-1] > 1 + mem_lane else 0.0
    else:
        input_total = 0.0
        memory_future_read = 0.0

    memory_write = float(gates[:, mem_lane].mean()) if gates.numel() and gates.shape[-1] > mem_lane else 0.0
    memory_overwrite = max(0.0, memory_write - float(args.memory_write_target)) ** 2
    lane_mass_mean = rep.get("lane_mass_mean", {}) or {}
    head_memory = _safe_float(lane_mass_mean.get(lane_name(mem_lane), 0.0))
    detail_attention = _safe_float(lane_mass_mean.get(lane_name(0), 0.0))
    detail_topread = _detail_topread_share(rep.get("class_top_reads", []))
    detail_cost = max(0.0, detail_attention - float(args.detail_head_shortcut_target)) ** 2

    boundary_list = [float(v) for v in boundary.tolist()] if boundary.numel() else []
    peak_thr = float(args.boundary_peak_threshold)
    peaks = [i for i, v in enumerate(boundary_list) if v >= peak_thr]
    boundary_mean = sum(boundary_list) / max(1, len(boundary_list))
    boundary_flatness = (sum((v - boundary_mean) ** 2 for v in boundary_list) / max(1, len(boundary_list))) ** 0.5 if boundary_list else 0.0
    primitive_weights = rep.get("primitive_weights", [])
    seq_nonflat = float(seq_change.mean()) if seq_change.numel() else 0.0
    memory_consumer_proxy = memory_future_read + head_memory

    flags = []
    entropy_mean = float(_tensor_from_list(rep.get("route_entropy", [])).mean()) if rep.get("route_entropy") else 0.0
    if boundary_mean > 0.90 and boundary_flatness < 0.05:
        flags.append("BOUNDARY_EXPLOIT")
    if boundary_mean < 0.05 or len(peaks) == 0:
        flags.append("BOUNDARY_DEAD")
    if entropy_mean > 1.30:
        flags.append("ROUTE_UNIFORM")
    if self_route_mass > 0.88 and useful_transition_mass < 0.12:
        flags.append("ROUTE_IDENTITY_COLLAPSE")
    if detail_attention > 0.55:
        flags.append("DETAIL_SHORTCUT")
    if memory_write < 0.03 and memory_consumer_proxy < 0.08:
        flags.append("MEMORY_DEAD")
    if memory_write > 0.30 and memory_consumer_proxy < 0.08:
        flags.append("MEMORY_JUNK")

    return {
        "epoch": int(epoch),
        "window": {"type": "epoch", "index": int(epoch)},
        "compare_to": str(getattr(args, "compare_to", "baseline_missing")),
        "version": "v4.3_min_heart_canonical",
        "collapse_flags": flags,
        "route": {
            "entropy_mean": entropy_mean,
            "matrix_by_step": rep.get("route_matrix", []),
            "offdiag_raw": offdiag_raw.tolist() if offdiag_raw.numel() else [],
            "offdiag_norm": offdiag_norm.tolist() if offdiag_norm.numel() else [],
            "offdiag_inside_boundary": inside.tolist() if inside.numel() else [],
            "offdiag_outside_boundary": outside.tolist() if outside.numel() else [],
            "offdiag_outside_boundary_cost": float(outside.mean()) if outside.numel() else 0.0,
            "self_route_by_step": self_route_by_step.tolist() if self_route_by_step.numel() else [],
            "useful_transition_by_step": useful_transition_by_step.tolist() if useful_transition_by_step.numel() else [],
            "self_route_mass": self_route_mass,
            "useful_transition_mass": useful_transition_mass,
            "boundary_by_step": boundary_list,
            "boundary_mean": boundary_mean,
            "boundary_flatness": boundary_flatness,
            "boundary_peak_count": len(peaks),
            "boundary_peaks": peaks,
            "boundary_usefulness": usefulness.tolist() if usefulness.numel() else [],
        },
        "sequence": {
            "route_delta_by_step": route_delta.tolist()[: int(args.tape_steps)] if route_delta.numel() else [],
            "primitive_delta_by_step": primitive_delta.tolist()[: int(args.tape_steps)] if primitive_delta.numel() else [],
            "trace_delta_by_step": trace_delta.tolist()[: int(args.tape_steps)] if trace_delta.numel() else [],
            "read_delta_by_step": read_delta.tolist()[: int(args.tape_steps)] if read_delta.numel() else [],
            "sequence_change_by_step": seq_change.tolist() if seq_change.numel() else [],
            "sequence_nonflat_score": seq_nonflat,
        },
        "read": {"late_input_by_step": [float(v) for v in late_by_step], "late_input_cost": float(sum(late_by_step) / max(1, len(late_by_step))) if late_by_step else 0.0, "input_read_total": input_total},
        "memory": {"write_mean": memory_write, "write_cost": memory_write, "overwrite_score": memory_overwrite, "future_read": memory_future_read, "head_consumer": head_memory, "consumer_score": memory_consumer_proxy, "memory_consumer_proxy": memory_consumer_proxy},
        "head": {"detail_attention_mass": detail_attention, "detail_head_shortcut_cost": detail_cost, "detail_topread_share": detail_topread, "class_lane_mass": rep.get("class_lane_mass", []), "lane_mass_mean": lane_mass_mean, "top_reads": rep.get("class_top_reads", [])},
        "operators": {"primitive_weights": primitive_weights, "primitive_names": rep.get("primitive_names", list(PRIMITIVES))},
        "budget": {"skip_gate_mean": 0.0, "skip_cost": "not_implemented", "operator_complexity_cost": "not_implemented", "update_collapse_proxy": None},
    }'''
    text = replace_block(text, r"def build_trace_feedback\(rep: Dict, args, epoch: int\).*?\n", r"def generate_candidate_suggestions\(", new_trace, "build_trace_feedback")

    text = text.replace('"duplicate_rate": 0.0,', '"attempted_candidates": len(candidates),\n            "skipped_duplicates": 0,\n            "duplicate_rate": 0.0,')

    text = text.replace('f"- offdiag_outside_boundary_cost: {float(route.get(\'offdiag_outside_boundary_cost\', 0.0)):.4f}",', 'f"- offdiag_outside_boundary_cost: {float(route.get(\'offdiag_outside_boundary_cost\', 0.0)):.4f}",\n        f"- self_route_mass: {float(route.get(\'self_route_mass\', 0.0)):.4f}",\n        f"- useful_transition_mass: {float(route.get(\'useful_transition_mass\', 0.0)):.4f}",\n        f"- collapse_flags: {\',\'.join(trace.get(\'collapse_flags\', []) or []) if trace.get(\'collapse_flags\') else \'NONE\'}",')
    text = text.replace('f"- memory_consumer_score: {float(memory.get(\'consumer_score\', 0.0)):.4f}",', 'f"- memory_consumer_proxy: {float(memory.get(\'memory_consumer_proxy\', memory.get(\'consumer_score\', 0.0))):.4f}",')
    text = text.replace('"skip_gate_mean", "skip_cost", "residual_dominance_proxy", "operator_complexity_cost",', '"skip_gate_mean", "skip_cost", "update_collapse_proxy", "residual_dominance_proxy", "operator_complexity_cost",')
    text = text.replace('p.add_argument("--memory-write-target", type=float, default=0.18)', 'p.add_argument("--memory-write-target", type=float, default=0.18)\n    p.add_argument("--memory-write-warmup-epochs", type=int, default=2)')
    text = text.replace('"version": "v4.3_min_heart",', '"version": "v4.3_min_heart_canonical",')

    TARGET.write_text(text, encoding="utf-8")
    changed = text != original
    print(f"[canonicalize] target={TARGET} changed={changed}")
    # Static assertions for critical markers.
    checks = [
        "sequence_route_delta_mean",
        "sequence_nonflat_score\": route_delta_mean + prim_delta_mean + upd_delta_mean + read_delta_mean",
        "read_delta_by_step",
        "boundary_usefulness",
        "self_route_mass",
        "useful_transition_mass",
        "ROUTE_IDENTITY_COLLAPSE",
        "memory_write_warmup_epochs",
        "memory_consumer_proxy",
        "update_collapse_proxy",
        "not_implemented",
    ]
    missing = [c for c in checks if c not in text]
    if missing:
        raise SystemExit(f"[canonicalize] missing markers: {missing}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
