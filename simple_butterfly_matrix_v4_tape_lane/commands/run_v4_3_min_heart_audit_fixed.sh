#!/usr/bin/env bash
set -euo pipefail
cd "$(git rev-parse --show-toplevel 2>/dev/null || pwd)"

python - "$@" <<'PY'
import sys, torch
from simple_butterfly_matrix_v4_tape_lane import tape_lane_transport_v4_3_min_heart as base

_old_build = base.build_trace_feedback
_old_gen = base.generate_candidate_suggestions
_old_run = base.run


def _step_delta(x, steps):
    if x.numel() == 0 or x.shape[0] <= 1:
        return torch.zeros(int(steps), dtype=torch.float32)
    d = (x[1:] - x[:-1]).abs()
    while d.dim() > 1:
        d = d.mean(dim=-1)
    return torch.cat([d, torch.zeros(1, dtype=d.dtype)], dim=0)[: int(steps)]


def sequence_terms(baux, args):
    device = baux.slots.device
    routes = baux.routes.float()
    prim = baux.primitive_weights.float()
    upd = baux.update_norms.float()
    read = baux.read_group_mass.float()
    route_delta = (routes[1:] - routes[:-1]).abs().mean(dim=(-2, -1)).mean() if routes.numel() and routes.shape[0] > 1 else torch.zeros((), device=device)
    prim_delta = (prim[1:] - prim[:-1]).abs().mean(dim=(-2, -1)).mean() if prim.numel() and prim.shape[0] > 1 else torch.zeros((), device=device)
    if upd.numel() and upd.shape[1] > 1:
        u = upd.mean(dim=(0, 3))
        trace_delta = (u[1:] - u[:-1]).abs().mean()
    else:
        trace_delta = torch.zeros((), device=device)
    read_delta = (read[1:] - read[:-1]).abs().mean() if read.numel() and read.shape[0] > 1 else torch.zeros((), device=device)
    return {
        "sequence_route_delta_mean": route_delta,
        "sequence_primitive_delta_mean": prim_delta,
        "sequence_update_delta_mean": trace_delta,
        "sequence_trace_delta_mean": trace_delta,
        "sequence_read_delta_mean": read_delta,
        "sequence_nonflat_score": route_delta + prim_delta + trace_delta + read_delta,
    }


def build_trace_feedback(rep, args, epoch):
    trace = _old_build(rep, args, epoch)
    read_group = base._tensor_from_list(rep.get("read_group_mass", []))
    read_delta = _step_delta(read_group, args.tape_steps)
    seq = trace.setdefault("sequence", {})
    seq["read_delta_by_step"] = read_delta.tolist()[: int(args.tape_steps)] if read_delta.numel() else []
    route_delta = torch.tensor(seq.get("route_delta_by_step", []), dtype=torch.float32)
    prim_delta = torch.tensor(seq.get("primitive_delta_by_step", []), dtype=torch.float32)
    trace_delta = torch.tensor(seq.get("trace_delta_by_step", []), dtype=torch.float32)
    n = min(len(route_delta), len(prim_delta), len(trace_delta), len(read_delta))
    if n:
        change = route_delta[:n] + prim_delta[:n] + trace_delta[:n] + read_delta[:n]
        seq["sequence_change_by_step"] = change.tolist()
        seq["sequence_nonflat_score"] = float(change.mean())
        boundary = torch.tensor(trace.get("route", {}).get("boundary_by_step", []), dtype=torch.float32)
        m = min(n, len(boundary))
        trace.setdefault("route", {})["boundary_usefulness"] = (boundary[:m] * change[:m]).tolist() if m else []
    trace["version"] = "v4.3_min_heart_audit_fixed"
    return trace


def generate_candidate_suggestions(trace, args, epoch):
    old = int(getattr(args, "max_candidates_per_epoch", 8))
    args.max_candidates_per_epoch = max(0, min(old, 12))
    out = _old_gen(trace, args, epoch)
    out.setdefault("diversity", {})["max_candidates_effective"] = int(args.max_candidates_per_epoch)
    args.max_candidates_per_epoch = old
    for c in out.get("candidates", []):
        c["deploy"] = False
    return out


def run(args):
    if getattr(args, "enable_counterfactual_screen", False):
        print("[v4.3 audit-fixed] counterfactual screen is not implemented in MVP0; suggestions only, no deploy.", flush=True)
    _old_run(args)

base.sequence_terms = sequence_terms
base.build_trace_feedback = build_trace_feedback
base.generate_candidate_suggestions = generate_candidate_suggestions
base.run = run
base.run(base.parser().parse_args(sys.argv[1:]))
PY
