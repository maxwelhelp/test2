#!/usr/bin/env python3
"""v4.4 context-controller entrypoint.

Bridge runtime for the v4.4 controller experiment. It reuses the working v4.3
context-controller wrapper, but patches it at runtime so v4.3 remains stable.

No Actor/Critic, no EditorLoop deploy, no MatrixMemory, no FeedbackBias auto-deploy.

Current bridge coverage:
    read/source controller      yes
    route/lane-flow controller  yes
    boundary controller         yes, but budget-normalized
    boundary route cheapness    capped/budgeted gate
    write gate controller       yes
    alive controller            present, disabled by default
    primitive controller        yes, patched around each transform unit
"""

from __future__ import annotations

import os
import subprocess
import sys
import tempfile
from pathlib import Path

PROJECT_DIR = "simple_butterfly_matrix_v4_tape_lane"


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _translate_args(argv: list[str]) -> tuple[list[str], dict[str, str]]:
    env = os.environ.copy()
    out: list[str] = []
    i = 0
    alpha_min = None
    alpha_max = None
    compare_to = None
    while i < len(argv):
        a = argv[i]
        if a == "--context-alpha-min" and i + 1 < len(argv):
            alpha_min = float(argv[i + 1]); i += 2; continue
        if a == "--context-alpha-max" and i + 1 < len(argv):
            alpha_max = float(argv[i + 1]); i += 2; continue
        if a == "--compare-to" and i + 1 < len(argv):
            compare_to = argv[i + 1]; i += 2; continue
        out.append(a); i += 1

    if alpha_min is None:
        alpha_min = float(env.get("CONTEXT_ALPHA_MIN", "0.01"))
    if alpha_max is None:
        alpha_max = float(env.get("CONTEXT_ALPHA_MAX", "0.10"))
    scale = 0.5 * (alpha_min + alpha_max)

    env["CONTEXT_CONTROLLER_SCALE"] = env.get("CONTEXT_CONTROLLER_SCALE", f"{scale:.6g}")
    env["PRIMITIVE_CONTROLLER_SCALE"] = env.get("PRIMITIVE_CONTROLLER_SCALE", env["CONTEXT_CONTROLLER_SCALE"])
    env["BOUNDARY_CONTROLLER_SCALE"] = env.get("BOUNDARY_CONTROLLER_SCALE", env["CONTEXT_CONTROLLER_SCALE"])
    # Mean boundary mass budget. 0.28 ≈ 3-4 effective peaks over T=12, but without a two-pass top-k planner.
    env["BOUNDARY_MEAN_TARGET"] = env.get("BOUNDARY_MEAN_TARGET", "0.28")
    # Even where boundary is high, it must not make cross-lane route fully free.
    env["BOUNDARY_ROUTE_GATE_SCALE"] = env.get("BOUNDARY_ROUTE_GATE_SCALE", "0.25")
    env["ALIVE_CONTROLLER_SCALE"] = env.get("ALIVE_CONTROLLER_SCALE", "0.0")
    env["V44_CONTEXT_ALPHA_MIN"] = f"{alpha_min:.6g}"
    env["V44_CONTEXT_ALPHA_MAX"] = f"{alpha_max:.6g}"
    if compare_to is not None:
        env["COMPARE_TO"] = compare_to
    return out, env


PRIMITIVE_CONTROLLER_CLASS = r'''

class ContextPrimitiveTransformUnit(nn.Module):
    """Context-aware wrapper around v4.2 TapeLaneTransformUnit.

    Adds an assembly-time primitive controller that sees lane_state, read_state,
    memory_summary, step_embed, and route_out_summary. It recomputes the same
    primitive candidates as the old unit and adds context bias before primitive
    softmax, so it changes the actual update.
    """

    def __init__(self, unit: nn.Module, primitive_controller_scale: float):
        super().__init__()
        self.unit = unit
        self.dim = int(unit.dim)
        self.lanes = int(unit.lanes)
        self.cells_per_lane = int(unit.cells_per_lane)
        self.primitive_controller_scale = float(primitive_controller_scale)
        p = len(base.PRIMITIVES)
        self.primitive_context_net = nn.Sequential(
            nn.LayerNorm(5 * self.dim),
            nn.Linear(5 * self.dim, self.dim),
            nn.SiLU(),
            nn.Linear(self.dim, p),
        )
        last = self.primitive_context_net[-1]
        nn.init.normal_(last.weight, std=1e-4)
        nn.init.zeros_(last.bias)

    def forward(self, x: torch.Tensor, read_packet: torch.Tensor, lane_embed: torch.Tensor):
        return self.forward_with_context(x, read_packet, lane_embed, None)

    def forward_with_context(self, x: torch.Tensor, read_packet: torch.Tensor, lane_embed: torch.Tensor, primitive_ctx: torch.Tensor | None):
        u = self.unit
        bsz, lanes, cells, dim = x.shape
        flat_x = x.reshape(bsz * lanes, cells, dim)
        flat_ctx = read_packet.reshape(bsz * lanes, cells, dim)

        ctx_m = flat_ctx @ u.ctx_w.to(device=x.device, dtype=x.dtype)
        channel = u.channel(flat_x + ctx_m)
        block = u.block(flat_x)
        low = (flat_x @ u.low_a.to(device=x.device, dtype=x.dtype)) @ u.low_b.to(device=x.device, dtype=x.dtype)
        gate = torch.sigmoid(
            flat_x @ u.gate_h.to(device=x.device, dtype=x.dtype)
            + ctx_m @ u.gate_c.to(device=x.device, dtype=x.dtype)
            + u.gate_bias.to(device=x.device, dtype=x.dtype)
        )
        product_gate = flat_x * torch.tanh(ctx_m)
        diff = (flat_x - ctx_m) @ u.diff_w.to(device=x.device, dtype=x.dtype)
        contrast = torch.tanh((flat_x - ctx_m) @ u.contrast_w.to(device=x.device, dtype=x.dtype)) * (flat_x + ctx_m)
        keep = torch.sigmoid(
            flat_x @ u.keep_h.to(device=x.device, dtype=x.dtype)
            + ctx_m @ u.keep_c.to(device=x.device, dtype=x.dtype)
        )
        memory_keep = keep * flat_x + (1.0 - keep) * ctx_m
        cands = torch.stack([channel, block, low, ctx_m, product_gate, diff, contrast, memory_keep], dim=2)

        lane_bias = u.lane_primitive_bias(lane_embed.to(device=x.device, dtype=x.dtype)).float().view(1, lanes, len(base.PRIMITIVES))
        ctx_summary = read_packet.mean(dim=2)
        ctx_bias = u.context_to_primitive(ctx_summary.to(dtype=x.dtype)).float()
        logits = u.primitive_logits.float().view(1, 1, -1) + lane_bias
        logits = logits + float(u.context_primitive_scale) * ctx_bias

        primitive_delta_norm = torch.zeros((), device=x.device)
        if primitive_ctx is not None:
            primitive_bias = self.primitive_context_net(primitive_ctx.to(dtype=x.dtype)).float()
            primitive_delta_norm = primitive_bias.detach().float().norm(dim=-1).mean()
            logits = logits + float(self.primitive_controller_scale) * primitive_bias

        weights = torch.softmax(logits, dim=-1).to(x.dtype)
        weights_full = weights.view(bsz, lanes, 1, len(base.PRIMITIVES), 1).expand(-1, -1, cells, -1, -1)
        weights_flat = weights_full.reshape(bsz * lanes, cells, len(base.PRIMITIVES), 1)
        update = (weights_flat * cands).sum(dim=2)
        update = u.update_norm(u.drop(update * gate)).reshape(bsz, lanes, cells, dim)
        return update, {
            "primitive_weights": weights.detach().float().mean(dim=0),
            "gate_mean": gate.detach().float().mean(),
            "keep_mean": keep.detach().float().mean(),
            "primitive_context_delta_norm": primitive_delta_norm,
        }
'''


SEQ_COMPAT_PATCH = r'''

_old_sequence_terms_v44 = base.sequence_terms

def sequence_terms_v44_compat(baux, args):
    out = _old_sequence_terms_v44(baux, args)
    if "boundary_usefulness_cost" not in out:
        device = torch.device("cpu")
        if getattr(baux, "boundaries", torch.empty(0)).numel():
            device = baux.boundaries.device
        proxy = torch.zeros((), device=device)
        out["boundary_usefulness_cost"] = F.relu(float(args.boundary_usefulness_target) - proxy).pow(2)
        out["boundary_usefulness_proxy"] = proxy.detach()
    return out

base.sequence_terms = sequence_terms_v44_compat
'''


def _patch_wrapper_text(text: str) -> str:
    bad_loader = "base.v42.make_loaders = make_loaders_fast"
    if bad_loader in text:
        text = text.replace(bad_loader, "# disabled by v4.4 bridge: " + bad_loader)

    marker = "class ContextTapeLaneRouterBackbone(_old_backbone):"
    if "class ContextPrimitiveTransformUnit" not in text and marker in text:
        text = text.replace(marker, PRIMITIVE_CONTROLLER_CLASS + "\n" + marker, 1)

    init_block = (
        '        self.context_controller_scale = float(os.environ.get("CONTEXT_CONTROLLER_SCALE", "0.15"))\n'
        '        self.read_context_net = nn.Sequential(nn.LayerNorm(4 * d), nn.Linear(4 * d, d), nn.SiLU(), nn.Linear(d, r))'
    )
    init_repl = (
        '        self.context_controller_scale = float(os.environ.get("CONTEXT_CONTROLLER_SCALE", "0.15"))\n'
        '        self.primitive_controller_scale = float(os.environ.get("PRIMITIVE_CONTROLLER_SCALE", str(self.context_controller_scale)))\n'
        '        self.boundary_controller_scale = float(os.environ.get("BOUNDARY_CONTROLLER_SCALE", str(self.context_controller_scale)))\n'
        '        self.boundary_mean_target = float(os.environ.get("BOUNDARY_MEAN_TARGET", "0.28"))\n'
        '        self.boundary_route_gate_scale = float(os.environ.get("BOUNDARY_ROUTE_GATE_SCALE", "0.25"))\n'
        '        self.alive_controller_scale = float(os.environ.get("ALIVE_CONTROLLER_SCALE", "0.0"))\n'
        '        self.read_context_net = nn.Sequential(nn.LayerNorm(4 * d), nn.Linear(4 * d, d), nn.SiLU(), nn.Linear(d, r))'
    )
    if init_block in text:
        text = text.replace(init_block, init_repl, 1)

    zero_block = (
        '        for net in [self.read_context_net, self.route_context_net, self.boundary_context_net, self.alive_context_net, self.write_context_net]:\n'
        '            last = net[-1]\n'
        '            nn.init.zeros_(last.weight)\n'
        '            nn.init.zeros_(last.bias)'
    )
    zero_repl = zero_block + '\n        self.units = nn.ModuleList([ContextPrimitiveTransformUnit(u, self.primitive_controller_scale) for u in self.units])'
    if zero_block in text and "ContextPrimitiveTransformUnit(u, self.primitive_controller_scale)" not in text:
        text = text.replace(zero_block, zero_repl, 1)

    old_alive = 'alive = torch.sigmoid(self.step_alive_logit[t].to(device=x.device, dtype=x.dtype) + self.context_controller_scale * alive_bias)  # [B]'
    new_alive = 'alive = torch.sigmoid(self.step_alive_logit[t].to(device=x.device, dtype=x.dtype) + self.alive_controller_scale * alive_bias)  # [B]'
    if old_alive in text:
        text = text.replace(old_alive, new_alive, 1)

    old_boundary = 'boundary = torch.sigmoid(self.boundary_logit[t].to(device=x.device, dtype=x.dtype) + self.context_controller_scale * boundary_bias)  # [B]'
    new_boundary = (
        'boundary_raw = torch.sigmoid(self.boundary_logit[t].to(device=x.device, dtype=x.dtype) + self.boundary_controller_scale * boundary_bias)  # [B]\n'
        '            # Budget normalization: prevents boundary_mean≈1.0 everywhere while keeping gradients and relative scores.\n'
        '            boundary_budget = torch.as_tensor(self.boundary_mean_target, device=x.device, dtype=boundary_raw.dtype)\n'
        '            boundary = (boundary_raw * boundary_budget / boundary_raw.detach().mean().clamp_min(1e-4)).clamp(0.0, 1.0)  # [B]'
    )
    if old_boundary in text:
        text = text.replace(old_boundary, new_boundary, 1)

    old_route_gate = 'route_logits = route_logits + boundary.view(-1, 1, 1) * self.boundary_route_bias.to(device=x.device, dtype=x.dtype).view(1, self.lanes, self.lanes)'
    new_route_gate = (
        'route_boundary_gate = (boundary * self.boundary_route_gate_scale).clamp(0.0, 1.0)\n'
        '            route_logits = route_logits + route_boundary_gate.view(-1, 1, 1) * self.boundary_route_bias.to(device=x.device, dtype=x.dtype).view(1, self.lanes, self.lanes)'
    )
    if old_route_gate in text:
        text = text.replace(old_route_gate, new_route_gate, 1)

    old_unit_call = '            update, unit_info = unit(x, read_packet, self.lane_embed)'
    new_unit_call = (
        '            lane_state_pre = x.mean(dim=2)\n'
        '            read_state_pre = read_packet.mean(dim=2)\n'
        '            mem_lane = min(self.lanes - 1, 3)\n'
        '            mem_summary = x[:, mem_lane].mean(dim=1).view(x.shape[0], 1, self.dim).expand(-1, self.lanes, -1)\n'
        '            step_for_primitive = self.step_embed[t].to(device=x.device, dtype=x.dtype).view(1, 1, self.dim).expand(x.shape[0], self.lanes, -1)\n'
        '            route_out_summary = torch.einsum("bft,bfd->btd", route, lane_state_pre)\n'
        '            primitive_ctx = torch.cat([lane_state_pre, read_state_pre, mem_summary, step_for_primitive, route_out_summary], dim=-1)\n'
        '            if hasattr(unit, "forward_with_context"):\n'
        '                update, unit_info = unit.forward_with_context(x, read_packet, self.lane_embed, primitive_ctx)\n'
        '            else:\n'
        '                update, unit_info = unit(x, read_packet, self.lane_embed)'
    )
    if old_unit_call in text:
        text = text.replace(old_unit_call, new_unit_call, 1)

    old_trace_ctx = 'trace["collapse_flags"] = flags; trace["version"] = "v4.3_context_controller_audit_fast"; trace["context_controller"] = {"active": True, "scale": float(os.environ.get("CONTEXT_CONTROLLER_SCALE", "0.15"))}'
    new_trace_ctx = 'trace["collapse_flags"] = flags; trace["version"] = "v4.4_context_controller_primitive_boundary_mean_budget_bridge"; trace["context_controller"] = {"active": True, "scale": float(os.environ.get("CONTEXT_CONTROLLER_SCALE", "0.15")), "primitive_controller": True, "primitive_scale": float(os.environ.get("PRIMITIVE_CONTROLLER_SCALE", os.environ.get("CONTEXT_CONTROLLER_SCALE", "0.15"))), "boundary_controller_scale": float(os.environ.get("BOUNDARY_CONTROLLER_SCALE", os.environ.get("CONTEXT_CONTROLLER_SCALE", "0.15"))), "boundary_mean_target": float(os.environ.get("BOUNDARY_MEAN_TARGET", "0.28")), "boundary_route_gate_scale": float(os.environ.get("BOUNDARY_ROUTE_GATE_SCALE", "0.25")), "alive_scale": float(os.environ.get("ALIVE_CONTROLLER_SCALE", "0.0"))}'
    if old_trace_ctx in text:
        text = text.replace(old_trace_ctx, new_trace_ctx, 1)

    old_report_line = 'f.write("- read/route/boundary/alive/write now receive context projections from current lane state, evidence and memory.\\n"); f.write("- fast eval computes loss/accuracy on all validation batches, but builds heavy trace/report once from last validation batch.\\n")'
    new_report_line = 'f.write("- read/route/boundary/write receive context projections from current lane state, evidence and memory.\\n"); f.write("- primitive_controller is active inside transform units and biases primitive/operator choice from lane/read/memory/step/route context.\\n"); f.write(f"- boundary_mean_target: {float(os.environ.get(\'BOUNDARY_MEAN_TARGET\', \'0.28\')):.3f}; boundary route_gate_scale={float(os.environ.get(\'BOUNDARY_ROUTE_GATE_SCALE\', \'0.25\')):.3f}.\\n"); f.write(f"- alive_controller_scale: {float(os.environ.get(\'ALIVE_CONTROLLER_SCALE\', \'0.0\')):.3f} (0.0 means disabled by default).\\n"); f.write("- fast eval computes loss/accuracy on all validation batches, but builds heavy trace/report once from last validation batch.\\n")'
    if old_report_line in text:
        text = text.replace(old_report_line, new_report_line, 1)

    seq_marker = "base.sequence_terms = sequence_terms"
    if "sequence_terms_v44_compat" not in text and seq_marker in text:
        text = text.replace(seq_marker, seq_marker + SEQ_COMPAT_PATCH, 1)
    return text


def _patched_wrapper_copy(wrapper: Path) -> Path:
    text = wrapper.read_text(encoding="utf-8")
    text = _patch_wrapper_text(text)
    tmp = tempfile.NamedTemporaryFile("w", encoding="utf-8", prefix="v44_context_wrapper_", suffix=".sh", delete=False)
    with tmp:
        tmp.write(text)
    os.chmod(tmp.name, 0o755)
    return Path(tmp.name)


def main(argv: list[str] | None = None) -> int:
    if argv is None:
        argv = sys.argv[1:]
    repo = _repo_root()
    wrapper = repo / PROJECT_DIR / "commands" / "run_v4_3_context_controller.sh"
    if not wrapper.exists():
        print(f"[v4.4] ERROR: missing existing context wrapper: {wrapper}", file=sys.stderr)
        return 2
    patched_wrapper = _patched_wrapper_copy(wrapper)
    forwarded, env = _translate_args(argv)
    print(
        "[v4.4] bridge entrypoint -> patched run_v4_3_context_controller.sh "
        f"CONTEXT_CONTROLLER_SCALE={env.get('CONTEXT_CONTROLLER_SCALE')} "
        f"PRIMITIVE_CONTROLLER_SCALE={env.get('PRIMITIVE_CONTROLLER_SCALE')} "
        f"BOUNDARY_CONTROLLER_SCALE={env.get('BOUNDARY_CONTROLLER_SCALE')} "
        f"BOUNDARY_MEAN_TARGET={env.get('BOUNDARY_MEAN_TARGET')} "
        f"BOUNDARY_ROUTE_GATE_SCALE={env.get('BOUNDARY_ROUTE_GATE_SCALE')} "
        f"ALIVE_CONTROLLER_SCALE={env.get('ALIVE_CONTROLLER_SCALE')} "
        f"alpha_min={env.get('V44_CONTEXT_ALPHA_MIN')} alpha_max={env.get('V44_CONTEXT_ALPHA_MAX')} "
        "fast_loader_override=disabled sequence_terms_compat=enabled primitive_controller=enabled boundary_mean_budget=enabled",
        flush=True,
    )
    cmd = ["bash", str(patched_wrapper), *forwarded]
    return subprocess.call(cmd, cwd=str(repo), env=env)


if __name__ == "__main__":
    raise SystemExit(main())
