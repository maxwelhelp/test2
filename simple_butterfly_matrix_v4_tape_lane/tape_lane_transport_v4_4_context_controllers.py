#!/usr/bin/env python3
"""v4.4 context-controller entrypoint.

This file is the Python runtime for the v4.4 controller experiment.
It reuses the already-working v4.3 context-controller wrapper, but patches it at
runtime so we can iterate safely without breaking the stable v4.3 baseline.

Architecture rule:
    base v4.3/v4.2 behavior + small context-dependent controller deltas

No Actor/Critic, no EditorLoop deploy, no MatrixMemory, no FeedbackBias auto-deploy.

Current bridge coverage:
    read/source controller      yes
    route/lane-flow controller  yes
    boundary controller         yes
    write gate controller       yes
    alive controller            present, but disabled by default
    primitive controller        yes, patched around each transform unit

The next clean step is to move this patched wrapper logic into a normal
TapeLaneRouterBackboneV44 class, but this bridge keeps the current working run path.
"""

from __future__ import annotations

import os
import subprocess
import sys
import tempfile
from pathlib import Path


PROJECT_DIR = "simple_butterfly_matrix_v4_tape_lane"


def _repo_root() -> Path:
    # file = <repo>/simple_butterfly_matrix_v4_tape_lane/tape_lane_transport_v4_4_context_controllers.py
    return Path(__file__).resolve().parents[1]


def _translate_args(argv: list[str]) -> tuple[list[str], dict[str, str]]:
    """Translate v4.4 CLI aliases to the existing v4.3 context wrapper."""
    env = os.environ.copy()
    out: list[str] = []
    i = 0
    alpha_min = None
    alpha_max = None
    compare_to = None

    while i < len(argv):
        a = argv[i]
        if a == "--context-alpha-min" and i + 1 < len(argv):
            alpha_min = float(argv[i + 1])
            i += 2
            continue
        if a == "--context-alpha-max" and i + 1 < len(argv):
            alpha_max = float(argv[i + 1])
            i += 2
            continue
        if a == "--compare-to" and i + 1 < len(argv):
            compare_to = argv[i + 1]
            i += 2
            continue
        out.append(a)
        i += 1

    if alpha_min is None:
        alpha_min = float(env.get("CONTEXT_ALPHA_MIN", "0.01"))
    if alpha_max is None:
        alpha_max = float(env.get("CONTEXT_ALPHA_MAX", "0.10"))
    scale = 0.5 * (alpha_min + alpha_max)

    env["CONTEXT_CONTROLLER_SCALE"] = env.get("CONTEXT_CONTROLLER_SCALE", f"{scale:.6g}")
    env["PRIMITIVE_CONTROLLER_SCALE"] = env.get("PRIMITIVE_CONTROLLER_SCALE", env["CONTEXT_CONTROLLER_SCALE"])
    # Adaptive depth is riskier than read/route/write/primitive. Keep alive context off unless explicitly enabled.
    env["ALIVE_CONTROLLER_SCALE"] = env.get("ALIVE_CONTROLLER_SCALE", "0.0")
    env["V44_CONTEXT_ALPHA_MIN"] = f"{alpha_min:.6g}"
    env["V44_CONTEXT_ALPHA_MAX"] = f"{alpha_max:.6g}"
    if compare_to is not None:
        env["COMPARE_TO"] = compare_to
    return out, env


PRIMITIVE_CONTROLLER_CLASS = r'''

class ContextPrimitiveTransformUnit(nn.Module):
    """Context-aware wrapper around v4.2 TapeLaneTransformUnit.

    The old unit already has global primitive logits, lane bias, and read-packet context bias.
    This wrapper adds a stronger assembly-time primitive controller that sees:
        lane_state, read_state, memory_summary, step_embed, route_out_summary.

    It recomputes the same primitive candidates as the old unit and adds a context bias before
    the primitive softmax, so it changes the actual update, not only the logs.
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


def _patch_wrapper_text(text: str) -> str:
    """Patch the legacy shell-embedded Python wrapper for v4.4 bridge smoke/run."""
    # 1) Disable broken data-loader monkey-patch only.
    bad_loader = "base.v42.make_loaders = make_loaders_fast"
    if bad_loader in text:
        text = text.replace(bad_loader, "# disabled by v4.4 bridge: " + bad_loader)
    else:
        print("[v4.4] warning: fast-loader monkey-patch line not found; wrapper unchanged", file=sys.stderr)

    # 2) Insert primitive transform wrapper before ContextTapeLaneRouterBackbone.
    if "class ContextPrimitiveTransformUnit" not in text:
        marker = "class ContextTapeLaneRouterBackbone(_old_backbone):"
        if marker in text:
            text = text.replace(marker, PRIMITIVE_CONTROLLER_CLASS + "\n" + marker, 1)
        else:
            print("[v4.4] warning: ContextTapeLaneRouterBackbone marker not found; primitive controller not injected", file=sys.stderr)

    # 3) Add primitive/alive controller scales and wrap transform units after base init.
    init_block = (
        '        self.context_controller_scale = float(os.environ.get("CONTEXT_CONTROLLER_SCALE", "0.15"))\n'
        '        self.read_context_net = nn.Sequential(nn.LayerNorm(4 * d), nn.Linear(4 * d, d), nn.SiLU(), nn.Linear(d, r))'
    )
    init_repl = (
        '        self.context_controller_scale = float(os.environ.get("CONTEXT_CONTROLLER_SCALE", "0.15"))\n'
        '        self.primitive_controller_scale = float(os.environ.get("PRIMITIVE_CONTROLLER_SCALE", str(self.context_controller_scale)))\n'
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
    zero_repl = zero_block + (
        '\n        self.units = nn.ModuleList([ContextPrimitiveTransformUnit(u, self.primitive_controller_scale) for u in self.units])'
    )
    if zero_block in text and "ContextPrimitiveTransformUnit(u, self.primitive_controller_scale)" not in text:
        text = text.replace(zero_block, zero_repl, 1)

    # 4) Keep alive controller disabled by default through separate scale.
    old_alive = 'alive = torch.sigmoid(self.step_alive_logit[t].to(device=x.device, dtype=x.dtype) + self.context_controller_scale * alive_bias)  # [B]'
    new_alive = 'alive = torch.sigmoid(self.step_alive_logit[t].to(device=x.device, dtype=x.dtype) + self.alive_controller_scale * alive_bias)  # [B]'
    if old_alive in text:
        text = text.replace(old_alive, new_alive, 1)

    # 5) Replace primitive call with context-aware primitive call.
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

    # 6) Add primitive controller status to trace and report.
    old_trace_ctx = 'trace["collapse_flags"] = flags; trace["version"] = "v4.3_context_controller_audit_fast"; trace["context_controller"] = {"active": True, "scale": float(os.environ.get("CONTEXT_CONTROLLER_SCALE", "0.15"))}'
    new_trace_ctx = 'trace["collapse_flags"] = flags; trace["version"] = "v4.4_context_controller_primitive_bridge"; trace["context_controller"] = {"active": True, "scale": float(os.environ.get("CONTEXT_CONTROLLER_SCALE", "0.15")), "primitive_controller": True, "primitive_scale": float(os.environ.get("PRIMITIVE_CONTROLLER_SCALE", os.environ.get("CONTEXT_CONTROLLER_SCALE", "0.15"))), "alive_scale": float(os.environ.get("ALIVE_CONTROLLER_SCALE", "0.0"))}'
    if old_trace_ctx in text:
        text = text.replace(old_trace_ctx, new_trace_ctx, 1)

    old_report_line = 'f.write("- read/route/boundary/alive/write now receive context projections from current lane state, evidence and memory.\\n"); f.write("- fast eval computes loss/accuracy on all validation batches, but builds heavy trace/report once from last validation batch.\\n")'
    new_report_line = 'f.write("- read/route/boundary/write receive context projections from current lane state, evidence and memory.\\n"); f.write("- primitive_controller is active inside transform units and biases primitive/operator choice from lane/read/memory/step/route context.\\n"); f.write(f"- alive_controller_scale: {float(os.environ.get(\'ALIVE_CONTROLLER_SCALE\', \'0.0\')):.3f} (0.0 means disabled by default).\\n"); f.write("- fast eval computes loss/accuracy on all validation batches, but builds heavy trace/report once from last validation batch.\\n")'
    if old_report_line in text:
        text = text.replace(old_report_line, new_report_line, 1)

    # 7) Current aux_losses_v43 expects these keys from base.sequence_terms.
    old_return = (
        '    return {"sequence_route_delta_mean": route_delta, "sequence_primitive_delta_mean": prim_delta, '
        '"sequence_update_delta_mean": trace_delta, "sequence_trace_delta_mean": trace_delta, '
        '"sequence_read_delta_mean": read_delta, "sequence_nonflat_score": route_delta + prim_delta + trace_delta + read_delta, '
        '"self_route_mass": route_extra["self_route_mass"].to(device), "useful_transition_mass": route_extra["useful_transition_mass"].to(device)}'
    )
    new_return = (
        '    if getattr(baux, "boundaries", torch.empty(0, device=device)).numel():\n'
        '        boundary = baux.boundaries.float().to(device)\n'
        '        routes_f = routes.to(device)\n'
        '        prim_f = prim.to(device)\n'
        '        read_f = read.to(device)\n'
        '        route_step = (routes_f[1:] - routes_f[:-1]).abs().mean(dim=(-2, -1)) if routes_f.numel() and routes_f.shape[0] > 1 else torch.empty(0, device=device)\n'
        '        prim_step = (prim_f[1:] - prim_f[:-1]).abs().mean(dim=(-2, -1)) if prim_f.numel() and prim_f.shape[0] > 1 else torch.empty(0, device=device)\n'
        '        read_step = (read_f[1:] - read_f[:-1]).abs().mean(dim=(-2, -1)) if read_f.numel() and read_f.shape[0] > 1 else torch.empty(0, device=device)\n'
        '        if upd.numel() and upd.shape[1] > 1:\n'
        '            upd_by_t = upd.mean(dim=(0, 3)).to(device)\n'
        '            upd_step = (upd_by_t[1:] - upd_by_t[:-1]).abs().mean(dim=-1)\n'
        '        else:\n'
        '            upd_step = torch.empty(0, device=device)\n'
        '        common = min(int(boundary.shape[0]), int(route_step.shape[0]), int(prim_step.shape[0]), int(read_step.shape[0]), int(upd_step.shape[0]))\n'
        '        if common:\n'
        '            seq_by_step = route_step[:common] + prim_step[:common] + read_step[:common] + upd_step[:common]\n'
        '            denom = seq_by_step.detach().mean().clamp_min(1e-6)\n'
        '            boundary_usefulness = (boundary[:common] * ((seq_by_step / denom) - 1.0)).mean()\n'
        '        else:\n'
        '            boundary_usefulness = torch.zeros((), device=device)\n'
        '    else:\n'
        '        boundary_usefulness = torch.zeros((), device=device)\n'
        '    return {"sequence_route_delta_mean": route_delta, "sequence_primitive_delta_mean": prim_delta, '
        '"sequence_update_delta_mean": trace_delta, "sequence_trace_delta_mean": trace_delta, '
        '"sequence_read_delta_mean": read_delta, "sequence_nonflat_score": route_delta + prim_delta + trace_delta + read_delta, '
        '"self_route_mass": route_extra["self_route_mass"].to(device), "useful_transition_mass": route_extra["useful_transition_mass"].to(device), '
        '"boundary_usefulness_cost": F.relu(float(args.boundary_usefulness_target) - boundary_usefulness).pow(2), '
        '"boundary_usefulness_proxy": boundary_usefulness.detach()}'
    )
    if old_return in text:
        text = text.replace(old_return, new_return)
    elif "boundary_usefulness_cost" not in text:
        print("[v4.4] warning: sequence_terms return pattern not found; boundary usefulness keys may still be missing", file=sys.stderr)
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
        f"ALIVE_CONTROLLER_SCALE={env.get('ALIVE_CONTROLLER_SCALE')} "
        f"alpha_min={env.get('V44_CONTEXT_ALPHA_MIN')} alpha_max={env.get('V44_CONTEXT_ALPHA_MAX')} "
        "fast_loader_override=disabled sequence_terms_compat=enabled primitive_controller=enabled",
        flush=True,
    )
    cmd = ["bash", str(patched_wrapper), *forwarded]
    return subprocess.call(cmd, cwd=str(repo), env=env)


if __name__ == "__main__":
    raise SystemExit(main())
