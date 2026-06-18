#!/usr/bin/env python3
"""v4.4 context-controller entrypoint.

This file is the Python runtime for the v4.4 controller experiment.
It intentionally reuses the already-implemented v4.3 context-controller launcher logic
instead of duplicating a large backbone patch here.

Architecture rule:
    base v4.3/v4.2 behavior + small context-dependent controller deltas

This entrypoint keeps v4.3 canonical main intact and provides a real file for:
    commands/sync_run_5ep_v4_4_context_controllers_push_logs_v2.sh

No Actor/Critic, no EditorLoop deploy, no MatrixMemory, no FeedbackBias auto-deploy.

Important: this is a bridge implementation. It covers read/route/boundary/write/alive
controllers through the existing wrapper. A full primitive_controller inside the
transform unit is the next code step after this bridge smoke passes.

The old wrapper also contains a broken experimental fast-loader override that looks for
SpeechCommandsBalanced in v4.2. This bridge disables only that one monkey-patch at runtime
and keeps the rest of the controller wrapper intact.

The wrapper also has an older sequence_terms() implementation. Current v4.3 aux_losses_v43
expects boundary_usefulness_cost/proxy, so this bridge injects those keys into the temporary
wrapper copy instead of editing the old wrapper in-place.
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
    """Translate v4.4 CLI aliases to the existing v4.3 context wrapper.

    The existing controller wrapper accepts CONTEXT_CONTROLLER_SCALE as an env var.
    The v4.4 sync passes --context-alpha-min/--context-alpha-max.
    For this bridge we use the midpoint as a fixed safe controller scale.
    """
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
            # v4.3 wrapper/base parser may not know this arg in all revisions.
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
    env["V44_CONTEXT_ALPHA_MIN"] = f"{alpha_min:.6g}"
    env["V44_CONTEXT_ALPHA_MAX"] = f"{alpha_max:.6g}"
    if compare_to is not None:
        env["COMPARE_TO"] = compare_to
    return out, env


def _patch_wrapper_text(text: str) -> str:
    """Patch the legacy shell-embedded Python wrapper for v4.4 bridge smoke."""
    # 1) Disable broken data-loader monkey-patch only.
    bad_loader = "base.v42.make_loaders = make_loaders_fast"
    if bad_loader in text:
        text = text.replace(bad_loader, "# disabled by v4.4 bridge: " + bad_loader)
    else:
        print("[v4.4] warning: fast-loader monkey-patch line not found; wrapper unchanged", file=sys.stderr)

    # 2) Current aux_losses_v43 expects these keys from base.sequence_terms.
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
        print("[v4.4] expected path is <repo>/simple_butterfly_matrix_v4_tape_lane/commands/run_v4_3_context_controller.sh", file=sys.stderr)
        return 2

    patched_wrapper = _patched_wrapper_copy(wrapper)
    forwarded, env = _translate_args(argv)
    print(
        "[v4.4] bridge entrypoint -> patched run_v4_3_context_controller.sh "
        f"CONTEXT_CONTROLLER_SCALE={env.get('CONTEXT_CONTROLLER_SCALE')} "
        f"alpha_min={env.get('V44_CONTEXT_ALPHA_MIN')} alpha_max={env.get('V44_CONTEXT_ALPHA_MAX')} "
        "fast_loader_override=disabled sequence_terms_compat=enabled",
        flush=True,
    )
    cmd = ["bash", str(patched_wrapper), *forwarded]
    return subprocess.call(cmd, cwd=str(repo), env=env)


if __name__ == "__main__":
    raise SystemExit(main())
