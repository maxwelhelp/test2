#!/usr/bin/env python3
"""v4.5 Projection Feedback runtime bridge.

Extends the v4.4 context-controller bridge with real influence from two scout layers:

1. Offline Projection Council feedback:
   - scout_runtime_feedback_epoch_XXX.json
   - best projected program gives detached bias into controller logits.

2. Online Scout:
   - tiny trainable MLP over current gctx inside forward;
   - runs right before boundary/transition/fanout/route choices;
   - gives differentiable bias to current choices;
   - has an explicit usage cost so the model pays for using it.

This is NOT permanent auto-deploy. It is a weak, paid, trainable prior. Main
loss/backprop still decides whether the projected direction survives.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

_REPO_ROOT_FOR_IMPORT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT_FOR_IMPORT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT_FOR_IMPORT))

from simple_butterfly_matrix_v4_tape_lane import tape_lane_transport_v4_4_context_controllers as v44

PROJECT_DIR = v44.PROJECT_DIR


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


def _find_latest_feedback(repo: Path) -> str:
    env_path = os.environ.get("SCOUT_FEEDBACK_PATH", "").strip()
    if env_path and Path(env_path).exists():
        return env_path
    reports = repo / PROJECT_DIR / "agent_reports"
    files = sorted(reports.glob("v4_4_context_controllers_*/scout_runtime_feedback_epoch_*.json"), key=lambda p: p.stat().st_mtime)
    if files:
        return str(files[-1])
    return env_path


_old_translate_args = v44._translate_args


def _translate_args_v45(argv: list[str]):
    forwarded, env = _old_translate_args(argv)
    repo = _repo_root()
    feedback_path = _find_latest_feedback(repo)
    env["SCOUT_FEEDBACK_PATH"] = env.get("SCOUT_FEEDBACK_PATH", feedback_path)
    env["SCOUT_FEEDBACK_SCALE"] = env.get("SCOUT_FEEDBACK_SCALE", "0.015")
    env["SCOUT_FEEDBACK_MODE"] = env.get("SCOUT_FEEDBACK_MODE", "detached_bias")
    env["ONLINE_SCOUT_SCALE"] = env.get("ONLINE_SCOUT_SCALE", "0.020")
    env["ONLINE_SCOUT_GATE_INIT"] = env.get("ONLINE_SCOUT_GATE_INIT", "-3.0")
    env["ONLINE_SCOUT_USE_COST"] = env.get("ONLINE_SCOUT_USE_COST", "0.0005")
    env["SCOUT_SKIP_COST_SCALE"] = env.get("SCOUT_SKIP_COST_SCALE", "0.001")
    return forwarded, env


_old_patch_wrapper_text = v44._patch_wrapper_text


def _patch_wrapper_text_v45(text: str) -> str:
    text = _old_patch_wrapper_text(text)

    # 1) Load offline feedback and define online scout controls.
    needle = '        self.alive_controller_scale = float(os.environ.get("ALIVE_CONTROLLER_SCALE", "0.0"))\n'
    repl = needle + (
        '        self.scout_feedback_scale = float(os.environ.get("SCOUT_FEEDBACK_SCALE", "0.0"))\n'
        '        self.online_scout_scale = float(os.environ.get("ONLINE_SCOUT_SCALE", "0.0"))\n'
        '        self.online_scout_use_cost = float(os.environ.get("ONLINE_SCOUT_USE_COST", "0.0"))\n'
        '        self.scout_skip_cost_scale = float(os.environ.get("SCOUT_SKIP_COST_SCALE", "0.0"))\n'
        '        self.scout_feedback_path = os.environ.get("SCOUT_FEEDBACK_PATH", "")\n'
        '        self.scout_feedback = {}\n'
        '        self.scout_feedback_active = False\n'
        '        if self.scout_feedback_path and os.path.exists(self.scout_feedback_path):\n'
        '            try:\n'
        '                import json as _scout_json\n'
        '                with open(self.scout_feedback_path, "r", encoding="utf-8") as _sf:\n'
        '                    _raw_scout = _scout_json.load(_sf)\n'
        '                self.scout_feedback = _raw_scout.get("feedback", _raw_scout.get("controller_feedback", {}))\n'
        '                self.scout_feedback_active = bool(self.scout_feedback) and abs(self.scout_feedback_scale) > 0.0\n'
        '                print(f"[v4.5 scout] loaded feedback: {self.scout_feedback_path} scale={self.scout_feedback_scale} active={self.scout_feedback_active}", flush=True)\n'
        '            except Exception as _e:\n'
        '                print(f"[v4.5 scout] failed to load feedback {self.scout_feedback_path}: {_e}", flush=True)\n'
    )
    if "self.online_scout_scale" not in text and needle in text:
        text = text.replace(needle, repl, 1)

    # 2) Add online scout network into backbone init after transition basis is registered.
    basis_anchor = '        self.register_buffer("transition_route_basis", basis, persistent=False)'
    basis_repl = basis_anchor + (
        '\n        # Online scout: current-step differentiable request/alternative scorer.\n'
        '        # Output layout: boundary(1), transition(8), fanout(4), route_edges(L*L).\n'
        '        self.online_scout_net = nn.Sequential(nn.LayerNorm(4 * d), nn.Linear(4 * d, d), nn.SiLU(), nn.Linear(d, 1 + 8 + 4 + l * l))\n'
        '        self.online_scout_gate_logit = nn.Parameter(torch.tensor(float(os.environ.get("ONLINE_SCOUT_GATE_INIT", "-3.0"))))\n'
        '        _online_last = self.online_scout_net[-1]\n'
        '        nn.init.normal_(_online_last.weight, std=1e-4)\n'
        '        nn.init.zeros_(_online_last.bias)'
    )
    if basis_anchor in text and "self.online_scout_net" not in text:
        text = text.replace(basis_anchor, basis_repl, 1)

    # 3) Primitive controller receives offline scout primitive bias; online primitive path already exists through primitive_context_net.
    old_sig = '    def forward_with_context(self, x: torch.Tensor, read_packet: torch.Tensor, lane_embed: torch.Tensor, primitive_ctx: torch.Tensor | None):'
    new_sig = '    def forward_with_context(self, x: torch.Tensor, read_packet: torch.Tensor, lane_embed: torch.Tensor, primitive_ctx: torch.Tensor | None, step_idx: int | None = None, scout_feedback: dict | None = None, scout_scale: float = 0.0):'
    if old_sig in text:
        text = text.replace(old_sig, new_sig, 1)

    prim_anchor = '            logits = logits + float(self.primitive_controller_scale) * primitive_bias\n'
    prim_inject = prim_anchor + (
        '        if scout_feedback and scout_scale and step_idx is not None:\n'
        '            try:\n'
        '                _pb = scout_feedback.get("primitive_bias", {}).get(str(int(step_idx)), {})\n'
        '                if _pb:\n'
        '                    _lane_names = ["detail", "state", "abstract", "memory"]\n'
        '                    _prim_names = list(base.PRIMITIVES)\n'
        '                    _sb = torch.zeros((1, lanes, len(base.PRIMITIVES)), device=x.device, dtype=logits.dtype)\n'
        '                    for _ln, _mp in _pb.items():\n'
        '                        if _ln in _lane_names and isinstance(_mp, dict):\n'
        '                            _li = _lane_names.index(_ln)\n'
        '                            for _pn, _val in _mp.items():\n'
        '                                if _pn in _prim_names:\n'
        '                                    _sb[:, _li, _prim_names.index(_pn)] += float(_val)\n'
        '                    logits = logits + float(scout_scale) * _sb.detach()\n'
        '            except Exception:\n'
        '                pass\n'
    )
    if prim_anchor in text and "_pb = scout_feedback.get(\"primitive_bias\"" not in text:
        text = text.replace(prim_anchor, prim_inject, 1)

    # 4) Boundary: remove hard per-step mean normalization and add offline+online scout bias.
    # Hard normalization was the reason boundary became flat 0.28 at every step.
    old_boundary_block = (
        'boundary_raw = torch.sigmoid(self.boundary_logit[t].to(device=x.device, dtype=x.dtype) + self.boundary_controller_scale * boundary_bias)  # [B]\n'
        '            boundary_budget = torch.as_tensor(self.boundary_mean_target, device=x.device, dtype=boundary_raw.dtype)\n'
        '            boundary = (boundary_raw * boundary_budget / boundary_raw.detach().mean().clamp_min(1e-4)).clamp(0.0, 1.0)  # [B]'
    )
    new_boundary_block = (
        'online_scout_logits = self.online_scout_net(gctx).float() if getattr(self, "online_scout_scale", 0.0) != 0.0 else None\n'
        '            online_scout_gate = torch.sigmoid(self.online_scout_gate_logit.float()).to(device=x.device, dtype=x.dtype) if online_scout_logits is not None else torch.zeros((), device=x.device, dtype=x.dtype)\n'
        '            online_boundary_bias = online_scout_logits[:, 0].to(dtype=x.dtype) if online_scout_logits is not None else torch.zeros_like(boundary_bias)\n'
        '            scout_boundary_bias = 0.0\n'
        '            if getattr(self, "scout_feedback_active", False):\n'
        '                try:\n'
        '                    _bt = self.scout_feedback.get("boundary_target", [])\n'
        '                    if t < len(_bt): scout_boundary_bias = float(_bt[t])\n'
        '                except Exception:\n'
        '                    scout_boundary_bias = 0.0\n'
        '            boundary_raw = torch.sigmoid(self.boundary_logit[t].to(device=x.device, dtype=x.dtype) + self.boundary_controller_scale * boundary_bias + self.scout_feedback_scale * scout_boundary_bias + self.online_scout_scale * online_scout_gate * online_boundary_bias)  # [B]\n'
        '            boundary = boundary_raw.clamp(0.0, 1.0)  # [B], budget is now a loss/price, not hard per-step flattening'
    )
    if old_boundary_block in text:
        text = text.replace(old_boundary_block, new_boundary_block, 1)

    # If an older v4.5 patch has already replaced only boundary_raw, still remove leftover hard budget lines.
    hard_budget = (
        '            boundary_budget = torch.as_tensor(self.boundary_mean_target, device=x.device, dtype=boundary_raw.dtype)\n'
        '            boundary = (boundary_raw * boundary_budget / boundary_raw.detach().mean().clamp_min(1e-4)).clamp(0.0, 1.0)  # [B]'
    )
    if hard_budget in text and 'budget is now a loss/price' not in text:
        text = text.replace(hard_budget, '            boundary = boundary_raw.clamp(0.0, 1.0)  # [B], budget is now a loss/price')

    # 5) Transition target: offline bias + online differentiable bias.
    trans_anchor = '            transition_type_logits = self.transition_type_net(gctx).float()\n'
    trans_inject = trans_anchor + (
        '            if online_scout_logits is not None:\n'
        '                transition_type_logits = transition_type_logits + self.online_scout_scale * online_scout_gate.float() * online_scout_logits[:, 1:9]\n'
        '            if getattr(self, "scout_feedback_active", False):\n'
        '                try:\n'
        '                    _tm = {"identity":0,"detail_to_state":1,"state_to_abstract":2,"state_to_memory":3,"memory_to_state":4,"fork_detail_to_state_memory":5,"join_detail_state_to_abstract":6,"head_prepare":7}\n'
        '                    _td = self.scout_feedback.get("transition_target", {}).get(str(t), {})\n'
        '                    for _name, _val in _td.items():\n'
        '                        if _name in _tm:\n'
        '                            transition_type_logits[:, _tm[_name]] += self.scout_feedback_scale * float(_val)\n'
        '                except Exception:\n'
        '                    pass\n'
    )
    if trans_anchor in text and "online_scout_logits[:, 1:9]" not in text:
        text = text.replace(trans_anchor, trans_inject, 1)

    # 6) Route edge basis: offline route edges + online route edge logits.
    route_anchor = '            route_logits = route_logits + self.transition_controller_scale * route_boundary_gate.view(-1, 1, 1) * transition_bias\n'
    route_inject = route_anchor + (
        '            if online_scout_logits is not None:\n'
        '                _or = online_scout_logits[:, 13:].view(x.shape[0], self.lanes, self.lanes).to(dtype=x.dtype)\n'
        '                route_logits = route_logits + self.online_scout_scale * online_scout_gate * _or\n'
        '            if getattr(self, "scout_feedback_active", False):\n'
        '                try:\n'
        '                    _lm = {"detail":0,"state":1,"abstract":2,"memory":3}\n'
        '                    _rb = self.scout_feedback.get("route_bias", {}).get(str(t), {})\n'
        '                    for _edge, _val in _rb.items():\n'
        '                        if "->" in _edge:\n'
        '                            _a, _b = [x.strip() for x in _edge.split("->", 1)]\n'
        '                            if _a in _lm and _b in _lm:\n'
        '                                route_logits[:, _lm[_a], _lm[_b]] += self.scout_feedback_scale * float(_val)\n'
        '                except Exception:\n'
        '                    pass\n'
    )
    if route_anchor in text and "online_scout_logits[:, 13:]" not in text:
        text = text.replace(route_anchor, route_inject, 1)

    # 7) Fanout target: offline fanout + online fanout logits.
    fan_anchor = '            fanout_logits = self.fanout_context_net(gctx).float()\n'
    fan_inject = fan_anchor + (
        '            if online_scout_logits is not None:\n'
        '                fanout_logits = fanout_logits + self.online_scout_scale * online_scout_gate.float() * online_scout_logits[:, 9:13]\n'
        '            if getattr(self, "scout_feedback_active", False):\n'
        '                try:\n'
        '                    _fm = {"one":0,"two":1,"three":2,"all_soft":3}\n'
        '                    _fd = self.scout_feedback.get("fanout_target", {}).get(str(t), {})\n'
        '                    for _name, _val in _fd.items():\n'
        '                        if _name in _fm:\n'
        '                            fanout_logits[:, _fm[_name]] += self.scout_feedback_scale * float(_val)\n'
        '                except Exception:\n'
        '                    pass\n'
    )
    if fan_anchor in text and "online_scout_logits[:, 9:13]" not in text:
        text = text.replace(fan_anchor, fan_inject, 1)

    # 8) Pass offline scout feedback into primitive unit.
    old_call = '                update, unit_info = unit.forward_with_context(x, read_packet, self.lane_embed, primitive_ctx)'
    new_call = '                update, unit_info = unit.forward_with_context(x, read_packet, self.lane_embed, primitive_ctx, t, self.scout_feedback if getattr(self, "scout_feedback_active", False) else None, self.scout_feedback_scale)'
    if old_call in text:
        text = text.replace(old_call, new_call, 1)

    # 9) Paid usage: add scout-gate cost into train loss, and make skip_cost nonzero.
    skip_zero = 'out["skip_cost"] = torch.zeros((), device=logits.device)'
    skip_real = 'out["skip_cost"] = torch.clamp(residual_proxy - float(os.environ.get("SKIP_COLLAPSE_TARGET", "4.0")), min=0.0).pow(2)'
    if skip_zero in text:
        text = text.replace(skip_zero, skip_real, 1)

    train_anchor = '            loss = loss + args.lambda_skip_cost * losses["skip_cost"]\n'
    train_inject = train_anchor + (
        '            try:\n'
        '                _gate = torch.sigmoid(model.backbone.online_scout_gate_logit.float()) if hasattr(model.backbone, "online_scout_gate_logit") else torch.zeros((), device=logits.device)\n'
        '                loss = loss + float(os.environ.get("ONLINE_SCOUT_USE_COST", "0.0")) * _gate.pow(2).to(device=logits.device)\n'
        '            except Exception:\n'
        '                pass\n'
    )
    if train_anchor in text and "ONLINE_SCOUT_USE_COST" not in text:
        text = text.replace(train_anchor, train_inject, 1)

    # 10) Make trace/report mention online scout and feedback.
    old_ctx = '"boundary_route_gate_scale": float(os.environ.get("BOUNDARY_ROUTE_GATE_SCALE", "0.25")), "alive_scale": float(os.environ.get("ALIVE_CONTROLLER_SCALE", "0.0"))}'
    new_ctx = '"boundary_route_gate_scale": float(os.environ.get("BOUNDARY_ROUTE_GATE_SCALE", "0.25")), "alive_scale": float(os.environ.get("ALIVE_CONTROLLER_SCALE", "0.0")), "scout_feedback_path": os.environ.get("SCOUT_FEEDBACK_PATH", ""), "scout_feedback_scale": float(os.environ.get("SCOUT_FEEDBACK_SCALE", "0.0")), "online_scout_scale": float(os.environ.get("ONLINE_SCOUT_SCALE", "0.0")), "online_scout_use_cost": float(os.environ.get("ONLINE_SCOUT_USE_COST", "0.0"))}'
    if old_ctx in text:
        text = text.replace(old_ctx, new_ctx, 1)

    return text


def main(argv: list[str] | None = None) -> int:
    v44._translate_args = _translate_args_v45
    v44._patch_wrapper_text = _patch_wrapper_text_v45
    if argv is None:
        argv = sys.argv[1:]
    return v44.main(argv)


if __name__ == "__main__":
    raise SystemExit(main())
