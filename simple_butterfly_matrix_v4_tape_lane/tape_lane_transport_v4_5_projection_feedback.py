#!/usr/bin/env python3
"""v4.5 Projection Feedback runtime bridge.

Extends the v4.4 context-controller bridge with actual influence from
Projection Council output.

Input feedback file:
    scout_runtime_feedback_epoch_XXX.json

Effect:
    Adds tiny detached scout bias into:
      - boundary logits
      - transition_type logits
      - fanout logits
      - route logits / route basis edges
      - primitive logits per step/lane/primitive

This is NOT permanent auto-deploy. It is a weak prior / extra basis push from
the best projected alternative. Main task loss/backprop still decides whether
that direction survives.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

# When this file is executed directly as
#   python simple_butterfly_matrix_v4_tape_lane/tape_lane_transport_v4_5_projection_feedback.py
# Python puts the package directory, not the repository root, on sys.path.
# Add repo root explicitly so package imports work from the sync scripts.
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
    return forwarded, env


_old_patch_wrapper_text = v44._patch_wrapper_text


def _patch_wrapper_text_v45(text: str) -> str:
    text = _old_patch_wrapper_text(text)

    # 1) Load feedback into the backbone once. It stays a detached dict; optimizer never touches it.
    needle = '        self.alive_controller_scale = float(os.environ.get("ALIVE_CONTROLLER_SCALE", "0.0"))\n'
    repl = needle + (
        '        self.scout_feedback_scale = float(os.environ.get("SCOUT_FEEDBACK_SCALE", "0.0"))\n'
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
    if "self.scout_feedback_scale" not in text and needle in text:
        text = text.replace(needle, repl, 1)

    # 2) Primitive controller: add per-step/lane/primitive scout bias from best projected alternative.
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

    # 3) Boundary feedback: push boundary positions selected by the projected council.
    old_boundary = 'boundary_raw = torch.sigmoid(self.boundary_logit[t].to(device=x.device, dtype=x.dtype) + self.boundary_controller_scale * boundary_bias)  # [B]'
    new_boundary = (
        'scout_boundary_bias = 0.0\n'
        '            if getattr(self, "scout_feedback_active", False):\n'
        '                try:\n'
        '                    _bt = self.scout_feedback.get("boundary_target", [])\n'
        '                    if t < len(_bt): scout_boundary_bias = float(_bt[t])\n'
        '                except Exception:\n'
        '                    scout_boundary_bias = 0.0\n'
        '            boundary_raw = torch.sigmoid(self.boundary_logit[t].to(device=x.device, dtype=x.dtype) + self.boundary_controller_scale * boundary_bias + self.scout_feedback_scale * scout_boundary_bias)  # [B]'
    )
    if old_boundary in text:
        text = text.replace(old_boundary, new_boundary, 1)

    # 4) Transition target: add bias to transition_type logits.
    trans_anchor = '            transition_type_logits = self.transition_type_net(gctx).float()\n'
    trans_inject = trans_anchor + (
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
    if trans_anchor in text and "_td = self.scout_feedback.get(\"transition_target\"" not in text:
        text = text.replace(trans_anchor, trans_inject, 1)

    # 5) Route edge basis: directly boost suggested route edges before route softmax.
    route_anchor = '            route_logits = route_logits + self.transition_controller_scale * route_boundary_gate.view(-1, 1, 1) * transition_bias\n'
    route_inject = route_anchor + (
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
    if route_anchor in text and "_rb = self.scout_feedback.get(\"route_bias\"" not in text:
        text = text.replace(route_anchor, route_inject, 1)

    # 6) Fanout target: add bias to fanout logits.
    fan_anchor = '            fanout_logits = self.fanout_context_net(gctx).float()\n'
    fan_inject = fan_anchor + (
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
    if fan_anchor in text and "_fd = self.scout_feedback.get(\"fanout_target\"" not in text:
        text = text.replace(fan_anchor, fan_inject, 1)

    # 7) Pass scout feedback into primitive unit.
    old_call = '                update, unit_info = unit.forward_with_context(x, read_packet, self.lane_embed, primitive_ctx)'
    new_call = '                update, unit_info = unit.forward_with_context(x, read_packet, self.lane_embed, primitive_ctx, t, self.scout_feedback if getattr(self, "scout_feedback_active", False) else None, self.scout_feedback_scale)'
    if old_call in text:
        text = text.replace(old_call, new_call, 1)

    # 8) Make report mention scout influence.
    old_ctx = '"boundary_route_gate_scale": float(os.environ.get("BOUNDARY_ROUTE_GATE_SCALE", "0.25")), "alive_scale": float(os.environ.get("ALIVE_CONTROLLER_SCALE", "0.0"))}'
    new_ctx = '"boundary_route_gate_scale": float(os.environ.get("BOUNDARY_ROUTE_GATE_SCALE", "0.25")), "alive_scale": float(os.environ.get("ALIVE_CONTROLLER_SCALE", "0.0")), "scout_feedback_path": os.environ.get("SCOUT_FEEDBACK_PATH", ""), "scout_feedback_scale": float(os.environ.get("SCOUT_FEEDBACK_SCALE", "0.0"))}'
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
