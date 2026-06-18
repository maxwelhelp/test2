#!/usr/bin/env python3
"""Compact reports for v4.6 loop core.

StepAnalyzer is deliberately report-only. It detaches summaries and never drives
training by files or external feedback.
"""
from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import torch

from .losses import allowed_route_mask, entropy_from_probs


def _float(x, default: float = 0.0) -> float:
    try:
        if isinstance(x, torch.Tensor):
            return float(x.detach().float().cpu())
        return float(x)
    except Exception:
        return float(default)


def _tolist(x: torch.Tensor):
    return x.detach().float().cpu().tolist()


def _json_dump(path: Path, data: Dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def append_metrics_csv(path: Path, row: Dict[str, object], fieldnames: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists()
    with path.open("a", newline="", encoding="utf-8") as f:
        wr = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        if not exists:
            wr.writeheader()
        wr.writerow(row)


class StepAnalyzer:
    def __init__(
        self,
        lane_names: Optional[Sequence[str]] = None,
        primitive_names: Optional[Sequence[str]] = None,
        boundary_peak_threshold: float = 0.35,
    ) -> None:
        self.lane_names = list(lane_names or ["detail", "state", "abstract", "memory"])
        self.primitive_names = list(primitive_names or [])
        self.boundary_peak_threshold = float(boundary_peak_threshold)

    def summarize(
        self,
        trace: Dict[str, torch.Tensor],
        *,
        epoch: int,
        gumbel_tau: float,
        metrics: Dict[str, float],
        aux_losses: Optional[Dict[str, float]] = None,
    ) -> Dict:
        boundary = trace["boundary"].detach().float()  # [B,T]
        route = trace["route"].detach().float()        # [B,T,L,L]
        prim_w = trace["primitive_weights"].detach().float()
        prim_s = trace["primitive_signs"].detach().float()
        compose = trace.get("compose_weights")
        write_gate = trace.get("write_gate")
        mem_write = trace.get("memory_write_norm")
        mem_read = trace.get("memory_read_norm")
        mem_infl = trace.get("memory_read_influence")

        b_step = boundary.mean(dim=0)
        b_mean = b_step.mean()
        b_std = b_step.std(unbiased=False) if b_step.numel() > 1 else torch.zeros(())
        b_peaks = (b_step > self.boundary_peak_threshold).nonzero(as_tuple=False).view(-1).tolist()

        lanes = int(route.shape[-1])
        r_mean = route.mean(dim=0)  # [T,L,L]
        r_ent_by_step = entropy_from_probs(route, dim=-1).mean(dim=(0, 2))
        eye = torch.eye(lanes, dtype=route.dtype, device=route.device)
        self_mass_by_step = (route * eye.view(1, 1, lanes, lanes)).sum(dim=(-2, -1)).mean(dim=0) / float(max(1, lanes))
        mask = allowed_route_mask(lanes, device=route.device, dtype=route.dtype)
        disallowed = (1.0 - mask).view(1, 1, lanes, lanes)
        disallowed_by_step = (route * disallowed).sum(dim=-1).mean(dim=(0, 2))
        useful_parts = []
        if lanes >= 2:
            useful_parts.append(route[..., 0, 1])
        if lanes >= 3:
            useful_parts.append(route[..., 1, 2])
        if lanes >= 4:
            useful_parts.append(route[..., 1, 3])
            useful_parts.append(route[..., 3, 1])
        useful_by_step = torch.stack(useful_parts, dim=0).mean(dim=(0, 1)) if useful_parts else torch.zeros(route.shape[1])

        pw_mean = prim_w.mean(dim=0)  # [T,L,K]
        ps_mean = prim_s.mean(dim=0)
        p_ent = entropy_from_probs(prim_w, dim=-1)
        p_ent_by_step_lane = p_ent.mean(dim=0)
        p_top1 = pw_mean.argmax(dim=-1)
        top1_counts = torch.bincount(p_top1.reshape(-1), minlength=prim_w.shape[-1]).float()
        top1_share = top1_counts.max() / float(max(1, p_top1.numel()))
        neg_share = (prim_s < 0).float().mean()
        sign_abs = prim_s.abs().mean()

        mem_write_step = mem_write.detach().float().mean(dim=0) if isinstance(mem_write, torch.Tensor) else torch.zeros(boundary.shape[1])
        mem_read_step = mem_read.detach().float().mean(dim=0) if isinstance(mem_read, torch.Tensor) else torch.zeros(boundary.shape[1])
        mem_infl_step = mem_infl.detach().float().mean(dim=0) if isinstance(mem_infl, torch.Tensor) else torch.zeros(boundary.shape[1])

        aux_losses = dict(aux_losses or {})
        task_loss = float(metrics.get("val_loss", metrics.get("train_ce", 0.0)) or 0.0)
        aux_total = float(aux_losses.get("aux_total", 0.0) or 0.0)
        aux_ratio = aux_total / max(1e-8, task_loss)
        flags: List[str] = []
        if _float(b_mean) > 0.80 and len(b_peaks) >= max(1, boundary.shape[1] - 1):
            flags.append("BOUNDARY_EXPLOIT")
        if _float(b_mean) < 0.05 or len(b_peaks) == 0:
            flags.append("BOUNDARY_DEAD")
        if _float(b_std) < 0.015:
            flags.append("BOUNDARY_FLAT")
        if _float(r_ent_by_step.mean()) > math.log(max(2, lanes)) * 0.90:
            flags.append("ROUTE_UNIFORM")
        if _float(self_mass_by_step.mean()) > 0.90:
            flags.append("ROUTE_IDENTITY_COLLAPSE")
        if _float(top1_share) > 0.85:
            flags.append("PRIMITIVE_COLLAPSE")
        if _float(p_ent.mean()) > math.log(max(2, prim_w.shape[-1])) * 0.92:
            flags.append("PRIMITIVE_UNIFORM")
        if _float(neg_share) < 0.02 or _float(sign_abs) < 0.02:
            flags.append("SIGN_DEAD")
        if _float(mem_write_step.mean()) < 1e-4 and _float(mem_read_step.mean()) < 1e-4:
            flags.append("MEMORY_DEAD")
        if _float(mem_write_step.mean()) > 0.10 and _float(mem_infl_step.mean()) < 1e-4:
            flags.append("MEMORY_JUNK")
        if aux_ratio > 0.50:
            flags.append("COST_DOMINANCE")

        return {
            "epoch": int(epoch),
            "gumbel_tau": float(gumbel_tau),
            "metrics": metrics,
            "cost_breakdown": aux_losses,
            "cost_dominance": aux_ratio,
            "lane_names": self.lane_names[:lanes],
            "primitive_names": self.primitive_names,
            "boundary": {
                "by_step": _tolist(b_step),
                "mean": _float(b_mean),
                "std": _float(b_std),
                "peak_count": len(b_peaks),
                "peaks": b_peaks,
            },
            "route": {
                "matrix_mean_by_step": _tolist(r_mean),
                "entropy_by_step": _tolist(r_ent_by_step),
                "entropy_mean": _float(r_ent_by_step.mean()),
                "self_route_by_step": _tolist(self_mass_by_step),
                "self_route_mass": _float(self_mass_by_step.mean()),
                "useful_transition_by_step": _tolist(useful_by_step),
                "useful_transition_mass": _float(useful_by_step.mean()),
                "disallowed_route_by_step": _tolist(disallowed_by_step),
                "disallowed_route_mass": _float(disallowed_by_step.mean()),
            },
            "primitive": {
                "weights_by_step_lane": _tolist(pw_mean),
                "top1_by_step_lane": p_top1.detach().cpu().tolist(),
                "entropy_by_step_lane": _tolist(p_ent_by_step_lane),
                "entropy_mean": _float(p_ent.mean()),
                "top1_share": _float(top1_share),
                "sign_mean_by_step_lane_primitive": _tolist(ps_mean),
                "sign_negative_share": _float(neg_share),
                "sign_abs_mean": _float(sign_abs),
            },
            "compose": {
                "weights_by_step_lane": _tolist(compose.detach().float().mean(dim=0)) if isinstance(compose, torch.Tensor) else [],
            },
            "memory": {
                "forget": _float(trace.get("memory_forget", torch.tensor(0.0))),
                "write_norm_by_step": _tolist(mem_write_step),
                "read_norm_by_step": _tolist(mem_read_step),
                "read_influence_by_step": _tolist(mem_infl_step),
                "write_norm_mean": _float(mem_write_step.mean()),
                "read_norm_mean": _float(mem_read_step.mean()),
                "read_influence_proxy": _float(mem_infl_step.mean()),
            },
            "write_gate_by_step_lane": _tolist(write_gate.detach().float().mean(dim=0)) if isinstance(write_gate, torch.Tensor) else [],
            "collapse_flags": flags,
        }

    def write_artifacts(
        self,
        out_dir: Path,
        summary: Dict,
        *,
        final_report: Optional[Dict] = None,
    ) -> None:
        out_dir.mkdir(parents=True, exist_ok=True)
        epoch = int(summary.get("epoch", 0))
        _json_dump(out_dir / f"trace_epoch_{epoch:03d}.json", summary)
        if final_report is not None:
            _json_dump(out_dir / "final_report.json", final_report)
        self.write_report_to_chatgpt(out_dir, summary)
        self.write_agent_status(out_dir, summary)

    def write_report_to_chatgpt(self, out_dir: Path, summary: Dict) -> None:
        b = summary.get("boundary", {})
        r = summary.get("route", {})
        p = summary.get("primitive", {})
        m = summary.get("memory", {})
        metrics = summary.get("metrics", {})
        flags = summary.get("collapse_flags") or []
        lines = [
            "REPORT_TO_CHATGPT",
            "",
            "Версия: v4.6_loop_core",
            f"epoch: {summary.get('epoch', 0)}  gumbel_tau={float(summary.get('gumbel_tau', 0.0)):.4f}",
            f"train_ce={float(metrics.get('train_ce', 0.0)):.4f} train_acc={100*float(metrics.get('train_acc', 0.0)):.2f}% "
            f"val_loss={float(metrics.get('val_loss', 0.0)):.4f} val_acc={100*float(metrics.get('val_acc', 0.0)):.2f}% "
            f"best={100*float(metrics.get('best_acc', 0.0)):.2f}%",
            "",
            "Closed loop:",
            "- The controller decision affects execution, execution affects loss, loss gradient updates the controller decision path.",
            "- Council/JSON feedback is not used for training; StepAnalyzer is report-only.",
            "",
            "Boundary / route:",
            f"- boundary_mean/std/peaks: {float(b.get('mean',0.0)):.4f}/{float(b.get('std',0.0)):.4f}/{b.get('peak_count',0)} peaks={b.get('peaks', [])}",
            f"- route_entropy_mean: {float(r.get('entropy_mean',0.0)):.4f}",
            f"- self_route_mass: {float(r.get('self_route_mass',0.0)):.4f}",
            f"- useful_transition_mass: {float(r.get('useful_transition_mass',0.0)):.4f}",
            f"- disallowed_route_mass: {float(r.get('disallowed_route_mass',0.0)):.4f}",
            "",
            "Primitive / signs:",
            f"- primitive_entropy_mean: {float(p.get('entropy_mean',0.0)):.4f}",
            f"- primitive_top1_share: {float(p.get('top1_share',0.0)):.4f}",
            f"- primitive_sign_negative_share: {float(p.get('sign_negative_share',0.0)):.4f}",
            f"- primitive_sign_abs_mean: {float(p.get('sign_abs_mean',0.0)):.4f}",
            "",
            "Memory:",
            f"- forget: {float(m.get('forget',0.0)):.4f}",
            f"- write_norm/read_norm/read_influence: {float(m.get('write_norm_mean',0.0)):.4f}/{float(m.get('read_norm_mean',0.0)):.4f}/{float(m.get('read_influence_proxy',0.0)):.4f}",
            "",
            f"cost_dominance aux/task: {float(summary.get('cost_dominance',0.0)):.4f}",
            f"collapse_flags: {','.join(flags) if flags else 'NONE'}",
            "",
            "Артефакты:",
            "- metrics.csv",
            "- trace_epoch_XXX.json",
            "- final_report.json",
            "- train.log если запускался sync script",
            "- AGENT_STATUS.md",
        ]
        (out_dir / "REPORT_TO_CHATGPT.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")

    def write_agent_status(self, out_dir: Path, summary: Dict) -> None:
        flags = summary.get("collapse_flags") or []
        status = "pass" if not flags else "needs_fix"
        lines = [
            "# Agent Status",
            "",
            "Current stage: v4.6 loop core MVP",
            "Entrypoint: simple_butterfly_matrix_v4_tape_lane/v4_6_loop_core/tape_lane_transport_v4_6_loop_core.py",
            f"Smoke status: {status}",
            f"Epoch: {summary.get('epoch', 0)}",
            "",
            "Closed-loop invariant:",
            "- The controller decision affects execution, execution affects loss, loss gradient updates the controller decision path.",
            "",
            "Latest structural flags:",
        ]
        if flags:
            lines.extend(f"- {f}" for f in flags)
        else:
            lines.append("- NONE")
        (out_dir / "AGENT_STATUS.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
