#!/usr/bin/env python3
"""v4.6.3 edge program with the convolutional frontend disabled.

This is an ablation entrypoint. It keeps the edge-conditioned program core, but
replaces ConvWaveFrontend with a non-convolutional adaptive pooling + linear
projection frontend so we can test how much Conv1D was doing.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from .tape_lane_transport_v4_6_3_edge_program import (
        EdgeProgramClassifier,
        build_penalty,
        train_epoch,
        evaluate,
    )
    from .modules.step_analyzer import StepAnalyzer, append_metrics_csv
    from .tape_lane_transport_v4_6_loop_core import (
        LANE_NAMES,
        amp_dtype,
        ensure_dir,
        make_loaders,
        metrics_fields,
        set_seed,
        sync_if_cuda,
        write_json,
    )
except ImportError:
    ROOT = Path(__file__).resolve().parents[2]
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    from simple_butterfly_matrix_v4_tape_lane.v4_6_loop_core.tape_lane_transport_v4_6_3_edge_program import EdgeProgramClassifier, build_penalty, train_epoch, evaluate  # type: ignore
    from simple_butterfly_matrix_v4_tape_lane.v4_6_loop_core.modules.step_analyzer import StepAnalyzer, append_metrics_csv  # type: ignore
    from simple_butterfly_matrix_v4_tape_lane.v4_6_loop_core.tape_lane_transport_v4_6_loop_core import LANE_NAMES, amp_dtype, ensure_dir, make_loaders, metrics_fields, set_seed, sync_if_cuda, write_json  # type: ignore


class LinearPatchWaveFrontend(nn.Module):
    """No-conv frontend: adaptive pooling windows followed by per-step linear maps."""

    def __init__(self, dim: int, steps: int, bins_per_step: int = 128, hidden_mult: int = 2) -> None:
        super().__init__()
        self.dim = int(dim)
        self.steps = int(steps)
        self.bins_per_step = int(bins_per_step)
        hidden = max(self.dim, int(hidden_mult) * self.dim)
        self.pool = nn.AdaptiveAvgPool1d(self.steps * self.bins_per_step)
        self.proj = nn.Sequential(
            nn.LayerNorm(self.bins_per_step),
            nn.Linear(self.bins_per_step, hidden),
            nn.SiLU(),
            nn.Linear(hidden, self.dim),
            nn.LayerNorm(self.dim),
        )

    def forward(self, wav: torch.Tensor) -> torch.Tensor:
        if wav.dim() == 3:
            wav = wav.squeeze(1)
        if wav.dim() != 2:
            wav = wav.reshape(wav.shape[0], -1)
        wav = wav.float()
        wav = wav - wav.mean(dim=-1, keepdim=True)
        wav = wav / wav.std(dim=-1, keepdim=True).clamp_min(1e-5)
        pooled = self.pool(wav.unsqueeze(1)).squeeze(1)
        patches = pooled.view(wav.shape[0], self.steps, self.bins_per_step)
        return self.proj(patches)


class NoConvEdgeProgramClassifier(EdgeProgramClassifier):
    def __init__(self, classes: int, args) -> None:
        super().__init__(classes, args)
        self.frontend = LinearPatchWaveFrontend(
            dim=int(args.dim),
            steps=int(args.steps),
            bins_per_step=int(args.no_conv_bins_per_step),
            hidden_mult=int(args.no_conv_hidden_mult),
        )


def run(args) -> None:
    set_seed(args.seed)
    if bool(getattr(args, "synthetic_data", False)) or bool(getattr(args, "allow_synthetic_fallback", False)):
        raise SystemExit("v4.6.3 no-conv evidence run forbids synthetic data/fallback")
    device = args.device if args.device != "cuda" or torch.cuda.is_available() else "cpu"
    if device.startswith("cuda"):
        torch.backends.cudnn.benchmark = True
        try:
            torch.set_float32_matmul_precision("high")
        except Exception:
            pass
    dtype = amp_dtype(args.amp)
    out_dir = ensure_dir(Path(args.out_dir))
    train_loader, val_loader, classes, train_counts, val_counts = make_loaders(args)
    args.num_classes = len(classes)
    model = NoConvEdgeProgramClassifier(len(classes), args).to(device)
    params = sum(p.numel() for p in model.parameters())
    print(f"loaded datasets: train={len(train_loader.dataset)} val={len(val_loader.dataset)} classes={classes}", flush=True)
    print(f"V463EdgeProgramNoConv params={params} steps={args.steps} lanes={args.lanes} edges={args.lanes*args.lanes} D={args.dim} bins={args.no_conv_bins_per_step} K={args.num_primitives} penalty={args.enable_edge_credit_penalty} device={device} amp={args.amp}", flush=True)

    decay, no_decay = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        (no_decay if name.endswith("bias") or "norm" in name.lower() or "forget_logit" in name else decay).append(p)
    opt = torch.optim.AdamW([{"params": decay, "weight_decay": float(args.weight_decay)}, {"params": no_decay, "weight_decay": 0.0}], lr=float(args.lr), betas=(0.9, 0.95))
    scaler = torch.amp.GradScaler("cuda", enabled=device.startswith("cuda") and dtype == torch.float16)
    analyzer = StepAnalyzer(lane_names=LANE_NAMES[: int(args.lanes)], primitive_names=model.primitive_bank.primitive_names, boundary_peak_threshold=float(args.boundary_peak_threshold))

    best = 0.0
    best_epoch = 0
    final_summary = None
    previous_credit = None
    fields = metrics_fields() + ["edge_gate_cost", "output_gate_cost", "dead_step_loss", "edge_credit_penalty", "bad_edge_loss", "bad_output_loss"]
    for epoch in range(1, int(args.epochs) + 1):
        penalty = build_penalty(model, previous_credit, args, device)
        tr = train_epoch(model, train_loader, opt, scaler, device, dtype, args, epoch, penalty)
        tau = float(tr["gumbel_tau"])
        va = evaluate(model, val_loader, device, dtype, args, tau)
        previous_credit = va.get("credit_ablation")
        if va["acc"] > best:
            best = float(va["acc"])
            best_epoch = int(epoch)
        row = {"epoch": epoch, "gumbel_tau": tau, "train_loss": tr["loss"], "train_ce": tr["ce"], "train_acc": tr["acc"], "val_loss": va["loss"], "val_acc": va["acc"], "best_acc": best, "train_seconds": tr.get("train_seconds", 0.0), "train_samples_per_sec": tr.get("train_samples_per_sec", 0.0), "eval_seconds": va.get("eval_seconds", 0.0), "eval_samples_per_sec": va.get("eval_samples_per_sec", 0.0)}
        for k in fields:
            if k in tr:
                row[k] = tr[k]
            elif k in va.get("aux", {}):
                row[k] = va["aux"][k]
        append_metrics_csv(out_dir / "metrics.csv", row, fields)
        metrics = {"train_loss": tr["loss"], "train_ce": tr["ce"], "train_acc": tr["acc"], "val_loss": va["loss"], "val_acc": va["acc"], "best_acc": best, "best_epoch": best_epoch}
        trace = va.get("trace")
        if trace is not None:
            summary = analyzer.summarize(trace, epoch=epoch, gumbel_tau=tau, metrics=metrics, aux_losses=va.get("aux") or {})
            edge_mass = trace["edge_gate"].float() * trace["edge_boundary"].float() * trace["edge_write_gate"].float()
            summary["edge"] = {
                "edge_gate_mean": float(trace["edge_gate"].float().mean().cpu()),
                "edge_boundary_mean": float(trace["edge_boundary"].float().mean().cpu()),
                "edge_write_mean": float(trace["edge_write_gate"].float().mean().cpu()),
                "edge_mass_by_step": edge_mass.mean(dim=(0, 2, 3)).cpu().tolist(),
            }
            summary["output"] = {"output_gate_by_step_slot": trace["output_gate"].float().mean(dim=0).cpu().tolist(), "output_gate_mean": float(trace["output_gate"].float().mean().cpu())}
            summary["credit_ablation"] = previous_credit
            summary["frontend"] = {"mode": "no_conv_linear_patch", "bins_per_step": int(args.no_conv_bins_per_step)}
            final_summary = summary
            analyzer.write_artifacts(out_dir, summary)
            write_json(out_dir / f"credit_ablation_epoch_{epoch:03d}.json", previous_credit or {})
        print(f"epoch {epoch:03d}/{args.epochs} train={tr['loss']:.4f}/{100*tr['acc']:.2f}% val={va['loss']:.4f}/{100*va['acc']:.2f}% best={100*best:.2f}%@{best_epoch} tau={tau:.3f}", flush=True)
    final_report = {"version": "v4.6.3_edge_program_no_conv", "frontend": "no_conv_linear_patch", "best_acc": best, "best_epoch": best_epoch, "args": vars(args), "classes": classes, "train_counts": train_counts, "val_counts": val_counts, "group_names": model.primitive_bank.group_names, "primitive_names": model.primitive_bank.primitive_names, "last_summary": final_summary, "closed_loop_invariant": "edge decisions affect execution; execution affects CE loss; delayed credit can penalize bad edge/step/output mass next epoch"}
    write_json(out_dir / "final_report.json", final_report)
    if final_summary is not None:
        analyzer.write_artifacts(out_dir, final_summary, final_report=final_report)


def parser() -> argparse.ArgumentParser:
    from simple_butterfly_matrix_v4_tape_lane.v4_6_loop_core.tape_lane_transport_v4_6_3_edge_program import parser as edge_parser  # type: ignore
    p = edge_parser()
    p.set_defaults(out_dir="./simple_butterfly_matrix_v4_tape_lane/agent_reports/v4_6_3_edge_program_no_conv")
    p.add_argument("--no-conv-bins-per-step", type=int, default=128)
    p.add_argument("--no-conv-hidden-mult", type=int, default=2)
    return p


if __name__ == "__main__":
    run(parser().parse_args())
