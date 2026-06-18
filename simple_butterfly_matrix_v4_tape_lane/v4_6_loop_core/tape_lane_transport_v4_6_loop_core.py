#!/usr/bin/env python3
"""v4.6 loop core: closed differentiable matrix program assembly loop.

The controller decision affects execution, execution affects loss, loss gradient
updates the controller decision path.
"""
from __future__ import annotations

import argparse
import json
import math
import random
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, Subset

try:  # package import when used as a module
    from .modules import JointController, MatrixMemory, ParallelPrimitiveSelector, closed_loop_aux_losses
    from .modules.step_analyzer import StepAnalyzer, append_metrics_csv
except ImportError:  # direct script execution from repo root
    ROOT = Path(__file__).resolve().parents[2]
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    from simple_butterfly_matrix_v4_tape_lane.v4_6_loop_core.modules import (  # type: ignore
        JointController,
        MatrixMemory,
        ParallelPrimitiveSelector,
        closed_loop_aux_losses,
    )
    from simple_butterfly_matrix_v4_tape_lane.v4_6_loop_core.modules.step_analyzer import (  # type: ignore
        StepAnalyzer,
        append_metrics_csv,
    )


LANE_NAMES = ["detail", "state", "abstract", "memory"]


def set_seed(seed: int) -> None:
    random.seed(int(seed))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def write_json(path: Path, data: Dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def amp_dtype(name: str) -> torch.dtype:
    n = str(name).lower()
    if n in {"fp16", "float16", "half"}:
        return torch.float16
    if n in {"bf16", "bfloat16"}:
        return torch.bfloat16
    return torch.float32


def sync_if_cuda(device: str) -> None:
    if str(device).startswith("cuda") and torch.cuda.is_available():
        torch.cuda.synchronize()


class SyntheticWaveCommands(Dataset):
    """Tiny deterministic fallback for import/smoke tests when real data is absent."""

    def __init__(self, n: int, classes: Sequence[str], sample_rate: int = 16000, seconds: float = 0.2, seed: int = 0) -> None:
        self.n = int(n)
        self.classes = list(classes)
        self.sample_rate = int(sample_rate)
        self.length = max(512, int(sample_rate * seconds))
        g = torch.Generator().manual_seed(int(seed))
        self.labels = torch.randint(0, len(self.classes), (self.n,), generator=g)
        self.phase = torch.rand(self.n, generator=g) * 2.0 * math.pi

    def __len__(self) -> int:
        return self.n

    def __getitem__(self, idx: int):
        y = int(self.labels[idx])
        t = torch.linspace(0.0, 1.0, self.length)
        base = 2.0 + float(y % max(1, len(self.classes)))
        wav = torch.sin(2 * math.pi * base * t + float(self.phase[idx]))
        wav = wav + 0.25 * torch.sin(2 * math.pi * (base * 2.7) * t)
        noise = torch.sin(2 * math.pi * (17.0 + idx % 13) * t) * 0.03
        return (wav + noise).float(), torch.tensor(y, dtype=torch.long)


def make_synthetic_loaders(args):
    classes = [c.strip() for c in str(args.classes).split(",") if c.strip()]
    if not classes:
        classes = ["yes", "no", "up", "down", "left", "right", "on", "off", "stop", "go"]
    train = SyntheticWaveCommands(max(int(args.train_limit), int(args.batch_size)), classes, args.sample_rate, seconds=float(args.synthetic_seconds), seed=args.seed)
    val = SyntheticWaveCommands(max(int(args.val_limit), int(args.eval_batch_size)), classes, args.sample_rate, seconds=float(args.synthetic_seconds), seed=args.seed + 17)
    train_loader = DataLoader(train, batch_size=int(args.batch_size), shuffle=True, num_workers=int(args.workers), pin_memory=bool(args.pin_memory), drop_last=False)
    val_loader = DataLoader(val, batch_size=int(args.eval_batch_size), shuffle=False, num_workers=int(args.workers), pin_memory=bool(args.pin_memory), drop_last=False)
    counts = {c: 0 for c in classes}
    for y in train.labels.tolist():
        counts[classes[int(y)]] += 1
    val_counts = {c: 0 for c in classes}
    for y in val.labels.tolist():
        val_counts[classes[int(y)]] += 1
    return train_loader, val_loader, classes, counts, val_counts


def _patch_v42_loader_args(args) -> None:
    """Compatibility fields expected by the old v4.2 SpeechCommands loader."""
    defaults = {
        "synthetic": False,
        "download": False,
        "seconds": 1.0,
        "synthetic_length": max(
            512,
            int(float(getattr(args, "synthetic_seconds", 0.2)) * int(getattr(args, "sample_rate", 16000))),
        ),
        "pin_memory": False,
        "workers": 0,
        "eval_batch_size": int(getattr(args, "batch_size", 128)),
    }
    for name, value in defaults.items():
        if not hasattr(args, name):
            setattr(args, name, value)


def make_loaders(args):
    if bool(args.synthetic_data):
        return make_synthetic_loaders(args)

    _patch_v42_loader_args(args)
    try:
        from simple_butterfly_matrix_v4_tape_lane import tape_lane_transport_v4_2_fixed as fixed  # type: ignore
        v42 = fixed.v42
        return v42.make_loaders(args)
    except Exception as exc:
        if bool(args.allow_synthetic_fallback):
            print(f"[v4.6] real loader failed, using synthetic fallback: {exc}", flush=True)
            return make_synthetic_loaders(args)
        raise RuntimeError(
            "failed to build real SpeechCommands loaders. Set DATA_ROOT/--data-root to the real dataset, "
            "or use SYNTHETIC_DATA=1 for a smoke-only run."
        ) from exc

class ConvWaveFrontend(nn.Module):
    """Simple matrix-friendly 1D frontend that returns [B,T,D]."""

    def __init__(self, dim: int, steps: int) -> None:
        super().__init__()
        self.dim = int(dim)
        self.steps = int(steps)
        mid = max(16, self.dim // 2)
        self.net = nn.Sequential(
            nn.Conv1d(1, mid, kernel_size=15, stride=8, padding=7),
            nn.GELU(),
            nn.Conv1d(mid, self.dim, kernel_size=9, stride=4, padding=4),
            nn.GELU(),
            nn.Conv1d(self.dim, self.dim, kernel_size=5, stride=2, padding=2),
            nn.GELU(),
        )
        self.norm = nn.LayerNorm(self.dim)

    def forward(self, wav: torch.Tensor) -> torch.Tensor:
        if wav.dim() == 3:
            wav = wav.squeeze(1)
        if wav.dim() != 2:
            wav = wav.reshape(wav.shape[0], -1)
        wav = wav.float()
        wav = wav - wav.mean(dim=-1, keepdim=True)
        wav = wav / wav.std(dim=-1, keepdim=True).clamp_min(1e-5)
        x = self.net(wav.unsqueeze(1))
        x = F.adaptive_avg_pool1d(x, self.steps).transpose(1, 2).contiguous()
        return self.norm(x)


class V46LoopCoreClassifier(nn.Module):
    def __init__(self, classes: int, args) -> None:
        super().__init__()
        self.classes = int(classes)
        self.dim = int(args.dim)
        self.steps = int(args.steps)
        self.lanes = int(args.lanes)
        self.compose_modes = int(args.compose_modes)
        self.frontend = ConvWaveFrontend(self.dim, self.steps)
        self.input_proj = nn.Linear(self.dim, self.dim)
        self.initial_lane = nn.Parameter(torch.randn(self.lanes, self.dim) * 0.02)
        self.controller = JointController(
            dim=self.dim,
            lanes=self.lanes,
            num_primitives=int(args.num_primitives),
            compose_modes=self.compose_modes,
            max_steps=max(self.steps, 32),
            dropout=float(args.dropout),
            route_prior_strength=float(args.route_prior_strength),
        )
        self.selector = ParallelPrimitiveSelector(dim=self.dim, num_primitives=int(args.num_primitives), rank=int(args.primitive_rank), dropout=float(args.dropout))
        self.memory = MatrixMemory(self.dim, init_forget_logit=float(args.memory_forget_logit_init))
        self.memory_into_state = nn.Linear(self.dim, self.dim, bias=False)
        self.output_memory_proj = nn.Linear(self.dim, self.dim, bias=False)
        self.output_memory_gate = nn.Parameter(torch.tensor(0.0))
        self.state_norm = nn.LayerNorm(self.dim)
        self.head = nn.Sequential(nn.LayerNorm(self.dim), nn.Linear(self.dim, self.dim), nn.SiLU(), nn.Dropout(float(args.dropout)), nn.Linear(self.dim, self.classes))

    def _compose(self, state: torch.Tensor, routed: torch.Tensor, write_gate: torch.Tensor, compose_logits: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        mode_w = torch.softmax(compose_logits.float(), dim=-1).to(dtype=state.dtype)
        replace = routed
        add = state + routed
        subtract = state - routed
        gated = state + write_gate.unsqueeze(-1) * routed
        modes = torch.stack([replace, add, subtract, gated], dim=-2)
        if mode_w.shape[-1] != modes.shape[-2]:
            mode_w = mode_w[..., : modes.shape[-2]]
        mixed = torch.einsum("blm,blmd->bld", mode_w, modes)
        return mixed, mode_w

    def forward(self, wav: torch.Tensor, tau: float = 1.0):
        features = self.input_proj(self.frontend(wav))
        bsz = features.shape[0]
        dtype = features.dtype
        device = features.device
        state = features[:, 0, :].unsqueeze(1) + self.initial_lane.to(device=device, dtype=dtype).unsqueeze(0)
        mem = self.memory.init_state(bsz, device=device, dtype=dtype)
        memory_read = torch.zeros_like(mem)
        previous_state: Optional[torch.Tensor] = None

        tr: Dict[str, List[torch.Tensor]] = {
            "boundary": [],
            "route": [],
            "primitive_weights": [],
            "primitive_signs": [],
            "compose_weights": [],
            "write_gate": [],
            "memory_write_norm": [],
            "memory_read_norm": [],
            "memory_read_influence": [],
            "update_norm": [],
            "controller_z_norm": [],
        }
        memory_forget = torch.sigmoid(self.memory.forget_logit)

        for t in range(self.steps):
            evidence = features[:, t, :]
            ctrl = self.controller(state, memory_read, t, evidence=evidence)
            prim = self.selector(state, ctrl.z, ctrl.primitive_scores, ctrl.primitive_sign_logits, tau=float(tau), memory_read=memory_read, previous_state=previous_state)
            boundary = torch.sigmoid(ctrl.boundary_logit)
            route = torch.softmax(ctrl.route_logits.float(), dim=-1).to(dtype=state.dtype)
            write_gate = torch.sigmoid(ctrl.write_logits)
            fanout = torch.sigmoid(ctrl.fanout_logits)
            update = prim.update * fanout.unsqueeze(-1)
            routed = torch.einsum("bst,bsd->btd", route, update)
            mixed, compose_w = self._compose(state, routed, write_gate, ctrl.compose_logits)
            self_update = state + write_gate.unsqueeze(-1) * update
            state_before_read = (1.0 - boundary.view(bsz, 1, 1)) * self_update + boundary.view(bsz, 1, 1) * mixed

            mem_out = self.memory(mem, ctrl.z, write_gate=ctrl.memory_write_logits, read_gate=ctrl.memory_read_gate_logits)
            mem = mem_out.mem_next
            memory_read = mem_out.read
            read_gate = torch.sigmoid(ctrl.memory_read_gate_logits).view(bsz, 1, 1).to(dtype=state.dtype)
            read_inject = read_gate * self.memory_into_state(memory_read).view(bsz, 1, self.dim)
            state = self.state_norm(state_before_read + read_inject)

            tr["boundary"].append(boundary)
            tr["route"].append(route)
            tr["primitive_weights"].append(prim.weights)
            tr["primitive_signs"].append(prim.signs)
            tr["compose_weights"].append(compose_w)
            tr["write_gate"].append(write_gate)
            tr["memory_write_norm"].append(mem_out.write_vec.float().norm(dim=-1))
            tr["memory_read_norm"].append(memory_read.float().norm(dim=-1))
            tr["memory_read_influence"].append(read_inject.float().norm(dim=-1).mean(dim=-1))
            tr["update_norm"].append(update.float().norm(dim=-1).mean(dim=-1))
            tr["controller_z_norm"].append(ctrl.stats["controller_z_norm"].expand(bsz))
            previous_state = state_before_read

        pooled = state.mean(dim=1)
        mem_component = self.output_memory_proj(memory_read)
        head_input_no_mem = self.state_norm(pooled)
        head_input = self.state_norm(pooled + torch.sigmoid(self.output_memory_gate).to(dtype=pooled.dtype) * mem_component)
        logits_without_mem = self.head(head_input_no_mem)
        logits = self.head(head_input)
        output_mem_influence = (logits - logits_without_mem).float().norm(dim=-1)

        trace = {
            "boundary": torch.stack(tr["boundary"], dim=1),
            "route": torch.stack(tr["route"], dim=1),
            "primitive_weights": torch.stack(tr["primitive_weights"], dim=1),
            "primitive_signs": torch.stack(tr["primitive_signs"], dim=1),
            "compose_weights": torch.stack(tr["compose_weights"], dim=1),
            "write_gate": torch.stack(tr["write_gate"], dim=1),
            "memory_write_norm": torch.stack(tr["memory_write_norm"], dim=1),
            "memory_read_norm": torch.stack(tr["memory_read_norm"], dim=1),
            "memory_read_influence": torch.stack(tr["memory_read_influence"], dim=1) + output_mem_influence.view(bsz, 1) / float(max(1, self.steps)),
            "update_norm": torch.stack(tr["update_norm"], dim=1),
            "controller_z_norm": torch.stack(tr["controller_z_norm"], dim=1),
            "memory_forget": memory_forget,
            "logits": logits,
        }
        return logits, trace


def tau_for_epoch(args, epoch: int) -> float:
    return max(float(args.gumbel_tau_min), float(args.gumbel_tau_start) * (float(args.gumbel_tau_decay) ** max(0, int(epoch) - 1)))


def weighted_loss(ce: torch.Tensor, aux: Dict[str, torch.Tensor], args) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    terms = {
        "boundary_budget_loss": float(args.lambda_boundary_budget),
        "boundary_flatness_loss": float(args.lambda_boundary_flatness),
        "route_entropy_band_loss": float(args.lambda_route_entropy),
        "route_allowed_loss": float(args.lambda_route_allowed),
        "route_identity_loss": float(args.lambda_route_identity),
        "primitive_uniform_loss": float(args.lambda_primitive_uniform),
        "primitive_diversity_loss": float(args.lambda_primitive_diversity),
        "sign_balance_loss": float(args.lambda_sign_balance),
        "program_cost": float(args.lambda_program_cost),
        "memory_write_cost": float(args.lambda_memory_write_cost),
        "logit_norm": float(args.lambda_logit_norm),
    }
    loss = ce
    aux_total = torch.zeros((), device=ce.device, dtype=ce.dtype)
    for name, weight in terms.items():
        if weight == 0.0 or name not in aux:
            continue
        val = aux[name].to(device=ce.device, dtype=ce.dtype)
        contrib = float(weight) * val
        loss = loss + contrib
        aux_total = aux_total + contrib.detach().float()
    aux["aux_total"] = aux_total
    return loss, aux


def train_epoch(model, loader, opt, scaler, device: str, dtype: torch.dtype, args, epoch: int) -> Dict[str, float]:
    model.train()
    use_amp = device.startswith("cuda") and dtype != torch.float32
    tau = tau_for_epoch(args, epoch)
    totals = {"loss": 0.0, "ce": 0.0, "correct": 0, "n": 0}
    aux_sum: Dict[str, float] = {}
    sync_if_cuda(device)
    t0 = time.perf_counter()
    batches = 0
    for step, (wav, y) in enumerate(loader, 1):
        if int(args.max_train_batches) > 0 and step > int(args.max_train_batches):
            break
        batches += 1
        wav = wav.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        opt.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.split(":")[0], dtype=dtype, enabled=use_amp):
            logits, trace = model(wav, tau=tau)
            ce = F.cross_entropy(logits.float(), y)
            aux = closed_loop_aux_losses(trace, args)
            loss, aux = weighted_loss(ce, aux, args)
        if not torch.isfinite(loss):
            print("NONFINITE_LOSS skip", flush=True)
            continue
        scaler.scale(loss).backward()
        if float(args.grad_clip) > 0:
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), float(args.grad_clip))
        scaler.step(opt)
        scaler.update()
        bs = int(y.numel())
        totals["loss"] += float(loss.detach().cpu()) * bs
        totals["ce"] += float(ce.detach().cpu()) * bs
        totals["correct"] += int((logits.argmax(-1) == y).sum().detach().cpu())
        totals["n"] += bs
        for k, v in aux.items():
            aux_sum[k] = aux_sum.get(k, 0.0) + float(v.detach().float().cpu()) * bs
        if int(args.log_every) > 0 and step % int(args.log_every) == 0:
            print(f"epoch {epoch:03d} step {step:05d} loss={totals['loss']/max(1, totals['n']):.4f} ce={totals['ce']/max(1, totals['n']):.4f} acc={100*totals['correct']/max(1, totals['n']):.2f}% tau={tau:.3f}", flush=True)
    sync_if_cuda(device)
    elapsed = max(1e-9, time.perf_counter() - t0)
    out = {
        "loss": totals["loss"] / max(1, totals["n"]),
        "ce": totals["ce"] / max(1, totals["n"]),
        "acc": totals["correct"] / max(1, totals["n"]),
        "gumbel_tau": tau,
        "train_seconds": elapsed,
        "train_batches_per_sec": batches / elapsed,
        "train_samples_per_sec": totals["n"] / elapsed,
    }
    for k, v in aux_sum.items():
        out[k] = v / max(1, totals["n"])
    return out


@torch.no_grad()
def evaluate(model, loader, device: str, dtype: torch.dtype, args, epoch: int, tau: float):
    model.eval()
    use_amp = device.startswith("cuda") and dtype != torch.float32
    total_loss, correct, n = 0.0, 0, 0
    conf = torch.zeros(int(args.num_classes), int(args.num_classes), dtype=torch.long)
    last_trace = None
    aux_last: Dict[str, float] = {}
    sync_if_cuda(device)
    t0 = time.perf_counter()
    batches = 0
    for step, (wav, y) in enumerate(loader, 1):
        if int(args.max_val_batches) > 0 and step > int(args.max_val_batches):
            break
        batches += 1
        wav = wav.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        with torch.autocast(device_type=device.split(":")[0], dtype=dtype, enabled=use_amp):
            logits, trace = model(wav, tau=tau)
            loss = F.cross_entropy(logits.float(), y)
            aux = closed_loop_aux_losses(trace, args)
            _, aux = weighted_loss(loss, aux, args)
        pred = logits.argmax(-1)
        bs = int(y.numel())
        total_loss += float(loss.detach().cpu()) * bs
        correct += int((pred == y).sum().detach().cpu())
        n += bs
        conf += torch.bincount((y.cpu() * int(args.num_classes) + pred.cpu()), minlength=int(args.num_classes) ** 2).view(int(args.num_classes), int(args.num_classes))
        last_trace = trace
        aux_last = {k: float(v.detach().float().cpu()) for k, v in aux.items()}
    sync_if_cuda(device)
    elapsed = max(1e-9, time.perf_counter() - t0)
    return {
        "loss": total_loss / max(1, n),
        "acc": correct / max(1, n),
        "n": n,
        "confusion": conf.tolist(),
        "trace": last_trace,
        "aux": aux_last,
        "eval_seconds": elapsed,
        "eval_batches_per_sec": batches / elapsed,
        "eval_samples_per_sec": n / elapsed,
    }


def metrics_fields() -> List[str]:
    return [
        "epoch", "gumbel_tau", "train_loss", "train_ce", "train_acc", "val_loss", "val_acc", "best_acc",
        "boundary_budget_loss", "boundary_flatness_loss", "boundary_mean", "boundary_std", "boundary_soft_peak_count",
        "route_entropy", "route_entropy_band_loss", "self_route_mass", "useful_transition_mass", "route_disallowed_mass",
        "route_allowed_loss", "route_identity_loss", "primitive_entropy", "primitive_top1_share",
        "primitive_sign_negative_share", "primitive_uniform_loss", "primitive_diversity_loss", "sign_balance_loss",
        "program_cost", "memory_write_cost", "logit_norm", "aux_total",
        "train_seconds", "train_samples_per_sec", "eval_seconds", "eval_samples_per_sec",
    ]


def run(args) -> None:
    set_seed(args.seed)
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
    model = V46LoopCoreClassifier(len(classes), args).to(device)
    params = sum(p.numel() for p in model.parameters())
    print(f"loaded datasets: train={len(train_loader.dataset)} val={len(val_loader.dataset)} classes={classes}", flush=True)
    print(f"V46LoopCore params={params} steps={args.steps} lanes={args.lanes} D={args.dim} K={args.num_primitives} rank={args.primitive_rank} device={device} amp={args.amp}", flush=True)

    decay, no_decay = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if name.endswith("bias") or "norm" in name.lower() or "forget_logit" in name:
            no_decay.append(p)
        else:
            decay.append(p)
    opt = torch.optim.AdamW([{"params": decay, "weight_decay": float(args.weight_decay)}, {"params": no_decay, "weight_decay": 0.0}], lr=float(args.lr), betas=(0.9, 0.95))
    scaler = torch.amp.GradScaler("cuda", enabled=device.startswith("cuda") and dtype == torch.float16)
    analyzer = StepAnalyzer(lane_names=LANE_NAMES[: int(args.lanes)], primitive_names=model.selector.primitive_names, boundary_peak_threshold=float(args.boundary_peak_threshold))

    best = 0.0
    best_epoch = 0
    final_summary: Optional[Dict] = None
    fields = metrics_fields()
    for epoch in range(1, int(args.epochs) + 1):
        tr = train_epoch(model, train_loader, opt, scaler, device, dtype, args, epoch)
        tau = float(tr["gumbel_tau"])
        va = evaluate(model, val_loader, device, dtype, args, epoch, tau)
        if va["acc"] > best:
            best = float(va["acc"])
            best_epoch = int(epoch)
        row = {"epoch": epoch, "gumbel_tau": tau, "train_loss": tr["loss"], "train_ce": tr["ce"], "train_acc": tr["acc"], "val_loss": va["loss"], "val_acc": va["acc"], "best_acc": best, "train_seconds": tr.get("train_seconds", 0.0), "train_samples_per_sec": tr.get("train_samples_per_sec", 0.0), "eval_seconds": va.get("eval_seconds", 0.0), "eval_samples_per_sec": va.get("eval_samples_per_sec", 0.0)}
        for k in fields:
            if k in tr:
                row[k] = tr[k]
        append_metrics_csv(out_dir / "metrics.csv", row, fields)

        metrics = {"train_loss": tr["loss"], "train_ce": tr["ce"], "train_acc": tr["acc"], "val_loss": va["loss"], "val_acc": va["acc"], "best_acc": best, "best_epoch": best_epoch}
        trace = va.get("trace")
        if trace is not None:
            summary = analyzer.summarize(trace, epoch=epoch, gumbel_tau=tau, metrics=metrics, aux_losses=va.get("aux") or {})
            final_summary = summary
            analyzer.write_artifacts(out_dir, summary)
        print(f"epoch {epoch:03d}/{args.epochs} train={tr['loss']:.4f}/{100*tr['acc']:.2f}% val={va['loss']:.4f}/{100*va['acc']:.2f}% best={100*best:.2f}%@{best_epoch} tau={tau:.3f}", flush=True)

    final_report = {"version": "v4.6_loop_core", "best_acc": best, "best_epoch": best_epoch, "args": vars(args), "classes": classes, "train_counts": train_counts, "val_counts": val_counts, "last_summary": final_summary, "closed_loop_invariant": "The controller decision affects execution, execution affects loss, loss gradient updates the controller decision path."}
    write_json(out_dir / "final_report.json", final_report)
    if final_summary is not None:
        analyzer.write_artifacts(out_dir, final_summary, final_report=final_report)


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="v4.6 differentiable loop core")
    p.add_argument("--data-root", default="../architecture_builder/data/speechcommands")
    p.add_argument("--classes", default="yes,no,up,down,left,right,on,off,stop,go")
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--train-limit", type=int, default=12000)
    p.add_argument("--val-limit", type=int, default=2000)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--eval-batch-size", type=int, default=256)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--pin-memory", action="store_true")
    p.add_argument("--download", action="store_true")
    p.add_argument("--seconds", type=float, default=1.0)
    p.add_argument("--synthetic", action="store_true", help="compat flag for old loader; prefer --synthetic-data")
    p.add_argument("--synthetic-length", type=int, default=3200)
    p.add_argument("--synthetic-data", action="store_true")
    p.add_argument("--allow-synthetic-fallback", action="store_true")
    p.add_argument("--sample-rate", type=int, default=16000)
    p.add_argument("--synthetic-seconds", type=float, default=0.2)
    p.add_argument("--n-mels", type=int, default=64)
    p.add_argument("--hop-length", type=int, default=160)
    p.add_argument("--device", default="cuda")
    p.add_argument("--amp", choices=["off", "fp32", "fp16", "bf16"], default="fp16")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--dim", type=int, default=128)
    p.add_argument("--steps", "--tape-steps", dest="steps", type=int, default=8)
    p.add_argument("--lanes", type=int, default=4)
    p.add_argument("--num-primitives", type=int, default=8)
    p.add_argument("--primitive-rank", type=int, default=32)
    p.add_argument("--compose-modes", type=int, default=4)
    p.add_argument("--dropout", type=float, default=0.05)
    p.add_argument("--route-prior-strength", type=float, default=0.35)
    p.add_argument("--memory-forget-logit-init", type=float, default=1.0)
    p.add_argument("--gumbel-tau-start", type=float, default=1.0)
    p.add_argument("--gumbel-tau-min", type=float, default=0.2)
    p.add_argument("--gumbel-tau-decay", type=float, default=0.92)
    p.add_argument("--lr", type=float, default=5e-4)
    p.add_argument("--weight-decay", type=float, default=0.01)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--lambda-boundary-budget", type=float, default=0.006)
    p.add_argument("--lambda-boundary-flatness", type=float, default=0.002)
    p.add_argument("--lambda-route-entropy", type=float, default=0.010)
    p.add_argument("--lambda-route-allowed", type=float, default=0.012)
    p.add_argument("--lambda-route-identity", type=float, default=0.008)
    p.add_argument("--lambda-primitive-uniform", type=float, default=0.004)
    p.add_argument("--lambda-primitive-diversity", type=float, default=0.002)
    p.add_argument("--lambda-sign-balance", type=float, default=0.001)
    p.add_argument("--lambda-program-cost", type=float, default=0.0005)
    p.add_argument("--lambda-memory-write-cost", type=float, default=0.0005)
    p.add_argument("--lambda-logit-norm", type=float, default=1e-5)
    p.add_argument("--boundary-min-peaks", type=float, default=1.0)
    p.add_argument("--boundary-max-peaks", type=float, default=4.0)
    p.add_argument("--boundary-peak-threshold", type=float, default=0.35)
    p.add_argument("--boundary-peak-tau", type=float, default=0.06)
    p.add_argument("--boundary-flatness-target", type=float, default=0.03)
    p.add_argument("--max-train-batches", type=int, default=0)
    p.add_argument("--max-val-batches", type=int, default=0)
    p.add_argument("--log-every", type=int, default=100)
    p.add_argument("--out-dir", default="./simple_butterfly_matrix_v4_tape_lane/agent_reports/v4_6_loop_core_smoke")
    p.add_argument("--no-save-checkpoints", action="store_true", default=True)
    return p


if __name__ == "__main__":
    run(parser().parse_args())
