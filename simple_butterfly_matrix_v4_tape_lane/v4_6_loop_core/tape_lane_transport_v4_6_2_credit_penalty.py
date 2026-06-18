#!/usr/bin/env python3
"""v4.6.2 credit-penalty loop core.

Adds delayed penalties for actions that validation ablation says are bad.
Epoch N ablation is applied only during epoch N+1, so the model is not trained
from future information inside the same epoch.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F

try:
    from .modules import closed_loop_aux_losses
    from .modules.step_analyzer import StepAnalyzer, append_metrics_csv
    from .tape_lane_transport_v4_6_loop_core import (
        LANE_NAMES, amp_dtype, ensure_dir, make_loaders, metrics_fields,
        set_seed, sync_if_cuda, tau_for_epoch, weighted_loss, write_json,
    )
    from .tape_lane_transport_v4_6_1_grouped_core import V461GroupedLoopCoreClassifier, build_context_stats
except ImportError:
    ROOT = Path(__file__).resolve().parents[2]
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    from simple_butterfly_matrix_v4_tape_lane.v4_6_loop_core.modules import closed_loop_aux_losses  # type: ignore
    from simple_butterfly_matrix_v4_tape_lane.v4_6_loop_core.modules.step_analyzer import StepAnalyzer, append_metrics_csv  # type: ignore
    from simple_butterfly_matrix_v4_tape_lane.v4_6_loop_core.tape_lane_transport_v4_6_loop_core import (  # type: ignore
        LANE_NAMES, amp_dtype, ensure_dir, make_loaders, metrics_fields,
        set_seed, sync_if_cuda, tau_for_epoch, weighted_loss, write_json,
    )
    from simple_butterfly_matrix_v4_tape_lane.v4_6_loop_core.tape_lane_transport_v4_6_1_grouped_core import V461GroupedLoopCoreClassifier, build_context_stats  # type: ignore


class V462CreditPenaltyLoopCoreClassifier(V461GroupedLoopCoreClassifier):
    """v4.6.1 grouped model plus optional step ablation."""

    def forward(self, wav: torch.Tensor, tau: float = 1.0, ablate_group: Optional[int] = None, ablate_primitive: Optional[int] = None, ablate_step: Optional[int] = None):
        features = self.input_proj(self.frontend(wav))
        bsz = features.shape[0]
        dtype = features.dtype
        device = features.device
        state = features[:, 0, :].unsqueeze(1) + self.initial_lane.to(device=device, dtype=dtype).unsqueeze(0)
        mem = self.memory.init_state(bsz, device=device, dtype=dtype)
        memory_read = torch.zeros_like(mem)
        previous_state: Optional[torch.Tensor] = None
        previous_choice_context = torch.zeros(bsz, self.lanes, self.dim, device=device, dtype=dtype)
        previous_update_norm: Optional[torch.Tensor] = None
        tr: Dict[str, List[torch.Tensor]] = {k: [] for k in [
            "boundary", "route", "primitive_weights", "primitive_signs", "group_weights", "compose_weights", "write_gate",
            "memory_write_norm", "memory_read_norm", "memory_read_influence", "update_norm", "controller_z_norm", "context_stats_norm",
        ]}
        memory_forget = torch.sigmoid(self.memory.forget_logit)

        for t in range(self.steps):
            evidence = features[:, t, :]
            context_stats = build_context_stats(evidence, state, memory_read, previous_update_norm, t, self.steps)
            ctrl = self.controller(state, memory_read, t, evidence=evidence, context_stats=context_stats, previous_choice_context=previous_choice_context)
            prim = self.selector(
                state, ctrl.z, ctrl.primitive_scores, ctrl.primitive_sign_logits,
                tau=float(tau), memory_read=memory_read, previous_state=previous_state,
                ablate_group=ablate_group, ablate_primitive=ablate_primitive,
            )
            boundary = torch.sigmoid(ctrl.boundary_logit)
            route = torch.softmax(ctrl.route_logits.float(), dim=-1).to(dtype=state.dtype)
            write_gate = torch.sigmoid(ctrl.write_logits)
            fanout = torch.sigmoid(ctrl.fanout_logits)
            update = prim.update * fanout.unsqueeze(-1)
            memory_write_logits = ctrl.memory_write_logits
            if ablate_step is not None and int(ablate_step) == int(t):
                update = torch.zeros_like(update)
                write_gate = torch.zeros_like(write_gate)
                boundary = torch.zeros_like(boundary)
                memory_write_logits = torch.full_like(ctrl.memory_write_logits, -20.0)

            routed = torch.einsum("bst,bsd->btd", route, update)
            mixed, compose_w = self._compose(state, routed, write_gate, ctrl.compose_logits)
            self_update = state + write_gate.unsqueeze(-1) * update
            state_before_read = (1.0 - boundary.view(bsz, 1, 1)) * self_update + boundary.view(bsz, 1, 1) * mixed
            mem_out = self.memory(mem, ctrl.z, write_gate=memory_write_logits, read_gate=ctrl.memory_read_gate_logits)
            mem = mem_out.mem_next
            memory_read = mem_out.read
            read_gate = torch.sigmoid(ctrl.memory_read_gate_logits).view(bsz, 1, 1).to(dtype=state.dtype)
            read_inject = read_gate * self.memory_into_state(memory_read).view(bsz, 1, self.dim)
            state = self.state_norm(state_before_read + read_inject)
            previous_state = state_before_read
            previous_choice_context = prim.choice_context
            previous_update_norm = update.float().norm(dim=-1).mean(dim=-1)

            tr["boundary"].append(boundary)
            tr["route"].append(route)
            tr["primitive_weights"].append(prim.weights)
            tr["primitive_signs"].append(prim.signs)
            tr["group_weights"].append(prim.group_weights)
            tr["compose_weights"].append(compose_w)
            tr["write_gate"].append(write_gate)
            tr["memory_write_norm"].append(mem_out.write_vec.float().norm(dim=-1))
            tr["memory_read_norm"].append(memory_read.float().norm(dim=-1))
            tr["memory_read_influence"].append(read_inject.float().norm(dim=-1).mean(dim=-1))
            tr["update_norm"].append(previous_update_norm)
            tr["controller_z_norm"].append(ctrl.stats["controller_z_norm"].expand(bsz))
            tr["context_stats_norm"].append(context_stats.float().norm(dim=-1).mean(dim=-1))

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
            "group_weights": torch.stack(tr["group_weights"], dim=1),
            "compose_weights": torch.stack(tr["compose_weights"], dim=1),
            "write_gate": torch.stack(tr["write_gate"], dim=1),
            "memory_write_norm": torch.stack(tr["memory_write_norm"], dim=1),
            "memory_read_norm": torch.stack(tr["memory_read_norm"], dim=1),
            "memory_read_influence": torch.stack(tr["memory_read_influence"], dim=1) + output_mem_influence.view(bsz, 1) / float(max(1, self.steps)),
            "update_norm": torch.stack(tr["update_norm"], dim=1),
            "controller_z_norm": torch.stack(tr["controller_z_norm"], dim=1),
            "context_stats_norm": torch.stack(tr["context_stats_norm"], dim=1),
            "memory_forget": memory_forget,
            "logits": logits,
        }
        return logits, trace


@torch.no_grad()
def report_credit_ablation(model, wav: torch.Tensor, y: torch.Tensor, tau: float) -> Dict[str, Dict[str, float]]:
    model.eval()
    logits, _ = model(wav, tau=tau)
    base = F.cross_entropy(logits.float(), y).float()
    out: Dict[str, Dict[str, float]] = {"groups": {}, "primitives": {}, "steps": {}}
    for gid, name in enumerate(model.selector.group_names):
        l2, _ = model(wav, tau=tau, ablate_group=gid)
        out["groups"][name] = float((F.cross_entropy(l2.float(), y).float() - base).detach().cpu())
    for pid, name in enumerate(model.selector.primitive_names):
        l2, _ = model(wav, tau=tau, ablate_primitive=pid)
        out["primitives"][name] = float((F.cross_entropy(l2.float(), y).float() - base).detach().cpu())
    for step_id in range(int(model.steps)):
        l2, _ = model(wav, tau=tau, ablate_step=step_id)
        out["steps"][f"step_{step_id:02d}"] = float((F.cross_entropy(l2.float(), y).float() - base).detach().cpu())
    return out


def _bad_vector(names: List[str], credit: Dict[str, float], device: torch.device, scale: float) -> torch.Tensor:
    vals = [max(0.0, -float(credit.get(name, 0.0))) for name in names]
    t = torch.tensor(vals, device=device, dtype=torch.float32)
    if t.numel() and float(t.max().detach().cpu()) > 0.0:
        t = t / t.max().clamp_min(1e-8)
    return t * float(scale)


def build_credit_penalty_state(model, credit: Optional[Dict], args, device: str) -> Optional[Dict[str, torch.Tensor]]:
    if not bool(getattr(args, "enable_credit_penalty", False)) or not credit:
        return None
    dev = torch.device(device)
    group_bad = _bad_vector(model.selector.group_names, credit.get("groups") or {}, dev, float(args.credit_bad_group_scale))
    prim_bad = _bad_vector(model.selector.primitive_names, credit.get("primitives") or {}, dev, float(args.credit_bad_primitive_scale))
    step_names = [f"step_{i:02d}" for i in range(int(model.steps))]
    step_bad = _bad_vector(step_names, credit.get("steps") or {}, dev, float(args.credit_bad_step_scale))
    if group_bad.sum() <= 0 and prim_bad.sum() <= 0 and step_bad.sum() <= 0:
        return None
    return {"group_bad": group_bad, "primitive_bad": prim_bad, "step_bad": step_bad}


def credit_penalty_loss(trace: Dict[str, torch.Tensor], penalty: Optional[Dict[str, torch.Tensor]]) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    device = trace["logits"].device
    if penalty is None:
        z = torch.zeros((), device=device)
        return z, {"credit_bad_group_loss": z, "credit_bad_primitive_loss": z, "credit_bad_step_loss": z, "credit_penalty_total": z}
    group_loss = (trace["group_weights"].float() * penalty["group_bad"].view(1, 1, 1, -1)).sum(dim=-1).mean()
    prim_loss = (trace["primitive_weights"].float() * penalty["primitive_bad"].view(1, 1, 1, -1)).sum(dim=-1).mean()
    action_mass = trace["boundary"].float() + trace["write_gate"].float().mean(dim=-1) + 0.1 * trace["update_norm"].float()
    step_loss = (action_mass * penalty["step_bad"].view(1, -1)).mean()
    total = group_loss + prim_loss + step_loss
    return total, {"credit_bad_group_loss": group_loss.detach(), "credit_bad_primitive_loss": prim_loss.detach(), "credit_bad_step_loss": step_loss.detach(), "credit_penalty_total": total.detach()}


def train_epoch(model, loader, opt, scaler, device: str, dtype: torch.dtype, args, epoch: int, credit_penalty: Optional[Dict[str, torch.Tensor]]) -> Dict[str, float]:
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
            cp_loss, cp_terms = credit_penalty_loss(trace, credit_penalty)
            if bool(getattr(args, "enable_credit_penalty", False)):
                loss = loss + cp_loss.to(dtype=loss.dtype)
                aux.update(cp_terms)
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
    out = {"loss": totals["loss"] / max(1, totals["n"]), "ce": totals["ce"] / max(1, totals["n"]), "acc": totals["correct"] / max(1, totals["n"]), "gumbel_tau": tau, "train_seconds": elapsed, "train_batches_per_sec": batches / elapsed, "train_samples_per_sec": totals["n"] / elapsed}
    for k, v in aux_sum.items():
        out[k] = v / max(1, totals["n"])
    return out


@torch.no_grad()
def evaluate(model, loader, device: str, dtype: torch.dtype, args, tau: float):
    model.eval()
    use_amp = device.startswith("cuda") and dtype != torch.float32
    total_loss, correct, n = 0.0, 0, 0
    conf = torch.zeros(int(args.num_classes), int(args.num_classes), dtype=torch.long)
    last_trace = None
    aux_last: Dict[str, float] = {}
    credit = None
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
        if bool(args.enable_credit_ablation) and credit is None:
            small_wav = wav[: min(int(args.credit_ablation_batch), wav.shape[0])]
            small_y = y[: small_wav.shape[0]]
            credit = report_credit_ablation(model, small_wav, small_y, tau)
    sync_if_cuda(device)
    elapsed = max(1e-9, time.perf_counter() - t0)
    return {"loss": total_loss / max(1, n), "acc": correct / max(1, n), "n": n, "confusion": conf.tolist(), "trace": last_trace, "aux": aux_last, "credit_ablation": credit, "eval_seconds": elapsed, "eval_batches_per_sec": batches / elapsed, "eval_samples_per_sec": n / elapsed}


def run(args) -> None:
    set_seed(args.seed)
    if bool(getattr(args, "synthetic_data", False)) or bool(getattr(args, "allow_synthetic_fallback", False)):
        raise SystemExit("v4.6.2 evidence run forbids synthetic data/fallback; use validate RUN_SYNTHETIC_SMOKE=1 only")
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
    model = V462CreditPenaltyLoopCoreClassifier(len(classes), args).to(device)
    params = sum(p.numel() for p in model.parameters())
    print(f"loaded datasets: train={len(train_loader.dataset)} val={len(val_loader.dataset)} classes={classes}", flush=True)
    print(f"V462CreditPenaltyLoopCore params={params} steps={args.steps} lanes={args.lanes} D={args.dim} groups={model.selector.group_names} K={args.num_primitives} rank={args.primitive_rank} route_prior={args.route_prior_strength} credit_penalty={bool(args.enable_credit_penalty)} device={device} amp={args.amp}", flush=True)

    decay, no_decay = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        (no_decay if name.endswith("bias") or "norm" in name.lower() or "forget_logit" in name else decay).append(p)
    opt = torch.optim.AdamW([{"params": decay, "weight_decay": float(args.weight_decay)}, {"params": no_decay, "weight_decay": 0.0}], lr=float(args.lr), betas=(0.9, 0.95))
    scaler = torch.amp.GradScaler("cuda", enabled=device.startswith("cuda") and dtype == torch.float16)
    analyzer = StepAnalyzer(lane_names=LANE_NAMES[: int(args.lanes)], primitive_names=model.selector.primitive_names, boundary_peak_threshold=float(args.boundary_peak_threshold))

    best = 0.0
    best_epoch = 0
    final_summary = None
    previous_credit = None
    fields = metrics_fields() + ["group_entropy", "group_top1_share", "credit_bad_group_loss", "credit_bad_primitive_loss", "credit_bad_step_loss", "credit_penalty_total"]
    for epoch in range(1, int(args.epochs) + 1):
        penalty_state = build_credit_penalty_state(model, previous_credit, args, device)
        tr = train_epoch(model, train_loader, opt, scaler, device, dtype, args, epoch, penalty_state)
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
        append_metrics_csv(out_dir / "metrics.csv", row, fields)
        metrics = {"train_loss": tr["loss"], "train_ce": tr["ce"], "train_acc": tr["acc"], "val_loss": va["loss"], "val_acc": va["acc"], "best_acc": best, "best_epoch": best_epoch}
        trace = va.get("trace")
        if trace is not None:
            summary = analyzer.summarize(trace, epoch=epoch, gumbel_tau=tau, metrics=metrics, aux_losses=va.get("aux") or {})
            summary["groups"] = {"names": model.selector.group_names, "weights_by_step_lane": trace["group_weights"].detach().float().mean(dim=0).cpu().tolist()}
            summary["credit_ablation"] = va.get("credit_ablation")
            summary["credit_penalty"] = {"enabled": bool(args.enable_credit_penalty), "bad_group_scale": float(args.credit_bad_group_scale), "bad_primitive_scale": float(args.credit_bad_primitive_scale), "bad_step_scale": float(args.credit_bad_step_scale)}
            summary["context"] = {"context_stats_norm_mean": float(trace["context_stats_norm"].detach().float().mean().cpu())}
            final_summary = summary
            analyzer.write_artifacts(out_dir, summary)
            write_json(out_dir / f"credit_ablation_epoch_{epoch:03d}.json", va.get("credit_ablation") or {})
        print(f"epoch {epoch:03d}/{args.epochs} train={tr['loss']:.4f}/{100*tr['acc']:.2f}% val={va['loss']:.4f}/{100*va['acc']:.2f}% best={100*best:.2f}%@{best_epoch} tau={tau:.3f}", flush=True)

    final_report = {"version": "v4.6.2_credit_penalty_core", "best_acc": best, "best_epoch": best_epoch, "args": vars(args), "classes": classes, "train_counts": train_counts, "val_counts": val_counts, "group_names": model.selector.group_names, "primitive_names": model.selector.primitive_names, "last_summary": final_summary, "closed_loop_invariant": "The controller decision affects execution, execution affects loss, loss gradient updates the controller decision path."}
    write_json(out_dir / "final_report.json", final_report)
    if final_summary is not None:
        analyzer.write_artifacts(out_dir, final_summary, final_report=final_report)


def parser() -> argparse.ArgumentParser:
    from simple_butterfly_matrix_v4_tape_lane.v4_6_loop_core.tape_lane_transport_v4_6_loop_core import parser as base_parser  # type: ignore
    p = base_parser()
    p.set_defaults(data_root="../architecture_builder/data/speechcommands", num_primitives=18, route_prior_strength=0.0, out_dir="./simple_butterfly_matrix_v4_tape_lane/agent_reports/v4_6_2_credit_penalty_core")
    p.add_argument("--enable-credit-ablation", action="store_true", default=True)
    p.add_argument("--credit-ablation-batch", type=int, default=32)
    p.add_argument("--enable-credit-penalty", action="store_true", default=False)
    p.add_argument("--credit-bad-group-scale", type=float, default=0.020)
    p.add_argument("--credit-bad-primitive-scale", type=float, default=0.012)
    p.add_argument("--credit-bad-step-scale", type=float, default=0.010)
    return p


if __name__ == "__main__":
    run(parser().parse_args())
