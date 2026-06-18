#!/usr/bin/env python3
"""v4.3_min_heart: conservative bridge from v4.2 to ProgramHeart.

This version does NOT implement actor/critic/planner/auto-deploy.
It keeps the working v4.2 tape-lane body and adds:
  - costed differentiable paths;
  - boundary-coupled route economics;
  - differentiable detail-head shortcut cost;
  - route/boundary/read/memory/head trace JSON;
  - deterministic sparse candidate suggestions;
  - reports prepared for ChatGPT/agent review.

The purpose is to test whether sequence + separators become meaningful before
building the full ProgramHeart.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Import fixed runner first: it patches the old BlockButterfly bug before v4.2 loads.
from simple_butterfly_matrix_v4_tape_lane import tape_lane_transport_v4_2_fixed as fixed  # noqa: E402

v42 = fixed.v42

LANE_NAMES = v42.LANE_NAMES
READ_GROUP_NAMES = v42.READ_GROUP_NAMES
PRIMITIVES = v42.PRIMITIVES


def lane_name(i: int) -> str:
    return v42.lane_name(i)


def read_group_name(i: int) -> str:
    return v42.read_group_name(i)


def _to_float_list(x: torch.Tensor):
    return x.detach().float().cpu().tolist()


def _safe_float(x, default: float = 0.0) -> float:
    try:
        return float(x)
    except Exception:
        return float(default)


class ClassMatrixLaneHeadV43(v42.ClassMatrixLaneHead):
    """v4.2 head with non-detached attention for differentiable shortcut losses.

    v4.2 detached class_slot_attention inside haux. That was fine for reports but
    not for the v4.3 differentiable detail_head_shortcut_cost. This override keeps
    tensors live for loss computation while reports explicitly detach when needed.
    """

    def forward(self, slots: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        bsz, slot_count, dim = slots.shape
        lane_map_prior = v42.lane_slot_matrix(
            self.tape_steps, self.lanes, self.cells_per_lane, slots.device, normalize_rows=True
        ).to(slots.dtype)
        lane_map_mass = v42.lane_slot_matrix(
            self.tape_steps, self.lanes, self.cells_per_lane, slots.device, normalize_rows=False
        ).to(slots.dtype)
        lane_w = torch.softmax(self.class_lane_logits.float(), dim=-1).to(slots.dtype)  # [C,L]
        slot_prior = torch.matmul(lane_w, lane_map_prior).clamp_min(1e-8)  # [C,S]

        keys = slots @ self.key_w.to(device=slots.device, dtype=slots.dtype)
        values = self.drop(slots @ self.value_w.to(device=slots.device, dtype=slots.dtype))
        q = self.class_state.to(device=slots.device, dtype=slots.dtype) @ self.class_q.to(device=slots.device, dtype=slots.dtype)
        score = torch.einsum("cd,bsd->bcs", q, keys) / math.sqrt(dim)
        if self.lane_prior_strength != 0.0:
            score = score + self.lane_prior_strength * slot_prior.log().view(1, self.classes, slot_count)
        attn = torch.softmax(score.float(), dim=-1).to(slots.dtype)
        class_read = torch.einsum("bcs,bsd->bcd", attn, values)

        pair_w = torch.softmax(self.class_pair_logits.float(), dim=-1).to(slots.dtype)
        pair_base = torch.matmul(pair_w, self.pair_state.to(device=slots.device, dtype=slots.dtype))
        pair_q = class_read @ self.pair_q.to(device=slots.device, dtype=slots.dtype)
        pair_k = pair_base @ self.pair_k.to(device=slots.device, dtype=slots.dtype)
        pair_score = torch.einsum("bcd,ed->bce", pair_q, pair_k) / math.sqrt(dim)
        pair_attn = torch.softmax(pair_score.float(), dim=-1).to(slots.dtype)
        pair_v = pair_base @ self.pair_v.to(device=slots.device, dtype=slots.dtype)
        pair_ctx = torch.einsum("bce,ed->bcd", pair_attn, pair_v)

        cls = self.class_state.to(device=slots.device, dtype=slots.dtype).view(1, self.classes, dim).expand(bsz, -1, -1)
        delta = self.class_update(torch.cat([cls, class_read, pair_ctx, cls * class_read, class_read - pair_ctx], dim=-1))
        write = torch.sigmoid(self.write_logit.to(device=slots.device, dtype=slots.dtype))
        class_next = self.class_norm(cls + write * self.drop(delta))
        logits = (class_next * class_read * self.logit_w.to(device=slots.device, dtype=slots.dtype).view(1, self.classes, dim)).sum(dim=-1)
        logits = logits + self.logit_bias.to(device=slots.device, dtype=slots.dtype)

        lane_mass = torch.einsum("bcs,ls->bcl", attn.float(), lane_map_mass.float())
        return logits, {
            "class_slot_attention": attn,
            "class_lane_mass": lane_mass,
            "class_lane_logits": self.class_lane_logits,
            "class_read": class_read,
            "class_next": class_next,
            "pair_attention": pair_attn,
            "pair_update_norm": pair_ctx.float().norm(dim=-1).mean(),
            "class_write": write,
        }


class TapeLaneRouterClassifierV43(nn.Module):
    def __init__(self, classes: int, args):
        super().__init__()
        self.backbone = v42.TapeLaneRouterBackbone(
            dim=args.dim,
            evidence_cells=args.evidence_cells,
            lanes=args.lanes,
            cells_per_lane=args.cells_per_lane,
            tape_steps=args.tape_steps,
            sample_rate=args.sample_rate,
            n_mels=args.n_mels,
            hop_length=args.hop_length,
            channel_stages=args.channel_stages,
            dropout=args.dropout,
            structured_init_strength=args.structured_init_strength,
            read_prior_mode=args.read_prior_mode,
            read_prior_strength=args.read_prior_strength,
            route_prior_mode=args.route_prior_mode,
            route_prior_strength=args.route_prior_strength,
            boundary_route_strength=args.boundary_route_strength,
            context_primitive_scale=args.context_primitive_scale,
        )
        self.head = ClassMatrixLaneHeadV43(
            dim=args.dim,
            classes=classes,
            tape_steps=args.tape_steps,
            lanes=args.lanes,
            cells_per_lane=args.cells_per_lane,
            pair_slots=args.pair_slots,
            dropout=args.head_dropout,
            lane_prior_strength=args.lane_prior_strength,
            class_lane_prior_mode=args.class_lane_prior_mode,
            class_lane_init_strength=args.class_lane_init_strength,
        )

    def forward(self, wav: torch.Tensor):
        slots, baux = self.backbone(wav)
        logits, haux = self.head(slots)
        return logits, baux, haux


def route_economy_terms(baux, args) -> Dict[str, torch.Tensor]:
    routes = baux.routes.float()
    if routes.numel() == 0:
        z = torch.zeros((), device=baux.slots.device)
        return {
            "route_offdiag_raw_mean": z,
            "route_offdiag_norm_mean": z,
            "route_offdiag_inside_boundary": z,
            "route_offdiag_outside_boundary": z,
            "route_offdiag_outside_boundary_cost": z,
            "boundary_budget_cost": z,
            "boundary_flatness": z,
            "boundary_peak_count": z,
            "sequence_route_delta_mean": z,
        }
    lanes = routes.shape[-1]
    eye = torch.eye(lanes, device=routes.device, dtype=routes.dtype)
    offdiag = routes * (1.0 - eye.view(1, lanes, lanes))
    raw = offdiag.sum(dim=(-2, -1))
    norm = offdiag.sum(dim=-1).mean(dim=-1).clamp(0.0, 1.0)
    boundary = baux.boundaries.float().to(routes.device)
    inside = norm * boundary
    outside = norm * (1.0 - boundary)
    if routes.shape[0] > 1:
        route_delta = (routes[1:] - routes[:-1]).abs().mean(dim=(-2, -1)).mean()
    else:
        route_delta = torch.zeros((), device=routes.device)
    return {
        "route_offdiag_raw_mean": raw.mean(),
        "route_offdiag_norm_mean": norm.mean(),
        "route_offdiag_inside_boundary": inside.mean(),
        "route_offdiag_outside_boundary": outside.mean(),
        "route_offdiag_outside_boundary_cost": outside.mean(),
        "boundary_budget_cost": boundary.mean() if boundary.numel() else torch.zeros((), device=routes.device),
        "boundary_flatness": boundary.std(unbiased=False) if boundary.numel() > 1 else torch.zeros((), device=routes.device),
        "boundary_peak_count": (boundary > float(args.boundary_peak_threshold)).float().sum(),
        "sequence_route_delta_mean": route_delta,
    }

def _route_extra_terms_from_routes(routes: torch.Tensor, lanes: int) -> Dict[str, torch.Tensor]:
    if routes.numel() == 0:
        z = torch.zeros((), device=routes.device if hasattr(routes, "device") else "cpu")
        return {
            "self_route_mass": z,
            "useful_transition_mass": z,
            "self_route_by_step": torch.empty(0, device=z.device),
            "useful_transition_by_step": torch.empty(0, device=z.device),
        }
    eye = torch.eye(routes.shape[-1], dtype=routes.dtype, device=routes.device)
    self_by_step = (routes * eye.view(1, routes.shape[-1], routes.shape[-1])).sum(dim=(-2, -1)) / float(max(1, lanes))
    useful_parts = []
    if lanes >= 2:
        useful_parts.append(routes[:, 0, 1])  # detail -> state
    if lanes >= 3:
        useful_parts.append(routes[:, 1, 2])  # state -> abstract
    if lanes >= 4:
        useful_parts.append(routes[:, 1, 3])  # state -> memory
        useful_parts.append(routes[:, 3, 1])  # memory -> state
    useful_by_step = torch.stack(useful_parts, dim=0).mean(dim=0) if useful_parts else torch.zeros(routes.shape[0], dtype=routes.dtype, device=routes.device)
    return {
        "self_route_mass": self_by_step.mean(),
        "useful_transition_mass": useful_by_step.mean(),
        "self_route_by_step": self_by_step,
        "useful_transition_by_step": useful_by_step,
    }


def late_input_cost_from_read(baux, args) -> torch.Tensor:
    rg = baux.read_group_mass.float()
    if rg.numel() == 0:
        return torch.zeros((), device=baux.slots.device)
    steps = rg.shape[0]
    progress = torch.arange(steps, device=rg.device, dtype=rg.dtype) / float(max(1, steps - 1))
    depth_weight = torch.sigmoid((progress - float(args.late_input_start)) / max(1e-6, float(args.late_input_tau)))
    input_mass = rg[:, :, 0]
    return (depth_weight.view(steps, 1) * input_mass).mean()


def detail_attention_mass(haux: Dict[str, torch.Tensor], args) -> torch.Tensor:
    attn = haux["class_slot_attention"].float()
    if attn.numel() == 0:
        return torch.zeros((), device=attn.device)
    lane_map_mass = v42.lane_slot_matrix(args.tape_steps, args.lanes, args.cells_per_lane, attn.device, normalize_rows=False).float()
    detail_lane = 0
    detail_mask = lane_map_mass[detail_lane].view(1, 1, -1)
    return (attn * detail_mask).sum(dim=-1).mean()


def memory_terms(baux, haux: Dict[str, torch.Tensor], args) -> Dict[str, torch.Tensor]:
    gate = baux.write_gates.float()
    mem_lane = min(int(args.lanes) - 1, 3)
    if gate.numel():
        mem_write = gate[:, :, mem_lane, :].mean()
    else:
        mem_write = torch.zeros((), device=baux.slots.device)
    overwrite = F.relu(mem_write - float(args.memory_write_target)).pow(2)
    rg = baux.read_group_mass.float()
    if rg.numel() and rg.shape[-1] > 1 + mem_lane:
        future_read = rg[:, :, 1 + mem_lane].mean()
    else:
        future_read = torch.zeros((), device=baux.slots.device)
    lane_mass = haux.get("class_lane_mass")
    if lane_mass is not None and lane_mass.numel() and lane_mass.shape[-1] > mem_lane:
        head_consumer = lane_mass.float()[..., mem_lane].mean()
    else:
        head_consumer = torch.zeros((), device=baux.slots.device)
    consumer = future_read + head_consumer
    return {
        "memory_write_cost": mem_write,
        "memory_overwrite_cost": overwrite,
        "memory_write_gate": mem_write.detach(),
        "memory_future_read": future_read.detach(),
        "memory_head_consumer": head_consumer.detach(),
        "memory_consumer_score": consumer.detach(),
    }


def sequence_terms(baux, args) -> Dict[str, torch.Tensor]:
    routes = baux.routes.float()
    prim = baux.primitive_weights.float()
    upd = baux.update_norms.float()
    read = baux.read_group_mass.float()
    device = baux.slots.device
    if routes.numel() and routes.shape[0] > 1:
        route_delta_mean = (routes[1:] - routes[:-1]).abs().mean(dim=(-2, -1)).mean()
    else:
        route_delta_mean = torch.zeros((), device=device)
    if prim.numel() and prim.shape[0] > 1:
        prim_delta_mean = (prim[1:] - prim[:-1]).abs().mean(dim=(-2, -1)).mean()
    else:
        prim_delta_mean = torch.zeros((), device=device)
    if upd.numel() and upd.shape[1] > 1:
        u = upd.mean(dim=(0, 3))  # [T,L]
        upd_delta_mean = (u[1:] - u[:-1]).abs().mean()
    else:
        upd_delta_mean = torch.zeros((), device=device)
    if read.numel() and read.shape[0] > 1:
        read_delta_mean = (read[1:] - read[:-1]).abs().mean()
    else:
        read_delta_mean = torch.zeros((), device=device)
    route_extra = _route_extra_terms_from_routes(routes, int(args.lanes))
    return {
        "sequence_route_delta_mean": route_delta_mean,
        "sequence_primitive_delta_mean": prim_delta_mean,
        "sequence_update_delta_mean": upd_delta_mean,
        "sequence_read_delta_mean": read_delta_mean,
        "sequence_nonflat_score": route_delta_mean + prim_delta_mean + upd_delta_mean + read_delta_mean,
        "self_route_mass": route_extra["self_route_mass"].to(device),
        "useful_transition_mass": route_extra["useful_transition_mass"].to(device),
    }


def aux_losses_v43(logits: torch.Tensor, baux, haux: Dict[str, torch.Tensor], args) -> Dict[str, torch.Tensor]:
    base = v42.aux_losses(logits, baux, haux, args)
    route = route_economy_terms(baux, args)
    mem = memory_terms(baux, haux, args)
    seq = sequence_terms(baux, args)
    detail_mass = detail_attention_mass(haux, args)
    detail_cost = F.relu(detail_mass - float(args.detail_head_shortcut_target)).pow(2)
    residual_proxy = 1.0 / (baux.update_norms.float().mean() + 1e-6) if baux.update_norms.numel() else torch.zeros((), device=logits.device)
    out = dict(base)
    out.update(route)
    out.update(mem)
    out.update(seq)
    out["late_input_read_cost"] = late_input_cost_from_read(baux, args)
    out["detail_attention_mass"] = detail_mass
    out["detail_head_shortcut_cost"] = detail_cost
    out["skip_gate_mean"] = torch.zeros((), device=logits.device)
    out["skip_cost"] = torch.zeros((), device=logits.device)
    out["update_collapse_proxy"] = residual_proxy.detach()
    out["update_collapse_proxy"] = residual_proxy.detach()
    out["update_collapse_proxy"] = residual_proxy.detach()
    out["update_collapse_proxy"] = residual_proxy.detach()
    out["update_collapse_proxy"] = residual_proxy.detach()
    out["update_collapse_proxy"] = residual_proxy.detach()
    out["update_collapse_proxy"] = residual_proxy.detach()
    out["update_collapse_proxy"] = residual_proxy.detach()
    out["update_collapse_proxy"] = residual_proxy.detach()
    out["update_collapse_proxy"] = residual_proxy.detach()
    out["residual_dominance_proxy"] = residual_proxy.detach()  # deprecated alias  # deprecated alias  # deprecated alias  # deprecated alias  # deprecated alias  # deprecated alias  # deprecated alias  # deprecated alias  # deprecated alias  # deprecated alias
    out["operator_complexity_cost"] = torch.zeros((), device=logits.device)
    return out


def train_epoch(model, loader, opt, scaler, device, dtype, args, epoch: int):
    model.train()
    use_amp = device.startswith("cuda") and dtype != torch.float32
    totals = {"loss": 0.0, "ce": 0.0, "correct": 0, "n": 0}
    aux_sum: Dict[str, float] = {}
    for step, (wav, y) in enumerate(loader, 1):
        if args.max_train_batches and step > args.max_train_batches:
            break
        wav = wav.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        opt.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.split(":")[0], dtype=dtype, enabled=use_amp):
            logits, baux, haux = model(wav)
            ce = F.cross_entropy(logits.float(), y)
            losses = aux_losses_v43(logits, baux, haux, args)
            loss = ce
            loss = loss + args.lambda_write_budget * losses["write_budget"]
            loss = loss + args.lambda_update_alive * losses["update_alive"]
            loss = loss + args.lambda_class_read_div * losses["class_read_div"]
            loss = loss + args.lambda_lane_balance * losses["lane_balance"]
            loss = loss + args.lambda_slot_div * losses["slot_div"]
            loss = loss + args.lambda_route_entropy * losses["route_entropy_band"]
            loss = loss + args.lambda_step_alive_budget * losses["step_alive_budget"]
            loss = loss + args.lambda_route_offdiag_outside_boundary * losses["route_offdiag_outside_boundary_cost"]
            loss = loss + args.lambda_boundary_budget * losses["boundary_budget_cost"]
            loss = loss + args.lambda_late_input_read * losses["late_input_read_cost"]
            memory_warmup = max(1, int(getattr(args, "memory_write_warmup_epochs", 2)))
            memory_scale = min(1.0, float(epoch) / float(memory_warmup))
            loss = loss + (args.lambda_memory_write_cost * memory_scale) * losses["memory_write_cost"]
            loss = loss + (args.lambda_memory_overwrite * memory_scale) * losses["memory_overwrite_cost"]
            loss = loss + args.lambda_detail_head_shortcut * losses["detail_head_shortcut_cost"]
            loss = loss + args.lambda_skip_cost * losses["skip_cost"]
            loss = loss + args.lambda_operator_complexity * losses["operator_complexity_cost"]
            loss = loss + args.lambda_logit_norm * losses["logit_norm"]
        if not torch.isfinite(loss):
            print("NONFINITE_LOSS skip", flush=True)
            continue
        scaler.scale(loss).backward()
        if args.grad_clip > 0:
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        scaler.step(opt)
        scaler.update()
        bs = y.numel()
        totals["loss"] += float(loss.detach().cpu()) * bs
        totals["ce"] += float(ce.detach().cpu()) * bs
        totals["correct"] += int((logits.argmax(-1) == y).sum().detach().cpu())
        totals["n"] += bs
        for k, v in losses.items():
            aux_sum[k] = aux_sum.get(k, 0.0) + float(v.detach().cpu()) * bs
        if args.log_every and step % args.log_every == 0:
            print(
                f"epoch {epoch:03d} step {step:05d} "
                f"loss={totals['loss']/max(1, totals['n']):.4f} "
                f"ce={totals['ce']/max(1, totals['n']):.4f} "
                f"acc={100*totals['correct']/max(1, totals['n']):.2f}%",
                flush=True,
            )
    out = {k: v / max(1, totals["n"]) for k, v in totals.items() if k != "correct"}
    out["acc"] = totals["correct"] / max(1, totals["n"])
    for k, v in aux_sum.items():
        out[k] = v / max(1, totals["n"])
    return out


def _tensor_from_list(x, dtype=torch.float32):
    if x is None:
        return torch.empty(0, dtype=dtype)
    try:
        return torch.tensor(x, dtype=dtype)
    except Exception:
        return torch.empty(0, dtype=dtype)


def _detail_topread_share(top_reads: Sequence[Sequence[Dict]]) -> float:
    total = 0.0
    detail = 0.0
    for class_reads in top_reads or []:
        for item in class_reads or []:
            w = _safe_float(item.get("weight", 0.0))
            total += w
            if ".detail." in str(item.get("slot", "")):
                detail += w
    return detail / max(total, 1e-8)


def build_trace_feedback(rep: Dict, args, epoch: int) -> Dict:
    routes = _tensor_from_list(rep.get("route_matrix", []))
    boundary = _tensor_from_list(rep.get("boundary", []))
    read_group = _tensor_from_list(rep.get("read_group_mass", []))
    primitive = _tensor_from_list(rep.get("primitive_weights", []))
    updates = _tensor_from_list(rep.get("update_norm_by_step_lane", []))
    gates = _tensor_from_list(rep.get("write_gate_by_step_lane", []))
    lanes = int(args.lanes)
    mem_lane = min(lanes - 1, 3)

    if routes.numel():
        eye = torch.eye(routes.shape[-1], dtype=routes.dtype)
        offdiag = routes * (1.0 - eye.view(1, routes.shape[-1], routes.shape[-1]))
        offdiag_raw = offdiag.sum(dim=(-2, -1))
        offdiag_norm = offdiag.sum(dim=-1).mean(dim=-1).clamp(0.0, 1.0)
        route_extra = _route_extra_terms_from_routes(routes, lanes)
        self_route_by_step = route_extra["self_route_by_step"]
        useful_transition_by_step = route_extra["useful_transition_by_step"]
        self_route_mass = float(route_extra["self_route_mass"])
        useful_transition_mass = float(route_extra["useful_transition_mass"])
    else:
        offdiag_raw = torch.empty(0)
        offdiag_norm = torch.empty(0)
        self_route_by_step = torch.empty(0)
        useful_transition_by_step = torch.empty(0)
        self_route_mass = 0.0
        useful_transition_mass = 0.0
    if boundary.numel() and offdiag_norm.numel():
        inside = offdiag_norm * boundary
        outside = offdiag_norm * (1.0 - boundary)
    else:
        inside = torch.empty(0)
        outside = torch.empty(0)

    def step_delta(x: torch.Tensor) -> torch.Tensor:
        if x.numel() == 0 or x.shape[0] <= 1:
            return torch.zeros(max(1, int(args.tape_steps)))[: int(args.tape_steps)]
        d = (x[1:] - x[:-1]).abs()
        while d.dim() > 1:
            d = d.mean(dim=-1)
        return torch.cat([d, torch.zeros(1, dtype=d.dtype)], dim=0)

    route_delta = step_delta(routes)
    primitive_delta = step_delta(primitive)
    read_delta = step_delta(read_group)
    trace_delta = step_delta(updates) if updates.numel() else torch.zeros_like(route_delta)
    common_len = min(len(route_delta), len(primitive_delta), len(trace_delta), len(read_delta), len(boundary)) if boundary.numel() else 0
    if common_len:
        seq_change = route_delta[:common_len] + primitive_delta[:common_len] + trace_delta[:common_len] + read_delta[:common_len]
        usefulness = boundary[:common_len] * seq_change
    else:
        usefulness = torch.empty(0)
        seq_change = torch.empty(0)

    late_by_step = []
    if read_group.numel():
        input_mass_by_step = read_group[:, :, 0].mean(dim=-1)
        progress = torch.arange(input_mass_by_step.numel(), dtype=torch.float32) / float(max(1, input_mass_by_step.numel() - 1))
        depth_weight = torch.sigmoid((progress - float(args.late_input_start)) / max(1e-6, float(args.late_input_tau)))
        late_by_step = (input_mass_by_step * depth_weight).tolist()
        input_total = float(input_mass_by_step.mean())
        memory_future_read = float(read_group[:, :, 1 + mem_lane].mean()) if read_group.shape[-1] > 1 + mem_lane else 0.0
    else:
        input_total = 0.0
        memory_future_read = 0.0

    memory_write = float(gates[:, mem_lane].mean()) if gates.numel() and gates.shape[-1] > mem_lane else 0.0
    memory_overwrite = max(0.0, memory_write - float(args.memory_write_target)) ** 2
    lane_mass_mean = rep.get("lane_mass_mean", {}) or {}
    head_memory = _safe_float(lane_mass_mean.get(lane_name(mem_lane), 0.0))
    detail_attention = _safe_float(lane_mass_mean.get(lane_name(0), 0.0))
    detail_topread = _detail_topread_share(rep.get("class_top_reads", []))
    detail_cost = max(0.0, detail_attention - float(args.detail_head_shortcut_target)) ** 2

    boundary_list = [float(v) for v in boundary.tolist()] if boundary.numel() else []
    peak_thr = float(args.boundary_peak_threshold)
    peaks = [i for i, v in enumerate(boundary_list) if v >= peak_thr]
    boundary_mean = sum(boundary_list) / max(1, len(boundary_list))
    boundary_flatness = (sum((v - boundary_mean) ** 2 for v in boundary_list) / max(1, len(boundary_list))) ** 0.5 if boundary_list else 0.0
    primitive_weights = rep.get("primitive_weights", [])
    seq_nonflat = float(seq_change.mean()) if seq_change.numel() else 0.0
    memory_consumer_proxy = memory_future_read + head_memory

    flags = []
    entropy_mean = float(_tensor_from_list(rep.get("route_entropy", [])).mean()) if rep.get("route_entropy") else 0.0
    if boundary_mean > 0.90 and boundary_flatness < 0.05:
        flags.append("BOUNDARY_EXPLOIT")
    if boundary_mean < 0.05 or len(peaks) == 0:
        flags.append("BOUNDARY_DEAD")
    if entropy_mean > 1.30:
        flags.append("ROUTE_UNIFORM")
    if self_route_mass > 0.88 and useful_transition_mass < 0.12:
        flags.append("ROUTE_IDENTITY_COLLAPSE")
    if detail_attention > 0.55:
        flags.append("DETAIL_SHORTCUT")
    if memory_write < 0.03 and memory_consumer_proxy < 0.08:
        flags.append("MEMORY_DEAD")
    if memory_write > 0.30 and memory_consumer_proxy < 0.08:
        flags.append("MEMORY_JUNK")

    return {
        "epoch": int(epoch),
        "window": {"type": "epoch", "index": int(epoch)},
        "compare_to": str(getattr(args, "compare_to", "baseline_missing")),
        "version": "v4.3_min_heart_canonical",
        "collapse_flags": flags,
        "route": {
            "entropy_mean": entropy_mean,
            "matrix_by_step": rep.get("route_matrix", []),
            "offdiag_raw": offdiag_raw.tolist() if offdiag_raw.numel() else [],
            "offdiag_norm": offdiag_norm.tolist() if offdiag_norm.numel() else [],
            "offdiag_inside_boundary": inside.tolist() if inside.numel() else [],
            "offdiag_outside_boundary": outside.tolist() if outside.numel() else [],
            "offdiag_outside_boundary_cost": float(outside.mean()) if outside.numel() else 0.0,
            "self_route_by_step": self_route_by_step.tolist() if self_route_by_step.numel() else [],
            "useful_transition_by_step": useful_transition_by_step.tolist() if useful_transition_by_step.numel() else [],
            "self_route_mass": self_route_mass,
            "useful_transition_mass": useful_transition_mass,
            "boundary_by_step": boundary_list,
            "boundary_mean": boundary_mean,
            "boundary_flatness": boundary_flatness,
            "boundary_peak_count": len(peaks),
            "boundary_peaks": peaks,
            "boundary_usefulness": usefulness.tolist() if usefulness.numel() else [],
        },
        "sequence": {
            "route_delta_by_step": route_delta.tolist()[: int(args.tape_steps)] if route_delta.numel() else [],
            "primitive_delta_by_step": primitive_delta.tolist()[: int(args.tape_steps)] if primitive_delta.numel() else [],
            "trace_delta_by_step": trace_delta.tolist()[: int(args.tape_steps)] if trace_delta.numel() else [],
            "read_delta_by_step": read_delta.tolist()[: int(args.tape_steps)] if read_delta.numel() else [],
            "sequence_change_by_step": seq_change.tolist() if seq_change.numel() else [],
            "sequence_nonflat_score": seq_nonflat,
        },
        "read": {"late_input_by_step": [float(v) for v in late_by_step], "late_input_cost": float(sum(late_by_step) / max(1, len(late_by_step))) if late_by_step else 0.0, "input_read_total": input_total},
        "memory": {"write_mean": memory_write, "write_cost": memory_write, "overwrite_score": memory_overwrite, "future_read": memory_future_read, "head_consumer": head_memory, "consumer_score": memory_consumer_proxy, "memory_consumer_proxy": memory_consumer_proxy},
        "head": {"detail_attention_mass": detail_attention, "detail_head_shortcut_cost": detail_cost, "detail_topread_share": detail_topread, "class_lane_mass": rep.get("class_lane_mass", []), "lane_mass_mean": lane_mass_mean, "top_reads": rep.get("class_top_reads", [])},
        "operators": {"primitive_weights": primitive_weights, "primitive_names": rep.get("primitive_names", list(PRIMITIVES))},
        "budget": {"skip_gate_mean": 0.0, "skip_cost": "not_implemented", "operator_complexity_cost": "not_implemented", "update_collapse_proxy": None},
    }


def generate_candidate_suggestions(trace: Dict, args, epoch: int) -> Dict:
    candidates: List[Dict] = []
    seen = set()

    def add(c: Dict):
        if len(candidates) >= int(args.max_candidates_per_epoch):
            return
        c.setdefault("delta_scale", 0.03)
        c.setdefault("risk", "low")
        c["deploy"] = False
        key = _candidate_key(c)
        if key in seen:
            return
        seen.add(key)
        candidates.append(c)

    route = trace.get("route", {})
    seq = trace.get("sequence", {})
    read = trace.get("read", {})
    memory = trace.get("memory", {})
    head = trace.get("head", {})
    boundary = route.get("boundary_by_step", []) or []
    usefulness = route.get("boundary_usefulness", []) or []
    seq_change = seq.get("sequence_change_by_step", []) or []
    outside = route.get("offdiag_outside_boundary", []) or []

    for t, b in enumerate(boundary):
        u = usefulness[t] if t < len(usefulness) else 0.0
        sc = seq_change[t] if t < len(seq_change) else 0.0
        if float(b) > 0.55 and float(u) < 0.03:
            add({
                "source": "boundary_rule",
                "target_type": "boundary",
                "location": {"t": t},
                "action": "decrease",
                "target": "boundary",
                "reason": "boundary is high but boundary_usefulness is low",
                "evidence_metrics": {"boundary": b, "boundary_usefulness": u},
                "risk": "low",
            })
        if float(b) < 0.25 and float(sc) > 0.05:
            add({
                "source": "sequence_boundary_rule",
                "target_type": "boundary",
                "location": {"t": t},
                "action": "increase",
                "target": "boundary",
                "reason": "sequence changes here but boundary is low",
                "evidence_metrics": {"boundary": b, "sequence_change": sc},
                "risk": "medium",
            })
    for t, out in enumerate(outside):
        if float(out) > 0.35:
            add({
                "source": "route_boundary_rule",
                "target_type": "route",
                "location": {"t": t},
                "action": "decrease",
                "target": "offdiag_outside_boundary",
                "reason": "cross-lane routing is high outside boundary",
                "evidence_metrics": {"offdiag_outside_boundary": out, "boundary": boundary[t] if t < len(boundary) else None},
                "risk": "medium",
            })

    late = read.get("late_input_by_step", []) or []
    if late:
        half = max(1, len(late) // 2)
        late_tail = sum(float(v) for v in late[half:]) / max(1, len(late[half:]))
        if late_tail > 0.20 and float(head.get("detail_attention_mass", 0.0)) > float(args.detail_head_shortcut_target):
            add({
                "source": "late_input_shortcut_rule",
                "target_type": "read",
                "location": {"t": f">={half}"},
                "action": "decrease",
                "target": "late_input_read",
                "reason": "late raw input read and detail head shortcut are both high",
                "evidence_metrics": {"late_tail_input": late_tail, "detail_attention_mass": head.get("detail_attention_mass", 0.0)},
                "risk": "medium",
            })

    mem_write = float(memory.get("write_mean", 0.0))
    mem_consumer = float(memory.get("consumer_score", 0.0))
    if mem_write > 0.35 and mem_consumer < 0.10:
        add({
            "source": "memory_consumer_rule",
            "target_type": "memory",
            "location": {"lane": lane_name(min(int(args.lanes) - 1, 3))},
            "action": "decrease",
            "target": "memory_write_or_overwrite",
            "reason": "memory write is high but future/head consumer is low",
            "evidence_metrics": {"memory_write": mem_write, "memory_consumer_score": mem_consumer},
            "risk": "medium",
        })
    if mem_write > 0.20 and mem_consumer > 0.15:
        add({
            "source": "memory_consumer_rule",
            "target_type": "route",
            "location": {"from_lane": "memory", "to_lane": "state"},
            "action": "increase",
            "target": "route.memory->state",
            "reason": "memory appears useful; future Heart should test memory->state route rather than killing memory",
            "evidence_metrics": {"memory_write": mem_write, "memory_consumer_score": mem_consumer},
            "risk": "low",
        })

    detail_mass = float(head.get("detail_attention_mass", 0.0))
    if detail_mass > float(args.detail_head_shortcut_target):
        add({
            "source": "head_shortcut_rule",
            "target_type": "head",
            "location": {"lane": "detail"},
            "action": "decrease",
            "target": "detail_head_shortcut",
            "reason": "class head attention mass on detail lane is above target",
            "evidence_metrics": {"detail_attention_mass": detail_mass, "target": float(args.detail_head_shortcut_target)},
            "risk": "medium",
        })

    if trace.get("sequence", {}).get("sequence_nonflat_score", 0.0) < 0.01:
        add({
            "source": "sequence_flatness_rule",
            "target_type": "budget",
            "location": {"t": "all"},
            "action": "increase",
            "target": "sequence_differentiation_pressure",
            "reason": "tape positions look too similar; future Heart should encourage non-flat sequence",
            "evidence_metrics": {"sequence_nonflat_score": trace.get("sequence", {}).get("sequence_nonflat_score", 0.0)},
            "risk": "low",
        })

    by_source: Dict[str, int] = {}
    by_target: Dict[str, int] = {}
    by_step: Dict[str, int] = {}
    by_lane: Dict[str, int] = {}
    for c in candidates:
        by_source[c.get("source", "?")] = by_source.get(c.get("source", "?"), 0) + 1
        by_target[c.get("target_type", "?")] = by_target.get(c.get("target_type", "?"), 0) + 1
        loc = c.get("location") or {}
        if "t" in loc:
            by_step[str(loc["t"])] = by_step.get(str(loc["t"]), 0) + 1
        if "lane" in loc and loc["lane"] is not None:
            by_lane[str(loc["lane"])] = by_lane.get(str(loc["lane"]), 0) + 1
    return {
        "epoch": int(epoch),
        "window": {"type": "epoch", "index": int(epoch)},
        "version": "v4.3_min_heart_canonical",
        "candidates": candidates,
        "diversity": {
            "by_source": by_source,
            "by_target_type": by_target,
            "by_step": by_step,
            "by_lane": by_lane,
            "attempted_candidates": len(candidates),
            "skipped_duplicates": 0,
            "attempted_candidates": len(candidates),
            "skipped_duplicates": 0,
            "attempted_candidates": len(candidates),
            "skipped_duplicates": 0,
            "attempted_candidates": len(candidates),
            "skipped_duplicates": 0,
            "attempted_candidates": len(candidates),
            "skipped_duplicates": 0,
            "attempted_candidates": len(candidates),
            "skipped_duplicates": 0,
            "attempted_candidates": len(candidates),
            "skipped_duplicates": 0,
            "attempted_candidates": len(candidates),
            "skipped_duplicates": 0,
            "attempted_candidates": len(candidates),
            "skipped_duplicates": 0,
            "attempted_candidates": len(candidates),
            "skipped_duplicates": 0,
            "duplicate_rate": 0.0,
        },
    }


def write_chatgpt_report(out_dir: Path, analysis: Dict, trace: Dict, candidates: Dict, args) -> None:
    rep = analysis.get("matrix_report") or {}
    route = trace.get("route", {})
    seq = trace.get("sequence", {})
    read = trace.get("read", {})
    memory = trace.get("memory", {})
    head = trace.get("head", {})
    active = [i for i, v in enumerate(rep.get("step_alive", []) or []) if float(v) >= 0.50]
    peaks = route.get("boundary_peaks", [])
    lines = [
        "REPORT_TO_CHATGPT",
        "",
        "Версия: v4.3_min_heart",
        f"Лучший val_acc: {100.0 * float(analysis.get('best_acc', 0.0)):.2f}% @ epoch {analysis.get('best_epoch', 0)}",
        f"Текущий epoch/window: {analysis.get('epoch', 0)}",
        f"compare_to: {trace.get('compare_to', 'baseline_missing')}",
        "",
        "Главное по v4.3:",
        "- Это НЕ actor/critic/planner и НЕ auto-deploy.",
        "- Это costed paths + trace + deterministic candidate suggestions.",
        "- Последовательность должна возникать через разные tape positions, boundary peaks и изменение route/read/primitive по t.",
        "",
        "Route / boundary:",
        f"- route_entropy_mean: {float(route.get('entropy_mean', 0.0)):.4f}",
        f"- boundary_mean: {float(route.get('boundary_mean', 0.0)):.4f}",
        f"- boundary_flatness/std: {float(route.get('boundary_flatness', 0.0)):.4f}",
        f"- boundary_peak_count: {int(route.get('boundary_peak_count', 0))} peaks={peaks}",
        f"- offdiag_outside_boundary_cost: {float(route.get('offdiag_outside_boundary_cost', 0.0)):.4f}",
        f"- self_route_mass: {float(route.get('self_route_mass', 0.0)):.4f}",
        f"- useful_transition_mass: {float(route.get('useful_transition_mass', 0.0)):.4f}",
        f"- collapse_flags: {','.join(trace.get('collapse_flags', []) or []) if trace.get('collapse_flags') else 'NONE'}",
        f"- self_route_mass: {float(route.get('self_route_mass', 0.0)):.4f}",
        f"- useful_transition_mass: {float(route.get('useful_transition_mass', 0.0)):.4f}",
        f"- collapse_flags: {','.join(trace.get('collapse_flags', []) or []) if trace.get('collapse_flags') else 'NONE'}",
        f"- self_route_mass: {float(route.get('self_route_mass', 0.0)):.4f}",
        f"- useful_transition_mass: {float(route.get('useful_transition_mass', 0.0)):.4f}",
        f"- collapse_flags: {','.join(trace.get('collapse_flags', []) or []) if trace.get('collapse_flags') else 'NONE'}",
        f"- self_route_mass: {float(route.get('self_route_mass', 0.0)):.4f}",
        f"- useful_transition_mass: {float(route.get('useful_transition_mass', 0.0)):.4f}",
        f"- collapse_flags: {','.join(trace.get('collapse_flags', []) or []) if trace.get('collapse_flags') else 'NONE'}",
        f"- self_route_mass: {float(route.get('self_route_mass', 0.0)):.4f}",
        f"- useful_transition_mass: {float(route.get('useful_transition_mass', 0.0)):.4f}",
        f"- collapse_flags: {','.join(trace.get('collapse_flags', []) or []) if trace.get('collapse_flags') else 'NONE'}",
        f"- self_route_mass: {float(route.get('self_route_mass', 0.0)):.4f}",
        f"- useful_transition_mass: {float(route.get('useful_transition_mass', 0.0)):.4f}",
        f"- collapse_flags: {','.join(trace.get('collapse_flags', []) or []) if trace.get('collapse_flags') else 'NONE'}",
        f"- self_route_mass: {float(route.get('self_route_mass', 0.0)):.4f}",
        f"- useful_transition_mass: {float(route.get('useful_transition_mass', 0.0)):.4f}",
        f"- collapse_flags: {','.join(trace.get('collapse_flags', []) or []) if trace.get('collapse_flags') else 'NONE'}",
        f"- self_route_mass: {float(route.get('self_route_mass', 0.0)):.4f}",
        f"- useful_transition_mass: {float(route.get('useful_transition_mass', 0.0)):.4f}",
        f"- collapse_flags: {','.join(trace.get('collapse_flags', []) or []) if trace.get('collapse_flags') else 'NONE'}",
        f"- self_route_mass: {float(route.get('self_route_mass', 0.0)):.4f}",
        f"- useful_transition_mass: {float(route.get('useful_transition_mass', 0.0)):.4f}",
        f"- collapse_flags: {','.join(trace.get('collapse_flags', []) or []) if trace.get('collapse_flags') else 'NONE'}",
        f"- self_route_mass: {float(route.get('self_route_mass', 0.0)):.4f}",
        f"- useful_transition_mass: {float(route.get('useful_transition_mass', 0.0)):.4f}",
        f"- collapse_flags: {','.join(trace.get('collapse_flags', []) or []) if trace.get('collapse_flags') else 'NONE'}",
        "",
        "Sequence:",
        f"- active tape steps step_alive>=0.50: {active}",
        f"- sequence_nonflat_score: {float(seq.get('sequence_nonflat_score', 0.0)):.4f}",
        "",
        "Shortcuts / memory:",
        f"- late_input_cost: {float(read.get('late_input_cost', 0.0)):.4f}",
        f"- detail_attention_mass: {float(head.get('detail_attention_mass', 0.0)):.4f}",
        f"- detail_topread_share(report-only): {float(head.get('detail_topread_share', 0.0)):.4f}",
        f"- memory_write_mean: {float(memory.get('write_mean', 0.0)):.4f}",
        f"- memory_consumer_proxy: {float(memory.get('memory_consumer_proxy', memory.get('consumer_score', 0.0))):.4f}",
        "",
        "Candidate suggestions:",
    ]
    for c in candidates.get("candidates", [])[: int(args.max_candidates_per_epoch)]:
        lines.append(f"- {c.get('target_type')} {c.get('action')} {c.get('target')} at {c.get('location')} | {c.get('reason')} | deploy={c.get('deploy')}")
    if not candidates.get("candidates"):
        lines.append("- no candidate suggestions")
    lines += [
        "",
        "Артефакты:",
        "- metrics.csv",
        "- analysis_epoch_XXX.json",
        "- trace_feedback_epoch_XXX.json",
        "- candidate_suggestions_epoch_XXX.json",
        "- final_report.json",
        "- train.log если запускался sync script",
    ]
    (out_dir / "REPORT_TO_CHATGPT.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(args) -> None:
    v42.set_seed(args.seed)
    device = args.device if args.device != "cuda" or torch.cuda.is_available() else "cpu"
    if device.startswith("cuda"):
        torch.backends.cudnn.benchmark = True
        torch.set_float32_matmul_precision("high")
    dtype = v42.amp_dtype(args.amp)
    out_dir = v42.ensure_dir(Path(args.out_dir))
    train_loader, val_loader, classes, train_counts, val_counts = v42.make_loaders(args)
    args.num_classes = len(classes)
    model = TapeLaneRouterClassifierV43(len(classes), args).to(device)
    if args.init_checkpoint:
        ckpt = torch.load(args.init_checkpoint, map_location=device)
        state = ckpt.get("model", ckpt)
        model.load_state_dict(state, strict=True)
        print(f"loaded init checkpoint: {args.init_checkpoint}", flush=True)
    params = sum(p.numel() for p in model.parameters())
    print(f"loaded datasets: train={len(train_loader.dataset)} val={len(val_loader.dataset)} classes={classes}", flush=True)
    print(
        f"TapeLaneRouter v4.3_min_heart params={params} T={args.tape_steps} lanes={args.lanes} "
        f"cells={args.cells_per_lane} D={args.dim} pairs={args.pair_slots} device={device} amp={args.amp}",
        flush=True,
    )

    decay, no_decay = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if "bias" in name or "norm" in name:
            no_decay.append(p)
        else:
            decay.append(p)
    opt = torch.optim.AdamW([
        {"params": decay, "weight_decay": args.weight_decay},
        {"params": no_decay, "weight_decay": 0.0},
    ], lr=args.lr, betas=(0.9, 0.95))
    scaler = torch.amp.GradScaler("cuda", enabled=device.startswith("cuda") and dtype == torch.float16)

    fields = [
        "epoch", "train_loss", "train_ce", "train_acc", "val_loss", "val_acc", "best_acc",
        "write_budget", "update_alive", "class_read_div", "lane_balance", "slot_div",
        "route_entropy_band", "step_alive_budget", "route_offdiag_outside_boundary_cost",
        "boundary_budget_cost", "boundary_flatness", "boundary_peak_count", "late_input_read_cost",
        "memory_write_cost", "memory_overwrite_cost", "memory_write_gate", "memory_future_read",
        "memory_head_consumer", "memory_consumer_score", "detail_attention_mass", "detail_head_shortcut_cost",
        "skip_gate_mean", "skip_cost", "update_collapse_proxy", "residual_dominance_proxy", "operator_complexity_cost",
        "sequence_nonflat_score", "sequence_route_delta_mean", "sequence_primitive_delta_mean", "sequence_read_delta_mean",
        "logit_norm", "pair_update_norm",
    ]
    with (out_dir / "metrics.csv").open("w", newline="", encoding="utf-8") as f:
        csv.DictWriter(f, fieldnames=fields).writeheader()

    best, best_epoch = -1.0, 0
    last_analysis: Dict = {}
    last_trace: Dict = {}
    last_candidates: Dict = {}
    for epoch in range(1, args.epochs + 1):
        tr = train_epoch(model, train_loader, opt, scaler, device, dtype, args, epoch)
        va = v42.evaluate(model, val_loader, device, dtype, args)
        if va["acc"] > best:
            best, best_epoch = va["acc"], epoch
            if not args.no_save_checkpoints:
                torch.save({"model": model.state_dict(), "args": vars(args), "classes": classes, "epoch": epoch, "best_acc": best}, out_dir / "best.pt")
        if not args.no_save_checkpoints:
            torch.save({"model": model.state_dict(), "args": vars(args), "classes": classes, "epoch": epoch, "best_acc": best}, out_dir / "last.pt")

        rep = va["report"] or {}
        if rep:
            rep["version"] = "v4.3_min_heart"
        trace = build_trace_feedback(rep, args, epoch)
        candidates = generate_candidate_suggestions(trace, args, epoch) if args.enable_candidate_suggestions else {"epoch": epoch, "candidates": [], "diversity": {}}

        row = {k: 0.0 for k in fields}
        row.update({
            "epoch": epoch,
            "train_loss": tr["loss"],
            "train_ce": tr["ce"],
            "train_acc": tr["acc"],
            "val_loss": va["loss"],
            "val_acc": va["acc"],
            "best_acc": best,
        })
        for k in fields:
            if k in tr:
                row[k] = tr.get(k, 0.0)
        with (out_dir / "metrics.csv").open("a", newline="", encoding="utf-8") as f:
            csv.DictWriter(f, fieldnames=fields).writerow(row)

        last_analysis = {
            "epoch": epoch,
            "train": tr,
            "val": {"loss": va["loss"], "acc": va["acc"], "n": va["n"], "confusion": va["confusion"]},
            "best_acc": best,
            "best_epoch": best_epoch,
            "classes": classes,
            "train_counts": train_counts,
            "val_counts": val_counts,
            "matrix_report": rep,
            "trace_feedback": trace,
            "candidate_suggestions": candidates,
        }
        v42.write_json(out_dir / f"analysis_epoch_{epoch:03d}.json", last_analysis)
        v42.write_json(out_dir / f"trace_feedback_epoch_{epoch:03d}.json", trace)
        v42.write_json(out_dir / f"candidate_suggestions_epoch_{epoch:03d}.json", candidates)
        write_chatgpt_report(out_dir, last_analysis, trace, candidates, args)
        last_trace, last_candidates = trace, candidates
        print(
            f"epoch {epoch:03d}/{args.epochs} train={tr['loss']:.4f}/{100*tr['acc']:.2f}% "
            f"val={va['loss']:.4f}/{100*va['acc']:.2f}% best={100*best:.2f}%@{best_epoch} "
            f"seq={float(trace.get('sequence', {}).get('sequence_nonflat_score', 0.0)):.4f} "
            f"boundary_peaks={trace.get('route', {}).get('boundary_peak_count', 0)}",
            flush=True,
        )

    v42.write_json(out_dir / "final_report.json", {
        "version": "v4.3_min_heart_canonical",
        "best_acc": best,
        "best_epoch": best_epoch,
        "args": vars(args),
        "classes": classes,
        "last_trace_feedback": last_trace,
        "last_candidate_suggestions": last_candidates,
    })
    if last_analysis:
        write_chatgpt_report(out_dir, last_analysis, last_trace, last_candidates, args)


def parser() -> argparse.ArgumentParser:
    p = v42.parser()
    p.set_defaults(out_dir="./simple_butterfly_matrix_v4_tape_lane/runs/speechcommands_5ep_v4_3_min_heart")
    p.add_argument("--lambda-route-offdiag-outside-boundary", type=float, default=0.012)
    # Reuses existing --lambda-boundary-budget, but v4.3 semantics are mean(boundary), not target MSE.
    p.add_argument("--lambda-detail-head-shortcut", type=float, default=0.010)
    p.add_argument("--lambda-skip-cost", type=float, default=0.0)
    p.add_argument("--lambda-memory-write-cost", type=float, default=0.003)
    p.add_argument("--lambda-operator-complexity", type=float, default=0.0)
    p.add_argument("--detail-head-shortcut-target", type=float, default=0.42)
    p.add_argument("--boundary-peak-threshold", type=float, default=0.35)
    p.add_argument("--late-input-start", type=float, default=0.45)
    p.add_argument("--late-input-tau", type=float, default=0.12)
    p.add_argument("--max-candidates-per-epoch", type=int, default=8)
    p.add_argument("--heart-window-type", choices=["epoch"], default="epoch")
    p.add_argument("--enable-candidate-suggestions", action="store_true", default=True)
    p.add_argument("--enable-counterfactual-screen", action="store_true", default=False)
    p.add_argument("--compare-to", default="v4.2_fixed_guided same seed/config or baseline_missing")
    return p


if __name__ == "__main__":
    run(parser().parse_args())
