#!/usr/bin/env python3
"""v4: tape-lane router matrix program.

This version keeps the useful v3 idea (trainable class matrices + class-pair
repair) but removes fixed extract/compare/suppress/aggregate phases.

Backbone:
  evidence -> lanes/cells -> T tape steps
  read -> transform -> route/write
  soft step_alive, soft boundary, soft lane route matrices

No hard router and no top-k are used.
"""

from __future__ import annotations

import argparse
import csv
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from simple_butterfly_matrix.simple_butterfly_matrix import (  # noqa: E402
    BlockButterfly,
    ChannelButterfly,
    MatrixEvidence,
    amp_dtype,
    ensure_dir,
    make_loaders,
    set_seed,
    slot_diversity_loss,
    write_json,
)


LANE_NAMES = ("detail", "state", "abstract", "memory")
READ_GROUP_NAMES = ("input", "detail", "state", "abstract", "memory")
PRIMITIVES = ("channel", "block", "low_rank", "ctx_matrix", "product", "generic")


def _to_float_list(x: torch.Tensor):
    return x.detach().float().cpu().tolist()


def entropy(p: torch.Tensor, dim: int = -1, eps: float = 1e-8) -> torch.Tensor:
    p = p.float().clamp_min(eps)
    p = p / p.sum(dim=dim, keepdim=True).clamp_min(eps)
    return -(p * p.log()).sum(dim=dim)


@dataclass
class TapeLaneAux:
    slots: torch.Tensor
    slot_names: List[str]
    write_gates: torch.Tensor         # [B,T,L,A]
    update_norms: torch.Tensor        # [B,T,L,A]
    step_alive: torch.Tensor          # [T]
    boundaries: torch.Tensor          # [T]
    routes: torch.Tensor              # [T,L,L]
    route_entropy: torch.Tensor       # [T,L]
    read_group_mass: torch.Tensor     # [T,L,L+1]
    late_input_read_mass: torch.Tensor


def lane_slot_matrix(steps: int, lanes: int, cells: int, device: torch.device, normalize_rows: bool) -> torch.Tensor:
    """Return [lanes, total_slots] map for slots shaped [(steps+1), lanes, cells]."""

    total_steps = int(steps) + 1
    slot_count = total_steps * int(lanes) * int(cells)
    mat = torch.zeros(int(lanes), slot_count, device=device)
    pos = 0
    for _t in range(total_steps):
        for lane in range(int(lanes)):
            for _a in range(int(cells)):
                mat[lane, pos] = 1.0
                pos += 1
    if normalize_rows:
        mat = mat / mat.sum(dim=-1, keepdim=True).clamp_min(1.0)
    return mat


def class_lane_prior(classes: int, lanes: int) -> torch.Tensor:
    """Weak class/lane coverage prior; not a hard assignment."""

    p = torch.zeros(int(classes), int(lanes))
    for c in range(int(classes)):
        p[c, c % int(lanes)] = 0.55
        p[c, (c + 1) % int(lanes)] = 0.25
        p[c, (c + 2) % int(lanes)] = 0.12
    p += 0.04
    return p


class TapeLaneTransformUnit(nn.Module):
    """Small neutral transform bank for one tape step.

    It deliberately has no phase-specific branches. All candidates are generic
    matrix operations and lanes learn soft mixtures over them.
    """

    def __init__(self, dim: int, lanes: int, cells_per_lane: int, channel_stages: int, dropout: float):
        super().__init__()
        self.dim = int(dim)
        self.lanes = int(lanes)
        self.cells_per_lane = int(cells_per_lane)
        self.channel = ChannelButterfly(dim, channel_stages)
        self.block = BlockButterfly(cells_per_lane)
        rank = max(8, dim // 4)
        self.low_a = nn.Parameter(torch.randn(dim, rank) * 0.04)
        self.low_b = nn.Parameter(torch.randn(rank, dim) * 0.04)
        self.ctx_w = nn.Parameter(torch.randn(dim, dim) * 0.04)
        self.gate_h = nn.Parameter(torch.randn(dim, dim) * 0.02)
        self.gate_c = nn.Parameter(torch.randn(dim, dim) * 0.02)
        self.gate_bias = nn.Parameter(torch.full((dim,), -0.35))
        self.primitive_logits = nn.Parameter(torch.zeros(len(PRIMITIVES)))
        self.lane_primitive_bias = nn.Linear(dim, len(PRIMITIVES), bias=False)
        self.drop = nn.Dropout(dropout)
        self.update_norm = nn.LayerNorm(dim)

    def forward(self, x: torch.Tensor, read_packet: torch.Tensor, lane_embed: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        # x/read_packet: [B,L,A,D]
        bsz, lanes, cells, dim = x.shape
        flat_x = x.reshape(bsz * lanes, cells, dim)
        flat_ctx = read_packet.reshape(bsz * lanes, cells, dim)

        ctx_m = flat_ctx @ self.ctx_w.to(device=x.device, dtype=x.dtype)
        channel = self.channel(flat_x + ctx_m)
        block = self.block(flat_x)
        low = (flat_x @ self.low_a.to(device=x.device, dtype=x.dtype)) @ self.low_b.to(device=x.device, dtype=x.dtype)
        product = flat_x * torch.tanh(ctx_m)
        gate = torch.sigmoid(
            flat_x @ self.gate_h.to(device=x.device, dtype=x.dtype)
            + ctx_m @ self.gate_c.to(device=x.device, dtype=x.dtype)
            + self.gate_bias.to(device=x.device, dtype=x.dtype)
        )
        generic = gate * channel + (1.0 - gate) * low

        cands = torch.stack([channel, block, low, ctx_m, product, generic], dim=2)  # [B*L,A,P,D]

        lane_bias = self.lane_primitive_bias(lane_embed.to(device=x.device, dtype=x.dtype)).float()  # [L,P]
        weights = torch.softmax(self.primitive_logits.float().view(1, -1) + lane_bias, dim=-1).to(x.dtype)
        weights = weights.view(1, lanes, 1, len(PRIMITIVES), 1).expand(bsz, -1, cells, -1, -1).reshape(bsz * lanes, cells, len(PRIMITIVES), 1)

        update = (weights * cands).sum(dim=2)
        update = self.update_norm(self.drop(update * gate)).reshape(bsz, lanes, cells, dim)

        return update, {
            "primitive_weights": weights.detach().reshape(bsz, lanes, cells, len(PRIMITIVES)).float().mean(dim=(0, 2)),
            "gate_mean": gate.detach().float().mean(),
        }


class TapeLaneRouterBackbone(nn.Module):
    """One growing program tape with soft lane route matrices."""

    def __init__(
        self,
        dim: int,
        evidence_cells: int,
        lanes: int,
        cells_per_lane: int,
        tape_steps: int,
        sample_rate: int,
        n_mels: int,
        hop_length: int,
        channel_stages: int,
        dropout: float,
    ):
        super().__init__()
        self.dim = int(dim)
        self.lanes = int(lanes)
        self.cells_per_lane = int(cells_per_lane)
        self.tape_steps = int(tape_steps)
        self.evidence = MatrixEvidence(sample_rate, n_mels, hop_length, evidence_cells, dim)

        self.lane_embed = nn.Parameter(torch.randn(lanes, dim) * 0.02)
        self.step_embed = nn.Parameter(torch.randn(tape_steps, dim) * 0.02)
        self.init_query = nn.Parameter(torch.randn(lanes, cells_per_lane, dim) * 0.04)

        alive_init = torch.linspace(1.15, -0.85, tape_steps)
        self.step_alive_logit = nn.Parameter(alive_init)
        self.boundary_logit = nn.Parameter(torch.full((tape_steps,), -1.25))
        self.route_logits = nn.Parameter(torch.eye(lanes).view(1, lanes, lanes).repeat(tape_steps, 1, 1) * 0.35)
        self.read_group_logits = nn.Parameter(torch.zeros(tape_steps, lanes, lanes + 1))
        self.read_query_w = nn.Parameter(torch.randn(tape_steps, lanes, dim, dim) * 0.025)
        self.write_gate_logit = nn.Parameter(torch.full((tape_steps, lanes), -0.35))

        # Weak useful bias only: when boundary grows, allow state->abstract/memory movement.
        bias = torch.zeros(lanes, lanes)
        if lanes >= 4:
            bias[1, 2] = 0.30
            bias[1, 3] = 0.18
            bias[0, 1] = 0.12
        self.register_buffer("boundary_route_bias", bias)

        self.units = nn.ModuleList([
            TapeLaneTransformUnit(dim, lanes, cells_per_lane, channel_stages, dropout)
            for _ in range(tape_steps)
        ])
        self.norm = nn.LayerNorm(dim)

    def init_lanes(self, evidence: torch.Tensor) -> torch.Tensor:
        # evidence: [B,E,D]
        q = self.init_query.to(device=evidence.device, dtype=evidence.dtype).reshape(self.lanes * self.cells_per_lane, self.dim)
        score = torch.einsum("ad,bed->bae", q, evidence) / math.sqrt(self.dim)
        attn = torch.softmax(score.float(), dim=-1).to(evidence.dtype)
        base = torch.einsum("bae,bed->bad", attn, evidence)
        x = base.reshape(evidence.shape[0], self.lanes, self.cells_per_lane, self.dim)
        x = x + self.lane_embed.to(device=evidence.device, dtype=evidence.dtype).view(1, self.lanes, 1, self.dim)
        return self.norm(x)

    def read_step(self, x: torch.Tensor, evidence: torch.Tensor, t: int) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        # input context per target lane/cell.
        w = self.read_query_w[t].to(device=x.device, dtype=x.dtype)  # [L,D,D]
        q = torch.einsum("blad,ldh->blah", x, w) + self.step_embed[t].to(device=x.device, dtype=x.dtype).view(1, 1, 1, self.dim)
        score = torch.einsum("blad,bed->blae", q, evidence) / math.sqrt(self.dim)
        input_attn = torch.softmax(score.float(), dim=-1).to(x.dtype)
        input_ctx = torch.einsum("blae,bed->blad", input_attn, evidence)

        # stack groups: input + each source lane, then mix per target lane.
        lane_groups = x.unsqueeze(1).expand(-1, self.lanes, -1, -1, -1)  # [B,target,source,A,D]
        groups = torch.cat([input_ctx.unsqueeze(2), lane_groups], dim=2)  # [B,target,L+1,A,D]

        group_w = torch.softmax(self.read_group_logits[t].float(), dim=-1).to(x.dtype)  # [target,L+1]
        read_packet = (group_w.view(1, self.lanes, self.lanes + 1, 1, 1) * groups).sum(dim=2)

        return read_packet, {
            "group_mass": group_w,
            "input_mass": group_w[:, 0].float().mean(),
        }

    def forward(self, wav: torch.Tensor) -> Tuple[torch.Tensor, TapeLaneAux]:
        evidence = self.evidence(wav)
        x = self.init_lanes(evidence)

        all_slots = [x]
        slot_names = [f"T0.{LANE_NAMES[l] if l < len(LANE_NAMES) else f'lane{l}'}.C{a}" for l in range(self.lanes) for a in range(self.cells_per_lane)]
        gate_list: List[torch.Tensor] = []
        update_norm_list: List[torch.Tensor] = []
        alive_list: List[torch.Tensor] = []
        boundary_list: List[torch.Tensor] = []
        route_list: List[torch.Tensor] = []
        route_entropy_list: List[torch.Tensor] = []
        read_group_list: List[torch.Tensor] = []
        late_input_cost_terms: List[torch.Tensor] = []

        for t, unit in enumerate(self.units):
            alive = torch.sigmoid(self.step_alive_logit[t])
            boundary = torch.sigmoid(self.boundary_logit[t])
            route_logits = self.route_logits[t] + boundary.float() * self.boundary_route_bias.to(device=x.device)
            route = torch.softmax(route_logits.float(), dim=-1).to(x.dtype)  # [from,to]

            read_packet, read_info = self.read_step(x, evidence, t)
            update, _unit_info = unit(x, read_packet, self.lane_embed)
            write_gate = torch.sigmoid(self.write_gate_logit[t]).to(device=x.device, dtype=x.dtype).view(1, self.lanes, 1, 1)

            routed = torch.einsum("ft,bfad->btad", route, update)
            x = self.norm(x + alive.to(x.dtype) * write_gate * routed)

            all_slots.append(x)
            for l in range(self.lanes):
                lname = LANE_NAMES[l] if l < len(LANE_NAMES) else f"lane{l}"
                for a in range(self.cells_per_lane):
                    slot_names.append(f"T{t+1}.{lname}.C{a}")

            gate_list.append(write_gate.expand(wav.shape[0], self.lanes, self.cells_per_lane, 1).squeeze(-1))
            update_norm_list.append(update.float().norm(dim=-1))
            alive_list.append(alive)
            boundary_list.append(boundary)
            route_list.append(route)
            route_entropy_list.append(entropy(route, dim=-1))
            read_group_list.append(read_info["group_mass"])

            depth_weight = torch.tensor(float(t + 1) / float(max(1, self.tape_steps)), device=x.device)
            late_input_cost_terms.append(depth_weight * read_info["input_mass"].to(x.device))

        slot_tensor = torch.stack(all_slots, dim=1)  # [B,T+1,L,A,D]
        flat_slots = slot_tensor.reshape(wav.shape[0], -1, self.dim)
        read_group = torch.stack(read_group_list, dim=0) if read_group_list else torch.empty(0, device=wav.device)
        late_input = torch.stack(late_input_cost_terms).mean() if late_input_cost_terms else torch.zeros((), device=wav.device)
        aux = TapeLaneAux(
            slots=flat_slots,
            slot_names=slot_names,
            write_gates=torch.stack(gate_list, dim=1) if gate_list else torch.empty(wav.shape[0], 0, self.lanes, self.cells_per_lane, device=wav.device),
            update_norms=torch.stack(update_norm_list, dim=1) if update_norm_list else torch.empty(wav.shape[0], 0, self.lanes, self.cells_per_lane, device=wav.device),
            step_alive=torch.stack(alive_list) if alive_list else torch.empty(0, device=wav.device),
            boundaries=torch.stack(boundary_list) if boundary_list else torch.empty(0, device=wav.device),
            routes=torch.stack(route_list, dim=0) if route_list else torch.empty(0, self.lanes, self.lanes, device=wav.device),
            route_entropy=torch.stack(route_entropy_list, dim=0) if route_entropy_list else torch.empty(0, self.lanes, device=wav.device),
            read_group_mass=read_group,
            late_input_read_mass=late_input,
        )
        return flat_slots, aux


class ClassMatrixLaneHead(nn.Module):
    """v3 ClassMatrix head adapted from phases to lanes."""

    def __init__(self, dim: int, classes: int, tape_steps: int, lanes: int, cells_per_lane: int, pair_slots: int, dropout: float, lane_prior_strength: float):
        super().__init__()
        self.dim = int(dim)
        self.classes = int(classes)
        self.tape_steps = int(tape_steps)
        self.lanes = int(lanes)
        self.cells_per_lane = int(cells_per_lane)
        self.pair_slots = int(pair_slots)
        self.lane_prior_strength = float(lane_prior_strength)

        self.class_state = nn.Parameter(torch.randn(classes, dim) * 0.05)
        self.class_lane_logits = nn.Parameter(class_lane_prior(classes, lanes))
        self.pair_state = nn.Parameter(torch.randn(pair_slots, dim) * 0.04)
        self.class_pair_logits = nn.Parameter(torch.randn(classes, pair_slots) * 0.04)

        self.key_w = nn.Parameter(torch.randn(dim, dim) * 0.04)
        self.value_w = nn.Parameter(torch.randn(dim, dim) * 0.04)
        self.class_q = nn.Parameter(torch.randn(dim, dim) * 0.04)
        self.pair_q = nn.Parameter(torch.randn(dim, dim) * 0.04)
        self.pair_k = nn.Parameter(torch.randn(dim, dim) * 0.04)
        self.pair_v = nn.Parameter(torch.randn(dim, dim) * 0.04)

        self.class_update = nn.Sequential(
            nn.LayerNorm(dim * 5),
            nn.Linear(dim * 5, dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim * 2, dim),
        )
        self.class_norm = nn.LayerNorm(dim)
        self.logit_w = nn.Parameter(torch.randn(classes, dim) * 0.04)
        self.logit_bias = nn.Parameter(torch.zeros(classes))
        self.write_logit = nn.Parameter(torch.tensor(-0.35))
        self.drop = nn.Dropout(dropout)

    def forward(self, slots: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        bsz, slot_count, dim = slots.shape
        lane_map_prior = lane_slot_matrix(self.tape_steps, self.lanes, self.cells_per_lane, slots.device, normalize_rows=True).to(slots.dtype)
        lane_map_mass = lane_slot_matrix(self.tape_steps, self.lanes, self.cells_per_lane, slots.device, normalize_rows=False).to(slots.dtype)
        lane_w = torch.softmax(self.class_lane_logits.float(), dim=-1).to(slots.dtype)  # [C,L]
        slot_prior = torch.matmul(lane_w, lane_map_prior).clamp_min(1e-8)  # [C,S]

        keys = slots @ self.key_w.to(device=slots.device, dtype=slots.dtype)
        values = self.drop(slots @ self.value_w.to(device=slots.device, dtype=slots.dtype))
        q = self.class_state.to(device=slots.device, dtype=slots.dtype) @ self.class_q.to(device=slots.device, dtype=slots.dtype)
        score = torch.einsum("cd,bsd->bcs", q, keys) / math.sqrt(dim)
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
            "class_slot_attention": attn.detach(),
            "class_lane_mass": lane_mass.detach(),
            "class_lane_logits": self.class_lane_logits.detach(),
            "class_read": class_read.detach(),
            "class_next": class_next.detach(),
            "pair_attention": pair_attn.detach(),
            "pair_update_norm": pair_ctx.detach().float().norm(dim=-1).mean(),
            "class_write": write.detach(),
        }


class TapeLaneRouterClassifier(nn.Module):
    def __init__(self, classes: int, args):
        super().__init__()
        self.backbone = TapeLaneRouterBackbone(
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
        )
        self.head = ClassMatrixLaneHead(
            dim=args.dim,
            classes=classes,
            tape_steps=args.tape_steps,
            lanes=args.lanes,
            cells_per_lane=args.cells_per_lane,
            pair_slots=args.pair_slots,
            dropout=args.head_dropout,
            lane_prior_strength=args.lane_prior_strength,
        )

    def forward(self, wav: torch.Tensor):
        slots, baux = self.backbone(wav)
        logits, haux = self.head(slots)
        return logits, baux, haux


def class_read_diversity_loss(attn: torch.Tensor) -> torch.Tensor:
    a = attn.float().mean(dim=0)
    a = F.normalize(a, dim=-1)
    sim = a @ a.t()
    offdiag = sim - torch.eye(sim.shape[0], device=sim.device)
    return F.relu(offdiag - 0.20).mean()


def lane_balance_loss(class_lane_mass: torch.Tensor) -> torch.Tensor:
    usage = class_lane_mass.float().mean(dim=(0, 1))
    usage = usage / usage.sum().clamp_min(1e-8)
    ent = entropy(usage, dim=0)
    ent_floor = F.relu(torch.tensor(1.10, device=usage.device) - ent).pow(2)
    overuse = F.relu(usage.max() - 0.55).pow(2)
    return ent_floor + overuse


def route_entropy_band_loss(route_entropy: torch.Tensor, min_ent: float, max_ent: float) -> torch.Tensor:
    if route_entropy.numel() == 0:
        return torch.zeros((), device=route_entropy.device)
    return F.relu(float(min_ent) - route_entropy.float()).pow(2).mean() + F.relu(route_entropy.float() - float(max_ent)).pow(2).mean()


def aux_losses(logits: torch.Tensor, baux: TapeLaneAux, haux: Dict[str, torch.Tensor], args) -> Dict[str, torch.Tensor]:
    gate = baux.write_gates.float()
    upd = baux.update_norms.float()
    target = torch.tensor(float(args.write_target), device=logits.device)
    alive_target = torch.tensor(float(args.step_alive_target), device=logits.device)
    boundary_target = torch.tensor(float(args.boundary_target), device=logits.device)
    memory_gate = gate[:, :, min(args.lanes - 1, 3), :].mean() if gate.numel() else torch.zeros((), device=logits.device)
    return {
        "write_budget": (gate.mean() - target).pow(2) if gate.numel() else torch.zeros((), device=logits.device),
        "update_alive": F.relu(torch.tensor(float(args.min_update_norm), device=logits.device) - upd.mean()).pow(2) if upd.numel() else torch.zeros((), device=logits.device),
        "class_read_div": class_read_diversity_loss(haux["class_slot_attention"]),
        "lane_balance": lane_balance_loss(haux["class_lane_mass"]),
        "slot_div": slot_diversity_loss(baux.slots),
        "route_entropy_band": route_entropy_band_loss(baux.route_entropy, args.route_entropy_min, args.route_entropy_max),
        "step_alive_budget": (baux.step_alive.float().mean() - alive_target).pow(2) if baux.step_alive.numel() else torch.zeros((), device=logits.device),
        "boundary_budget": (baux.boundaries.float().mean() - boundary_target).pow(2) if baux.boundaries.numel() else torch.zeros((), device=logits.device),
        "late_input_read_cost": baux.late_input_read_mass.float(),
        "memory_overwrite_cost": memory_gate,
        "logit_norm": logits.float().pow(2).mean(),
        "pair_update_norm": haux["pair_update_norm"].float(),
    }


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
            losses = aux_losses(logits, baux, haux, args)
            loss = ce
            loss = loss + args.lambda_write_budget * losses["write_budget"]
            loss = loss + args.lambda_update_alive * losses["update_alive"]
            loss = loss + args.lambda_class_read_div * losses["class_read_div"]
            loss = loss + args.lambda_lane_balance * losses["lane_balance"]
            loss = loss + args.lambda_slot_div * losses["slot_div"]
            loss = loss + args.lambda_route_entropy * losses["route_entropy_band"]
            loss = loss + args.lambda_step_alive_budget * losses["step_alive_budget"]
            loss = loss + args.lambda_boundary_budget * losses["boundary_budget"]
            loss = loss + args.lambda_late_input_read * losses["late_input_read_cost"]
            loss = loss + args.lambda_memory_overwrite * losses["memory_overwrite_cost"]
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


def _slot_top_reads(attn: torch.Tensor, slot_names: Sequence[str], k: int = 5):
    top_reads = []
    for ci in range(attn.shape[0]):
        vals, idxs = torch.topk(attn[ci], k=min(k, attn.shape[1]))
        top_reads.append([{"slot": slot_names[int(j)], "weight": float(v)} for v, j in zip(vals.tolist(), idxs.tolist())])
    return top_reads


@torch.no_grad()
def evaluate(model, loader, device, dtype, args):
    model.eval()
    use_amp = device.startswith("cuda") and dtype != torch.float32
    total_loss, correct, n = 0.0, 0, 0
    conf = torch.zeros(args.num_classes, args.num_classes, dtype=torch.long)
    last_report = None
    for step, (wav, y) in enumerate(loader, 1):
        if args.max_val_batches and step > args.max_val_batches:
            break
        wav = wav.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        with torch.autocast(device_type=device.split(":")[0], dtype=dtype, enabled=use_amp):
            logits, baux, haux = model(wav)
            loss = F.cross_entropy(logits.float(), y)
        pred = logits.argmax(-1)
        bs = y.numel()
        total_loss += float(loss.detach().cpu()) * bs
        correct += int((pred == y).sum().detach().cpu())
        n += bs
        conf += torch.bincount((y.cpu() * args.num_classes + pred.cpu()), minlength=args.num_classes ** 2).view(args.num_classes, args.num_classes)

        attn = haux["class_slot_attention"].float().mean(dim=0).cpu()
        lane_mass = haux["class_lane_mass"].float().mean(dim=0).cpu()
        gates = baux.write_gates.float().mean(dim=(0, 3)).cpu() if baux.write_gates.numel() else torch.empty(0)
        updates = baux.update_norms.float().mean(dim=(0, 3)).cpu() if baux.update_norms.numel() else torch.empty(0)
        read_group = baux.read_group_mass.float().cpu() if baux.read_group_mass.numel() else torch.empty(0)
        top_reads = _slot_top_reads(attn, baux.slot_names)

        last_report = {
            "lane_names": list(LANE_NAMES[: args.lanes]),
            "read_group_names": list(READ_GROUP_NAMES[: args.lanes + 1]),
            "step_alive": _to_float_list(baux.step_alive),
            "boundary": _to_float_list(baux.boundaries),
            "route_matrix": _to_float_list(baux.routes),
            "route_entropy": _to_float_list(baux.route_entropy),
            "read_group_mass": _to_float_list(read_group),
            "write_gate_by_step_lane": _to_float_list(gates),
            "update_norm_by_step_lane": _to_float_list(updates),
            "late_input_read_mass": float(baux.late_input_read_mass.detach().float().cpu()),
            "class_top_reads": top_reads,
            "class_lane_mass": [
                {LANE_NAMES[li] if li < len(LANE_NAMES) else f"lane{li}": float(lane_mass[ci, li]) for li in range(lane_mass.shape[1])}
                for ci in range(lane_mass.shape[0])
            ],
            "lane_mass_mean": {
                LANE_NAMES[li] if li < len(LANE_NAMES) else f"lane{li}": float(lane_mass[:, li].mean())
                for li in range(lane_mass.shape[1])
            },
            "pair_update_norm": float(haux["pair_update_norm"].detach().cpu()),
            "class_write": float(haux["class_write"].detach().cpu()),
            "slot_count": len(baux.slot_names),
        }
    return {"loss": total_loss / max(1, n), "acc": correct / max(1, n), "n": n, "confusion": conf.tolist(), "report": last_report}


def write_chatgpt_report(out_dir: Path, analysis: Dict, args) -> None:
    rep = analysis.get("matrix_report") or {}
    step_alive = rep.get("step_alive", [])
    boundaries = rep.get("boundary", [])
    route_entropy = rep.get("route_entropy", [])
    lane_mass = rep.get("lane_mass_mean", {})
    late_input = rep.get("late_input_read_mass", 0.0)

    active = [i for i, v in enumerate(step_alive) if float(v) >= 0.50]
    boundary_hot = [i for i, v in enumerate(boundaries) if float(v) >= 0.35]
    ent_mean = []
    for row in route_entropy:
        if row:
            ent_mean.append(sum(float(v) for v in row) / len(row))
    ent_text = ", ".join(f"T{i}:{v:.2f}" for i, v in enumerate(ent_mean[: args.tape_steps]))

    lines = [
        "REPORT_TO_CHATGPT",
        "",
        f"Лучший val_acc: {100.0 * float(analysis.get('best_acc', 0.0)):.2f}% @ epoch {analysis.get('best_epoch', 0)}",
        f"Текущий epoch: {analysis.get('epoch', 0)}",
        "",
        "Что смотреть:",
        f"- Активные tape steps по step_alive>=0.50: {active}",
        f"- Заметные soft-boundaries boundary>=0.35: {boundary_hot}",
        f"- Late input read mass: {float(late_input):.4f} (если растёт сильно — модель читает сырой input поздно)",
        f"- Route entropy mean: {ent_text}",
        f"- Средняя class lane mass: {lane_mass}",
        "",
        "Интерпретация:",
        "- Хорошо: не все route_matrix identity, class_lane_mass распределяется по нескольким lanes.",
        "- Плохо: step_alive все одинаковые, boundary все 0/1, late_input_read_mass высокий, class_top_reads только из последних T.",
        "",
        "Артефакты:",
        "- metrics.csv",
        "- analysis_epoch_XXX.json",
        "- final_report.json",
        "- train.log если запускался sync_run_5ep_push_logs.sh",
    ]
    (out_dir / "REPORT_TO_CHATGPT.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(args) -> None:
    set_seed(args.seed)
    device = args.device if args.device != "cuda" or torch.cuda.is_available() else "cpu"
    if device.startswith("cuda"):
        torch.backends.cudnn.benchmark = True
        torch.set_float32_matmul_precision("high")
    dtype = amp_dtype(args.amp)
    out_dir = ensure_dir(Path(args.out_dir))
    train_loader, val_loader, classes, train_counts, val_counts = make_loaders(args)
    args.num_classes = len(classes)
    model = TapeLaneRouterClassifier(len(classes), args).to(device)
    if args.init_checkpoint:
        ckpt = torch.load(args.init_checkpoint, map_location=device)
        state = ckpt.get("model", ckpt)
        model.load_state_dict(state, strict=True)
        print(f"loaded init checkpoint: {args.init_checkpoint}", flush=True)
    params = sum(p.numel() for p in model.parameters())
    print(f"loaded datasets: train={len(train_loader.dataset)} val={len(val_loader.dataset)} classes={classes}", flush=True)
    print(
        f"TapeLaneRouter params={params} T={args.tape_steps} lanes={args.lanes} "
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
        "route_entropy_band", "step_alive_budget", "boundary_budget", "late_input_read_cost",
        "memory_overwrite_cost", "logit_norm", "pair_update_norm",
    ]
    with (out_dir / "metrics.csv").open("w", newline="", encoding="utf-8") as f:
        csv.DictWriter(f, fieldnames=fields).writeheader()

    best, best_epoch = -1.0, 0
    last_analysis = {}
    for epoch in range(1, args.epochs + 1):
        tr = train_epoch(model, train_loader, opt, scaler, device, dtype, args, epoch)
        va = evaluate(model, val_loader, device, dtype, args)
        if va["acc"] > best:
            best, best_epoch = va["acc"], epoch
            if not args.no_save_checkpoints:
                torch.save({"model": model.state_dict(), "args": vars(args), "classes": classes, "epoch": epoch, "best_acc": best}, out_dir / "best.pt")
        if not args.no_save_checkpoints:
            torch.save({"model": model.state_dict(), "args": vars(args), "classes": classes, "epoch": epoch, "best_acc": best}, out_dir / "last.pt")

        row = {
            "epoch": epoch,
            "train_loss": tr["loss"],
            "train_ce": tr["ce"],
            "train_acc": tr["acc"],
            "val_loss": va["loss"],
            "val_acc": va["acc"],
            "best_acc": best,
            "write_budget": tr.get("write_budget", 0.0),
            "update_alive": tr.get("update_alive", 0.0),
            "class_read_div": tr.get("class_read_div", 0.0),
            "lane_balance": tr.get("lane_balance", 0.0),
            "slot_div": tr.get("slot_div", 0.0),
            "route_entropy_band": tr.get("route_entropy_band", 0.0),
            "step_alive_budget": tr.get("step_alive_budget", 0.0),
            "boundary_budget": tr.get("boundary_budget", 0.0),
            "late_input_read_cost": tr.get("late_input_read_cost", 0.0),
            "memory_overwrite_cost": tr.get("memory_overwrite_cost", 0.0),
            "logit_norm": tr.get("logit_norm", 0.0),
            "pair_update_norm": tr.get("pair_update_norm", 0.0),
        }
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
            "matrix_report": va["report"],
        }
        write_json(out_dir / f"analysis_epoch_{epoch:03d}.json", last_analysis)
        write_chatgpt_report(out_dir, last_analysis, args)
        print(f"epoch {epoch:03d}/{args.epochs} train={tr['loss']:.4f}/{100*tr['acc']:.2f}% val={va['loss']:.4f}/{100*va['acc']:.2f}% best={100*best:.2f}%@{best_epoch}", flush=True)

    write_json(out_dir / "final_report.json", {"best_acc": best, "best_epoch": best_epoch, "args": vars(args), "classes": classes})
    if last_analysis:
        write_chatgpt_report(out_dir, last_analysis, args)


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser()
    p.add_argument("--synthetic", action="store_true")
    p.add_argument("--synthetic-length", type=int, default=512)
    p.add_argument("--data-root", default="../architecture_builder/data/speechcommands")
    p.add_argument("--download", action="store_true")
    p.add_argument("--classes", default="yes,no,up,down,left,right,on,off,stop,go")
    p.add_argument("--train-limit", type=int, default=12000)
    p.add_argument("--val-limit", type=int, default=2000)
    p.add_argument("--seconds", type=float, default=1.0)
    p.add_argument("--sample-rate", type=int, default=16000)
    p.add_argument("--n-mels", type=int, default=64)
    p.add_argument("--hop-length", type=int, default=160)

    p.add_argument("--dim", type=int, default=96)
    p.add_argument("--evidence-cells", type=int, default=48)
    p.add_argument("--lanes", type=int, default=4)
    p.add_argument("--cells-per-lane", type=int, default=12)
    p.add_argument("--tape-steps", type=int, default=12)
    p.add_argument("--pair-slots", type=int, default=12)
    p.add_argument("--channel-stages", type=int, default=3)
    p.add_argument("--dropout", type=float, default=0.04)
    p.add_argument("--head-dropout", type=float, default=0.05)

    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--eval-batch-size", type=int, default=256)
    p.add_argument("--workers", type=int, default=2)
    p.add_argument("--pin-memory", action="store_true")
    p.add_argument("--lr", type=float, default=5e-4)
    p.add_argument("--weight-decay", type=float, default=0.012)
    p.add_argument("--grad-clip", type=float, default=0.75)

    p.add_argument("--write-target", type=float, default=0.42)
    p.add_argument("--min-update-norm", type=float, default=0.18)
    p.add_argument("--lane-prior-strength", type=float, default=0.65)
    p.add_argument("--route-entropy-min", type=float, default=0.35)
    p.add_argument("--route-entropy-max", type=float, default=1.35)
    p.add_argument("--step-alive-target", type=float, default=0.52)
    p.add_argument("--boundary-target", type=float, default=0.22)

    p.add_argument("--lambda-write-budget", type=float, default=0.020)
    p.add_argument("--lambda-update-alive", type=float, default=0.004)
    p.add_argument("--lambda-class-read-div", type=float, default=0.030)
    p.add_argument("--lambda-lane-balance", type=float, default=0.035)
    p.add_argument("--lambda-slot-div", type=float, default=0.002)
    p.add_argument("--lambda-route-entropy", type=float, default=0.010)
    p.add_argument("--lambda-step-alive-budget", type=float, default=0.010)
    p.add_argument("--lambda-boundary-budget", type=float, default=0.004)
    p.add_argument("--lambda-late-input-read", type=float, default=0.012)
    p.add_argument("--lambda-memory-overwrite", type=float, default=0.001)
    p.add_argument("--lambda-logit-norm", type=float, default=0.0007)

    p.add_argument("--amp", choices=["fp16", "bf16", "fp32", "off"], default="fp16")
    p.add_argument("--device", default="cuda")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max-train-batches", type=int, default=0)
    p.add_argument("--max-val-batches", type=int, default=0)
    p.add_argument("--log-every", type=int, default=50)
    p.add_argument("--out-dir", default="./simple_butterfly_matrix_v4_tape_lane/runs/speechcommands_5ep")
    p.add_argument("--init-checkpoint", default="")
    p.add_argument("--no-save-checkpoints", action="store_true")
    return p


if __name__ == "__main__":
    run(parser().parse_args())
