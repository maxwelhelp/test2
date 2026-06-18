#!/usr/bin/env python3
"""v4.6.3 edge-conditioned matrix program.

Core change:
    old: primitive[t, source] -> route[source,target]
    new: edge_context[t, source, target] -> edge_gate/write/boundary/primitive[source,target]

Training stays soft/differentiable. Credit ablation is delayed: validation credit from
one epoch can be used as a soft penalty in the next epoch.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from .modules import MatrixMemory, closed_loop_aux_losses
    from .modules.primitive_selector import PRIMITIVE_GROUPS, DEFAULT_GROUPS, DEFAULT_PRIMITIVES
    from .modules.step_analyzer import StepAnalyzer, append_metrics_csv
    from .tape_lane_transport_v4_6_loop_core import (
        LANE_NAMES,
        ConvWaveFrontend,
        amp_dtype,
        ensure_dir,
        make_loaders,
        metrics_fields,
        set_seed,
        sync_if_cuda,
        tau_for_epoch,
        weighted_loss,
        write_json,
    )
except ImportError:
    ROOT = Path(__file__).resolve().parents[2]
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    from simple_butterfly_matrix_v4_tape_lane.v4_6_loop_core.modules import MatrixMemory, closed_loop_aux_losses  # type: ignore
    from simple_butterfly_matrix_v4_tape_lane.v4_6_loop_core.modules.primitive_selector import PRIMITIVE_GROUPS, DEFAULT_GROUPS, DEFAULT_PRIMITIVES  # type: ignore
    from simple_butterfly_matrix_v4_tape_lane.v4_6_loop_core.modules.step_analyzer import StepAnalyzer, append_metrics_csv  # type: ignore
    from simple_butterfly_matrix_v4_tape_lane.v4_6_loop_core.tape_lane_transport_v4_6_loop_core import (  # type: ignore
        LANE_NAMES,
        ConvWaveFrontend,
        amp_dtype,
        ensure_dir,
        make_loaders,
        metrics_fields,
        set_seed,
        sync_if_cuda,
        tau_for_epoch,
        weighted_loss,
        write_json,
    )


CONTEXT_STATS_DIM = 8


def build_context_stats(evidence: torch.Tensor, state: torch.Tensor, memory_read: torch.Tensor, previous_update_norm: Optional[torch.Tensor], step_index: int, steps: int) -> torch.Tensor:
    bsz, lanes, _ = state.shape
    ev = evidence.float()
    st = state.float()
    mem = memory_read.float()
    density = ev.abs().mean(dim=-1, keepdim=True)
    variance = ev.var(dim=-1, unbiased=False, keepdim=True)
    ev_norm = ev.norm(dim=-1, keepdim=True) / max(1.0, float(evidence.shape[-1]) ** 0.5)
    mem_norm = mem.norm(dim=-1, keepdim=True) / max(1.0, float(memory_read.shape[-1]) ** 0.5)
    if previous_update_norm is None:
        upd = torch.zeros(bsz, 1, device=state.device, dtype=st.dtype)
    else:
        upd = previous_update_norm.float().view(bsz, 1)
    progress = torch.full((bsz, 1), float(step_index) / float(max(1, steps - 1)), device=state.device, dtype=st.dtype)
    shared = torch.cat([density, variance, ev_norm, mem_norm, upd, progress], dim=-1).unsqueeze(1).expand(-1, lanes, -1)
    lane = torch.stack([
        st.norm(dim=-1) / max(1.0, float(state.shape[-1]) ** 0.5),
        st.var(dim=-1, unbiased=False),
    ], dim=-1)
    return torch.cat([shared, lane], dim=-1).to(dtype=state.dtype)


class EdgePrimitiveBank(nn.Module):
    """Grouped primitive bank applied per directed edge source->target."""

    def __init__(self, dim: int, num_primitives: int = 18, rank: int = 32, dropout: float = 0.0) -> None:
        super().__init__()
        self.dim = int(dim)
        self.rank = max(1, min(int(rank), self.dim))
        self.group_names = list(DEFAULT_GROUPS)
        self.primitive_names = list(DEFAULT_PRIMITIVES[: int(num_primitives)])
        while len(self.primitive_names) < int(num_primitives):
            self.primitive_names.append(f"low_rank_extra_{len(self.primitive_names)}")
        self.num_primitives = len(self.primitive_names)
        self.num_groups = len(self.group_names)
        group_ids = []
        for name in self.primitive_names:
            gid = 1
            for j, g in enumerate(self.group_names):
                if name in PRIMITIVE_GROUPS[g]:
                    gid = j
                    break
            group_ids.append(gid)
        self.register_buffer("primitive_group_id", torch.tensor(group_ids, dtype=torch.long), persistent=False)
        scale = 1.0 / math.sqrt(max(1, self.dim))
        self.low_u = nn.Parameter(torch.randn(self.num_primitives, self.dim, self.rank) * scale)
        self.low_v = nn.Parameter(torch.randn(self.num_primitives, self.rank, self.dim) * scale)
        self.channel = nn.Sequential(nn.LayerNorm(self.dim), nn.Linear(self.dim, self.dim * 2), nn.SiLU(), nn.Dropout(dropout), nn.Linear(self.dim * 2, self.dim))
        self.ctx_gate = nn.Sequential(nn.LayerNorm(self.dim), nn.Linear(self.dim, 1))
        self.product_a = nn.Linear(self.dim, self.rank, bias=False)
        self.product_b = nn.Linear(self.dim, self.rank, bias=False)
        self.product_out = nn.Linear(self.rank, self.dim, bias=False)
        self.memory_proj = nn.Linear(self.dim, self.dim, bias=False)
        self.merge = nn.Sequential(nn.LayerNorm(self.dim * 2), nn.Linear(self.dim * 2, self.dim), nn.SiLU(), nn.Linear(self.dim, self.dim))
        self.group_embedding = nn.Embedding(self.num_groups, self.dim)
        self.primitive_embedding = nn.Embedding(self.num_primitives, self.dim)
        self.choice_norm = nn.LayerNorm(self.dim)
        self.out_norm = nn.LayerNorm(self.dim)

    def _low_rank_all(self, src: torch.Tensor) -> torch.Tensor:
        h = torch.einsum("bijd,kdr->bijkr", src, self.low_u.to(device=src.device, dtype=src.dtype))
        return torch.einsum("bijkr,krd->bijkd", h, self.low_v.to(device=src.device, dtype=src.dtype))

    def grouped_weights(self, scores: torch.Tensor, tau: float, ablate_group: Optional[int] = None, ablate_primitive: Optional[int] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        sf = scores.float()
        k = sf.shape[-1]
        group_logits = []
        gid_vec = self.primitive_group_id.to(sf.device)
        for gid in range(self.num_groups):
            mask = (gid_vec == gid).view(1, 1, 1, k)
            group_logits.append(torch.logsumexp(sf.masked_fill(~mask, -1e4), dim=-1))
        gl = torch.stack(group_logits, dim=-1)
        group_w = F.gumbel_softmax(gl, tau=float(tau), hard=False, dim=-1)
        parts = []
        for gid in range(self.num_groups):
            mask = (gid_vec == gid).view(1, 1, 1, k)
            inside = F.softmax(sf.masked_fill(~mask, -1e4), dim=-1)
            parts.append(group_w[..., gid:gid + 1] * inside)
        w = torch.stack(parts, dim=0).sum(dim=0)
        if ablate_group is not None:
            gid = int(ablate_group)
            mask = (gid_vec != gid).to(dtype=w.dtype).view(1, 1, 1, k)
            w = w * mask
            w = w / w.sum(dim=-1, keepdim=True).clamp_min(1e-8)
            gm = torch.ones_like(group_w)
            gm[..., gid] = 0.0
            group_w = group_w * gm
            group_w = group_w / group_w.sum(dim=-1, keepdim=True).clamp_min(1e-8)
        if ablate_primitive is not None:
            pid = int(ablate_primitive)
            w = w.clone()
            w[..., pid] = 0.0
            w = w / w.sum(dim=-1, keepdim=True).clamp_min(1e-8)
        return w.to(dtype=scores.dtype), group_w.to(dtype=scores.dtype)

    def expert_bank(self, src: torch.Tensor, tgt: torch.Tensor, edge_z: torch.Tensor, memory_read: torch.Tensor) -> torch.Tensor:
        bsz, lanes, _, dim = src.shape
        bank = self._low_rank_all(src)
        mem = self.memory_proj(memory_read).view(bsz, 1, 1, dim).expand(-1, lanes, lanes, -1)
        gate = torch.sigmoid(self.ctx_gate(edge_z))
        prod = self.product_out(torch.tanh(self.product_a(src)) * torch.tanh(self.product_b(tgt)))
        merged = self.merge(torch.cat([src, tgt], dim=-1))
        mean_src = src.mean(dim=1, keepdim=True).expand_as(src)
        named = {
            "identity": src,
            "gated_keep": gate * src,
            "channel": self.channel(src),
            "low_rank": bank[..., min(1, self.num_primitives - 1), :],
            "ctx_matrix": gate * bank[..., min(2, self.num_primitives - 1), :],
            "diff_prev": src - tgt,
            "contrast": src - mean_src,
            "subtract_memory": src - mem,
            "lane_mean": mean_src,
            "smooth_lanes": 0.5 * (src + tgt),
            "pool_context": merged,
            "memory_read": mem,
            "memory_write_candidate": gate * merged + mem,
            "forget_like": src - gate * mem,
            "product_gate": prod,
            "gated_add": tgt + gate * prod,
            "mul_filter": src * torch.tanh(prod),
        }
        outs = []
        for i, name in enumerate(self.primitive_names):
            outs.append(named.get(name, bank[..., i, :]))
        return torch.stack(outs, dim=-2)

    def choice_context(self, weights: torch.Tensor, group_weights: torch.Tensor) -> torch.Tensor:
        pids = torch.arange(self.num_primitives, device=weights.device)
        gids = torch.arange(self.num_groups, device=weights.device)
        pe = self.primitive_embedding(pids).to(dtype=weights.dtype)
        ge = self.group_embedding(gids).to(dtype=weights.dtype)
        ctx = torch.einsum("bijk,kd->bijd", weights, pe) + torch.einsum("bijg,gd->bijd", group_weights, ge)
        return self.choice_norm(ctx)

    def forward(self, src: torch.Tensor, tgt: torch.Tensor, edge_z: torch.Tensor, memory_read: torch.Tensor, scores: torch.Tensor, sign_logits: torch.Tensor, tau: float, ablate_group: Optional[int] = None, ablate_primitive: Optional[int] = None) -> Dict[str, torch.Tensor]:
        experts = self.expert_bank(src, tgt, edge_z, memory_read)
        weights, group_weights = self.grouped_weights(scores, tau, ablate_group=ablate_group, ablate_primitive=ablate_primitive)
        signs = torch.tanh(sign_logits)
        update = torch.einsum("bijk,bijk,bijkd->bijd", weights, signs, experts)
        update = self.out_norm(update)
        return {
            "update": update,
            "weights": weights,
            "group_weights": group_weights,
            "signs": signs,
            "choice_context": self.choice_context(weights, group_weights),
        }


class EdgeProgramClassifier(nn.Module):
    def __init__(self, classes: int, args) -> None:
        super().__init__()
        self.classes = int(classes)
        self.dim = int(args.dim)
        self.steps = int(args.steps)
        self.lanes = int(args.lanes)
        self.memory_slot = min(int(args.memory_slot), self.lanes - 1)
        self.frontend = ConvWaveFrontend(self.dim, self.steps)
        self.input_proj = nn.Linear(self.dim, self.dim)
        self.slot_embedding = nn.Parameter(torch.randn(self.lanes, self.dim) * 0.02)
        self.step_embedding = nn.Embedding(max(32, self.steps), self.dim)
        self.memory = MatrixMemory(self.dim, init_forget_logit=float(args.memory_forget_logit_init))
        self.memory_proj = nn.Linear(self.dim, self.dim, bias=False)
        self.evidence_proj = nn.Linear(self.dim, self.dim, bias=False)
        self.stats_proj = nn.Sequential(nn.LayerNorm(CONTEXT_STATS_DIM), nn.Linear(CONTEXT_STATS_DIM, self.dim, bias=False))
        self.edge_in = nn.Sequential(nn.LayerNorm(self.dim * 10), nn.Linear(self.dim * 10, self.dim), nn.SiLU(), nn.Linear(self.dim, self.dim), nn.LayerNorm(self.dim))
        heads = 4 if self.dim % 4 == 0 else 1
        self.edge_attn = nn.MultiheadAttention(self.dim, heads, batch_first=True, dropout=float(args.dropout))
        self.edge_ff = nn.Sequential(nn.LayerNorm(self.dim), nn.Linear(self.dim, self.dim * 2), nn.SiLU(), nn.Dropout(float(args.dropout)), nn.Linear(self.dim * 2, self.dim), nn.LayerNorm(self.dim))
        self.edge_gate_head = nn.Linear(self.dim, 1)
        self.edge_boundary_head = nn.Linear(self.dim, 1)
        self.write_gate_head = nn.Linear(self.dim, 1)
        self.primitive_score_head = nn.Linear(self.dim, int(args.num_primitives))
        self.primitive_sign_head = nn.Linear(self.dim, int(args.num_primitives))
        self.step_boundary_head = nn.Linear(self.dim, 1)
        self.mem_write_gate_head = nn.Linear(self.dim, 1)
        self.mem_read_gate_head = nn.Linear(self.dim, 1)
        self.primitive_bank = EdgePrimitiveBank(self.dim, int(args.num_primitives), int(args.primitive_rank), float(args.dropout))
        self.read_to_state = nn.Linear(self.dim, self.dim, bias=False)
        self.state_norm = nn.LayerNorm(self.dim)
        self.output_gate_head = nn.Linear(self.dim, 1)
        self.output_op_head = nn.Linear(self.dim, 3)
        self.output_low = nn.Linear(self.dim, self.dim, bias=False)
        self.output_mem = nn.Linear(self.dim, self.dim, bias=False)
        self.output_norm = nn.LayerNorm(self.dim)
        self.head = nn.Sequential(nn.LayerNorm(self.dim), nn.Linear(self.dim, self.dim), nn.SiLU(), nn.Dropout(float(args.dropout)), nn.Linear(self.dim, self.classes))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        for head in (self.edge_gate_head, self.edge_boundary_head, self.write_gate_head, self.output_gate_head, self.mem_write_gate_head):
            if head.bias is not None:
                nn.init.normal_(head.bias, mean=0.0, std=0.03)

    def build_edge_z(self, state: torch.Tensor, evidence: torch.Tensor, memory_read: torch.Tensor, context_stats: torch.Tensor, previous_choice: torch.Tensor, step_id: int) -> torch.Tensor:
        bsz, lanes, dim = state.shape
        src = state.unsqueeze(2).expand(-1, lanes, lanes, -1)
        tgt = state.unsqueeze(1).expand(-1, lanes, lanes, -1)
        slot = self.slot_embedding.to(device=state.device, dtype=state.dtype)
        src_e = slot.view(1, lanes, 1, dim).expand(bsz, -1, lanes, -1)
        tgt_e = slot.view(1, 1, lanes, dim).expand(bsz, lanes, -1, -1)
        step = self.step_embedding(torch.full((bsz,), int(step_id) % self.step_embedding.num_embeddings, device=state.device, dtype=torch.long)).to(dtype=state.dtype).view(bsz, 1, 1, dim).expand(-1, lanes, lanes, -1)
        ev = self.evidence_proj(evidence).view(bsz, 1, 1, dim).expand(-1, lanes, lanes, -1)
        mem = self.memory_proj(memory_read).view(bsz, 1, 1, dim).expand(-1, lanes, lanes, -1)
        stats = self.stats_proj(context_stats).unsqueeze(2).expand(-1, lanes, lanes, -1)
        x = torch.cat([src, tgt, src - tgt, src * tgt, src_e, tgt_e, step, ev, mem + stats, previous_choice], dim=-1)
        edge = self.edge_in(x)
        flat = edge.view(bsz, lanes * lanes, dim)
        attn, _ = self.edge_attn(flat, flat, flat, need_weights=False)
        edge = self.edge_ff((flat + attn).view(bsz, lanes, lanes, dim))
        return edge

    def forward(self, wav: torch.Tensor, tau: float = 1.0, ablate_group: Optional[int] = None, ablate_primitive: Optional[int] = None, ablate_edge: Optional[Tuple[int, int]] = None, ablate_step: Optional[int] = None, ablate_output: Optional[Tuple[int, int]] = None, ablate_memory: bool = False):
        features = self.input_proj(self.frontend(wav))
        bsz = features.shape[0]
        device, dtype = features.device, features.dtype
        state = features[:, 0, :].unsqueeze(1) + self.slot_embedding.to(device=device, dtype=dtype).unsqueeze(0)
        mem = self.memory.init_state(bsz, device=device, dtype=dtype)
        memory_read = torch.zeros_like(mem)
        previous_choice = torch.zeros(bsz, self.lanes, self.lanes, self.dim, device=device, dtype=dtype)
        previous_update_norm: Optional[torch.Tensor] = None
        output_acc = torch.zeros(bsz, self.dim, device=device, dtype=dtype)
        output_den = torch.zeros(bsz, 1, device=device, dtype=dtype)
        tr: Dict[str, List[torch.Tensor]] = {k: [] for k in [
            "step_boundary", "edge_gate", "edge_boundary", "edge_write_gate", "edge_primitive_weights", "edge_group_weights", "edge_signs", "output_gate", "memory_write_norm", "memory_read_norm", "memory_read_influence", "update_norm", "context_stats_norm",
        ]}
        memory_forget = torch.sigmoid(self.memory.forget_logit)
        for t in range(self.steps):
            evidence = features[:, t, :]
            context_stats = build_context_stats(evidence, state, memory_read, previous_update_norm, t, self.steps)
            edge_z = self.build_edge_z(state, evidence, memory_read, context_stats, previous_choice, t)
            src = state.unsqueeze(2).expand(-1, self.lanes, self.lanes, -1)
            tgt = state.unsqueeze(1).expand(-1, self.lanes, self.lanes, -1)
            edge_gate = torch.sigmoid(self.edge_gate_head(edge_z).squeeze(-1))
            edge_boundary = torch.sigmoid(self.edge_boundary_head(edge_z).squeeze(-1))
            write_gate = torch.sigmoid(self.write_gate_head(edge_z).squeeze(-1))
            step_boundary = torch.sigmoid(self.step_boundary_head(edge_z.mean(dim=(1, 2))).squeeze(-1))
            prim = self.primitive_bank(src, tgt, edge_z, memory_read, self.primitive_score_head(edge_z), self.primitive_sign_head(edge_z), tau=float(tau), ablate_group=ablate_group, ablate_primitive=ablate_primitive)
            if ablate_edge is not None:
                i, j = int(ablate_edge[0]), int(ablate_edge[1])
                edge_gate = edge_gate.clone(); edge_gate[:, i, j] = 0.0
            if ablate_step is not None and int(ablate_step) == t:
                edge_gate = torch.zeros_like(edge_gate)
                write_gate = torch.zeros_like(write_gate)
                edge_boundary = torch.zeros_like(edge_boundary)
            gate = edge_gate * write_gate * edge_boundary * (0.5 + 0.5 * step_boundary.view(bsz, 1, 1))
            message = gate.unsqueeze(-1) * prim["update"]
            incoming = message.sum(dim=1) / math.sqrt(float(max(1, self.lanes)))
            mem_edge_z = edge_z[:, :, self.memory_slot, :]
            mem_write_logits = self.mem_write_gate_head(mem_edge_z).squeeze(-1)
            if ablate_memory or (ablate_step is not None and int(ablate_step) == t):
                mem_write_logits = torch.full_like(mem_write_logits, -20.0)
            mem_out = self.memory(mem, mem_edge_z, write_gate=mem_write_logits, read_gate=self.mem_read_gate_head(edge_z.mean(dim=(1, 2))).squeeze(-1))
            mem = mem_out.mem_next
            memory_read = torch.zeros_like(mem_out.read) if ablate_memory else mem_out.read
            read_inject = self.read_to_state(memory_read).view(bsz, 1, self.dim)
            state = self.state_norm(state + incoming + read_inject)
            previous_choice = prim["choice_context"]
            previous_update_norm = incoming.float().norm(dim=-1).mean(dim=-1)
            slot_context = edge_z.mean(dim=1)
            output_gate = torch.sigmoid(self.output_gate_head(slot_context).squeeze(-1))
            if ablate_step is not None and int(ablate_step) == t:
                output_gate = torch.zeros_like(output_gate)
            if ablate_output is not None and int(ablate_output[0]) == t:
                output_gate = output_gate.clone(); output_gate[:, int(ablate_output[1])] = 0.0
            ow = torch.softmax(self.output_op_head(slot_context).float(), dim=-1).to(dtype=dtype)
            mem_slot = self.output_mem(memory_read).view(bsz, 1, self.dim).expand(-1, self.lanes, -1)
            out_bank = torch.stack([state, self.output_low(state), mem_slot], dim=-2)
            out_candidate = torch.einsum("blm,blmd->bld", ow, out_bank)
            output_acc = output_acc + (output_gate.unsqueeze(-1) * out_candidate).sum(dim=1)
            output_den = output_den + output_gate.sum(dim=1, keepdim=True)
            tr["step_boundary"].append(step_boundary)
            tr["edge_gate"].append(edge_gate)
            tr["edge_boundary"].append(edge_boundary)
            tr["edge_write_gate"].append(write_gate)
            tr["edge_primitive_weights"].append(prim["weights"])
            tr["edge_group_weights"].append(prim["group_weights"])
            tr["edge_signs"].append(prim["signs"])
            tr["output_gate"].append(output_gate)
            tr["memory_write_norm"].append(mem_out.write_vec.float().norm(dim=-1))
            tr["memory_read_norm"].append(memory_read.float().norm(dim=-1))
            tr["memory_read_influence"].append(read_inject.float().norm(dim=-1).mean(dim=-1))
            tr["update_norm"].append(incoming.float().norm(dim=-1).mean(dim=-1))
            tr["context_stats_norm"].append(context_stats.float().norm(dim=-1).mean(dim=-1))
        output_state = self.output_norm(output_acc / output_den.clamp_min(1e-6))
        logits = self.head(output_state)
        edge_gate_t = torch.stack(tr["edge_gate"], dim=1)
        edge_boundary_t = torch.stack(tr["edge_boundary"], dim=1)
        edge_write_t = torch.stack(tr["edge_write_gate"], dim=1)
        edge_prim_t = torch.stack(tr["edge_primitive_weights"], dim=1)
        edge_group_t = torch.stack(tr["edge_group_weights"], dim=1)
        edge_sign_t = torch.stack(tr["edge_signs"], dim=1)
        trace = {
            "boundary": torch.stack(tr["step_boundary"], dim=1),
            "route": edge_gate_t / edge_gate_t.sum(dim=-1, keepdim=True).clamp_min(1e-8),
            "primitive_weights": edge_prim_t.mean(dim=3),
            "primitive_signs": edge_sign_t.mean(dim=3),
            "group_weights": edge_group_t.mean(dim=3),
            "write_gate": edge_write_t.mean(dim=3),
            "memory_write_norm": torch.stack(tr["memory_write_norm"], dim=1),
            "memory_read_norm": torch.stack(tr["memory_read_norm"], dim=1),
            "memory_read_influence": torch.stack(tr["memory_read_influence"], dim=1),
            "update_norm": torch.stack(tr["update_norm"], dim=1),
            "context_stats_norm": torch.stack(tr["context_stats_norm"], dim=1),
            "memory_forget": memory_forget,
            "logits": logits,
            "edge_gate": edge_gate_t,
            "edge_boundary": edge_boundary_t,
            "edge_write_gate": edge_write_t,
            "edge_primitive_weights": edge_prim_t,
            "edge_group_weights": edge_group_t,
            "edge_signs": edge_sign_t,
            "output_gate": torch.stack(tr["output_gate"], dim=1),
        }
        return logits, trace


@torch.no_grad()
def report_credit_ablation(model: EdgeProgramClassifier, wav: torch.Tensor, y: torch.Tensor, tau: float, args) -> Dict[str, Dict[str, float]]:
    logits, _ = model(wav, tau=tau)
    base = F.cross_entropy(logits.float(), y).float()
    out: Dict[str, Dict[str, float]] = {"groups": {}, "primitives": {}, "steps": {}, "edges": {}, "outputs": {}, "memory": {}}
    for gid, name in enumerate(model.primitive_bank.group_names):
        l2, _ = model(wav, tau=tau, ablate_group=gid)
        out["groups"][name] = float((F.cross_entropy(l2.float(), y).float() - base).detach().cpu())
    for pid, name in enumerate(model.primitive_bank.primitive_names):
        l2, _ = model(wav, tau=tau, ablate_primitive=pid)
        out["primitives"][name] = float((F.cross_entropy(l2.float(), y).float() - base).detach().cpu())
    for t in range(model.steps):
        l2, _ = model(wav, tau=tau, ablate_step=t)
        out["steps"][f"step_{t:02d}"] = float((F.cross_entropy(l2.float(), y).float() - base).detach().cpu())
    if bool(args.credit_ablate_edges):
        for i in range(model.lanes):
            for j in range(model.lanes):
                l2, _ = model(wav, tau=tau, ablate_edge=(i, j))
                out["edges"][f"edge_{i}_{j}"] = float((F.cross_entropy(l2.float(), y).float() - base).detach().cpu())
    if bool(args.credit_ablate_outputs):
        for t in range(model.steps):
            for j in range(model.lanes):
                l2, _ = model(wav, tau=tau, ablate_output=(t, j))
                out["outputs"][f"out_{t:02d}_{j}"] = float((F.cross_entropy(l2.float(), y).float() - base).detach().cpu())
    l2, _ = model(wav, tau=tau, ablate_memory=True)
    out["memory"]["no_memory"] = float((F.cross_entropy(l2.float(), y).float() - base).detach().cpu())
    return out


def bad_vector(names: List[str], credit: Dict[str, float], device: torch.device, scale: float) -> torch.Tensor:
    vals = [max(0.0, -float(credit.get(n, 0.0))) for n in names]
    t = torch.tensor(vals, device=device, dtype=torch.float32)
    if t.numel() and float(t.max().detach().cpu()) > 0:
        t = t / t.max().clamp_min(1e-8)
    return t * float(scale)


def build_penalty(model: EdgeProgramClassifier, credit: Optional[Dict], args, device: str) -> Optional[Dict[str, torch.Tensor]]:
    if not bool(args.enable_edge_credit_penalty) or not credit:
        return None
    dev = torch.device(device)
    groups = bad_vector(model.primitive_bank.group_names, credit.get("groups") or {}, dev, float(args.credit_bad_group_scale))
    prims = bad_vector(model.primitive_bank.primitive_names, credit.get("primitives") or {}, dev, float(args.credit_bad_primitive_scale))
    steps = bad_vector([f"step_{i:02d}" for i in range(model.steps)], credit.get("steps") or {}, dev, float(args.credit_bad_step_scale))
    edge_names = [f"edge_{i}_{j}" for i in range(model.lanes) for j in range(model.lanes)]
    edges = bad_vector(edge_names, credit.get("edges") or {}, dev, float(args.credit_bad_edge_scale)).view(model.lanes, model.lanes)
    out_names = [f"out_{t:02d}_{j}" for t in range(model.steps) for j in range(model.lanes)]
    outs = bad_vector(out_names, credit.get("outputs") or {}, dev, float(args.credit_bad_output_scale)).view(model.steps, model.lanes)
    return {"groups": groups, "prims": prims, "steps": steps, "edges": edges, "outputs": outs}


def credit_penalty_loss(trace: Dict[str, torch.Tensor], penalty: Optional[Dict[str, torch.Tensor]]) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    dev = trace["logits"].device
    if penalty is None:
        z = torch.zeros((), device=dev)
        return z, {"edge_credit_penalty": z, "bad_edge_loss": z, "bad_output_loss": z}
    gw = trace["edge_group_weights"].float()
    pw = trace["edge_primitive_weights"].float()
    edge_mass = (trace["edge_gate"].float() * trace["edge_write_gate"].float() * trace["edge_boundary"].float())
    group_loss = (gw * penalty["groups"].view(1, 1, 1, 1, -1)).sum(dim=-1).mean()
    prim_loss = (pw * penalty["prims"].view(1, 1, 1, 1, -1)).sum(dim=-1).mean()
    step_loss = ((edge_mass.mean(dim=(2, 3)) + trace["output_gate"].float().mean(dim=2)) * penalty["steps"].view(1, -1)).mean()
    edge_loss = (edge_mass * penalty["edges"].view(1, 1, penalty["edges"].shape[0], penalty["edges"].shape[1])).mean()
    out_loss = (trace["output_gate"].float() * penalty["outputs"].view(1, penalty["outputs"].shape[0], penalty["outputs"].shape[1])).mean()
    total = group_loss + prim_loss + step_loss + edge_loss + out_loss
    return total, {"edge_credit_penalty": total.detach(), "bad_edge_loss": edge_loss.detach(), "bad_output_loss": out_loss.detach()}


def edge_aux_losses(trace: Dict[str, torch.Tensor], args) -> Dict[str, torch.Tensor]:
    eg = trace["edge_gate"].float()
    eb = trace["edge_boundary"].float()
    wg = trace["edge_write_gate"].float()
    og = trace["output_gate"].float()
    edge_mass = eg * eb * wg
    return {
        "edge_gate_cost": edge_mass.mean(),
        "output_gate_cost": og.mean(),
        "dead_step_loss": F.relu(float(args.dead_step_min_mass) - (edge_mass.mean(dim=(2, 3)) + og.mean(dim=2))).pow(2).mean(),
    }


def train_epoch(model, loader, opt, scaler, device: str, dtype: torch.dtype, args, epoch: int, penalty: Optional[Dict[str, torch.Tensor]]) -> Dict[str, float]:
    model.train()
    use_amp = device.startswith("cuda") and dtype != torch.float32
    tau = tau_for_epoch(args, epoch)
    totals = {"loss": 0.0, "ce": 0.0, "correct": 0, "n": 0}
    aux_sum: Dict[str, float] = {}
    sync_if_cuda(device); t0 = time.perf_counter(); batches = 0
    for step, (wav, y) in enumerate(loader, 1):
        if int(args.max_train_batches) > 0 and step > int(args.max_train_batches):
            break
        batches += 1
        wav = wav.to(device, non_blocking=True); y = y.to(device, non_blocking=True)
        opt.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.split(":")[0], dtype=dtype, enabled=use_amp):
            logits, trace = model(wav, tau=tau)
            ce = F.cross_entropy(logits.float(), y)
            aux = closed_loop_aux_losses(trace, args)
            aux.update(edge_aux_losses(trace, args))
            loss, aux = weighted_loss(ce, aux, args)
            loss = loss + float(args.lambda_edge_gate_cost) * aux["edge_gate_cost"].to(dtype=loss.dtype)
            loss = loss + float(args.lambda_output_gate_cost) * aux["output_gate_cost"].to(dtype=loss.dtype)
            loss = loss + float(args.lambda_dead_step) * aux["dead_step_loss"].to(dtype=loss.dtype)
            cp, cp_terms = credit_penalty_loss(trace, penalty)
            if bool(args.enable_edge_credit_penalty):
                loss = loss + cp.to(dtype=loss.dtype)
                aux.update(cp_terms)
        scaler.scale(loss).backward()
        if float(args.grad_clip) > 0:
            scaler.unscale_(opt); torch.nn.utils.clip_grad_norm_(model.parameters(), float(args.grad_clip))
        scaler.step(opt); scaler.update()
        bs = int(y.numel())
        totals["loss"] += float(loss.detach().cpu()) * bs; totals["ce"] += float(ce.detach().cpu()) * bs
        totals["correct"] += int((logits.argmax(-1) == y).sum().detach().cpu()); totals["n"] += bs
        for k, v in aux.items():
            aux_sum[k] = aux_sum.get(k, 0.0) + float(v.detach().float().cpu()) * bs
        if int(args.log_every) > 0 and step % int(args.log_every) == 0:
            print(f"epoch {epoch:03d} step {step:05d} loss={totals['loss']/max(1, totals['n']):.4f} ce={totals['ce']/max(1, totals['n']):.4f} acc={100*totals['correct']/max(1, totals['n']):.2f}% tau={tau:.3f}", flush=True)
    sync_if_cuda(device); elapsed = max(1e-9, time.perf_counter() - t0)
    out = {"loss": totals["loss"] / max(1, totals["n"]), "ce": totals["ce"] / max(1, totals["n"]), "acc": totals["correct"] / max(1, totals["n"]), "gumbel_tau": tau, "train_seconds": elapsed, "train_samples_per_sec": totals["n"] / elapsed, "train_batches_per_sec": batches / elapsed}
    for k, v in aux_sum.items():
        out[k] = v / max(1, totals["n"])
    return out


@torch.no_grad()
def evaluate(model, loader, device: str, dtype: torch.dtype, args, tau: float):
    model.eval(); use_amp = device.startswith("cuda") and dtype != torch.float32
    total_loss = 0.0; correct = 0; n = 0; last_trace = None; aux_last = {}; credit = None
    sync_if_cuda(device); t0 = time.perf_counter(); batches = 0
    for step, (wav, y) in enumerate(loader, 1):
        if int(args.max_val_batches) > 0 and step > int(args.max_val_batches):
            break
        batches += 1
        wav = wav.to(device, non_blocking=True); y = y.to(device, non_blocking=True)
        with torch.autocast(device_type=device.split(":")[0], dtype=dtype, enabled=use_amp):
            logits, trace = model(wav, tau=tau)
            loss = F.cross_entropy(logits.float(), y)
            aux = closed_loop_aux_losses(trace, args); aux.update(edge_aux_losses(trace, args))
        pred = logits.argmax(-1); bs = int(y.numel())
        total_loss += float(loss.detach().cpu()) * bs; correct += int((pred == y).sum().detach().cpu()); n += bs
        last_trace = trace; aux_last = {k: float(v.detach().float().cpu()) for k, v in aux.items()}
        if bool(args.enable_credit_ablation) and credit is None:
            small_wav = wav[: min(int(args.credit_ablation_batch), wav.shape[0])]
            small_y = y[: small_wav.shape[0]]
            credit = report_credit_ablation(model, small_wav, small_y, tau, args)
    sync_if_cuda(device); elapsed = max(1e-9, time.perf_counter() - t0)
    return {"loss": total_loss / max(1, n), "acc": correct / max(1, n), "n": n, "trace": last_trace, "aux": aux_last, "credit_ablation": credit, "eval_seconds": elapsed, "eval_samples_per_sec": n / elapsed, "eval_batches_per_sec": batches / elapsed}


def run(args) -> None:
    set_seed(args.seed)
    if bool(getattr(args, "synthetic_data", False)) or bool(getattr(args, "allow_synthetic_fallback", False)):
        raise SystemExit("v4.6.3 evidence run forbids synthetic data/fallback")
    device = args.device if args.device != "cuda" or torch.cuda.is_available() else "cpu"
    if device.startswith("cuda"):
        torch.backends.cudnn.benchmark = True
        try: torch.set_float32_matmul_precision("high")
        except Exception: pass
    dtype = amp_dtype(args.amp); out_dir = ensure_dir(Path(args.out_dir))
    train_loader, val_loader, classes, train_counts, val_counts = make_loaders(args); args.num_classes = len(classes)
    model = EdgeProgramClassifier(len(classes), args).to(device)
    params = sum(p.numel() for p in model.parameters())
    print(f"loaded datasets: train={len(train_loader.dataset)} val={len(val_loader.dataset)} classes={classes}", flush=True)
    print(f"V463EdgeProgram params={params} steps={args.steps} lanes={args.lanes} edges={args.lanes*args.lanes} D={args.dim} K={args.num_primitives} penalty={args.enable_edge_credit_penalty} device={device} amp={args.amp}", flush=True)
    decay, no_decay = [], []
    for name, p in model.named_parameters():
        if not p.requires_grad: continue
        (no_decay if name.endswith("bias") or "norm" in name.lower() or "forget_logit" in name else decay).append(p)
    opt = torch.optim.AdamW([{"params": decay, "weight_decay": float(args.weight_decay)}, {"params": no_decay, "weight_decay": 0.0}], lr=float(args.lr), betas=(0.9, 0.95))
    scaler = torch.amp.GradScaler("cuda", enabled=device.startswith("cuda") and dtype == torch.float16)
    analyzer = StepAnalyzer(lane_names=LANE_NAMES[: int(args.lanes)], primitive_names=model.primitive_bank.primitive_names, boundary_peak_threshold=float(args.boundary_peak_threshold))
    best = 0.0; best_epoch = 0; final_summary = None; previous_credit = None
    fields = metrics_fields() + ["edge_gate_cost", "output_gate_cost", "dead_step_loss", "edge_credit_penalty", "bad_edge_loss", "bad_output_loss"]
    for epoch in range(1, int(args.epochs) + 1):
        penalty = build_penalty(model, previous_credit, args, device)
        tr = train_epoch(model, train_loader, opt, scaler, device, dtype, args, epoch, penalty)
        tau = float(tr["gumbel_tau"]); va = evaluate(model, val_loader, device, dtype, args, tau)
        previous_credit = va.get("credit_ablation")
        if va["acc"] > best: best = float(va["acc"]); best_epoch = int(epoch)
        row = {"epoch": epoch, "gumbel_tau": tau, "train_loss": tr["loss"], "train_ce": tr["ce"], "train_acc": tr["acc"], "val_loss": va["loss"], "val_acc": va["acc"], "best_acc": best, "train_seconds": tr.get("train_seconds", 0.0), "train_samples_per_sec": tr.get("train_samples_per_sec", 0.0), "eval_seconds": va.get("eval_seconds", 0.0), "eval_samples_per_sec": va.get("eval_samples_per_sec", 0.0)}
        for k in fields:
            if k in tr: row[k] = tr[k]
            elif k in va.get("aux", {}): row[k] = va["aux"][k]
        append_metrics_csv(out_dir / "metrics.csv", row, fields)
        metrics = {"train_loss": tr["loss"], "train_ce": tr["ce"], "train_acc": tr["acc"], "val_loss": va["loss"], "val_acc": va["acc"], "best_acc": best, "best_epoch": best_epoch}
        trace = va.get("trace")
        if trace is not None:
            summary = analyzer.summarize(trace, epoch=epoch, gumbel_tau=tau, metrics=metrics, aux_losses=va.get("aux") or {})
            summary["edge"] = {"edge_gate_mean": float(trace["edge_gate"].float().mean().cpu()), "edge_boundary_mean": float(trace["edge_boundary"].float().mean().cpu()), "edge_write_mean": float(trace["edge_write_gate"].float().mean().cpu()), "edge_mass_by_step": (trace["edge_gate"].float() * trace["edge_boundary"].float() * trace["edge_write_gate"].float()).mean(dim=(0, 2, 3)).cpu().tolist()}
            summary["output"] = {"output_gate_by_step_slot": trace["output_gate"].float().mean(dim=0).cpu().tolist(), "output_gate_mean": float(trace["output_gate"].float().mean().cpu())}
            summary["credit_ablation"] = previous_credit
            summary["edge_credit_penalty"] = {"enabled": bool(args.enable_edge_credit_penalty), "delayed_epoch_credit": True}
            final_summary = summary
            analyzer.write_artifacts(out_dir, summary)
            write_json(out_dir / f"credit_ablation_epoch_{epoch:03d}.json", previous_credit or {})
        print(f"epoch {epoch:03d}/{args.epochs} train={tr['loss']:.4f}/{100*tr['acc']:.2f}% val={va['loss']:.4f}/{100*va['acc']:.2f}% best={100*best:.2f}%@{best_epoch} tau={tau:.3f}", flush=True)
    final_report = {"version": "v4.6.3_edge_program", "best_acc": best, "best_epoch": best_epoch, "args": vars(args), "classes": classes, "train_counts": train_counts, "val_counts": val_counts, "group_names": model.primitive_bank.group_names, "primitive_names": model.primitive_bank.primitive_names, "last_summary": final_summary, "closed_loop_invariant": "edge decisions affect execution; execution affects CE loss; delayed credit can penalize bad edge/step/output mass next epoch"}
    write_json(out_dir / "final_report.json", final_report)
    if final_summary is not None: analyzer.write_artifacts(out_dir, final_summary, final_report=final_report)


def parser() -> argparse.ArgumentParser:
    from simple_butterfly_matrix_v4_tape_lane.v4_6_loop_core.tape_lane_transport_v4_6_loop_core import parser as base_parser  # type: ignore
    p = base_parser()
    p.set_defaults(data_root="../architecture_builder/data/speechcommands", num_primitives=18, route_prior_strength=0.0, out_dir="./simple_butterfly_matrix_v4_tape_lane/agent_reports/v4_6_3_edge_program")
    p.add_argument("--memory-slot", type=int, default=3)
    p.add_argument("--enable-credit-ablation", action="store_true", default=True)
    p.add_argument("--credit-ablation-batch", type=int, default=16)
    p.add_argument("--credit-ablate-edges", action="store_true", default=True)
    p.add_argument("--credit-ablate-outputs", action="store_true", default=True)
    p.add_argument("--enable-edge-credit-penalty", action="store_true", default=False)
    p.add_argument("--credit-bad-group-scale", type=float, default=0.012)
    p.add_argument("--credit-bad-primitive-scale", type=float, default=0.008)
    p.add_argument("--credit-bad-step-scale", type=float, default=0.008)
    p.add_argument("--credit-bad-edge-scale", type=float, default=0.008)
    p.add_argument("--credit-bad-output-scale", type=float, default=0.006)
    p.add_argument("--lambda-edge-gate-cost", type=float, default=0.001)
    p.add_argument("--lambda-output-gate-cost", type=float, default=0.0005)
    p.add_argument("--lambda-dead-step", type=float, default=0.002)
    p.add_argument("--dead-step-min-mass", type=float, default=0.05)
    return p


if __name__ == "__main__":
    run(parser().parse_args())
