#!/usr/bin/env python3
"""Grouped signed primitive selector for v4.6.1 loop core."""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


PRIMITIVE_GROUPS: Dict[str, List[str]] = {
    "keep": ["identity", "gated_keep"],
    "channel": ["channel", "low_rank", "ctx_matrix"],
    "correction": ["diff_prev", "contrast", "subtract_memory"],
    "aggregation": ["lane_mean", "smooth_lanes", "pool_context"],
    "memory": ["memory_read", "memory_write_candidate", "forget_like"],
    "composition": ["product_gate", "gated_add", "mul_filter"],
}

DEFAULT_PRIMITIVES: List[str] = [p for group in PRIMITIVE_GROUPS.values() for p in group]
DEFAULT_GROUPS: List[str] = list(PRIMITIVE_GROUPS.keys())


@dataclass
class PrimitiveOutput:
    update: torch.Tensor
    weights: torch.Tensor
    signs: torch.Tensor
    expert_outputs: torch.Tensor
    top1: torch.Tensor
    entropy: torch.Tensor
    group_weights: torch.Tensor
    group_top1: torch.Tensor
    group_entropy: torch.Tensor
    choice_context: torch.Tensor
    stats: Dict[str, torch.Tensor]


class ParallelPrimitiveSelector(nn.Module):
    """Two-level differentiable selector: group -> primitive inside group.

    The controller may still emit a flat primitive score vector. The selector
    turns it into a logical two-level choice by pooling scores per action group,
    then multiplying group probability by primitive-in-group probability.
    """

    def __init__(
        self,
        dim: int,
        num_primitives: int = 18,
        rank: int = 32,
        dropout: float = 0.0,
    ) -> None:
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
        self.register_buffer("primitive_exists", torch.ones(self.num_primitives), persistent=False)

        scale = 1.0 / math.sqrt(max(1, self.dim))
        self.low_u = nn.Parameter(torch.randn(self.num_primitives, self.dim, self.rank) * scale)
        self.low_v = nn.Parameter(torch.randn(self.num_primitives, self.rank, self.dim) * scale)
        self.channel = nn.Sequential(
            nn.LayerNorm(self.dim), nn.Linear(self.dim, self.dim * 2), nn.SiLU(), nn.Dropout(dropout), nn.Linear(self.dim * 2, self.dim)
        )
        self.ctx_gate = nn.Sequential(nn.LayerNorm(self.dim), nn.Linear(self.dim, 1))
        self.product_a = nn.Linear(self.dim, self.rank, bias=False)
        self.product_b = nn.Linear(self.rank, self.dim, bias=False)
        self.product_gate = nn.Sequential(nn.LayerNorm(self.dim), nn.Linear(self.dim, 1))
        self.memory_proj = nn.Linear(self.dim, self.dim, bias=False)
        self.group_embedding = nn.Embedding(self.num_groups, self.dim)
        self.primitive_embedding = nn.Embedding(self.num_primitives, self.dim)
        self.choice_norm = nn.LayerNorm(self.dim)
        self.out_norm = nn.LayerNorm(self.dim)

    def _low_rank_all(self, x: torch.Tensor) -> torch.Tensor:
        h = torch.einsum("bld,kdr->blkr", x, self.low_u.to(device=x.device, dtype=x.dtype))
        return torch.einsum("blkr,krd->blkd", h, self.low_v.to(device=x.device, dtype=x.dtype))

    def _expert_bank(self, x: torch.Tensor, z: torch.Tensor, memory_read: Optional[torch.Tensor], previous_state: Optional[torch.Tensor]) -> torch.Tensor:
        bsz, lanes, dim = x.shape
        bank = self._low_rank_all(x)
        prev_lane = torch.cat([x[:, :1, :], x[:, :-1, :]], dim=1)
        diff = x - previous_state if previous_state is not None and previous_state.shape == x.shape else x - prev_lane
        smooth = (x + torch.roll(x, 1, dims=1) + torch.roll(x, -1, dims=1)) / 3.0
        lane_mean = x.mean(dim=1, keepdim=True).expand(-1, lanes, -1)
        contrast = x - lane_mean
        ctx = torch.sigmoid(self.ctx_gate(z)) * bank[:, :, min(2, self.num_primitives - 1), :]
        product = torch.sigmoid(self.product_gate(z)) * self.product_b(F.silu(self.product_a(x)))
        mem = torch.zeros(bsz, lanes, dim, device=x.device, dtype=x.dtype) if memory_read is None else self.memory_proj(memory_read).view(bsz, 1, dim).expand(-1, lanes, -1)
        named = {
            "identity": x,
            "gated_keep": torch.sigmoid(self.ctx_gate(z)) * x,
            "channel": self.channel(x),
            "low_rank": bank[:, :, min(1, self.num_primitives - 1), :],
            "ctx_matrix": ctx,
            "diff_prev": diff,
            "contrast": contrast,
            "subtract_memory": x - mem,
            "lane_mean": lane_mean,
            "smooth_lanes": smooth,
            "pool_context": 0.5 * (smooth + lane_mean),
            "memory_read": mem,
            "memory_write_candidate": ctx + mem,
            "forget_like": x - torch.sigmoid(self.ctx_gate(z)) * mem,
            "product_gate": product,
            "gated_add": x + torch.sigmoid(self.product_gate(z)) * product,
            "mul_filter": x * torch.tanh(product),
        }
        outs = []
        for i, name in enumerate(self.primitive_names):
            outs.append(named.get(name, bank[:, :, i, :]))
        return torch.stack(outs, dim=2)

    def _grouped_weights(self, scores: torch.Tensor, tau: float) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        score_f = scores.float()
        bsz, lanes, k = score_f.shape
        group_logits = []
        for gid in range(self.num_groups):
            mask = (self.primitive_group_id.to(score_f.device) == gid).view(1, 1, k)
            masked = score_f.masked_fill(~mask, -1e4)
            group_logits.append(torch.logsumexp(masked, dim=-1))
        group_logits_t = torch.stack(group_logits, dim=-1)
        group_w = F.gumbel_softmax(group_logits_t, tau=float(tau), hard=False, dim=-1)
        prim_parts = []
        for gid in range(self.num_groups):
            mask = (self.primitive_group_id.to(score_f.device) == gid).view(1, 1, k)
            inside = F.softmax(score_f.masked_fill(~mask, -1e4), dim=-1)
            prim_parts.append(group_w[..., gid:gid + 1] * inside)
        final_w = torch.stack(prim_parts, dim=0).sum(dim=0).to(dtype=scores.dtype)
        return final_w, group_w.to(dtype=scores.dtype), group_logits_t

    def choice_context_from_weights(self, weights: torch.Tensor, group_weights: torch.Tensor) -> torch.Tensor:
        pids = torch.arange(self.num_primitives, device=weights.device)
        gids = torch.arange(self.num_groups, device=weights.device)
        prim_emb = self.primitive_embedding(pids).to(dtype=weights.dtype)
        group_emb = self.group_embedding(gids).to(dtype=weights.dtype)
        ctx = torch.einsum("blk,kd->bld", weights, prim_emb) + torch.einsum("blg,gd->bld", group_weights, group_emb)
        return self.choice_norm(ctx)

    def forward(
        self,
        x: torch.Tensor,
        z: torch.Tensor,
        scores: torch.Tensor,
        sign_logits: torch.Tensor,
        tau: float,
        memory_read: Optional[torch.Tensor] = None,
        previous_state: Optional[torch.Tensor] = None,
        ablate_group: Optional[int] = None,
        ablate_primitive: Optional[int] = None,
    ) -> PrimitiveOutput:
        if x.dim() != 3 or z.shape != x.shape:
            raise ValueError("x and z must be [B,L,D] and match")
        if scores.shape[:2] != x.shape[:2] or scores.shape[-1] != self.num_primitives:
            raise ValueError(f"scores must be [B,L,{self.num_primitives}], got {tuple(scores.shape)}")
        experts = self._expert_bank(x, z, memory_read, previous_state)
        weights, group_weights, _ = self._grouped_weights(scores, tau)
        if ablate_group is not None:
            gid = int(ablate_group)
            mask = (self.primitive_group_id.to(weights.device) != gid).to(dtype=weights.dtype).view(1, 1, -1)
            weights = weights * mask
            weights = weights / weights.sum(dim=-1, keepdim=True).clamp_min(1e-8)
            gmask = torch.ones_like(group_weights)
            gmask[..., gid] = 0.0
            group_weights = group_weights * gmask
            group_weights = group_weights / group_weights.sum(dim=-1, keepdim=True).clamp_min(1e-8)
        if ablate_primitive is not None:
            pid = int(ablate_primitive)
            weights = weights.clone()
            weights[..., pid] = 0.0
            weights = weights / weights.sum(dim=-1, keepdim=True).clamp_min(1e-8)
        signs = torch.tanh(sign_logits)
        update = torch.einsum("blk,blk,blkd->bld", weights, signs, experts)
        update = self.out_norm(update)
        entropy = -(weights.float().clamp_min(1e-8) * weights.float().clamp_min(1e-8).log()).sum(dim=-1)
        group_entropy = -(group_weights.float().clamp_min(1e-8) * group_weights.float().clamp_min(1e-8).log()).sum(dim=-1)
        top1 = weights.argmax(dim=-1)
        group_top1 = group_weights.argmax(dim=-1)
        choice_context = self.choice_context_from_weights(weights, group_weights)
        stats = {
            "primitive_entropy": entropy.mean(),
            "group_entropy": group_entropy.mean(),
            "primitive_top1_share": torch.bincount(top1.reshape(-1), minlength=self.num_primitives).float().max() / float(max(1, top1.numel())),
            "group_top1_share": torch.bincount(group_top1.reshape(-1), minlength=self.num_groups).float().max() / float(max(1, group_top1.numel())),
            "primitive_sign_negative_share": (signs.float() < 0).float().mean(),
            "primitive_sign_abs_mean": signs.float().abs().mean(),
            "expert_output_norm": experts.float().norm(dim=-1).mean(),
        }
        return PrimitiveOutput(update, weights, signs, experts, top1, entropy, group_weights, group_top1, group_entropy, choice_context, stats)


if __name__ == "__main__":
    torch.manual_seed(0)
    sel = ParallelPrimitiveSelector(dim=16, num_primitives=18, rank=4)
    x = torch.randn(2, 4, 16)
    z = torch.randn(2, 4, 16)
    scores = torch.randn(2, 4, 18)
    signs = torch.randn(2, 4, 18)
    out = sel(x, z, scores, signs, tau=1.0, memory_read=torch.zeros(2, 16))
    assert out.update.shape == x.shape
    assert out.group_weights.shape == (2, 4, sel.num_groups)
    print("grouped primitive_selector smoke ok")
