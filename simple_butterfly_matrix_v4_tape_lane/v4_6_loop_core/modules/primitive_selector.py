#!/usr/bin/env python3
"""Parallel signed primitive selector for v4.6 loop core."""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


DEFAULT_PRIMITIVES: List[str] = [
    "identity",
    "channel",
    "low_rank",
    "diff_prev",
    "contrast",
    "ctx_matrix",
    "product_gate",
    "memory_read",
]


@dataclass
class PrimitiveOutput:
    update: torch.Tensor
    weights: torch.Tensor
    signs: torch.Tensor
    expert_outputs: torch.Tensor
    top1: torch.Tensor
    entropy: torch.Tensor
    stats: Dict[str, torch.Tensor]


class ParallelPrimitiveSelector(nn.Module):
    """Compute all primitive projections in one pass and mix by Gumbel weights.

    Tensor contract:
        x:            [B,L,D]
        z:            [B,L,D]
        scores/signs: [B,L,K]
        expert_outputs: [B,L,K,D]
        update:       [B,L,D]
    """

    def __init__(
        self,
        dim: int,
        num_primitives: int = 8,
        rank: int = 32,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.dim = int(dim)
        self.num_primitives = int(num_primitives)
        self.rank = max(1, min(int(rank), self.dim))
        self.primitive_names = list(DEFAULT_PRIMITIVES[: self.num_primitives])
        while len(self.primitive_names) < self.num_primitives:
            self.primitive_names.append(f"low_rank_extra_{len(self.primitive_names)}")

        scale = 1.0 / math.sqrt(max(1, self.dim))
        self.low_u = nn.Parameter(torch.randn(self.num_primitives, self.dim, self.rank) * scale)
        self.low_v = nn.Parameter(torch.randn(self.num_primitives, self.rank, self.dim) * scale)
        self.channel = nn.Sequential(
            nn.LayerNorm(self.dim),
            nn.Linear(self.dim, self.dim * 2),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(self.dim * 2, self.dim),
        )
        self.ctx_gate = nn.Sequential(nn.LayerNorm(self.dim), nn.Linear(self.dim, 1))
        self.product_a = nn.Linear(self.dim, self.rank, bias=False)
        self.product_b = nn.Linear(self.rank, self.dim, bias=False)
        self.product_gate = nn.Sequential(nn.LayerNorm(self.dim), nn.Linear(self.dim, 1))
        self.memory_proj = nn.Linear(self.dim, self.dim, bias=False)
        self.out_norm = nn.LayerNorm(self.dim)

    def _low_rank_all(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B,L,D], U: [K,D,R], V: [K,R,D]
        h = torch.einsum("bld,kdr->blkr", x, self.low_u.to(device=x.device, dtype=x.dtype))
        y = torch.einsum("blkr,krd->blkd", h, self.low_v.to(device=x.device, dtype=x.dtype))
        return y

    def _expert_bank(
        self,
        x: torch.Tensor,
        z: torch.Tensor,
        memory_read: Optional[torch.Tensor],
        previous_state: Optional[torch.Tensor],
    ) -> torch.Tensor:
        bsz, lanes, dim = x.shape
        bank = self._low_rank_all(x)
        outs = []
        prev_lane = torch.cat([x[:, :1, :], x[:, :-1, :]], dim=1)
        smooth = (x + torch.roll(x, shifts=1, dims=1) + torch.roll(x, shifts=-1, dims=1)) / 3.0
        if previous_state is not None and previous_state.shape == x.shape:
            diff = x - previous_state
        else:
            diff = x - prev_lane
        contrast = x - x.mean(dim=1, keepdim=True)
        ctx_gate = torch.sigmoid(self.ctx_gate(z))
        product = torch.sigmoid(self.product_gate(z)) * self.product_b(F.silu(self.product_a(x)))
        if memory_read is None:
            mem = torch.zeros(bsz, lanes, dim, device=x.device, dtype=x.dtype)
        else:
            mem = self.memory_proj(memory_read).view(bsz, 1, dim).expand(-1, lanes, -1)

        named = {
            "identity": x,
            "channel": self.channel(x),
            "low_rank": bank[:, :, min(2, self.num_primitives - 1), :],
            "diff_prev": diff,
            "smooth": smooth,
            "contrast": contrast,
            "ctx_matrix": ctx_gate * bank[:, :, min(5, self.num_primitives - 1), :],
            "product_gate": product,
            "memory_read": mem,
        }
        for i, name in enumerate(self.primitive_names):
            outs.append(named.get(name, bank[:, :, i, :]))
        return torch.stack(outs, dim=2)

    def forward(
        self,
        x: torch.Tensor,
        z: torch.Tensor,
        scores: torch.Tensor,
        sign_logits: torch.Tensor,
        tau: float,
        memory_read: Optional[torch.Tensor] = None,
        previous_state: Optional[torch.Tensor] = None,
    ) -> PrimitiveOutput:
        if x.dim() != 3:
            raise ValueError(f"x must be [B,L,D], got {tuple(x.shape)}")
        if z.shape != x.shape:
            raise ValueError(f"z must match x, got z={tuple(z.shape)} x={tuple(x.shape)}")
        if scores.shape != sign_logits.shape or scores.shape[:2] != x.shape[:2] or scores.shape[-1] != self.num_primitives:
            raise ValueError("scores/sign_logits must be [B,L,K] with configured K")

        experts = self._expert_bank(x, z, memory_read, previous_state)
        weights = F.gumbel_softmax(scores.float(), tau=float(tau), hard=False, dim=-1).to(dtype=x.dtype)
        signs = torch.tanh(sign_logits)
        update = torch.einsum("blk,blk,blkd->bld", weights, signs, experts)
        update = self.out_norm(update)
        entropy = -(weights.float().clamp_min(1e-8) * weights.float().clamp_min(1e-8).log()).sum(dim=-1)
        top1 = weights.argmax(dim=-1)
        stats = {
            "primitive_entropy": entropy.mean(),
            "primitive_top1_share": torch.bincount(top1.reshape(-1), minlength=self.num_primitives).float().max()
            / float(max(1, top1.numel())),
            "primitive_sign_negative_share": (signs.float() < 0).float().mean(),
            "primitive_sign_abs_mean": signs.float().abs().mean(),
            "expert_output_norm": experts.float().norm(dim=-1).mean(),
        }
        return PrimitiveOutput(
            update=update,
            weights=weights,
            signs=signs,
            expert_outputs=experts,
            top1=top1,
            entropy=entropy,
            stats=stats,
        )


if __name__ == "__main__":
    torch.manual_seed(0)
    sel = ParallelPrimitiveSelector(dim=16, num_primitives=8, rank=4)
    x = torch.randn(2, 4, 16)
    z = torch.randn(2, 4, 16)
    scores = torch.randn(2, 4, 8)
    signs = torch.randn(2, 4, 8)
    out = sel(x, z, scores, signs, tau=1.0, memory_read=torch.zeros(2, 16))
    assert out.update.shape == x.shape
    assert out.expert_outputs.shape == (2, 4, 8, 16)
    print("primitive_selector smoke ok")
