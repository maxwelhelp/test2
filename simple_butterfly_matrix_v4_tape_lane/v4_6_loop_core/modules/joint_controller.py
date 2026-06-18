#!/usr/bin/env python3
"""Joint controller for v4.6 loop core.

Tensor contract:
    lane_state:  [B, L, D]
    memory_read: [B, D]
    evidence:    [B, D] or None
    z:           [B, L, D]
    route_logits:[B, L, L]  source lane -> target lane
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Optional, Union

import torch
import torch.nn as nn


@dataclass
class ControllerOutput:
    z: torch.Tensor
    boundary_logit: torch.Tensor
    route_logits: torch.Tensor
    write_logits: torch.Tensor
    fanout_logits: torch.Tensor
    primitive_scores: torch.Tensor
    primitive_sign_logits: torch.Tensor
    compose_logits: torch.Tensor
    memory_write_logits: torch.Tensor
    memory_read_gate_logits: torch.Tensor
    stats: Dict[str, torch.Tensor]


def build_route_prior(lanes: int, *, device=None, dtype=None) -> torch.Tensor:
    """Small structural prior, not a hard mask.

    Positive values mark useful transitions; negative values mark usually wasteful
    all-to-all paths. Identity is kept mildly positive so the model can preserve
    state when no boundary transition is useful.
    """
    prior = torch.full((lanes, lanes), -0.10, device=device, dtype=dtype or torch.float32)
    eye = torch.eye(lanes, device=device, dtype=dtype or torch.float32)
    prior = prior + 0.15 * eye
    if lanes >= 2:
        prior[0, 1] = 0.25  # detail -> state
    if lanes >= 3:
        prior[1, 2] = 0.25  # state -> abstract
    if lanes >= 4:
        prior[1, 3] = 0.15  # state -> memory interface
        prior[3, 1] = 0.25  # memory interface -> state
    return prior


class JointController(nn.Module):
    """One shared latent produces all coupled step decisions.

    All heads branch from the same `z`, so boundary/route/write/primitive/memory
    decisions are not independent unrelated controllers.
    """

    def __init__(
        self,
        dim: int,
        lanes: int = 4,
        num_primitives: int = 8,
        compose_modes: int = 4,
        max_steps: int = 32,
        hidden_mult: int = 2,
        dropout: float = 0.0,
        route_prior_strength: float = 0.35,
    ) -> None:
        super().__init__()
        self.dim = int(dim)
        self.lanes = int(lanes)
        self.num_primitives = int(num_primitives)
        self.compose_modes = int(compose_modes)
        self.max_steps = int(max_steps)
        self.route_prior_strength = float(route_prior_strength)

        hidden = max(self.dim, int(hidden_mult) * self.dim)
        self.lane_embedding = nn.Parameter(torch.randn(self.lanes, self.dim) * 0.02)
        self.step_embedding = nn.Embedding(self.max_steps, self.dim)
        self.memory_read_proj = nn.Linear(self.dim, self.dim, bias=False)
        self.evidence_proj = nn.Linear(self.dim, self.dim, bias=False)

        self.trunk = nn.Sequential(
            nn.LayerNorm(self.dim),
            nn.Linear(self.dim, hidden),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, self.dim),
            nn.LayerNorm(self.dim),
        )

        self.boundary_head = nn.Sequential(nn.LayerNorm(self.dim), nn.Linear(self.dim, 1))
        self.route_q = nn.Linear(self.dim, self.dim, bias=False)
        self.route_k = nn.Linear(self.dim, self.dim, bias=False)
        self.route_pair_bias = nn.Linear(self.dim, self.lanes, bias=True)
        self.write_head = nn.Linear(self.dim, 1)
        self.fanout_head = nn.Linear(self.dim, 1)
        self.primitive_score_head = nn.Linear(self.dim, self.num_primitives)
        self.primitive_sign_head = nn.Linear(self.dim, self.num_primitives)
        self.compose_head = nn.Linear(self.dim, self.compose_modes)
        self.memory_write_head = nn.Linear(self.dim, 1)
        self.memory_read_gate_head = nn.Sequential(nn.LayerNorm(self.dim), nn.Linear(self.dim, 1))

        self.register_buffer("route_prior", build_route_prior(self.lanes), persistent=False)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.normal_(self.step_embedding.weight, std=0.02)
        # Mildly prefer not writing everything at the start.
        nn.init.constant_(self.write_head.bias, -0.5)
        nn.init.constant_(self.fanout_head.bias, -0.2)
        nn.init.constant_(self.memory_write_head.bias, -0.8)
        # Boundary starts neither dead nor all-on.
        last = self.boundary_head[-1]
        if isinstance(last, nn.Linear):
            nn.init.constant_(last.bias, -0.4)

    def _step_ids(self, step_index: Union[int, torch.Tensor], batch: int, device: torch.device) -> torch.Tensor:
        if isinstance(step_index, int):
            idx = torch.full((batch,), int(step_index) % self.max_steps, device=device, dtype=torch.long)
        else:
            idx = step_index.to(device=device, dtype=torch.long).view(-1)
            if idx.numel() == 1:
                idx = idx.expand(batch)
            idx = idx.remainder(self.max_steps)
        return idx

    def forward(
        self,
        lane_state: torch.Tensor,
        memory_read: torch.Tensor,
        step_index: Union[int, torch.Tensor],
        evidence: Optional[torch.Tensor] = None,
    ) -> ControllerOutput:
        if lane_state.dim() != 3:
            raise ValueError(f"lane_state must be [B,L,D], got {tuple(lane_state.shape)}")
        bsz, lanes, dim = lane_state.shape
        if lanes != self.lanes or dim != self.dim:
            raise ValueError(f"expected [B,{self.lanes},{self.dim}], got {tuple(lane_state.shape)}")
        if memory_read.dim() != 2 or memory_read.shape != (bsz, dim):
            raise ValueError(f"memory_read must be [B,D], got {tuple(memory_read.shape)}")

        step_ids = self._step_ids(step_index, bsz, lane_state.device)
        step = self.step_embedding(step_ids).to(dtype=lane_state.dtype).view(bsz, 1, dim)
        lane = self.lane_embedding.to(device=lane_state.device, dtype=lane_state.dtype).view(1, lanes, dim)
        mem = self.memory_read_proj(memory_read).view(bsz, 1, dim)
        ctx = lane_state + lane + step + mem
        if evidence is not None:
            if evidence.dim() != 2 or evidence.shape != (bsz, dim):
                raise ValueError(f"evidence must be [B,D], got {tuple(evidence.shape)}")
            ctx = ctx + self.evidence_proj(evidence).view(bsz, 1, dim)

        z = self.trunk(ctx)
        pooled = z.mean(dim=1)

        boundary_logit = self.boundary_head(pooled).squeeze(-1)
        q = self.route_q(z)
        k = self.route_k(z)
        route_logits = torch.einsum("bld,bmd->blm", q, k) / math.sqrt(max(1, dim))
        route_logits = route_logits + self.route_pair_bias(z)
        route_logits = route_logits + self.route_prior.to(device=z.device, dtype=z.dtype).view(1, lanes, lanes) * self.route_prior_strength

        write_logits = self.write_head(z).squeeze(-1)
        fanout_logits = self.fanout_head(z).squeeze(-1)
        primitive_scores = self.primitive_score_head(z)
        primitive_sign_logits = self.primitive_sign_head(z)
        compose_logits = self.compose_head(z)
        memory_write_logits = self.memory_write_head(z).squeeze(-1)
        memory_read_gate_logits = self.memory_read_gate_head(pooled).squeeze(-1)

        stats = {
            "controller_z_norm": z.float().norm(dim=-1).mean(),
            "boundary_logit_mean": boundary_logit.float().mean(),
            "route_logit_std": route_logits.float().std(unbiased=False),
            "primitive_score_std": primitive_scores.float().std(unbiased=False),
            "memory_read_gate_logit_mean": memory_read_gate_logits.float().mean(),
        }
        return ControllerOutput(
            z=z,
            boundary_logit=boundary_logit,
            route_logits=route_logits,
            write_logits=write_logits,
            fanout_logits=fanout_logits,
            primitive_scores=primitive_scores,
            primitive_sign_logits=primitive_sign_logits,
            compose_logits=compose_logits,
            memory_write_logits=memory_write_logits,
            memory_read_gate_logits=memory_read_gate_logits,
            stats=stats,
        )


if __name__ == "__main__":
    torch.manual_seed(0)
    m = JointController(dim=16, lanes=4, num_primitives=8, max_steps=6)
    x = torch.randn(3, 4, 16)
    mem = torch.zeros(3, 16)
    out = m(x, mem, 2, evidence=torch.randn(3, 16))
    assert out.z.shape == (3, 4, 16)
    assert out.route_logits.shape == (3, 4, 4)
    assert out.primitive_scores.shape == (3, 4, 8)
    print("joint_controller smoke ok")
