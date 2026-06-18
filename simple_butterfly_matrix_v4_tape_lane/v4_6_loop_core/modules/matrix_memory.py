#!/usr/bin/env python3
"""Real matrix memory lane for v4.6."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn


@dataclass
class MemoryOutput:
    mem_next: torch.Tensor
    read: torch.Tensor
    write_vec: torch.Tensor
    stats: Dict[str, torch.Tensor]


class MatrixMemory(nn.Module):
    """Causal vector memory with learnable write/read matrices and forget."""

    def __init__(self, dim: int, init_forget_logit: float = 1.0) -> None:
        super().__init__()
        self.dim = int(dim)
        self.W_write = nn.Linear(self.dim, self.dim, bias=False)
        self.W_read = nn.Linear(self.dim, self.dim, bias=False)
        self.forget_logit = nn.Parameter(torch.tensor(float(init_forget_logit)))
        self.write_norm = nn.LayerNorm(self.dim)
        self.read_norm = nn.LayerNorm(self.dim)

    def init_state(self, batch_size: int, *, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        return torch.zeros(int(batch_size), self.dim, device=device, dtype=dtype)

    def _pool_write(self, write_input: torch.Tensor, write_gate: Optional[torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        if write_input.dim() == 2:
            gate = torch.ones(write_input.shape[0], 1, device=write_input.device, dtype=write_input.dtype)
            return write_input, gate.squeeze(-1)
        if write_input.dim() != 3:
            raise ValueError(f"write_input must be [B,D] or [B,L,D], got {tuple(write_input.shape)}")
        if write_gate is None:
            gate = torch.ones(write_input.shape[:2], device=write_input.device, dtype=write_input.dtype)
        else:
            if write_gate.shape != write_input.shape[:2]:
                raise ValueError(f"write_gate must be [B,L], got {tuple(write_gate.shape)}")
            gate = torch.sigmoid(write_gate)
        denom = gate.sum(dim=1, keepdim=True).clamp_min(1e-6)
        pooled = (gate.unsqueeze(-1) * write_input).sum(dim=1) / denom
        return pooled, gate.mean(dim=1)

    def forward(
        self,
        mem_prev: torch.Tensor,
        write_input: torch.Tensor,
        write_gate: Optional[torch.Tensor] = None,
        read_gate: Optional[torch.Tensor] = None,
    ) -> MemoryOutput:
        if mem_prev.dim() != 2 or mem_prev.shape[-1] != self.dim:
            raise ValueError(f"mem_prev must be [B,D], got {tuple(mem_prev.shape)}")
        pooled, gate_mean = self._pool_write(write_input, write_gate)
        pooled = self.write_norm(pooled)
        forget = torch.sigmoid(self.forget_logit).to(device=mem_prev.device, dtype=mem_prev.dtype)
        write_vec = self.W_write(pooled)
        mem_next = forget * mem_prev + (1.0 - forget) * write_vec
        read = self.read_norm(self.W_read(mem_next))
        if read_gate is not None:
            gate = torch.sigmoid(read_gate).view(-1, 1).to(dtype=read.dtype)
            read = gate * read
        stats = {
            "memory_forget": forget.float(),
            "memory_gate_mean": gate_mean.float().mean(),
            "memory_write_norm": write_vec.float().norm(dim=-1).mean(),
            "memory_read_norm": read.float().norm(dim=-1).mean(),
            "memory_state_norm": mem_next.float().norm(dim=-1).mean(),
        }
        return MemoryOutput(mem_next=mem_next, read=read, write_vec=write_vec, stats=stats)


if __name__ == "__main__":
    mem = MatrixMemory(16)
    s = mem.init_state(2, device=torch.device("cpu"), dtype=torch.float32)
    out = mem(s, torch.randn(2, 4, 16), write_gate=torch.randn(2, 4), read_gate=torch.randn(2))
    assert out.mem_next.shape == (2, 16)
    assert out.read.shape == (2, 16)
    print("matrix_memory smoke ok")
