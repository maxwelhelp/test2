#!/usr/bin/env python3
"""Hotfix runner for TapeLaneRouter v4.2.

The original v4.2 uses BlockButterfly from simple_butterfly_matrix.py.
That older BlockButterfly can break for cells_per_lane=12 because one stage may
produce more pairs than its parameter tensor stores. This file patches the
BlockButterfly symbol before importing v4.2, so the rest of v4.2 stays unchanged.

Run this file instead of tape_lane_transport_v4_2.py.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import simple_butterfly_matrix.simple_butterfly_matrix as base  # noqa: E402


class SafeBlockButterfly(nn.Module):
    """Power-of-two padded block butterfly over the cell axis.

    Input:  [batch, blocks, dim]
    Output: [batch, blocks, dim]

    This is fully differentiable and matrix-based. It pads the block axis to the
    next power of two, applies pair butterflies, then slices back to the original
    block count. It avoids the old wrap-around stage-order bug for non-power-of-two
    sizes such as 12.
    """

    def __init__(self, blocks: int):
        super().__init__()
        self.blocks = int(blocks)
        self.padded_blocks = 1 << math.ceil(math.log2(max(2, self.blocks)))
        self.stages = int(math.ceil(math.log2(max(2, self.padded_blocks))))
        pairs = max(1, self.padded_blocks // 2)
        self.weight = nn.Parameter(
            torch.eye(2).view(1, 1, 2, 2).repeat(self.stages, pairs, 1, 1)
            + 0.03 * torch.randn(self.stages, pairs, 2, 2)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.blocks < 2:
            return x
        bsz, blocks, dim = x.shape
        if blocks != self.blocks:
            # Keep it robust if caller passes a slightly different block count.
            # This path creates a temporary padded view and uses the largest safe slice.
            target_blocks = int(blocks)
        else:
            target_blocks = self.blocks
        padded = 1 << math.ceil(math.log2(max(2, target_blocks)))
        y = x
        if y.shape[1] < padded:
            pad_n = padded - y.shape[1]
            y = torch.cat([y, y[:, -1:, :].expand(-1, pad_n, -1)], dim=1)
        elif y.shape[1] > padded:
            y = y[:, :padded, :]

        # If runtime padded differs from init padded, use a safe prefix of weights.
        # For normal v4.2 cells_per_lane=12, both are 16.
        stages = int(math.ceil(math.log2(max(2, padded))))
        pairs = padded // 2
        for si in range(stages):
            stride = 2 ** si
            order = self._stage_order(padded, stride, y.device)
            inv = torch.empty_like(order)
            inv[order] = torch.arange(order.numel(), device=y.device)
            yp = y.index_select(1, order).reshape(bsz, pairs, 2, dim)
            if si < self.weight.shape[0] and pairs <= self.weight.shape[1]:
                w = self.weight[si, :pairs].to(device=x.device, dtype=x.dtype)
            else:
                eye = torch.eye(2, device=x.device, dtype=x.dtype).view(1, 2, 2).expand(pairs, -1, -1)
                w = eye
            mixed = torch.einsum("bpjd,pij->bpid", yp, w)
            y = mixed.reshape(bsz, padded, dim).index_select(1, inv)
        return y[:, :target_blocks, :]

    @staticmethod
    def _stage_order(blocks: int, stride: int, device: torch.device) -> torch.Tensor:
        # For power-of-two padded length, xor pairing gives a clean butterfly stage.
        idx = torch.arange(blocks, device=device)
        partner = idx ^ int(stride)
        used = torch.zeros(blocks, dtype=torch.bool, device=device)
        order = []
        for i in range(blocks):
            if bool(used[i]):
                continue
            j = int(partner[i])
            order.extend([i, j])
            used[i] = True
            used[j] = True
        return torch.tensor(order, dtype=torch.long, device=device)


# Patch before importing v4.2. Its `from ... import BlockButterfly` will now bind SafeBlockButterfly.
base.BlockButterfly = SafeBlockButterfly

from simple_butterfly_matrix_v4_tape_lane import tape_lane_transport_v4_2 as v42  # noqa: E402


if __name__ == "__main__":
    v42.run(v42.parser().parse_args())
