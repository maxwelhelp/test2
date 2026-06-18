#!/usr/bin/env python3
"""
program_assembly_curriculum_v1.py

Standalone synthetic curriculum for teaching a small controller to assemble matrix programs.
It does NOT replace v4.3/v4.4. It is a fast school for the controller grammar:
  - repeat a clean teacher assembly
  - repair missing/wrong read/primitive/route/write/boundary elements
  - denoise extra boundary/write/route noise
  - execute the assembled program through a differentiable matrix executor
  - measure whether repair improves real execution loss versus the corrupted assembly

The checkpoint is small and can later be adapted into v4.4/v4.5
read/primitive/route/boundary/write controllers.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


PRIMITIVES = ["identity", "shift_left", "shift_right", "diff_prev", "smooth3", "contrast", "gate_state", "memory_keep"]
LANE_NAMES = ["detail", "state", "abstract", "memory"]
READ_NAMES = ["input", "detail", "state", "abstract", "memory"]
MODE_NAMES = ["repeat", "repair", "denoise", "mixed"]

P_IDENTITY, P_SHIFT_LEFT, P_SHIFT_RIGHT, P_DIFF, P_SMOOTH, P_CONTRAST, P_GATE, P_MEMORY = range(8)


@dataclass
class ProgramBatch:
    x: torch.Tensor
    target_y: torch.Tensor
    read_t: torch.Tensor
    prim_t: torch.Tensor
    route_t: torch.Tensor
    write_t: torch.Tensor
    boundary_t: torch.Tensor
    active_mask: torch.Tensor
    corrupt_read: torch.Tensor
    corrupt_prim: torch.Tensor
    corrupt_route: torch.Tensor
    corrupt_write: torch.Tensor
    corrupt_boundary: torch.Tensor
    corrupt_mask: torch.Tensor
    write_corrupt_mask: torch.Tensor
    boundary_corrupt_mask: torch.Tensor
    mode_id: torch.Tensor
    task_id: torch.Tensor


# ----------------------------- executor -----------------------------

def apply_primitives(read_vec: torch.Tensor, state: torch.Tensor) -> torch.Tensor:
    """Return primitive outputs [B,L,P,D]."""
    memory = state[:, 3:4, :] if state.shape[1] >= 4 else state[:, -1:, :]
    memory = memory.expand(-1, state.shape[1], -1)
    mean = read_vec.mean(dim=-1, keepdim=True)
    smooth = (torch.roll(read_vec, 1, dims=-1) + read_vec + torch.roll(read_vec, -1, dims=-1)) / 3.0
    outs = [
        read_vec,
        torch.roll(read_vec, -1, dims=-1),
        torch.roll(read_vec, 1, dims=-1),
        read_vec - torch.roll(read_vec, 1, dims=-1),
        smooth,
        read_vec - mean,
        torch.sigmoid(state) * read_vec,
        0.55 * read_vec + 0.45 * memory,
    ]
    return torch.stack(outs, dim=2)


def execute_soft_program(x, read_prob, prim_prob, route_prob, write_prob, boundary_prob):
    """Differentiable matrix-program executor.

    Boundary controls whether cross-lane route is allowed. If boundary is low,
    route becomes identity-like. This prevents the curriculum from teaching
    boundary=ON everywhere as a free all-to-all shortcut.
    """
    bsz, steps, lanes, _ = read_prob.shape
    dim = x.shape[-1]
    state = torch.zeros(bsz, lanes, dim, device=x.device, dtype=x.dtype)
    state[:, 0] = x
    eye = torch.eye(lanes, device=x.device, dtype=x.dtype).view(1, lanes, lanes)
    boundary_used, route_entropy = [], []
    history = [state]
    for t in range(steps):
        sources = torch.cat([x.view(bsz, 1, dim), state], dim=1)
        read_vec = torch.einsum("blr,brd->bld", read_prob[:, t], sources)
        prim_all = apply_primitives(read_vec, state)
        prim_out = torch.einsum("blp,blpd->bld", prim_prob[:, t], prim_all)
        b = boundary_prob[:, t].view(bsz, 1, 1)
        route_eff = b * route_prob[:, t] + (1.0 - b) * eye
        routed = torch.einsum("bft,bfd->btd", route_eff, prim_out)
        state = F.layer_norm(state + write_prob[:, t].view(bsz, lanes, 1) * routed, (dim,))
        history.append(state)
        boundary_used.append(boundary_prob[:, t])
        ent = -(route_prob[:, t].clamp_min(1e-8) * route_prob[:, t].clamp_min(1e-8).log()).sum(dim=-1).mean(dim=-1)
        route_entropy.append(ent)
    out = state[:, 2] + 0.25 * state[:, 3]
    return out, {"state_history": torch.stack(history, dim=1), "boundary_used": torch.stack(boundary_used, dim=1), "route_entropy": torch.stack(route_entropy, dim=1)}


def hard_to_probs(read_t, prim_t, route_t, write_t, boundary_t, lanes, primitives):
    return (
        F.one_hot(read_t, num_classes=lanes + 1).float(),
        F.one_hot(prim_t, num_classes=primitives).float(),
        F.one_hot(route_t, num_classes=lanes).float(),
        write_t.float(),
        boundary_t.float(),
    )


def corrupted_to_probs(batch: ProgramBatch, lanes: int, primitives: int):
    """Safe execution of a corrupted program. Unknowns are replaced with no-op/self/off."""
    cr = batch.corrupt_read.clone()
    cp = batch.corrupt_prim.clone()
    cro = batch.corrupt_route.clone()
    cw = batch.corrupt_write.clone()
    cb = batch.corrupt_boundary.clone()
    # Unknown read -> lane self group, primitive -> identity, route -> self, write/boundary -> off.
    for l in range(lanes):
        cr[..., l] = torch.where(cr[..., l] > lanes, torch.full_like(cr[..., l], 1 + l), cr[..., l])
        cro[..., l] = torch.where(cro[..., l] >= lanes, torch.full_like(cro[..., l], l), cro[..., l])
    cp = torch.where(cp >= primitives, torch.zeros_like(cp), cp)
    cw = torch.where(cw > 1, torch.zeros_like(cw), cw).float()
    cb = torch.where(cb > 1, torch.zeros_like(cb), cb).float()
    return hard_to_probs(cr.clamp(0, lanes), cp.clamp(0, primitives - 1), cro.clamp(0, lanes - 1), cw, cb, lanes, primitives)


# ----------------------------- teacher/corruption -----------------------------

def _defaults(bsz, steps, lanes, device):
    read = torch.zeros(bsz, steps, lanes, dtype=torch.long, device=device)
    prim = torch.full((bsz, steps, lanes), P_IDENTITY, dtype=torch.long, device=device)
    route = torch.arange(lanes, device=device).view(1, 1, lanes).expand(bsz, steps, lanes).clone()
    write = torch.zeros(bsz, steps, lanes, dtype=torch.float32, device=device)
    boundary = torch.zeros(bsz, steps, dtype=torch.float32, device=device)
    active = torch.zeros(bsz, steps, lanes, dtype=torch.bool, device=device)
    for l in range(lanes):
        read[:, :, l] = 1 + l
    return read, prim, route, write, boundary, active


def _set_step(read, prim, route, write, boundary, active, b, t, src_lane, read_group, primitive, target_lane, boundary_on=1.0):
    read[b, t, src_lane] = int(read_group)
    prim[b, t, src_lane] = int(primitive)
    route[b, t, src_lane] = int(target_lane)
    write[b, t, target_lane] = 1.0
    boundary[b, t] = float(boundary_on)
    active[b, t, src_lane] = True


def make_teacher_batch(args, bsz, device, stage):
    lanes, steps, dim, p = int(args.lanes), int(args.steps), int(args.dim), len(PRIMITIVES)
    x = torch.randn(bsz, dim, device=device)
    if args.signal_structure:
        grid = torch.linspace(-1, 1, dim, device=device).view(1, dim)
        freq = torch.randint(1, 5, (bsz, 1), device=device).float()
        x = x + 0.35 * torch.sin(freq * math.pi * grid)

    read, prim, route, write, boundary, active = _defaults(bsz, steps, lanes, device)
    max_task = min(5, max(2, stage + 2))
    task_id = torch.randint(0, max_task, (bsz,), device=device)
    # Modes: 0 repeat/clean, 1 repair missing/wrong, 2 denoise extras, 3 mixed.
    if stage <= 0:
        mode_id = torch.zeros(bsz, dtype=torch.long, device=device)
    elif stage == 1:
        mode_id = torch.randint(0, 2, (bsz,), device=device)
    elif stage == 2:
        mode_id = torch.randint(0, 3, (bsz,), device=device)
    else:
        mode_id = torch.randint(0, 4, (bsz,), device=device)

    for b in range(bsz):
        tid = int(task_id[b].item())
        if tid == 0:
            _set_step(read, prim, route, write, boundary, active, b, 0, 0, 0, P_DIFF, 1, 1)
            _set_step(read, prim, route, write, boundary, active, b, 1, 1, 2, P_SMOOTH, 2, 1)
        elif tid == 1:
            _set_step(read, prim, route, write, boundary, active, b, 0, 0, 0, P_SHIFT_LEFT, 1, 1)
            _set_step(read, prim, route, write, boundary, active, b, 1, 1, 2, P_CONTRAST, 2, 1)
        elif tid == 2:
            _set_step(read, prim, route, write, boundary, active, b, 0, 0, 0, P_SMOOTH, 3, 1)
            _set_step(read, prim, route, write, boundary, active, b, 1, 0, 0, P_DIFF, 1, 1)
            _set_step(read, prim, route, write, boundary, active, b, 2, 3, 4, P_MEMORY, 2, 1)
        elif tid == 3:
            _set_step(read, prim, route, write, boundary, active, b, 0, 0, 0, P_CONTRAST, 1, 1)
            _set_step(read, prim, route, write, boundary, active, b, 1, 1, 2, P_GATE, 2, 1)
            _set_step(read, prim, route, write, boundary, active, b, 2, 2, 3, P_SMOOTH, 3, 1)
            if steps > 3:
                _set_step(read, prim, route, write, boundary, active, b, 3, 3, 4, P_MEMORY, 2, 1)
        else:
            _set_step(read, prim, route, write, boundary, active, b, 0, 0, 0, P_SHIFT_RIGHT, 1, 1)
            _set_step(read, prim, route, write, boundary, active, b, 1, 1, 2, P_DIFF, 2, 1)
            _set_step(read, prim, route, write, boundary, active, b, 2, 2, 3, P_CONTRAST, 3, 1)
            if steps > 3:
                _set_step(read, prim, route, write, boundary, active, b, 3, 3, 4, P_MEMORY, 2, 1)

    with torch.no_grad():
        y, _ = execute_soft_program(x, *hard_to_probs(read, prim, route, write, boundary, lanes, p))
    corrupt = corrupt_program(read, prim, route, write, boundary, active, mode_id, args)
    return ProgramBatch(x=x, target_y=y.detach(), read_t=read, prim_t=prim, route_t=route, write_t=write, boundary_t=boundary,
                        active_mask=active, corrupt_read=corrupt[0], corrupt_prim=corrupt[1], corrupt_route=corrupt[2],
                        corrupt_write=corrupt[3], corrupt_boundary=corrupt[4], corrupt_mask=corrupt[5],
                        write_corrupt_mask=corrupt[6], boundary_corrupt_mask=corrupt[7], mode_id=mode_id, task_id=task_id)


def corrupt_program(read, prim, route, write, boundary, active, mode_id, args):
    lanes, p = read.shape[-1], len(PRIMITIVES)
    cr, cp, cro = read.clone(), prim.clone(), route.clone()
    cw, cb = write.long().clone(), boundary.long().clone()
    cmask = torch.zeros_like(active)
    wmask = torch.zeros_like(write, dtype=torch.bool)
    bmask = torch.zeros_like(boundary, dtype=torch.bool)
    device = read.device
    bsz, steps, _ = read.shape
    base_miss = float(args.corrupt_prob)
    base_wrong = float(args.wrong_prob)
    base_extra = float(args.extra_prob)

    for b in range(bsz):
        mode = int(mode_id[b].item())
        miss_p = 0.0 if mode == 0 else base_miss
        wrong_p = 0.0 if mode == 0 else base_wrong
        extra_p = 0.0 if mode in (0, 1) else base_extra
        rand = torch.rand(steps, lanes, device=device)
        miss = (rand < miss_p) & active[b]
        wrong = (rand >= miss_p) & (rand < miss_p + wrong_p) & active[b]
        cr[b][miss] = lanes + 1
        cp[b][miss] = p
        cro[b][miss] = lanes
        cmask[b] |= miss
        if wrong.any():
            n = int(wrong.sum().item())
            cr[b][wrong] = torch.randint(0, lanes + 1, (n,), device=device)
            cp[b][wrong] = torch.randint(0, p, (n,), device=device)
            cro[b][wrong] = torch.randint(0, lanes, (n,), device=device)
            cmask[b] |= wrong
        # Denoise: extra writes/boundaries and wrong inactive route/read/primitive tokens.
        if extra_p > 0:
            extra_w = (torch.rand(steps, lanes, device=device) < extra_p) & (~active[b])
            cw[b][extra_w] = 1
            wmask[b] |= extra_w
            extra_b = (torch.rand(steps, device=device) < extra_p) & (boundary[b] < 0.5)
            cb[b][extra_b] = 1
            bmask[b] |= extra_b
            extra_r = (torch.rand(steps, lanes, device=device) < extra_p) & (~active[b])
            n = int(extra_r.sum().item())
            if n:
                cr[b][extra_r] = torch.randint(0, lanes + 1, (n,), device=device)
                cp[b][extra_r] = torch.randint(0, p, (n,), device=device)
                cro[b][extra_r] = torch.randint(0, lanes, (n,), device=device)
                cmask[b] |= extra_r
        # Unknown write/boundary in repair/mixed modes.
        if mode in (1, 3):
            unk_w = (torch.rand(steps, lanes, device=device) < miss_p * 0.25)
            cw[b][unk_w] = 2
            wmask[b] |= unk_w
            unk_b = (torch.rand(steps, device=device) < miss_p * 0.25)
            cb[b][unk_b] = 2
            bmask[b] |= unk_b
    return cr.clamp(0, lanes + 1), cp.clamp(0, p), cro.clamp(0, lanes), cw.clamp(0, 2), cb.clamp(0, 2), cmask, wmask, bmask


# ----------------------------- model -----------------------------

class ProgramBuilderStudent(nn.Module):
    def __init__(self, dim, steps, lanes, primitives, hidden=192):
        super().__init__()
        self.dim, self.steps, self.lanes, self.primitives, self.hidden = dim, steps, lanes, primitives, hidden
        self.x_enc = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, hidden), nn.SiLU(), nn.Linear(hidden, hidden))
        self.step_emb = nn.Embedding(steps, hidden)
        self.lane_emb = nn.Embedding(lanes, hidden)
        self.mode_emb = nn.Embedding(len(MODE_NAMES), hidden)
        self.read_emb = nn.Embedding(lanes + 2, hidden)
        self.prim_emb = nn.Embedding(primitives + 1, hidden)
        self.route_emb = nn.Embedding(lanes + 1, hidden)
        self.write_emb = nn.Embedding(3, hidden)
        self.boundary_emb = nn.Embedding(3, hidden)
        self.trunk = nn.Sequential(nn.LayerNorm(hidden), nn.Linear(hidden, hidden), nn.SiLU(), nn.Linear(hidden, hidden), nn.SiLU())
        self.read_head = nn.Linear(hidden, lanes + 1)
        self.prim_head = nn.Linear(hidden, primitives)
        self.route_head = nn.Linear(hidden, lanes)
        self.write_head = nn.Linear(hidden, 1)
        self.boundary_head = nn.Sequential(nn.LayerNorm(hidden), nn.Linear(hidden, hidden), nn.SiLU(), nn.Linear(hidden, 1))

    def forward(self, batch: ProgramBatch):
        bsz, steps, lanes = batch.read_t.shape
        device = batch.x.device
        h = self.x_enc(batch.x).view(bsz, 1, 1, self.hidden)
        h = h + self.step_emb(torch.arange(steps, device=device).view(1, steps, 1))
        h = h + self.lane_emb(torch.arange(lanes, device=device).view(1, 1, lanes))
        h = h + self.mode_emb(batch.mode_id.clamp(0, len(MODE_NAMES) - 1)).view(bsz, 1, 1, self.hidden)
        h = h + self.read_emb(batch.corrupt_read.clamp(0, lanes + 1))
        h = h + self.prim_emb(batch.corrupt_prim.clamp(0, self.primitives))
        h = h + self.route_emb(batch.corrupt_route.clamp(0, lanes))
        h = h + self.write_emb(batch.corrupt_write.clamp(0, 2))
        h = h + self.boundary_emb(batch.corrupt_boundary.clamp(0, 2)).unsqueeze(2)
        h = self.trunk(h)
        return {
            "read_logits": self.read_head(h),
            "prim_logits": self.prim_head(h),
            "route_logits": self.route_head(h),
            "write_logits": self.write_head(h).squeeze(-1),
            "boundary_logits": self.boundary_head(h.mean(dim=2)).squeeze(-1),
        }


# ----------------------------- losses/metrics -----------------------------

def weighted_ce(logits, target, weight):
    flat_loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), target.reshape(-1), reduction="none")
    w = weight.reshape(-1).float()
    return (flat_loss * w).sum() / w.sum().clamp_min(1.0)


def binary_f1(pred, target):
    pred, target = pred.float(), target.float()
    tp = (pred * target).sum()
    fp = (pred * (1.0 - target)).sum()
    fn = ((1.0 - pred) * target).sum()
    precision = tp / (tp + fp).clamp_min(1.0)
    recall = tp / (tp + fn).clamp_min(1.0)
    f1 = 2 * precision * recall / (precision + recall).clamp_min(1e-6)
    return precision, recall, f1


def compute_loss_and_metrics(model_out, batch: ProgramBatch, args):
    read_logits, prim_logits, route_logits = model_out["read_logits"], model_out["prim_logits"], model_out["route_logits"]
    write_logits, boundary_logits = model_out["write_logits"], model_out["boundary_logits"]
    active = batch.active_mask.float()
    inactive_w = float(args.inactive_token_weight)
    token_weight = active + inactive_w * (1.0 - active)

    read_loss = weighted_ce(read_logits, batch.read_t, token_weight)
    prim_loss = weighted_ce(prim_logits, batch.prim_t, token_weight)
    route_loss = weighted_ce(route_logits, batch.route_t, token_weight)
    write_loss = F.binary_cross_entropy_with_logits(write_logits, batch.write_t.float(), pos_weight=torch.tensor(float(args.write_pos_weight), device=batch.x.device))
    boundary_loss = F.binary_cross_entropy_with_logits(boundary_logits, batch.boundary_t.float(), pos_weight=torch.tensor(float(args.boundary_pos_weight), device=batch.x.device))

    read_prob = torch.softmax(read_logits, dim=-1)
    prim_prob = torch.softmax(prim_logits, dim=-1)
    route_prob = torch.softmax(route_logits, dim=-1)
    write_prob = torch.sigmoid(write_logits)
    boundary_prob = torch.sigmoid(boundary_logits)
    y_pred, aux = execute_soft_program(batch.x, read_prob, prim_prob, route_prob, write_prob, boundary_prob)
    exec_loss = F.mse_loss(y_pred, batch.target_y)

    boundary_count = boundary_prob.sum(dim=1)
    boundary_budget = (F.relu(boundary_count - float(args.boundary_max_peaks)).pow(2) + F.relu(float(args.boundary_min_peaks) - boundary_count).pow(2)).mean()
    eye = torch.eye(args.lanes, device=batch.x.device, dtype=route_prob.dtype).view(1, 1, args.lanes, args.lanes)
    offdiag = (route_prob * (1.0 - eye)).sum(dim=-1).mean(dim=-1)
    offdiag_without_boundary = (offdiag * (1.0 - boundary_prob)).mean()
    route_ent = -(route_prob.clamp_min(1e-8) * route_prob.clamp_min(1e-8).log()).sum(dim=-1).mean()
    route_entropy_cost = F.relu(route_ent - float(args.route_entropy_max)).pow(2)
    write_cost = write_prob.mean()

    loss = (float(args.lambda_trace) * (read_loss + prim_loss + route_loss + write_loss + boundary_loss)
            + float(args.lambda_exec) * exec_loss
            + float(args.lambda_boundary_budget) * boundary_budget
            + float(args.lambda_offdiag_no_boundary) * offdiag_without_boundary
            + float(args.lambda_route_entropy) * route_entropy_cost
            + float(args.lambda_write_cost) * write_cost)

    with torch.no_grad():
        pred_read, pred_prim, pred_route = read_logits.argmax(-1), prim_logits.argmax(-1), route_logits.argmax(-1)
        pred_write, pred_boundary = (write_prob > 0.5).float(), (boundary_prob > 0.5).float()
        denom_active = batch.active_mask.sum().clamp_min(1).float()
        read_acc = ((pred_read == batch.read_t) & batch.active_mask).sum().float() / denom_active
        prim_acc = ((pred_prim == batch.prim_t) & batch.active_mask).sum().float() / denom_active
        route_acc = ((pred_route == batch.route_t) & batch.active_mask).sum().float() / denom_active
        read_all_acc = (pred_read == batch.read_t).float().mean()
        prim_all_acc = (pred_prim == batch.prim_t).float().mean()
        route_all_acc = (pred_route == batch.route_t).float().mean()
        write_acc = (pred_write == batch.write_t).float().mean()
        boundary_acc = (pred_boundary == batch.boundary_t).float().mean()
        b_prec, b_rec, b_f1 = binary_f1(pred_boundary, batch.boundary_t)
        corrupt_positions = batch.corrupt_mask
        repair_acc = (((pred_read == batch.read_t) & (pred_prim == batch.prim_t) & (pred_route == batch.route_t) & corrupt_positions).sum().float() / corrupt_positions.sum().clamp_min(1).float())
        write_repair_acc = ((pred_write == batch.write_t).float() * batch.write_corrupt_mask.float()).sum() / batch.write_corrupt_mask.float().sum().clamp_min(1.0)
        boundary_repair_acc = ((pred_boundary == batch.boundary_t).float() * batch.boundary_corrupt_mask.float()).sum() / batch.boundary_corrupt_mask.float().sum().clamp_min(1.0)
        active_exact = (((pred_read == batch.read_t) | (~batch.active_mask)).all(dim=(1, 2)) & ((pred_prim == batch.prim_t) | (~batch.active_mask)).all(dim=(1, 2)) & ((pred_route == batch.route_t) | (~batch.active_mask)).all(dim=(1, 2))).float().mean()
        full_exact = ((pred_read == batch.read_t).all(dim=(1, 2)) & (pred_prim == batch.prim_t).all(dim=(1, 2)) & (pred_route == batch.route_t).all(dim=(1, 2)) & (pred_write == batch.write_t).all(dim=(1, 2)) & (pred_boundary == batch.boundary_t).all(dim=1)).float().mean()
        boundary_exploit_rate = (boundary_count > (batch.boundary_t.shape[1] * 0.75)).float().mean()
        corrupt_probs = corrupted_to_probs(batch, args.lanes, len(PRIMITIVES))
        corrupted_y, _ = execute_soft_program(batch.x, *corrupt_probs)
        corrupted_exec_loss = F.mse_loss(corrupted_y, batch.target_y)
        exec_improvement = corrupted_exec_loss - exec_loss
        metrics = {
            "loss": float(loss.detach().cpu()), "exec_loss": float(exec_loss.detach().cpu()), "corrupted_exec_loss": float(corrupted_exec_loss.detach().cpu()), "exec_improvement": float(exec_improvement.detach().cpu()),
            "read_loss": float(read_loss.detach().cpu()), "prim_loss": float(prim_loss.detach().cpu()), "route_loss": float(route_loss.detach().cpu()), "write_loss": float(write_loss.detach().cpu()), "boundary_loss": float(boundary_loss.detach().cpu()),
            "read_acc": float(read_acc.cpu()), "prim_acc": float(prim_acc.cpu()), "route_acc": float(route_acc.cpu()), "read_all_acc": float(read_all_acc.cpu()), "prim_all_acc": float(prim_all_acc.cpu()), "route_all_acc": float(route_all_acc.cpu()),
            "write_acc": float(write_acc.cpu()), "boundary_acc": float(boundary_acc.cpu()), "boundary_precision": float(b_prec.cpu()), "boundary_recall": float(b_rec.cpu()), "boundary_f1": float(b_f1.cpu()),
            "repair_acc": float(repair_acc.cpu()), "write_repair_acc": float(write_repair_acc.cpu()), "boundary_repair_acc": float(boundary_repair_acc.cpu()), "active_program_exact": float(active_exact.cpu()), "full_program_exact": float(full_exact.cpu()),
            "boundary_budget": float(boundary_budget.detach().cpu()), "boundary_exploit_rate": float(boundary_exploit_rate.cpu()), "offdiag_without_boundary": float(offdiag_without_boundary.detach().cpu()), "route_entropy": float(route_ent.detach().cpu()), "write_mean": float(write_prob.mean().detach().cpu()), "boundary_mean": float(boundary_prob.mean().detach().cpu()), "boundary_count_mean": float(boundary_count.mean().detach().cpu()),
        }
    return loss, metrics, {"y_pred": y_pred.detach(), "aux": aux, "pred": (pred_read.detach(), pred_prim.detach(), pred_route.detach(), pred_write.detach(), pred_boundary.detach())}


@torch.no_grad()
def evaluate(model, args, device, stage, batches=20):
    model.eval(); sums: Dict[str, float] = {}; examples = None
    for _ in range(batches):
        batch = make_teacher_batch(args, args.batch_size, device, stage)
        out = model(batch)
        _, metrics, extra = compute_loss_and_metrics(out, batch, args)
        for k, v in metrics.items(): sums[k] = sums.get(k, 0.0) + float(v)
        if examples is None: examples = make_examples(batch, extra, args, 5)
    return {k: v / max(1, batches) for k, v in sums.items()}, examples or []


def make_examples(batch, extra, args, max_examples=5):
    pred_read, pred_prim, pred_route, pred_write, pred_boundary = extra["pred"]
    out = []
    for i in range(min(max_examples, batch.x.shape[0])):
        steps = []
        for t in range(args.steps):
            active_lanes = torch.where(batch.active_mask[i, t])[0].tolist()
            step = {"t": int(t), "target_boundary": float(batch.boundary_t[i, t].cpu()), "pred_boundary": float(pred_boundary[i, t].cpu()), "lanes": []}
            for l in active_lanes:
                step["lanes"].append({"lane": LANE_NAMES[l], "target": {"read": READ_NAMES[int(batch.read_t[i, t, l].cpu())], "primitive": PRIMITIVES[int(batch.prim_t[i, t, l].cpu())], "route_to": LANE_NAMES[int(batch.route_t[i, t, l].cpu())]}, "pred": {"read": READ_NAMES[int(pred_read[i, t, l].cpu())], "primitive": PRIMITIVES[int(pred_prim[i, t, l].cpu())], "route_to": LANE_NAMES[int(pred_route[i, t, l].cpu())]}})
            steps.append(step)
        out.append({"task_id": int(batch.task_id[i].cpu()), "mode": MODE_NAMES[int(batch.mode_id[i].cpu())], "steps": steps})
    return out


# ----------------------------- train/report -----------------------------

def stage_for_epoch(epoch, args):
    if epoch <= args.stage0_epochs: return 0
    if epoch <= args.stage0_epochs + args.stage1_epochs: return 1
    if epoch <= args.stage0_epochs + args.stage1_epochs + args.stage2_epochs: return 2
    return 4


def append_csv(path, row, header_written):
    with path.open("a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(row.keys()))
        if not header_written: w.writeheader()
        w.writerow(row)


def train(args):
    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    random.seed(args.seed); torch.manual_seed(args.seed)
    device = torch.device(args.device if (args.device == "cpu" or torch.cuda.is_available()) else "cpu")
    if device.type == "cuda": torch.cuda.manual_seed_all(args.seed)
    model = ProgramBuilderStudent(args.dim, args.steps, args.lanes, len(PRIMITIVES), args.hidden).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scaler = torch.cuda.amp.GradScaler(enabled=(device.type == "cuda" and args.amp == "fp16"))
    metrics_path = out_dir / "metrics.csv"; header = False; best = {"score": -1.0, "epoch": 0}; t0 = time.time()
    for epoch in range(1, args.epochs + 1):
        stage = stage_for_epoch(epoch, args); model.train(); sums: Dict[str, float] = {}
        for step in range(1, args.train_steps_per_epoch + 1):
            batch = make_teacher_batch(args, args.batch_size, device, stage)
            opt.zero_grad(set_to_none=True)
            use_amp = device.type == "cuda" and args.amp == "fp16"
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
                loss, metrics, _ = compute_loss_and_metrics(model(batch), batch, args)
            scaler.scale(loss).backward()
            if args.grad_clip > 0:
                scaler.unscale_(opt); nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            scaler.step(opt); scaler.update()
            for k, v in metrics.items(): sums[k] = sums.get(k, 0.0) + float(v)
            if args.log_every and step % args.log_every == 0:
                print(f"epoch={epoch:03d} step={step:05d} stage={stage} loss={sums['loss']/step:.4f} prim={sums['prim_acc']/step:.3f} route={sums['route_acc']/step:.3f} bF1={sums['boundary_f1']/step:.3f} exact={sums['full_program_exact']/step:.3f}", flush=True)
        train_avg = {"train_" + k: v / max(1, args.train_steps_per_epoch) for k, v in sums.items()}
        val, examples = evaluate(model, args, device, stage, args.eval_batches)
        row = {"epoch": epoch, "stage": stage}; row.update(train_avg); row.update({"val_" + k: v for k, v in val.items()})
        append_csv(metrics_path, row, header); header = True
        score = val.get("full_program_exact", 0.0) + val.get("prim_acc", 0.0) + val.get("route_acc", 0.0) + val.get("boundary_f1", 0.0) + max(0.0, val.get("exec_improvement", 0.0)) - val.get("boundary_exploit_rate", 0.0)
        if score > best["score"]:
            best = {"score": float(score), "epoch": epoch, "val": val}
            if args.save_checkpoint:
                torch.save({"model": model.state_dict(), "args": vars(args), "primitives": PRIMITIVES, "lane_names": LANE_NAMES, "read_names": READ_NAMES, "mode_names": MODE_NAMES}, out_dir / "program_builder_curriculum_v1_best.pt")
        (out_dir / f"examples_epoch_{epoch:03d}.json").write_text(json.dumps(examples, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"EPOCH {epoch:03d} stage={stage} full_exact={val['full_program_exact']:.3f} prim={val['prim_acc']:.3f} route={val['route_acc']:.3f} bF1={val['boundary_f1']:.3f} improve={val['exec_improvement']:.4f} exploit={val['boundary_exploit_rate']:.3f}", flush=True)
    final = {"version": "program_assembly_curriculum_v1", "elapsed_sec": time.time() - t0, "best": best, "args": vars(args), "what_it_trains": ["repeat", "repair", "denoise", "real_execution", "read", "primitive", "route", "write", "boundary"], "not_yet": ["actor", "critic", "auto_deploy", "v4.4_weight_adapter", "audio_transfer"]}
    (out_dir / "final_report.json").write_text(json.dumps(final, ensure_ascii=False, indent=2), encoding="utf-8")
    write_chatgpt_report(out_dir, final)
    print(f"DONE out_dir={out_dir} best_epoch={best['epoch']} best_score={best['score']:.4f}", flush=True)


def write_chatgpt_report(out_dir, final):
    best = final.get("best", {}); val = best.get("val", {}) or {}
    lines = [
        "REPORT_TO_CHATGPT: program_assembly_curriculum_v1", "",
        "Цель: школа сборщика программ на синтетике с реальным differentiable executor.",
        "Режимы: repeat / repair / denoise / mixed.",
        "Обучает: read, primitive, route, write, boundary; проверяет, улучшает ли восстановление execution_loss против corrupted program.", "",
        f"best_epoch: {best.get('epoch', 0)}", f"best_score: {best.get('score', 0.0):.4f}",
        f"full_program_exact: {val.get('full_program_exact', 0.0):.4f}", f"active_program_exact: {val.get('active_program_exact', 0.0):.4f}",
        f"primitive_acc: {val.get('prim_acc', 0.0):.4f}", f"route_acc: {val.get('route_acc', 0.0):.4f}", f"read_acc: {val.get('read_acc', 0.0):.4f}",
        f"write_acc: {val.get('write_acc', 0.0):.4f}", f"boundary_f1: {val.get('boundary_f1', 0.0):.4f}", f"repair_acc: {val.get('repair_acc', 0.0):.4f}",
        f"write_repair_acc: {val.get('write_repair_acc', 0.0):.4f}", f"boundary_repair_acc: {val.get('boundary_repair_acc', 0.0):.4f}",
        f"exec_loss: {val.get('exec_loss', 0.0):.6f}", f"corrupted_exec_loss: {val.get('corrupted_exec_loss', 0.0):.6f}", f"exec_improvement: {val.get('exec_improvement', 0.0):.6f}",
        f"boundary_exploit_rate: {val.get('boundary_exploit_rate', 0.0):.4f}", f"offdiag_without_boundary: {val.get('offdiag_without_boundary', 0.0):.4f}", "",
        "Smoke success:", "- primitive_acc > 0.80", "- route_acc > 0.75", "- boundary_f1 > 0.80", "- full_program_exact grows", "- exec_improvement > 0", "- boundary_exploit_rate < 0.10",
    ]
    (Path(out_dir) / "REPORT_TO_CHATGPT.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")


def parser():
    p = argparse.ArgumentParser()
    p.add_argument("--out-dir", default="simple_butterfly_matrix_v4_tape_lane/agent_reports/program_assembly_curriculum_v1_smoke")
    p.add_argument("--device", default="cuda"); p.add_argument("--amp", default="fp16", choices=["fp16", "off"]); p.add_argument("--seed", type=int, default=42)
    p.add_argument("--dim", type=int, default=32); p.add_argument("--hidden", type=int, default=192); p.add_argument("--steps", type=int, default=6); p.add_argument("--lanes", type=int, default=4)
    p.add_argument("--epochs", type=int, default=8); p.add_argument("--batch-size", type=int, default=512); p.add_argument("--train-steps-per-epoch", type=int, default=250); p.add_argument("--eval-batches", type=int, default=30)
    p.add_argument("--lr", type=float, default=8e-4); p.add_argument("--weight-decay", type=float, default=0.01); p.add_argument("--grad-clip", type=float, default=1.0); p.add_argument("--log-every", type=int, default=50)
    p.add_argument("--stage0-epochs", type=int, default=2); p.add_argument("--stage1-epochs", type=int, default=2); p.add_argument("--stage2-epochs", type=int, default=2)
    p.add_argument("--corrupt-prob", type=float, default=0.35); p.add_argument("--wrong-prob", type=float, default=0.20); p.add_argument("--extra-prob", type=float, default=0.08)
    p.add_argument("--signal-structure", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--inactive-token-weight", type=float, default=0.08); p.add_argument("--write-pos-weight", type=float, default=2.0); p.add_argument("--boundary-pos-weight", type=float, default=1.5)
    p.add_argument("--lambda-trace", type=float, default=1.0); p.add_argument("--lambda-exec", type=float, default=1.0); p.add_argument("--lambda-boundary-budget", type=float, default=0.25); p.add_argument("--lambda-offdiag-no-boundary", type=float, default=0.15); p.add_argument("--lambda-route-entropy", type=float, default=0.05); p.add_argument("--lambda-write-cost", type=float, default=0.01)
    p.add_argument("--boundary-min-peaks", type=float, default=1.0); p.add_argument("--boundary-max-peaks", type=float, default=4.0); p.add_argument("--route-entropy-max", type=float, default=1.25)
    p.add_argument("--save-checkpoint", action="store_true")
    return p


if __name__ == "__main__":
    train(parser().parse_args())
