#!/usr/bin/env python3
"""
program_assembly_curriculum_v1.py

Standalone curriculum for teaching a small controller to assemble matrix programs.
It does NOT replace v4.3/v4.4. It is a fast synthetic school for:
  - repeat teacher program
  - repair missing elements
  - denoise extra/wrong elements
  - execute the assembled program through a differentiable matrix executor
  - log whether boundary/route/primitive/write choices become logical

The checkpoint is intentionally small and can later be adapted into v4.4/v4.5
read/primitive/route/boundary/write controllers.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


PRIMITIVES = [
    "identity",
    "shift_left",
    "shift_right",
    "diff_prev",
    "smooth3",
    "contrast",
    "gate_state",
    "memory_keep",
]

LANE_NAMES = ["detail", "state", "abstract", "memory"]
READ_NAMES = ["input", "detail", "state", "abstract", "memory"]

P_IDENTITY = 0
P_SHIFT_LEFT = 1
P_SHIFT_RIGHT = 2
P_DIFF = 3
P_SMOOTH = 4
P_CONTRAST = 5
P_GATE = 6
P_MEMORY = 7


@dataclass
class ProgramBatch:
    x: torch.Tensor                         # [B,D]
    target_y: torch.Tensor                  # [B,D]
    read_t: torch.Tensor                    # [B,S,L]
    prim_t: torch.Tensor                    # [B,S,L]
    route_t: torch.Tensor                   # [B,S,L]
    write_t: torch.Tensor                   # [B,S,L]
    boundary_t: torch.Tensor                # [B,S]
    active_mask: torch.Tensor               # [B,S,L]
    corrupt_read: torch.Tensor              # [B,S,L]
    corrupt_prim: torch.Tensor              # [B,S,L]
    corrupt_route: torch.Tensor             # [B,S,L]
    corrupt_write: torch.Tensor             # [B,S,L]
    corrupt_boundary: torch.Tensor          # [B,S]
    corrupt_mask: torch.Tensor              # [B,S,L]
    task_id: torch.Tensor                   # [B]


# ----------------------------- primitives/executor -----------------------------

def apply_primitives(read_vec: torch.Tensor, state: torch.Tensor) -> torch.Tensor:
    """Return all primitive outputs: [B,L,P,D].

    read_vec/state: [B,L,D]
    """
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


def execute_soft_program(
    x: torch.Tensor,
    read_prob: torch.Tensor,       # [B,S,L,R]
    prim_prob: torch.Tensor,       # [B,S,L,P]
    route_prob: torch.Tensor,      # [B,S,L,L] source->target
    write_prob: torch.Tensor,      # [B,S,L]
    boundary_prob: torch.Tensor,   # [B,S]
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """Differentiable executor.

    Boundary gates route freedom. If boundary is low, route becomes mostly identity.
    This is the main anti-shortcut rule for the curriculum.
    """
    bsz, steps, lanes, read_groups = read_prob.shape
    dim = x.shape[-1]
    state = torch.zeros(bsz, lanes, dim, device=x.device, dtype=x.dtype)
    state[:, 0, :] = x
    eye = torch.eye(lanes, device=x.device, dtype=x.dtype).view(1, 1, lanes, lanes)
    state_history = [state]
    boundary_used = []
    route_entropy = []

    for t in range(steps):
        sources = torch.cat([x.view(bsz, 1, dim), state], dim=1)  # [B,R,D]
        read_vec = torch.einsum("blr,brd->bld", read_prob[:, t], sources)
        prim_all = apply_primitives(read_vec, state)
        prim_out = torch.einsum("blp,blpd->bld", prim_prob[:, t], prim_all)

        b = boundary_prob[:, t].view(bsz, 1, 1)
        route_eff = b * route_prob[:, t] + (1.0 - b) * eye[:, 0]
        routed = torch.einsum("bft,bfd->btd", route_eff, prim_out)
        write = write_prob[:, t].view(bsz, lanes, 1)
        state = state + write * routed
        state = F.layer_norm(state, (dim,))
        state_history.append(state)
        boundary_used.append(boundary_prob[:, t])
        ent = -(route_prob[:, t].clamp_min(1e-8) * route_prob[:, t].clamp_min(1e-8).log()).sum(dim=-1).mean(dim=-1)
        route_entropy.append(ent)

    # target/output lives in abstract lane, with a little memory contribution.
    out = state[:, 2, :] + 0.25 * state[:, 3, :]
    aux = {
        "state_history": torch.stack(state_history, dim=1),
        "boundary_used": torch.stack(boundary_used, dim=1),
        "route_entropy": torch.stack(route_entropy, dim=1),
    }
    return out, aux


def hard_to_probs(read_t, prim_t, route_t, write_t, boundary_t, lanes: int, primitives: int):
    read_prob = F.one_hot(read_t, num_classes=lanes + 1).float()
    prim_prob = F.one_hot(prim_t, num_classes=primitives).float()
    route_prob = F.one_hot(route_t, num_classes=lanes).float()
    write_prob = write_t.float()
    boundary_prob = boundary_t.float()
    return read_prob, prim_prob, route_prob, write_prob, boundary_prob


# ----------------------------- teacher programs -----------------------------

def _defaults(bsz: int, steps: int, lanes: int, device):
    read = torch.zeros(bsz, steps, lanes, dtype=torch.long, device=device)
    prim = torch.full((bsz, steps, lanes), P_IDENTITY, dtype=torch.long, device=device)
    route = torch.arange(lanes, device=device).view(1, 1, lanes).expand(bsz, steps, lanes).clone()
    write = torch.zeros(bsz, steps, lanes, dtype=torch.float32, device=device)
    boundary = torch.zeros(bsz, steps, dtype=torch.float32, device=device)
    active = torch.zeros(bsz, steps, lanes, dtype=torch.bool, device=device)
    # default read-self for each lane: read group 1+lane
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


def make_teacher_batch(args, bsz: int, device, stage: int) -> ProgramBatch:
    lanes = int(args.lanes)
    steps = int(args.steps)
    dim = int(args.dim)
    p = len(PRIMITIVES)
    x = torch.randn(bsz, dim, device=device)
    # Add simple structured signals so primitives matter.
    if args.signal_structure:
        grid = torch.linspace(-1, 1, dim, device=device).view(1, dim)
        x = x + 0.35 * torch.sin((torch.randint(1, 5, (bsz, 1), device=device).float()) * math.pi * grid)

    read, prim, route, write, boundary, active = _defaults(bsz, steps, lanes, device)
    task_id = torch.randint(0, min(5, max(2, stage + 2)), (bsz,), device=device)

    for b in range(bsz):
        tid = int(task_id[b].item())
        if tid == 0:
            # input diff -> state, smooth -> abstract
            _set_step(read, prim, route, write, boundary, active, b, 0, 0, 0, P_DIFF, 1, 1)
            _set_step(read, prim, route, write, boundary, active, b, 1, 1, 2, P_SMOOTH, 2, 1)
        elif tid == 1:
            # input shift -> state, contrast -> abstract
            _set_step(read, prim, route, write, boundary, active, b, 0, 0, 0, P_SHIFT_LEFT, 1, 1)
            _set_step(read, prim, route, write, boundary, active, b, 1, 1, 2, P_CONTRAST, 2, 1)
        elif tid == 2:
            # input smooth -> memory, input diff -> state, memory_keep -> abstract
            _set_step(read, prim, route, write, boundary, active, b, 0, 0, 0, P_SMOOTH, 3, 1)
            _set_step(read, prim, route, write, boundary, active, b, 1, 0, 0, P_DIFF, 1, 1)
            _set_step(read, prim, route, write, boundary, active, b, 2, 3, 4, P_MEMORY, 2, 1)
        elif tid == 3:
            # longer chain: input -> state -> abstract -> memory -> abstract refine
            _set_step(read, prim, route, write, boundary, active, b, 0, 0, 0, P_CONTRAST, 1, 1)
            _set_step(read, prim, route, write, boundary, active, b, 1, 1, 2, P_GATE, 2, 1)
            _set_step(read, prim, route, write, boundary, active, b, 2, 2, 3, P_SMOOTH, 3, 1)
            if steps > 3:
                _set_step(read, prim, route, write, boundary, active, b, 3, 3, 4, P_MEMORY, 2, 1)
        else:
            # harder: two reads and memory recall
            _set_step(read, prim, route, write, boundary, active, b, 0, 0, 0, P_SHIFT_RIGHT, 1, 1)
            _set_step(read, prim, route, write, boundary, active, b, 1, 1, 2, P_DIFF, 2, 1)
            _set_step(read, prim, route, write, boundary, active, b, 2, 2, 3, P_CONTRAST, 3, 1)
            if steps > 3:
                _set_step(read, prim, route, write, boundary, active, b, 3, 3, 4, P_MEMORY, 2, 1)

    # Teacher execution.
    target_probs = hard_to_probs(read, prim, route, write, boundary, lanes, p)
    with torch.no_grad():
        y, _ = execute_soft_program(x, *target_probs)

    corrupt = corrupt_program(read, prim, route, write, boundary, active, args)
    return ProgramBatch(x=x, target_y=y.detach(), read_t=read, prim_t=prim, route_t=route, write_t=write, boundary_t=boundary,
                        active_mask=active, corrupt_read=corrupt[0], corrupt_prim=corrupt[1], corrupt_route=corrupt[2],
                        corrupt_write=corrupt[3], corrupt_boundary=corrupt[4], corrupt_mask=corrupt[5], task_id=task_id)


def corrupt_program(read, prim, route, write, boundary, active, args):
    """Return corrupted visible program tokens.

    Unknown ids:
      read unknown = lanes+1 token in embedding input, but clamped before CE.
      prim unknown = P token
      route unknown = lanes token
      write/boundary unknown = 2 token
    """
    lanes = read.shape[-1]
    p = len(PRIMITIVES)
    cr = read.clone()
    cp = prim.clone()
    cro = route.clone()
    cw = write.long().clone()
    cb = boundary.long().clone()
    cmask = torch.zeros_like(active)
    prob = float(args.corrupt_prob)
    bsz, steps, _ = read.shape
    device = read.device

    rand = torch.rand(bsz, steps, lanes, device=device)
    miss = (rand < prob) & active
    wrong = (rand >= prob) & (rand < prob + float(args.wrong_prob)) & active
    cr[miss] = lanes + 1
    cp[miss] = p
    cro[miss] = lanes
    cmask |= miss
    if wrong.any():
        cr[wrong] = torch.randint(0, lanes + 1, (int(wrong.sum().item()),), device=device)
        cp[wrong] = torch.randint(0, p, (int(wrong.sum().item()),), device=device)
        cro[wrong] = torch.randint(0, lanes, (int(wrong.sum().item()),), device=device)
        cmask |= wrong

    # Corrupt write tokens at active target lanes and add extra writes sometimes.
    wmiss = (torch.rand_like(write) < prob * 0.4)
    cw[wmiss] = 2
    extra_w = (torch.rand_like(write) < float(args.extra_prob)) & (~active)
    cw[extra_w] = 1

    # Boundary corruption: missing true boundaries, extra boundaries, unknown tokens.
    bmiss = (torch.rand_like(boundary) < prob * 0.5) & (boundary > 0.5)
    bextra = (torch.rand_like(boundary) < float(args.extra_prob)) & (boundary < 0.5)
    bunknown = (torch.rand_like(boundary) < prob * 0.25)
    cb[bmiss] = 0
    cb[bextra] = 1
    cb[bunknown] = 2
    return cr, cp, cro, cw.clamp(0, 2), cb.clamp(0, 2), cmask


# ----------------------------- builder model -----------------------------

class ProgramBuilderStudent(nn.Module):
    def __init__(self, dim: int, steps: int, lanes: int, primitives: int, hidden: int = 192):
        super().__init__()
        self.dim = dim
        self.steps = steps
        self.lanes = lanes
        self.primitives = primitives
        self.hidden = hidden
        self.x_enc = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, hidden), nn.SiLU(), nn.Linear(hidden, hidden))
        self.step_emb = nn.Embedding(steps, hidden)
        self.lane_emb = nn.Embedding(lanes, hidden)
        self.read_emb = nn.Embedding(lanes + 2, hidden)       # groups + unknown
        self.prim_emb = nn.Embedding(primitives + 1, hidden)  # primitives + unknown
        self.route_emb = nn.Embedding(lanes + 1, hidden)      # lanes + unknown
        self.write_emb = nn.Embedding(3, hidden)              # 0/1/unknown
        self.boundary_emb = nn.Embedding(3, hidden)           # 0/1/unknown
        self.trunk = nn.Sequential(nn.LayerNorm(hidden), nn.Linear(hidden, hidden), nn.SiLU(), nn.Linear(hidden, hidden), nn.SiLU())
        self.read_head = nn.Linear(hidden, lanes + 1)
        self.prim_head = nn.Linear(hidden, primitives)
        self.route_head = nn.Linear(hidden, lanes)
        self.write_head = nn.Linear(hidden, 1)
        self.boundary_head = nn.Sequential(nn.LayerNorm(hidden), nn.Linear(hidden, hidden), nn.SiLU(), nn.Linear(hidden, 1))

    def forward(self, batch: ProgramBatch):
        bsz, steps, lanes = batch.read_t.shape
        device = batch.x.device
        xh = self.x_enc(batch.x).view(bsz, 1, 1, self.hidden)
        st = torch.arange(steps, device=device).view(1, steps, 1)
        ln = torch.arange(lanes, device=device).view(1, 1, lanes)
        h = xh
        h = h + self.step_emb(st)
        h = h + self.lane_emb(ln)
        h = h + self.read_emb(batch.corrupt_read.clamp(0, lanes + 1))
        h = h + self.prim_emb(batch.corrupt_prim.clamp(0, self.primitives))
        h = h + self.route_emb(batch.corrupt_route.clamp(0, lanes))
        h = h + self.write_emb(batch.corrupt_write.clamp(0, 2))
        h = h + self.boundary_emb(batch.corrupt_boundary.clamp(0, 2)).unsqueeze(2)
        h = self.trunk(h)
        read_logits = self.read_head(h)
        prim_logits = self.prim_head(h)
        route_logits = self.route_head(h)
        write_logits = self.write_head(h).squeeze(-1)
        step_h = h.mean(dim=2)
        boundary_logits = self.boundary_head(step_h).squeeze(-1)
        return {
            "read_logits": read_logits,
            "prim_logits": prim_logits,
            "route_logits": route_logits,
            "write_logits": write_logits,
            "boundary_logits": boundary_logits,
        }


# ----------------------------- losses/metrics -----------------------------

def masked_ce(logits, target, mask):
    if mask.sum().item() == 0:
        return logits.sum() * 0.0
    return F.cross_entropy(logits[mask], target[mask])


def compute_loss_and_metrics(model_out, batch: ProgramBatch, args):
    read_logits = model_out["read_logits"]
    prim_logits = model_out["prim_logits"]
    route_logits = model_out["route_logits"]
    write_logits = model_out["write_logits"]
    boundary_logits = model_out["boundary_logits"]
    active = batch.active_mask

    read_loss = masked_ce(read_logits, batch.read_t, active)
    prim_loss = masked_ce(prim_logits, batch.prim_t, active)
    route_loss = masked_ce(route_logits, batch.route_t, active)
    # supervise write over all lanes, but positives get a little more weight.
    write_loss = F.binary_cross_entropy_with_logits(write_logits, batch.write_t.float(), pos_weight=torch.tensor(2.0, device=batch.x.device))
    boundary_loss = F.binary_cross_entropy_with_logits(boundary_logits, batch.boundary_t.float(), pos_weight=torch.tensor(1.5, device=batch.x.device))

    read_prob = torch.softmax(read_logits, dim=-1)
    prim_prob = torch.softmax(prim_logits, dim=-1)
    route_prob = torch.softmax(route_logits, dim=-1)
    write_prob = torch.sigmoid(write_logits)
    boundary_prob = torch.sigmoid(boundary_logits)
    y_pred, aux = execute_soft_program(batch.x, read_prob, prim_prob, route_prob, write_prob, boundary_prob)
    exec_loss = F.mse_loss(y_pred, batch.target_y)

    # Program economy losses.
    boundary_count = boundary_prob.sum(dim=1)
    boundary_budget = (F.relu(boundary_count - float(args.boundary_max_peaks)).pow(2) + F.relu(float(args.boundary_min_peaks) - boundary_count).pow(2)).mean()
    boundary_exploit = ((boundary_count > (batch.boundary_t.shape[1] * 0.75)).float()).mean()
    eye = torch.eye(args.lanes, device=batch.x.device, dtype=route_prob.dtype).view(1, 1, args.lanes, args.lanes)
    offdiag = (route_prob * (1.0 - eye)).sum(dim=-1).mean(dim=-1)
    offdiag_without_boundary = (offdiag * (1.0 - boundary_prob)).mean()
    route_ent = -(route_prob.clamp_min(1e-8) * route_prob.clamp_min(1e-8).log()).sum(dim=-1).mean()
    route_entropy_cost = F.relu(route_ent - float(args.route_entropy_max)).pow(2)
    write_cost = write_prob.mean()

    loss = (
        float(args.lambda_trace) * (read_loss + prim_loss + route_loss + write_loss + boundary_loss)
        + float(args.lambda_exec) * exec_loss
        + float(args.lambda_boundary_budget) * boundary_budget
        + float(args.lambda_offdiag_no_boundary) * offdiag_without_boundary
        + float(args.lambda_route_entropy) * route_entropy_cost
        + float(args.lambda_write_cost) * write_cost
    )

    with torch.no_grad():
        pred_read = read_logits.argmax(dim=-1)
        pred_prim = prim_logits.argmax(dim=-1)
        pred_route = route_logits.argmax(dim=-1)
        pred_write = (write_prob > 0.5).float()
        pred_boundary = (boundary_prob > 0.5).float()
        denom = active.sum().clamp_min(1).float()
        read_acc = ((pred_read == batch.read_t) & active).sum().float() / denom
        prim_acc = ((pred_prim == batch.prim_t) & active).sum().float() / denom
        route_acc = ((pred_route == batch.route_t) & active).sum().float() / denom
        write_acc = (pred_write == batch.write_t).float().mean()
        boundary_acc = (pred_boundary == batch.boundary_t).float().mean()
        corrupt_positions = batch.corrupt_mask
        repair_acc = (((pred_read == batch.read_t) & (pred_prim == batch.prim_t) & (pred_route == batch.route_t) & corrupt_positions).sum().float() / corrupt_positions.sum().clamp_min(1).float())
        exact = ((pred_read == batch.read_t) | (~active)).all(dim=(1, 2)) & ((pred_prim == batch.prim_t) | (~active)).all(dim=(1, 2)) & ((pred_route == batch.route_t) | (~active)).all(dim=(1, 2))
        exact = exact.float().mean()
        metrics = {
            "loss": float(loss.detach().cpu()),
            "exec_loss": float(exec_loss.detach().cpu()),
            "read_loss": float(read_loss.detach().cpu()),
            "prim_loss": float(prim_loss.detach().cpu()),
            "route_loss": float(route_loss.detach().cpu()),
            "write_loss": float(write_loss.detach().cpu()),
            "boundary_loss": float(boundary_loss.detach().cpu()),
            "read_acc": float(read_acc.detach().cpu()),
            "prim_acc": float(prim_acc.detach().cpu()),
            "route_acc": float(route_acc.detach().cpu()),
            "write_acc": float(write_acc.detach().cpu()),
            "boundary_acc": float(boundary_acc.detach().cpu()),
            "repair_acc": float(repair_acc.detach().cpu()),
            "program_exact": float(exact.detach().cpu()),
            "boundary_budget": float(boundary_budget.detach().cpu()),
            "boundary_exploit_rate": float(boundary_exploit.detach().cpu()),
            "offdiag_without_boundary": float(offdiag_without_boundary.detach().cpu()),
            "route_entropy": float(route_ent.detach().cpu()),
            "write_mean": float(write_prob.mean().detach().cpu()),
            "boundary_mean": float(boundary_prob.mean().detach().cpu()),
            "boundary_count_mean": float(boundary_count.mean().detach().cpu()),
        }
    return loss, metrics, {"y_pred": y_pred.detach(), "aux": aux, "pred": (pred_read.detach(), pred_prim.detach(), pred_route.detach(), pred_write.detach(), pred_boundary.detach())}


@torch.no_grad()
def evaluate(model, args, device, stage: int, batches: int = 20):
    model.eval()
    sums: Dict[str, float] = {}
    n = 0
    examples = None
    for _ in range(batches):
        batch = make_teacher_batch(args, args.batch_size, device, stage)
        out = model(batch)
        _, metrics, extra = compute_loss_and_metrics(out, batch, args)
        for k, v in metrics.items():
            sums[k] = sums.get(k, 0.0) + float(v)
        n += 1
        if examples is None:
            examples = make_examples(batch, extra, args, max_examples=5)
    avg = {k: v / max(1, n) for k, v in sums.items()}
    return avg, examples or []


def make_examples(batch: ProgramBatch, extra, args, max_examples: int = 5):
    pred_read, pred_prim, pred_route, pred_write, pred_boundary = extra["pred"]
    out = []
    for i in range(min(max_examples, batch.x.shape[0])):
        steps = []
        for t in range(args.steps):
            active_lanes = torch.where(batch.active_mask[i, t])[0].tolist()
            step = {"t": int(t), "target_boundary": float(batch.boundary_t[i, t].cpu()), "pred_boundary": float(pred_boundary[i, t].cpu()), "lanes": []}
            for l in active_lanes:
                step["lanes"].append({
                    "lane": LANE_NAMES[l] if l < len(LANE_NAMES) else str(l),
                    "target": {
                        "read": READ_NAMES[int(batch.read_t[i, t, l].cpu())] if int(batch.read_t[i, t, l].cpu()) < len(READ_NAMES) else "?",
                        "primitive": PRIMITIVES[int(batch.prim_t[i, t, l].cpu())],
                        "route_to": LANE_NAMES[int(batch.route_t[i, t, l].cpu())],
                        "write_to": [LANE_NAMES[j] for j in torch.where(batch.write_t[i, t] > 0.5)[0].tolist()],
                    },
                    "pred": {
                        "read": READ_NAMES[int(pred_read[i, t, l].cpu())] if int(pred_read[i, t, l].cpu()) < len(READ_NAMES) else "?",
                        "primitive": PRIMITIVES[int(pred_prim[i, t, l].cpu())],
                        "route_to": LANE_NAMES[int(pred_route[i, t, l].cpu())],
                    },
                })
            steps.append(step)
        out.append({"task_id": int(batch.task_id[i].cpu()), "steps": steps})
    return out


# ----------------------------- train/report -----------------------------

def stage_for_epoch(epoch: int, args) -> int:
    # Gradual curriculum: simple -> chain -> repair/denoise/memory.
    if epoch <= args.stage0_epochs:
        return 0
    if epoch <= args.stage0_epochs + args.stage1_epochs:
        return 1
    if epoch <= args.stage0_epochs + args.stage1_epochs + args.stage2_epochs:
        return 2
    return 4


def append_csv(path: Path, row: Dict[str, float], header_written: bool):
    keys = list(row.keys())
    with path.open("a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        if not header_written:
            w.writeheader()
        w.writerow(row)


def train(args):
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    seed = int(args.seed)
    random.seed(seed)
    torch.manual_seed(seed)
    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)
    model = ProgramBuilderStudent(args.dim, args.steps, args.lanes, len(PRIMITIVES), args.hidden).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scaler = torch.cuda.amp.GradScaler(enabled=(device.type == "cuda" and args.amp == "fp16"))
    metrics_path = out_dir / "metrics.csv"
    header = False
    best = {"score": -1.0, "epoch": 0}
    t0 = time.time()

    for epoch in range(1, args.epochs + 1):
        stage = stage_for_epoch(epoch, args)
        model.train()
        sums: Dict[str, float] = {}
        for step in range(1, args.train_steps_per_epoch + 1):
            batch = make_teacher_batch(args, args.batch_size, device, stage)
            opt.zero_grad(set_to_none=True)
            use_amp = device.type == "cuda" and args.amp == "fp16"
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
                out = model(batch)
                loss, metrics, _ = compute_loss_and_metrics(out, batch, args)
            scaler.scale(loss).backward()
            if args.grad_clip > 0:
                scaler.unscale_(opt)
                nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            scaler.step(opt)
            scaler.update()
            for k, v in metrics.items():
                sums[k] = sums.get(k, 0.0) + float(v)
            if args.log_every and step % args.log_every == 0:
                print(f"epoch={epoch:03d} step={step:05d} stage={stage} loss={sums['loss']/step:.4f} prim_acc={sums['prim_acc']/step:.3f} route_acc={sums['route_acc']/step:.3f} boundary_acc={sums['boundary_acc']/step:.3f}", flush=True)
        train_avg = {"train_" + k: v / max(1, args.train_steps_per_epoch) for k, v in sums.items()}
        val, examples = evaluate(model, args, device, stage, args.eval_batches)
        row = {"epoch": epoch, "stage": stage}
        row.update(train_avg)
        row.update({"val_" + k: v for k, v in val.items()})
        append_csv(metrics_path, row, header)
        header = True
        score = val.get("program_exact", 0.0) + val.get("prim_acc", 0.0) + val.get("route_acc", 0.0) + val.get("boundary_acc", 0.0) - val.get("boundary_exploit_rate", 0.0)
        if score > best["score"]:
            best = {"score": float(score), "epoch": epoch, "val": val}
            if args.save_checkpoint:
                torch.save({"model": model.state_dict(), "args": vars(args), "primitives": PRIMITIVES, "lane_names": LANE_NAMES, "read_names": READ_NAMES}, out_dir / "program_builder_curriculum_v1_best.pt")
        (out_dir / f"examples_epoch_{epoch:03d}.json").write_text(json.dumps(examples, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"EPOCH {epoch:03d} stage={stage} val_exact={val['program_exact']:.3f} val_prim={val['prim_acc']:.3f} val_route={val['route_acc']:.3f} val_boundary={val['boundary_acc']:.3f} exploit={val['boundary_exploit_rate']:.3f}", flush=True)

    final = {
        "version": "program_assembly_curriculum_v1",
        "elapsed_sec": time.time() - t0,
        "best": best,
        "args": vars(args),
        "what_it_trains": ["read", "primitive", "route", "write", "boundary", "repair_missing", "denoise_extra", "real_soft_executor"],
        "not_yet": ["actor", "critic", "auto_deploy", "v4.4_weight_adapter", "causal_audio_memory_patch"],
    }
    (out_dir / "final_report.json").write_text(json.dumps(final, ensure_ascii=False, indent=2), encoding="utf-8")
    write_chatgpt_report(out_dir, final)
    print(f"DONE out_dir={out_dir} best_epoch={best['epoch']} best_score={best['score']:.4f}", flush=True)


def write_chatgpt_report(out_dir: Path, final: Dict):
    best = final.get("best", {})
    val = best.get("val", {}) or {}
    lines = [
        "REPORT_TO_CHATGPT: program_assembly_curriculum_v1",
        "",
        "Цель: учим сборщик программ на синтетике через реальные differentiable executions.",
        "Это не SpeechCommands и не actor/critic. Это школа для read/primitive/route/write/boundary controllers.",
        "",
        f"best_epoch: {best.get('epoch', 0)}",
        f"best_score: {best.get('score', 0.0):.4f}",
        f"program_exact: {val.get('program_exact', 0.0):.4f}",
        f"primitive_acc: {val.get('prim_acc', 0.0):.4f}",
        f"route_acc: {val.get('route_acc', 0.0):.4f}",
        f"read_acc: {val.get('read_acc', 0.0):.4f}",
        f"boundary_acc: {val.get('boundary_acc', 0.0):.4f}",
        f"repair_acc: {val.get('repair_acc', 0.0):.4f}",
        f"exec_loss: {val.get('exec_loss', 0.0):.6f}",
        f"boundary_exploit_rate: {val.get('boundary_exploit_rate', 0.0):.4f}",
        f"offdiag_without_boundary: {val.get('offdiag_without_boundary', 0.0):.4f}",
        "",
        "Смотреть:",
        "- metrics.csv",
        "- examples_epoch_XXX.json",
        "- final_report.json",
        "",
        "Критерии успеха smoke:",
        "- primitive_acc > 0.80",
        "- route_acc > 0.75",
        "- boundary_acc > 0.85",
        "- boundary_exploit_rate < 0.10",
        "- exec_loss падает",
    ]
    (out_dir / "REPORT_TO_CHATGPT.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")


def parser():
    p = argparse.ArgumentParser()
    p.add_argument("--out-dir", type=str, default="simple_butterfly_matrix_v4_tape_lane/agent_reports/program_assembly_curriculum_v1_smoke")
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--amp", type=str, default="fp16", choices=["fp16", "off"])
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--dim", type=int, default=32)
    p.add_argument("--hidden", type=int, default=192)
    p.add_argument("--steps", type=int, default=6)
    p.add_argument("--lanes", type=int, default=4)
    p.add_argument("--epochs", type=int, default=8)
    p.add_argument("--batch-size", type=int, default=512)
    p.add_argument("--train-steps-per-epoch", type=int, default=250)
    p.add_argument("--eval-batches", type=int, default=30)
    p.add_argument("--lr", type=float, default=8e-4)
    p.add_argument("--weight-decay", type=float, default=0.01)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--log-every", type=int, default=50)
    p.add_argument("--stage0-epochs", type=int, default=2)
    p.add_argument("--stage1-epochs", type=int, default=2)
    p.add_argument("--stage2-epochs", type=int, default=2)
    p.add_argument("--corrupt-prob", type=float, default=0.35)
    p.add_argument("--wrong-prob", type=float, default=0.20)
    p.add_argument("--extra-prob", type=float, default=0.08)
    p.add_argument("--signal-structure", action="store_true", default=True)
    p.add_argument("--lambda-trace", type=float, default=1.0)
    p.add_argument("--lambda-exec", type=float, default=1.0)
    p.add_argument("--lambda-boundary-budget", type=float, default=0.25)
    p.add_argument("--lambda-offdiag-no-boundary", type=float, default=0.15)
    p.add_argument("--lambda-route-entropy", type=float, default=0.05)
    p.add_argument("--lambda-write-cost", type=float, default=0.01)
    p.add_argument("--boundary-min-peaks", type=float, default=1.0)
    p.add_argument("--boundary-max-peaks", type=float, default=3.0)
    p.add_argument("--route-entropy-max", type=float, default=1.25)
    p.add_argument("--save-checkpoint", action="store_true")
    return p


if __name__ == "__main__":
    train(parser().parse_args())
