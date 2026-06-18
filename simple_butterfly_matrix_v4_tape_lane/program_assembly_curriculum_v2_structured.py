#!/usr/bin/env python3
"""
program_assembly_curriculum_v2_structured.py

Separate curriculum/training file. It does not touch v4.3/v4.4 main code.

Purpose:
  1) synthetic program assembly school: repeat / repair / denoise / mixed
  2) structured head/input data school: convert traces from real heads/layers into program labels
  3) quality-filter examples before training
  4) keep program-builder pretraining separate from normal task training

Supported modes:
  --mode train_online       generate examples on the fly and train
  --mode generate_jsonl     write normalized program examples to JSONL
  --mode inspect_jsonl      print/validate examples from JSONL
  --mode train_jsonl        train from normalized JSONL or raw structured head JSONL

Raw structured input format is intentionally flexible. Records may contain either:
  teacher_program.steps[] already normalized, or head_trace / structured_trace fields.

Example structured record:
{
  "x": [.. optional float vector ..],
  "target_y": [.. optional float vector ..],
  "head_trace": {
    "boundary_peaks": [1,3],
    "steps": [
      {"t":0,"read":"input","primitive_scores":{"diff_prev":0.9},"route_scores":{"detail->state":0.8},"write":"state"},
      {"t":1,"read":"state","primitive":"smooth3","route":"state->abstract","write":"abstract"}
    ]
  }
}
"""

from __future__ import annotations

import argparse, csv, json, math, random, time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

PRIMITIVES = ["identity", "shift_left", "shift_right", "diff_prev", "smooth3", "contrast", "gate_state", "memory_keep"]
LANE_NAMES = ["detail", "state", "abstract", "memory"]
READ_NAMES = ["input", "detail", "state", "abstract", "memory"]
MODE_NAMES = ["repeat", "repair", "denoise", "mixed", "structured"]
P_IDENTITY, P_SHIFT_LEFT, P_SHIFT_RIGHT, P_DIFF, P_SMOOTH, P_CONTRAST, P_GATE, P_MEMORY = range(8)
LANE = {n: i for i, n in enumerate(LANE_NAMES)}
READ = {n: i for i, n in enumerate(READ_NAMES)}
PRIM = {n: i for i, n in enumerate(PRIMITIVES)}

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
    memory = state[:, 3:4] if state.shape[1] >= 4 else state[:, -1:]
    memory = memory.expand(-1, state.shape[1], -1)
    mean = read_vec.mean(dim=-1, keepdim=True)
    smooth = (torch.roll(read_vec, 1, -1) + read_vec + torch.roll(read_vec, -1, -1)) / 3.0
    outs = [read_vec, torch.roll(read_vec, -1, -1), torch.roll(read_vec, 1, -1), read_vec - torch.roll(read_vec, 1, -1), smooth, read_vec - mean, torch.sigmoid(state) * read_vec, 0.55 * read_vec + 0.45 * memory]
    return torch.stack(outs, dim=2)

def execute_soft_program(x, read_prob, prim_prob, route_prob, write_prob, boundary_prob):
    bsz, steps, lanes, _ = read_prob.shape
    dim = x.shape[-1]
    state = torch.zeros(bsz, lanes, dim, device=x.device, dtype=x.dtype)
    state[:, 0] = x
    eye = torch.eye(lanes, device=x.device, dtype=x.dtype).view(1, lanes, lanes)
    route_entropy, boundary_used, history = [], [], [state]
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
        ent = -(route_prob[:, t].clamp_min(1e-8) * route_prob[:, t].clamp_min(1e-8).log()).sum(-1).mean(-1)
        route_entropy.append(ent)
    return state[:, 2] + 0.25 * state[:, 3], {"state_history": torch.stack(history, 1), "boundary_used": torch.stack(boundary_used, 1), "route_entropy": torch.stack(route_entropy, 1)}

def hard_to_probs(read_t, prim_t, route_t, write_t, boundary_t, lanes, primitives):
    return F.one_hot(read_t, lanes + 1).float(), F.one_hot(prim_t, primitives).float(), F.one_hot(route_t, lanes).float(), write_t.float(), boundary_t.float()

def safe_corrupted_probs(batch: ProgramBatch, lanes: int, primitives: int):
    cr, cp, cro = batch.corrupt_read.clone(), batch.corrupt_prim.clone(), batch.corrupt_route.clone()
    cw, cb = batch.corrupt_write.clone(), batch.corrupt_boundary.clone()
    for l in range(lanes):
        cr[..., l] = torch.where(cr[..., l] > lanes, torch.full_like(cr[..., l], 1 + l), cr[..., l])
        cro[..., l] = torch.where(cro[..., l] >= lanes, torch.full_like(cro[..., l], l), cro[..., l])
    cp = torch.where(cp >= primitives, torch.zeros_like(cp), cp)
    cw = torch.where(cw > 1, torch.zeros_like(cw), cw).float()
    cb = torch.where(cb > 1, torch.zeros_like(cb), cb).float()
    return hard_to_probs(cr.clamp(0, lanes), cp.clamp(0, primitives - 1), cro.clamp(0, lanes - 1), cw, cb, lanes, primitives)

# ----------------------------- program construction -----------------------------

def defaults(bsz, steps, lanes, device):
    read = torch.zeros(bsz, steps, lanes, dtype=torch.long, device=device)
    prim = torch.zeros(bsz, steps, lanes, dtype=torch.long, device=device)
    route = torch.arange(lanes, device=device).view(1, 1, lanes).expand(bsz, steps, lanes).clone()
    write = torch.zeros(bsz, steps, lanes, dtype=torch.float32, device=device)
    boundary = torch.zeros(bsz, steps, dtype=torch.float32, device=device)
    active = torch.zeros(bsz, steps, lanes, dtype=torch.bool, device=device)
    for l in range(lanes):
        read[:, :, l] = 1 + l
    return read, prim, route, write, boundary, active

def set_step(read, prim, route, write, boundary, active, b, t, src_lane, read_group, primitive, target_lane, boundary_on=1.0):
    read[b, t, src_lane] = int(read_group)
    prim[b, t, src_lane] = int(primitive)
    route[b, t, src_lane] = int(target_lane)
    write[b, t, target_lane] = 1.0
    boundary[b, t] = float(boundary_on)
    active[b, t, src_lane] = True

def parse_lane(v, default=0):
    if isinstance(v, int): return max(0, min(len(LANE_NAMES) - 1, v))
    return LANE.get(str(v), default)

def parse_read(v, default=0):
    if isinstance(v, int): return max(0, min(len(READ_NAMES) - 1, v))
    return READ.get(str(v), default)

def parse_prim(v, default=0):
    if isinstance(v, int): return max(0, min(len(PRIMITIVES) - 1, v))
    return PRIM.get(str(v), default)

def parse_route(v, default_src=0):
    if isinstance(v, int): return max(0, min(len(LANE_NAMES) - 1, v))
    s = str(v)
    if "->" in s: return parse_lane(s.split("->", 1)[1], default_src)
    return parse_lane(s, default_src)

def best_key(d: Dict, default: str):
    if not isinstance(d, dict) or not d: return default
    return max(d.items(), key=lambda kv: float(kv[1]))[0]

def infer_steps_from_structured(rec: Dict, max_steps: int) -> List[Dict]:
    # 1) explicit normalized teacher program wins
    tp = rec.get("teacher_program") or rec.get("program") or {}
    if isinstance(tp, dict) and isinstance(tp.get("steps"), list):
        return tp["steps"][:max_steps]
    if isinstance(tp, list):
        return tp[:max_steps]
    # 2) infer from head_trace/structured_trace
    trace = rec.get("head_trace") or rec.get("structured_trace") or rec.get("trace") or {}
    raw_steps = trace.get("steps") or rec.get("steps") or []
    boundary_peaks = set(int(x) for x in trace.get("boundary_peaks", rec.get("boundary_peaks", [])) if int(x) < max_steps)
    out = []
    for i, st in enumerate(raw_steps[:max_steps]):
        t = int(st.get("t", i))
        prim = st.get("primitive", best_key(st.get("primitive_scores", st.get("operator_scores", {})), "identity"))
        route = st.get("route", best_key(st.get("route_scores", {}), "detail->state"))
        write = st.get("write", None)
        if write is None:
            write = route.split("->", 1)[1] if isinstance(route, str) and "->" in route else "state"
        read = st.get("read", st.get("read_source", "input" if t == 0 else "state"))
        src = st.get("lane", st.get("src_lane", "detail" if t == 0 else "state"))
        # boundary if explicit, peak, or deltas/contribution says step transition.
        b = st.get("boundary", None)
        if b is None:
            delta = float(st.get("route_delta", 0.0)) + float(st.get("read_delta", 0.0)) + float(st.get("primitive_delta", 0.0)) + float(st.get("trace_delta", 0.0))
            b = 1.0 if (t in boundary_peaks or delta > 0.05 or st.get("patch_effect", 0.0) > 0.01) else 0.0
        out.append({"t": t, "lane": src, "read": read, "primitive": prim, "route": route, "write": write, "boundary": b})
    return out[:max_steps]

def record_to_program_tensors(rec: Dict, args, device) -> Optional[Tuple[torch.Tensor, ...]]:
    lanes, steps, dim, p = args.lanes, args.steps, args.dim, len(PRIMITIVES)
    read, prim, route, write, boundary, active = defaults(1, steps, lanes, device)
    raw_x = rec.get("x", rec.get("input", None))
    if raw_x is None:
        x = torch.randn(1, dim, device=device)
    else:
        x0 = torch.tensor(raw_x, dtype=torch.float32, device=device).flatten()
        if x0.numel() < dim: x0 = F.pad(x0, (0, dim - x0.numel()))
        x = x0[:dim].view(1, dim)
    steps_list = infer_steps_from_structured(rec, steps)
    if not steps_list: return None
    for st in steps_list:
        t = max(0, min(steps - 1, int(st.get("t", 0))))
        src = parse_lane(st.get("lane", st.get("src_lane", "detail")), 0)
        r = parse_read(st.get("read", "input" if t == 0 else LANE_NAMES[src]), 0)
        pr = parse_prim(st.get("primitive", "identity"), 0)
        rt = parse_route(st.get("route", st.get("write", src)), src)
        wr = parse_lane(st.get("write", rt), rt)
        b = float(st.get("boundary", 1.0))
        set_step(read, prim, route, write, boundary, active, 0, t, src, r, pr, rt, b)
        write[0, t, wr] = 1.0
    if not quality_ok(read, prim, route, write, boundary, active, args): return None
    target_y = rec.get("target_y", None)
    if target_y is None:
        with torch.no_grad(): y, _ = execute_soft_program(x, *hard_to_probs(read, prim, route, write, boundary, lanes, p))
    else:
        y0 = torch.tensor(target_y, dtype=torch.float32, device=device).flatten()
        if y0.numel() < dim: y0 = F.pad(y0, (0, dim - y0.numel()))
        y = y0[:dim].view(1, dim)
    return x, y.detach(), read, prim, route, write, boundary, active

def quality_ok(read, prim, route, write, boundary, active, args) -> bool:
    bc = float(boundary.sum().item())
    if bc < args.quality_min_boundary or bc > args.quality_max_boundary: return False
    lanes = route.shape[-1]
    eye = torch.arange(lanes, device=route.device).view(1, 1, lanes)
    active_route = route[active]
    if active_route.numel() == 0: return False
    # Reject all-self and all-cross/all-boundary garbage.
    self_share = float((route == eye).float()[active].mean().item()) if active.any() else 1.0
    if self_share > args.quality_max_self_route: return False
    if float(active.float().mean().item()) > args.quality_max_active_density: return False
    return True

# ----------------------------- batches -----------------------------

def synthetic_batch(args, bsz, device, stage):
    lanes, steps, dim = args.lanes, args.steps, args.dim
    x = torch.randn(bsz, dim, device=device)
    if args.signal_structure:
        grid = torch.linspace(-1, 1, dim, device=device).view(1, dim)
        freq = torch.randint(1, 5, (bsz, 1), device=device).float()
        x = x + 0.35 * torch.sin(freq * math.pi * grid)
    read, prim, route, write, boundary, active = defaults(bsz, steps, lanes, device)
    max_task = min(5, max(2, stage + 2))
    task_id = torch.randint(0, max_task, (bsz,), device=device)
    if stage <= 0: mode_id = torch.zeros(bsz, dtype=torch.long, device=device)
    elif stage == 1: mode_id = torch.randint(0, 2, (bsz,), device=device)
    elif stage == 2: mode_id = torch.randint(0, 3, (bsz,), device=device)
    else: mode_id = torch.randint(0, 4, (bsz,), device=device)
    for b in range(bsz):
        tid = int(task_id[b])
        if tid == 0:
            set_step(read, prim, route, write, boundary, active, b, 0, 0, 0, P_DIFF, 1, 1)
            set_step(read, prim, route, write, boundary, active, b, 1, 1, 2, P_SMOOTH, 2, 1)
        elif tid == 1:
            set_step(read, prim, route, write, boundary, active, b, 0, 0, 0, P_SHIFT_LEFT, 1, 1)
            set_step(read, prim, route, write, boundary, active, b, 1, 1, 2, P_CONTRAST, 2, 1)
        elif tid == 2:
            set_step(read, prim, route, write, boundary, active, b, 0, 0, 0, P_SMOOTH, 3, 1)
            set_step(read, prim, route, write, boundary, active, b, 1, 0, 0, P_DIFF, 1, 1)
            set_step(read, prim, route, write, boundary, active, b, 2, 3, 4, P_MEMORY, 2, 1)
        elif tid == 3:
            set_step(read, prim, route, write, boundary, active, b, 0, 0, 0, P_CONTRAST, 1, 1)
            set_step(read, prim, route, write, boundary, active, b, 1, 1, 2, P_GATE, 2, 1)
            set_step(read, prim, route, write, boundary, active, b, 2, 2, 3, P_SMOOTH, 3, 1)
            if steps > 3: set_step(read, prim, route, write, boundary, active, b, 3, 3, 4, P_MEMORY, 2, 1)
        else:
            set_step(read, prim, route, write, boundary, active, b, 0, 0, 0, P_SHIFT_RIGHT, 1, 1)
            set_step(read, prim, route, write, boundary, active, b, 1, 1, 2, P_DIFF, 2, 1)
            set_step(read, prim, route, write, boundary, active, b, 2, 2, 3, P_CONTRAST, 3, 1)
            if steps > 3: set_step(read, prim, route, write, boundary, active, b, 3, 3, 4, P_MEMORY, 2, 1)
    with torch.no_grad(): y, _ = execute_soft_program(x, *hard_to_probs(read, prim, route, write, boundary, lanes, len(PRIMITIVES)))
    return make_batch_from_tensors(x, y.detach(), read, prim, route, write, boundary, active, mode_id, task_id, args)

def corrupt_program(read, prim, route, write, boundary, active, mode_id, args):
    lanes, p = read.shape[-1], len(PRIMITIVES)
    cr, cp, cro = read.clone(), prim.clone(), route.clone()
    cw, cb = write.long().clone(), boundary.long().clone()
    cmask, wmask, bmask = torch.zeros_like(active), torch.zeros_like(write, dtype=torch.bool), torch.zeros_like(boundary, dtype=torch.bool)
    bsz, steps, _ = read.shape; device = read.device
    for b in range(bsz):
        mode = int(mode_id[b])
        miss_p = 0.0 if mode == 0 else float(args.corrupt_prob)
        wrong_p = 0.0 if mode == 0 else float(args.wrong_prob)
        extra_p = 0.0 if mode in (0, 1) else float(args.extra_prob)
        rand = torch.rand(steps, lanes, device=device)
        miss = (rand < miss_p) & active[b]
        wrong = (rand >= miss_p) & (rand < miss_p + wrong_p) & active[b]
        cr[b][miss] = lanes + 1; cp[b][miss] = p; cro[b][miss] = lanes; cmask[b] |= miss
        if wrong.any():
            n = int(wrong.sum())
            cr[b][wrong] = torch.randint(0, lanes + 1, (n,), device=device)
            cp[b][wrong] = torch.randint(0, p, (n,), device=device)
            cro[b][wrong] = torch.randint(0, lanes, (n,), device=device)
            cmask[b] |= wrong
        if extra_p > 0:
            extra_w = (torch.rand(steps, lanes, device=device) < extra_p) & (~active[b])
            cw[b][extra_w] = 1; wmask[b] |= extra_w
            extra_b = (torch.rand(steps, device=device) < extra_p) & (boundary[b] < 0.5)
            cb[b][extra_b] = 1; bmask[b] |= extra_b
            extra_r = (torch.rand(steps, lanes, device=device) < extra_p) & (~active[b])
            n = int(extra_r.sum())
            if n:
                cr[b][extra_r] = torch.randint(0, lanes + 1, (n,), device=device)
                cp[b][extra_r] = torch.randint(0, p, (n,), device=device)
                cro[b][extra_r] = torch.randint(0, lanes, (n,), device=device)
                cmask[b] |= extra_r
        if mode in (1, 3, 4):
            unk_w = torch.rand(steps, lanes, device=device) < miss_p * 0.25
            cw[b][unk_w] = 2; wmask[b] |= unk_w
            unk_b = torch.rand(steps, device=device) < miss_p * 0.25
            cb[b][unk_b] = 2; bmask[b] |= unk_b
    return cr.clamp(0, lanes + 1), cp.clamp(0, p), cro.clamp(0, lanes), cw.clamp(0, 2), cb.clamp(0, 2), cmask, wmask, bmask

def make_batch_from_tensors(x, y, read, prim, route, write, boundary, active, mode_id, task_id, args):
    corrupt = corrupt_program(read, prim, route, write, boundary, active, mode_id, args)
    return ProgramBatch(x, y, read, prim, route, write, boundary, active, corrupt[0], corrupt[1], corrupt[2], corrupt[3], corrupt[4], corrupt[5], corrupt[6], corrupt[7], mode_id, task_id)

def load_jsonl_records(path: str) -> List[Dict]:
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line: records.append(json.loads(line))
    return records

def structured_batch(records: List[Dict], args, bsz, device) -> ProgramBatch:
    xs, ys, reads, prims, routes, writes, boundaries, actives, modes, tasks = [], [], [], [], [], [], [], [], [], []
    tries = 0
    while len(xs) < bsz and tries < bsz * 20:
        tries += 1
        rec = random.choice(records)
        parsed = record_to_program_tensors(rec, args, device)
        if parsed is None: continue
        x, y, read, prim, route, write, boundary, active = parsed
        xs.append(x); ys.append(y); reads.append(read); prims.append(prim); routes.append(route); writes.append(write); boundaries.append(boundary); actives.append(active)
        modes.append(torch.full((1,), 4, dtype=torch.long, device=device)); tasks.append(torch.full((1,), int(rec.get("task_id", 999)), dtype=torch.long, device=device))
    if not xs: return synthetic_batch(args, bsz, device, 3)
    return make_batch_from_tensors(torch.cat(xs, 0), torch.cat(ys, 0), torch.cat(reads, 0), torch.cat(prims, 0), torch.cat(routes, 0), torch.cat(writes, 0), torch.cat(boundaries, 0), torch.cat(actives, 0), torch.cat(modes, 0), torch.cat(tasks, 0), args)

# ----------------------------- model/loss -----------------------------

class ProgramBuilderStudent(nn.Module):
    def __init__(self, dim, steps, lanes, primitives, hidden=192):
        super().__init__(); self.dim=dim; self.steps=steps; self.lanes=lanes; self.primitives=primitives; self.hidden=hidden
        self.x_enc=nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, hidden), nn.SiLU(), nn.Linear(hidden, hidden))
        self.step_emb=nn.Embedding(steps, hidden); self.lane_emb=nn.Embedding(lanes, hidden); self.mode_emb=nn.Embedding(len(MODE_NAMES), hidden)
        self.read_emb=nn.Embedding(lanes+2, hidden); self.prim_emb=nn.Embedding(primitives+1, hidden); self.route_emb=nn.Embedding(lanes+1, hidden)
        self.write_emb=nn.Embedding(3, hidden); self.boundary_emb=nn.Embedding(3, hidden)
        self.trunk=nn.Sequential(nn.LayerNorm(hidden), nn.Linear(hidden, hidden), nn.SiLU(), nn.Linear(hidden, hidden), nn.SiLU())
        self.read_head=nn.Linear(hidden, lanes+1); self.prim_head=nn.Linear(hidden, primitives); self.route_head=nn.Linear(hidden, lanes); self.write_head=nn.Linear(hidden, 1)
        self.boundary_head=nn.Sequential(nn.LayerNorm(hidden), nn.Linear(hidden, hidden), nn.SiLU(), nn.Linear(hidden, 1))
    def forward(self, batch):
        bsz, steps, lanes = batch.read_t.shape; device=batch.x.device
        h=self.x_enc(batch.x).view(bsz,1,1,self.hidden)
        h=h+self.step_emb(torch.arange(steps,device=device).view(1,steps,1))+self.lane_emb(torch.arange(lanes,device=device).view(1,1,lanes))
        h=h+self.mode_emb(batch.mode_id.clamp(0,len(MODE_NAMES)-1)).view(bsz,1,1,self.hidden)
        h=h+self.read_emb(batch.corrupt_read.clamp(0,lanes+1))+self.prim_emb(batch.corrupt_prim.clamp(0,self.primitives))+self.route_emb(batch.corrupt_route.clamp(0,lanes))
        h=h+self.write_emb(batch.corrupt_write.clamp(0,2))+self.boundary_emb(batch.corrupt_boundary.clamp(0,2)).unsqueeze(2)
        h=self.trunk(h)
        return {"read_logits":self.read_head(h),"prim_logits":self.prim_head(h),"route_logits":self.route_head(h),"write_logits":self.write_head(h).squeeze(-1),"boundary_logits":self.boundary_head(h.mean(2)).squeeze(-1)}

def weighted_ce(logits, target, weight):
    loss=F.cross_entropy(logits.reshape(-1,logits.shape[-1]), target.reshape(-1), reduction="none"); w=weight.reshape(-1).float()
    return (loss*w).sum()/w.sum().clamp_min(1.0)

def binary_f1(pred,target):
    pred=pred.float(); target=target.float(); tp=(pred*target).sum(); fp=(pred*(1-target)).sum(); fn=((1-pred)*target).sum()
    prec=tp/(tp+fp).clamp_min(1.0); rec=tp/(tp+fn).clamp_min(1.0); return prec, rec, 2*prec*rec/(prec+rec).clamp_min(1e-6)

def compute_loss(model_out,batch,args):
    read_logits,prim_logits,route_logits=model_out["read_logits"],model_out["prim_logits"],model_out["route_logits"]
    write_logits,boundary_logits=model_out["write_logits"],model_out["boundary_logits"]
    active=batch.active_mask.float(); token_weight=active+float(args.inactive_token_weight)*(1-active)
    read_loss=weighted_ce(read_logits,batch.read_t,token_weight); prim_loss=weighted_ce(prim_logits,batch.prim_t,token_weight); route_loss=weighted_ce(route_logits,batch.route_t,token_weight)
    write_loss=F.binary_cross_entropy_with_logits(write_logits,batch.write_t.float(),pos_weight=torch.tensor(float(args.write_pos_weight),device=batch.x.device))
    boundary_loss=F.binary_cross_entropy_with_logits(boundary_logits,batch.boundary_t.float(),pos_weight=torch.tensor(float(args.boundary_pos_weight),device=batch.x.device))
    read_prob=torch.softmax(read_logits,-1); prim_prob=torch.softmax(prim_logits,-1); route_prob=torch.softmax(route_logits,-1); write_prob=torch.sigmoid(write_logits); boundary_prob=torch.sigmoid(boundary_logits)
    y_pred,_=execute_soft_program(batch.x,read_prob,prim_prob,route_prob,write_prob,boundary_prob); exec_loss=F.mse_loss(y_pred,batch.target_y)
    boundary_count=boundary_prob.sum(1); boundary_budget=(F.relu(boundary_count-float(args.boundary_max_peaks)).pow(2)+F.relu(float(args.boundary_min_peaks)-boundary_count).pow(2)).mean()
    eye=torch.eye(args.lanes,device=batch.x.device,dtype=route_prob.dtype).view(1,1,args.lanes,args.lanes); offdiag=(route_prob*(1-eye)).sum(-1).mean(-1); offdiag_no_boundary=(offdiag*(1-boundary_prob)).mean()
    route_ent=-(route_prob.clamp_min(1e-8)*route_prob.clamp_min(1e-8).log()).sum(-1).mean(); route_ent_cost=F.relu(route_ent-float(args.route_entropy_max)).pow(2)
    loss=float(args.lambda_trace)*(read_loss+prim_loss+route_loss+write_loss+boundary_loss)+float(args.lambda_exec)*exec_loss+float(args.lambda_boundary_budget)*boundary_budget+float(args.lambda_offdiag_no_boundary)*offdiag_no_boundary+float(args.lambda_route_entropy)*route_ent_cost+float(args.lambda_write_cost)*write_prob.mean()
    with torch.no_grad():
        pr=read_logits.argmax(-1); pp=prim_logits.argmax(-1); pro=route_logits.argmax(-1); pw=(write_prob>0.5).float(); pb=(boundary_prob>0.5).float(); denom=batch.active_mask.sum().clamp_min(1).float()
        bprec,brec,bf1=binary_f1(pb,batch.boundary_t)
        corrupted_y,_=execute_soft_program(batch.x,*safe_corrupted_probs(batch,args.lanes,len(PRIMITIVES))); corr_loss=F.mse_loss(corrupted_y,batch.target_y); improve=corr_loss-exec_loss
        corrupt_positions=batch.corrupt_mask
        metrics={"loss":float(loss.cpu()),"exec_loss":float(exec_loss.cpu()),"corrupted_exec_loss":float(corr_loss.cpu()),"exec_improvement":float(improve.cpu()),"read_acc":float((((pr==batch.read_t)&batch.active_mask).sum().float()/denom).cpu()),"prim_acc":float((((pp==batch.prim_t)&batch.active_mask).sum().float()/denom).cpu()),"route_acc":float((((pro==batch.route_t)&batch.active_mask).sum().float()/denom).cpu()),"write_acc":float((pw==batch.write_t).float().mean().cpu()),"boundary_acc":float((pb==batch.boundary_t).float().mean().cpu()),"boundary_f1":float(bf1.cpu()),"boundary_precision":float(bprec.cpu()),"boundary_recall":float(brec.cpu()),"repair_acc":float(((((pr==batch.read_t)&(pp==batch.prim_t)&(pro==batch.route_t)&corrupt_positions).sum().float())/corrupt_positions.sum().clamp_min(1).float()).cpu()),"active_program_exact":float((((pr==batch.read_t)|(~batch.active_mask)).all((1,2))&((pp==batch.prim_t)|(~batch.active_mask)).all((1,2))&((pro==batch.route_t)|(~batch.active_mask)).all((1,2))).float().mean().cpu()),"full_program_exact":float(((pr==batch.read_t).all((1,2))&(pp==batch.prim_t).all((1,2))&(pro==batch.route_t).all((1,2))&(pw==batch.write_t).all((1,2))&(pb==batch.boundary_t).all(1)).float().mean().cpu()),"boundary_exploit_rate":float((boundary_count>(batch.boundary_t.shape[1]*0.75)).float().mean().cpu()),"boundary_mean":float(boundary_prob.mean().cpu()),"boundary_count_mean":float(boundary_count.mean().cpu()),"offdiag_without_boundary":float(offdiag_no_boundary.cpu()),"route_entropy":float(route_ent.cpu()),"write_mean":float(write_prob.mean().cpu())}
    return loss, metrics, {"pred":(pr,pp,pro,pw,pb)}

# ----------------------------- train / dataset modes -----------------------------

def stage_for_epoch(epoch,args):
    if epoch<=args.stage0_epochs: return 0
    if epoch<=args.stage0_epochs+args.stage1_epochs: return 1
    if epoch<=args.stage0_epochs+args.stage1_epochs+args.stage2_epochs: return 2
    return 4

def get_train_batch(args,device,stage,records=None):
    if records and args.task_source in ("structured","mixed") and (args.task_source=="structured" or random.random()<float(args.structured_mix_prob)):
        return structured_batch(records,args,args.batch_size,device)
    return synthetic_batch(args,args.batch_size,device,stage)

def append_csv(path,row,header):
    with open(path,"a",newline="",encoding="utf-8") as f:
        w=csv.DictWriter(f,fieldnames=list(row.keys()))
        if not header: w.writeheader()
        w.writerow(row)

def examples_from_batch(batch,extra,args,max_examples=5):
    pr,pp,pro,pw,pb=extra["pred"]; out=[]
    for i in range(min(max_examples,batch.x.shape[0])):
        steps=[]
        for t in range(args.steps):
            lanes=[]
            for l in torch.where(batch.active_mask[i,t])[0].tolist():
                lanes.append({"lane":LANE_NAMES[l],"target":{"read":READ_NAMES[int(batch.read_t[i,t,l])],"primitive":PRIMITIVES[int(batch.prim_t[i,t,l])],"route_to":LANE_NAMES[int(batch.route_t[i,t,l])]},"pred":{"read":READ_NAMES[int(pr[i,t,l])],"primitive":PRIMITIVES[int(pp[i,t,l])],"route_to":LANE_NAMES[int(pro[i,t,l])]}})
            steps.append({"t":t,"target_boundary":float(batch.boundary_t[i,t].cpu()),"pred_boundary":float(pb[i,t].cpu()),"lanes":lanes})
        out.append({"mode":MODE_NAMES[int(batch.mode_id[i])],"task_id":int(batch.task_id[i]),"steps":steps})
    return out

@torch.no_grad()
def evaluate(model,args,device,stage,records=None):
    model.eval(); sums={}; examples=None
    for _ in range(args.eval_batches):
        batch=get_train_batch(args,device,stage,records); loss,metrics,extra=compute_loss(model(batch),batch,args)
        for k,v in metrics.items(): sums[k]=sums.get(k,0.0)+float(v)
        if examples is None: examples=examples_from_batch(batch,extra,args)
    return {k:v/max(1,args.eval_batches) for k,v in sums.items()}, examples or []

def train(args):
    out=Path(args.out_dir); out.mkdir(parents=True,exist_ok=True); random.seed(args.seed); torch.manual_seed(args.seed)
    device=torch.device(args.device if (args.device=="cpu" or torch.cuda.is_available()) else "cpu")
    records=load_jsonl_records(args.structured_jsonl) if args.structured_jsonl else None
    model=ProgramBuilderStudent(args.dim,args.steps,args.lanes,len(PRIMITIVES),args.hidden).to(device); opt=torch.optim.AdamW(model.parameters(),lr=args.lr,weight_decay=args.weight_decay)
    scaler=torch.cuda.amp.GradScaler(enabled=(device.type=="cuda" and args.amp=="fp16")); header=False; best={"score":-1,"epoch":0}; t0=time.time()
    for epoch in range(1,args.epochs+1):
        stage=stage_for_epoch(epoch,args); model.train(); sums={}
        for step in range(1,args.train_steps_per_epoch+1):
            batch=get_train_batch(args,device,stage,records); opt.zero_grad(set_to_none=True); use_amp=device.type=="cuda" and args.amp=="fp16"
            with torch.autocast(device_type=device.type,dtype=torch.float16,enabled=use_amp): loss,metrics,_=compute_loss(model(batch),batch,args)
            scaler.scale(loss).backward(); scaler.unscale_(opt); nn.utils.clip_grad_norm_(model.parameters(),args.grad_clip); scaler.step(opt); scaler.update()
            for k,v in metrics.items(): sums[k]=sums.get(k,0.0)+float(v)
            if args.log_every and step%args.log_every==0: print(f"epoch={epoch:03d} step={step:05d} stage={stage} loss={sums['loss']/step:.4f} prim={sums['prim_acc']/step:.3f} route={sums['route_acc']/step:.3f} bF1={sums['boundary_f1']/step:.3f} exact={sums['full_program_exact']/step:.3f}",flush=True)
        val,examples=evaluate(model,args,device,stage,records); row={"epoch":epoch,"stage":stage}; row.update({"train_"+k:v/max(1,args.train_steps_per_epoch) for k,v in sums.items()}); row.update({"val_"+k:v for k,v in val.items()}); append_csv(out/"metrics.csv",row,header); header=True
        score=val.get("full_program_exact",0)+val.get("prim_acc",0)+val.get("route_acc",0)+val.get("boundary_f1",0)+max(0,val.get("exec_improvement",0))-val.get("boundary_exploit_rate",0)
        if score>best["score"]:
            best={"score":float(score),"epoch":epoch,"val":val}
            if args.save_checkpoint: torch.save({"model":model.state_dict(),"args":vars(args),"primitives":PRIMITIVES,"lanes":LANE_NAMES,"reads":READ_NAMES,"modes":MODE_NAMES},out/"program_builder_curriculum_v2_best.pt")
        (out/f"examples_epoch_{epoch:03d}.json").write_text(json.dumps(examples,ensure_ascii=False,indent=2),encoding="utf-8")
        print(f"EPOCH {epoch:03d} stage={stage} full_exact={val['full_program_exact']:.3f} prim={val['prim_acc']:.3f} route={val['route_acc']:.3f} bF1={val['boundary_f1']:.3f} improve={val['exec_improvement']:.4f} exploit={val['boundary_exploit_rate']:.3f}",flush=True)
    final={"version":"program_assembly_curriculum_v2_structured","elapsed_sec":time.time()-t0,"best":best,"args":vars(args),"records_loaded":len(records) if records else 0,"separation":"This is builder pretraining only; do not load whole checkpoint into task model. Transfer controller parts/adapters only."}
    (out/"final_report.json").write_text(json.dumps(final,ensure_ascii=False,indent=2),encoding="utf-8"); write_report(out,final); print(f"DONE out_dir={out} best_epoch={best['epoch']} best_score={best['score']:.4f}")

def write_report(out,final):
    val=(final.get("best",{}).get("val",{}) or {})
    lines=["REPORT_TO_CHATGPT: program_assembly_curriculum_v2_structured","","Цель: обучение сборщика программ отдельно от основной task-модели.","Источники: synthetic или structured head/input JSONL.","Важно: checkpoint не является task-checkpoint; переносить только controller/adapters.","",f"records_loaded: {final.get('records_loaded',0)}",f"best_epoch: {final.get('best',{}).get('epoch',0)}",f"best_score: {final.get('best',{}).get('score',0):.4f}",f"full_program_exact: {val.get('full_program_exact',0):.4f}",f"primitive_acc: {val.get('prim_acc',0):.4f}",f"route_acc: {val.get('route_acc',0):.4f}",f"boundary_f1: {val.get('boundary_f1',0):.4f}",f"exec_improvement: {val.get('exec_improvement',0):.6f}",f"boundary_exploit_rate: {val.get('boundary_exploit_rate',0):.4f}","","Success smoke:","- primitive_acc > 0.80","- route_acc > 0.75","- boundary_f1 > 0.80","- exec_improvement > 0","- boundary_exploit_rate < 0.10"]
    (out/"REPORT_TO_CHATGPT.txt").write_text("\n".join(lines)+"\n",encoding="utf-8")

def normalized_record_from_synthetic(args, device, stage):
    b=synthetic_batch(args,1,device,stage)
    i=0; steps=[]
    for t in range(args.steps):
        for l in torch.where(b.active_mask[i,t])[0].tolist():
            steps.append({"t":int(t),"lane":LANE_NAMES[l],"read":READ_NAMES[int(b.read_t[i,t,l])],"primitive":PRIMITIVES[int(b.prim_t[i,t,l])],"route":f"{LANE_NAMES[l]}->{LANE_NAMES[int(b.route_t[i,t,l])]}","write":LANE_NAMES[int(torch.where(b.write_t[i,t]>0.5)[0][0])],"boundary":float(b.boundary_t[i,t])})
    return {"x":b.x[i].cpu().tolist(),"target_y":b.target_y[i].cpu().tolist(),"teacher_program":{"steps":steps},"source":"synthetic","task_id":int(b.task_id[i]),"mode":MODE_NAMES[int(b.mode_id[i])]}

def generate_jsonl(args):
    out=Path(args.dataset_out); out.parent.mkdir(parents=True,exist_ok=True); random.seed(args.seed); torch.manual_seed(args.seed); device=torch.device("cpu")
    accepted=0; rejected=0
    with out.open("w",encoding="utf-8") as f:
        while accepted<args.num_examples and rejected<args.num_examples*20:
            rec=normalized_record_from_synthetic(args,device,stage=random.choice([0,1,2,4]))
            parsed=record_to_program_tensors(rec,args,device)
            if parsed is None: rejected+=1; continue
            f.write(json.dumps(rec,ensure_ascii=False)+"\n"); accepted+=1
    manifest={"version":"program_assembly_curriculum_v2_structured","num_examples":accepted,"rejected":rejected,"seed":args.seed,"dim":args.dim,"steps":args.steps,"quality":{"min_boundary":args.quality_min_boundary,"max_boundary":args.quality_max_boundary}}
    (out.parent/"manifest.json").write_text(json.dumps(manifest,ensure_ascii=False,indent=2),encoding="utf-8"); print(json.dumps(manifest,ensure_ascii=False,indent=2))

def inspect_jsonl(args):
    records=load_jsonl_records(args.structured_jsonl or args.dataset_out); device=torch.device("cpu"); ok=0; bad=0; examples=[]
    for rec in records[:args.inspect_limit]:
        parsed=record_to_program_tensors(rec,args,device)
        if parsed is None: bad+=1
        else:
            ok+=1; examples.append({"source":rec.get("source","structured"),"steps":infer_steps_from_structured(rec,args.steps)[:args.steps]})
    print(json.dumps({"checked":min(len(records),args.inspect_limit),"ok":ok,"bad":bad,"examples":examples[:5]},ensure_ascii=False,indent=2))

def parser():
    p=argparse.ArgumentParser()
    p.add_argument("--mode",default="train_online",choices=["train_online","generate_jsonl","inspect_jsonl","train_jsonl"])
    p.add_argument("--out-dir",default="simple_butterfly_matrix_v4_tape_lane/agent_reports/program_assembly_curriculum_v2_structured_smoke")
    p.add_argument("--dataset-out",default="simple_butterfly_matrix_v4_tape_lane/program_datasets/program_assembly_v2/examples.jsonl")
    p.add_argument("--structured-jsonl",default="")
    p.add_argument("--task-source",default="synthetic",choices=["synthetic","structured","mixed"])
    p.add_argument("--structured-mix-prob",type=float,default=0.5)
    p.add_argument("--device",default="cuda"); p.add_argument("--amp",default="fp16",choices=["fp16","off"]); p.add_argument("--seed",type=int,default=42)
    p.add_argument("--dim",type=int,default=32); p.add_argument("--hidden",type=int,default=192); p.add_argument("--steps",type=int,default=6); p.add_argument("--lanes",type=int,default=4)
    p.add_argument("--epochs",type=int,default=8); p.add_argument("--batch-size",type=int,default=512); p.add_argument("--train-steps-per-epoch",type=int,default=250); p.add_argument("--eval-batches",type=int,default=30)
    p.add_argument("--lr",type=float,default=8e-4); p.add_argument("--weight-decay",type=float,default=0.01); p.add_argument("--grad-clip",type=float,default=1.0); p.add_argument("--log-every",type=int,default=50)
    p.add_argument("--stage0-epochs",type=int,default=2); p.add_argument("--stage1-epochs",type=int,default=2); p.add_argument("--stage2-epochs",type=int,default=2)
    p.add_argument("--corrupt-prob",type=float,default=0.35); p.add_argument("--wrong-prob",type=float,default=0.20); p.add_argument("--extra-prob",type=float,default=0.08)
    p.add_argument("--signal-structure",action=argparse.BooleanOptionalAction,default=True)
    p.add_argument("--inactive-token-weight",type=float,default=0.08); p.add_argument("--write-pos-weight",type=float,default=2.0); p.add_argument("--boundary-pos-weight",type=float,default=1.5)
    p.add_argument("--lambda-trace",type=float,default=1.0); p.add_argument("--lambda-exec",type=float,default=1.0); p.add_argument("--lambda-boundary-budget",type=float,default=0.25); p.add_argument("--lambda-offdiag-no-boundary",type=float,default=0.15); p.add_argument("--lambda-route-entropy",type=float,default=0.05); p.add_argument("--lambda-write-cost",type=float,default=0.01)
    p.add_argument("--boundary-min-peaks",type=float,default=1.0); p.add_argument("--boundary-max-peaks",type=float,default=4.0); p.add_argument("--route-entropy-max",type=float,default=1.25)
    p.add_argument("--quality-min-boundary",type=float,default=1.0); p.add_argument("--quality-max-boundary",type=float,default=4.0); p.add_argument("--quality-max-self-route",type=float,default=0.95); p.add_argument("--quality-max-active-density",type=float,default=0.35)
    p.add_argument("--num-examples",type=int,default=1000); p.add_argument("--inspect-limit",type=int,default=100); p.add_argument("--save-checkpoint",action="store_true")
    return p

if __name__=="__main__":
    args=parser().parse_args()
    if args.mode=="generate_jsonl": generate_jsonl(args)
    elif args.mode=="inspect_jsonl": inspect_jsonl(args)
    else:
        if args.mode=="train_jsonl": args.task_source="structured"
        train(args)
