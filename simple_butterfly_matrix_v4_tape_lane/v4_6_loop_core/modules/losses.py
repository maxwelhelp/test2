#!/usr/bin/env python3
"""Differentiable structural objectives and metrics for v4.6/v4.6.1."""
from __future__ import annotations

from typing import Dict

import torch
import torch.nn.functional as F


def entropy_from_probs(p: torch.Tensor, dim: int = -1) -> torch.Tensor:
    q = p.float().clamp_min(1e-8)
    return -(q * q.log()).sum(dim=dim)


def allowed_route_mask(lanes: int, *, device=None, dtype=None) -> torch.Tensor:
    m = torch.eye(lanes, device=device, dtype=dtype or torch.float32)
    if lanes >= 2:
        m[0, 1] = 1.0
    if lanes >= 3:
        m[1, 2] = 1.0
    if lanes >= 4:
        m[1, 3] = 1.0
        m[3, 1] = 1.0
    return m


def closed_loop_aux_losses(trace: Dict[str, torch.Tensor], args) -> Dict[str, torch.Tensor]:
    boundary = trace["boundary"].float()
    route = trace["route"].float()
    weights = trace["primitive_weights"].float()
    signs = trace["primitive_signs"].float()
    device = boundary.device
    lanes = int(route.shape[-1])
    k = int(weights.shape[-1])

    threshold = float(getattr(args, "boundary_peak_threshold", 0.35))
    tau = max(1e-6, float(getattr(args, "boundary_peak_tau", 0.06)))
    min_peaks = float(getattr(args, "boundary_min_peaks", 1.0))
    max_peaks = float(getattr(args, "boundary_max_peaks", 4.0))
    soft_count = torch.sigmoid((boundary - threshold) / tau).sum(dim=-1)
    boundary_budget_loss = F.relu(min_peaks - soft_count).pow(2).mean() + F.relu(soft_count - max_peaks).pow(2).mean()
    boundary_std = boundary.std(dim=-1, unbiased=False).mean() if boundary.shape[-1] > 1 else torch.zeros((), device=device)
    boundary_flatness_loss = F.relu(float(getattr(args, "boundary_flatness_target", 0.03)) - boundary_std).pow(2)

    route_entropy = entropy_from_probs(route, dim=-1).mean()
    entropy_max = 0.82 * torch.log(torch.tensor(float(lanes), device=device))
    route_entropy_band_loss = F.relu(route_entropy - entropy_max).pow(2) + F.relu(0.20 - route_entropy).pow(2)
    eye = torch.eye(lanes, device=device, dtype=route.dtype)
    self_route_mass = (route * eye.view(1, 1, lanes, lanes)).sum(dim=(-2, -1)).mean() / float(max(1, lanes))
    mask = allowed_route_mask(lanes, device=device, dtype=route.dtype)
    route_allowed_mass = (route * mask.view(1, 1, lanes, lanes)).sum(dim=-1).mean()
    route_disallowed_mass = (route * (1.0 - mask).view(1, 1, lanes, lanes)).sum(dim=-1).mean()
    route_identity_loss = F.relu(self_route_mass - 0.90).pow(2)
    useful = []
    if lanes >= 2:
        useful.append(route[..., 0, 1])
    if lanes >= 3:
        useful.append(route[..., 1, 2])
    if lanes >= 4:
        useful.extend([route[..., 1, 3], route[..., 3, 1]])
    useful_transition_mass = torch.stack(useful, dim=0).mean() if useful else torch.zeros((), device=device)

    primitive_entropy = entropy_from_probs(weights, dim=-1).mean()
    max_prim_entropy = torch.log(torch.tensor(float(k), device=device))
    primitive_uniform_loss = F.relu(primitive_entropy - 0.92 * max_prim_entropy).pow(2)
    usage = weights.mean(dim=(0, 1, 2))
    primitive_diversity_loss = (usage * usage.clamp_min(1e-8).log()).sum() + max_prim_entropy
    primitive_top1 = weights.argmax(dim=-1)
    primitive_top1_share = torch.bincount(primitive_top1.reshape(-1), minlength=k).float().max() / float(max(1, primitive_top1.numel()))
    primitive_sign_negative_share = (signs < 0).float().mean()
    sign_balance_loss = signs.mean().pow(2)

    group_weights = trace.get("group_weights")
    if group_weights is not None:
        gw = group_weights.float()
        group_entropy = entropy_from_probs(gw, dim=-1).mean()
        g = int(gw.shape[-1])
        group_max_ent = torch.log(torch.tensor(float(g), device=device))
        group_uniform_loss = F.relu(group_entropy - 0.92 * group_max_ent).pow(2)
        group_top1 = gw.argmax(dim=-1)
        group_top1_share = torch.bincount(group_top1.reshape(-1), minlength=g).float().max() / float(max(1, group_top1.numel()))
    else:
        group_entropy = torch.zeros((), device=device)
        group_uniform_loss = torch.zeros((), device=device)
        group_top1_share = torch.zeros((), device=device)

    write_gate = trace.get("write_gate")
    mem_write = trace.get("memory_write_norm")
    mem_read = trace.get("memory_read_norm")
    mem_infl = trace.get("memory_read_influence")
    logits = trace.get("logits")

    if mem_write is not None and mem_read is not None and mem_infl is not None:
        target = float(getattr(args, "memory_useful_influence_target", 0.015))
        # Penalize writing/reading memory only when the measured output/state influence is tiny.
        memory_junk_loss = (mem_write.float() * mem_read.float() * F.relu(target - mem_infl.float())).mean()
        memory_influence_mean = mem_infl.float().mean()
    else:
        memory_junk_loss = torch.zeros((), device=device)
        memory_influence_mean = torch.zeros((), device=device)

    return {
        "boundary_budget_loss": boundary_budget_loss,
        "boundary_flatness_loss": boundary_flatness_loss,
        "boundary_soft_peak_count": soft_count.mean(),
        "boundary_mean": boundary.mean(),
        "boundary_std": boundary_std,
        "route_entropy_band_loss": route_entropy_band_loss,
        "route_allowed_loss": route_disallowed_mass,
        "route_identity_loss": route_identity_loss,
        "route_entropy": route_entropy,
        "self_route_mass": self_route_mass,
        "useful_transition_mass": useful_transition_mass,
        "route_allowed_mass": route_allowed_mass,
        "route_disallowed_mass": route_disallowed_mass,
        "primitive_uniform_loss": primitive_uniform_loss,
        "primitive_diversity_loss": primitive_diversity_loss,
        "sign_balance_loss": sign_balance_loss,
        "primitive_entropy": primitive_entropy,
        "primitive_top1_share": primitive_top1_share,
        "primitive_sign_negative_share": primitive_sign_negative_share,
        "group_entropy": group_entropy,
        "group_uniform_loss": group_uniform_loss,
        "group_top1_share": group_top1_share,
        "program_cost": write_gate.float().mean() if write_gate is not None else torch.zeros((), device=device),
        "memory_write_cost": mem_write.float().mean() if mem_write is not None else torch.zeros((), device=device),
        "memory_influence_mean": memory_influence_mean,
        "memory_junk_loss": memory_junk_loss,
        "logit_norm": logits.float().pow(2).mean() if logits is not None else torch.zeros((), device=device),
    }
