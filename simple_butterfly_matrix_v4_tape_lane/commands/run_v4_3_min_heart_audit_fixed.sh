#!/usr/bin/env bash
set -euo pipefail
cd "$(git rev-parse --show-toplevel 2>/dev/null || pwd)"

python - "$@" <<'PY'
import os, sys, torch
import torch.nn.functional as F
from typing import Dict
from torch.utils.data import DataLoader
from simple_butterfly_matrix_v4_tape_lane import tape_lane_transport_v4_3_min_heart as base

_old_build = base.build_trace_feedback
_old_gen = base.generate_candidate_suggestions
_old_report = base.write_chatgpt_report
_old_run = base.run


def _step_delta(x, steps):
    if x.numel() == 0 or x.shape[0] <= 1:
        return torch.zeros(int(steps), dtype=torch.float32)
    d = (x[1:] - x[:-1]).abs()
    while d.dim() > 1:
        d = d.mean(dim=-1)
    return torch.cat([d, torch.zeros(1, dtype=d.dtype)], dim=0)[: int(steps)]


def _route_extra_terms(routes, lanes):
    if routes.numel() == 0:
        z = torch.zeros((), dtype=torch.float32)
        return {"self_route_mass": z, "useful_transition_mass": z, "self_route_by_step": [], "useful_transition_by_step": []}
    eye = torch.eye(routes.shape[-1], dtype=routes.dtype, device=routes.device)
    self_by_step = (routes * eye.view(1, routes.shape[-1], routes.shape[-1])).sum(dim=(-2, -1)) / float(max(1, lanes))
    useful_parts = []
    if lanes >= 2:
        useful_parts.append(routes[:, 0, 1])
    if lanes >= 3:
        useful_parts.append(routes[:, 1, 2])
    if lanes >= 4:
        useful_parts.append(routes[:, 1, 3])
        useful_parts.append(routes[:, 3, 1])
    useful_by_step = torch.stack(useful_parts, dim=0).mean(dim=0) if useful_parts else torch.zeros(routes.shape[0], dtype=routes.dtype, device=routes.device)
    return {
        "self_route_mass": self_by_step.mean(),
        "useful_transition_mass": useful_by_step.mean(),
        "self_route_by_step": self_by_step.detach().float().cpu().tolist(),
        "useful_transition_by_step": useful_by_step.detach().float().cpu().tolist(),
    }


def make_loaders_fast(args):
    classes = [c.strip() for c in args.classes.split(",") if c.strip()]
    if args.synthetic:
        train_ds = base.v42.SyntheticMatrixTask(args.train_limit or 1024, len(classes), args.synthetic_length)
        val_ds = base.v42.SyntheticMatrixTask(args.val_limit or 256, len(classes), args.synthetic_length)
        counts = {c: (args.train_limit or 1024) // len(classes) for c in classes}
        train_counts, val_counts = counts, {c: (args.val_limit or 256) // len(classes) for c in classes}
        target_length = args.synthetic_length
    else:
        train_ds = base.v42.SpeechCommandsBalanced(args.data_root, "training", classes, args.train_limit, args.download)
        val_ds = base.v42.SpeechCommandsBalanced(args.data_root, "validation", classes, args.val_limit, args.download)
        train_counts = getattr(train_ds, "counts", {})
        val_counts = getattr(val_ds, "counts", {})
        target_length = int(args.sample_rate * args.seconds)
    collate = lambda b: base.v42.collate_wavs(b, target_length)
    common = dict(num_workers=args.workers, pin_memory=args.pin_memory, collate_fn=collate)
    if int(args.workers) > 0:
        common.update(persistent_workers=True, prefetch_factor=int(getattr(args, "prefetch_factor", 2)))
    train = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, **common)
    val = DataLoader(val_ds, batch_size=args.eval_batch_size, shuffle=False, **common)
    return train, val, classes, train_counts, val_counts


@torch.no_grad()
def evaluate_fast(model, loader, device, dtype, args):
    model.eval()
    use_amp = device.startswith("cuda") and dtype != torch.float32
    total_loss, correct, n = 0.0, 0, 0
    conf = torch.zeros(args.num_classes, args.num_classes, dtype=torch.long)
    last_baux = None
    last_haux = None
    for step, (wav, y) in enumerate(loader, 1):
        if args.max_val_batches and step > args.max_val_batches:
            break
        wav = wav.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        with torch.autocast(device_type=device.split(":")[0], dtype=dtype, enabled=use_amp):
            logits, baux, haux = model(wav)
            loss = F.cross_entropy(logits.float(), y)
        pred = logits.argmax(-1)
        bs = y.numel()
        total_loss += float(loss.detach().cpu()) * bs
        correct += int((pred == y).sum().detach().cpu())
        n += bs
        conf += torch.bincount((y.cpu() * args.num_classes + pred.cpu()), minlength=args.num_classes ** 2).view(args.num_classes, args.num_classes)
        last_baux, last_haux = baux, haux

    last_report = None
    if last_baux is not None and last_haux is not None:
        attn = last_haux["class_slot_attention"].float().mean(dim=0).cpu()
        lane_mass = last_haux["class_lane_mass"].float().mean(dim=0).cpu()
        gates = last_baux.write_gates.float().mean(dim=(0, 3)).cpu() if last_baux.write_gates.numel() else torch.empty(0)
        updates = last_baux.update_norms.float().mean(dim=(0, 3)).cpu() if last_baux.update_norms.numel() else torch.empty(0)
        read_group = last_baux.read_group_mass.float().cpu() if last_baux.read_group_mass.numel() else torch.empty(0)
        primitive_weights = last_baux.primitive_weights.float().cpu() if last_baux.primitive_weights.numel() else torch.empty(0)
        top_reads = base.v42._slot_top_reads(attn, last_baux.slot_names)
        last_report = {
            "version": "v4.3_fast_eval_report_last_batch",
            "lane_names": [base.v42.lane_name(i) for i in range(args.lanes)],
            "read_group_names": [base.v42.read_group_name(i) for i in range(args.lanes + 1)],
            "primitive_names": list(base.PRIMITIVES),
            "prior_config": {
                "structured_init_strength": args.structured_init_strength,
                "read_prior_mode": args.read_prior_mode,
                "read_prior_strength": args.read_prior_strength,
                "route_prior_mode": args.route_prior_mode,
                "route_prior_strength": args.route_prior_strength,
                "boundary_route_strength": args.boundary_route_strength,
                "lane_prior_strength": args.lane_prior_strength,
                "class_lane_prior_mode": args.class_lane_prior_mode,
                "class_lane_init_strength": args.class_lane_init_strength,
                "context_primitive_scale": args.context_primitive_scale,
            },
            "lane_init_norm": base.v42._to_float_list(last_baux.lane_init_norm),
            "step_alive": base.v42._to_float_list(last_baux.step_alive),
            "boundary": base.v42._to_float_list(last_baux.boundaries),
            "route_matrix": base.v42._to_float_list(last_baux.routes),
            "route_entropy": base.v42._to_float_list(last_baux.route_entropy),
            "read_group_mass": base.v42._to_float_list(read_group),
            "primitive_weights": base.v42._to_float_list(primitive_weights),
            "write_gate_by_step_lane": base.v42._to_float_list(gates),
            "update_norm_by_step_lane": base.v42._to_float_list(updates),
            "late_input_read_mass": float(last_baux.late_input_read_mass.detach().float().cpu()),
            "class_top_reads": top_reads,
            "class_lane_mass": [{base.v42.lane_name(li): float(lane_mass[ci, li]) for li in range(lane_mass.shape[1])} for ci in range(lane_mass.shape[0])],
            "lane_mass_mean": {base.v42.lane_name(li): float(lane_mass[:, li].mean()) for li in range(lane_mass.shape[1])},
            "pair_update_norm": float(last_haux["pair_update_norm"].detach().cpu()),
            "class_write": float(last_haux["class_write"].detach().cpu()),
            "slot_count": len(last_baux.slot_names),
        }
    return {"loss": total_loss / max(1, n), "acc": correct / max(1, n), "n": n, "confusion": conf.tolist(), "report": last_report}


def sequence_terms(baux, args):
    device = baux.slots.device
    routes = baux.routes.float()
    prim = baux.primitive_weights.float()
    upd = baux.update_norms.float()
    read = baux.read_group_mass.float()
    route_delta = (routes[1:] - routes[:-1]).abs().mean(dim=(-2, -1)).mean() if routes.numel() and routes.shape[0] > 1 else torch.zeros((), device=device)
    prim_delta = (prim[1:] - prim[:-1]).abs().mean(dim=(-2, -1)).mean() if prim.numel() and prim.shape[0] > 1 else torch.zeros((), device=device)
    if upd.numel() and upd.shape[1] > 1:
        u = upd.mean(dim=(0, 3))
        trace_delta = (u[1:] - u[:-1]).abs().mean()
    else:
        trace_delta = torch.zeros((), device=device)
    read_delta = (read[1:] - read[:-1]).abs().mean() if read.numel() and read.shape[0] > 1 else torch.zeros((), device=device)
    route_extra = _route_extra_terms(routes, int(args.lanes))
    return {
        "sequence_route_delta_mean": route_delta,
        "sequence_primitive_delta_mean": prim_delta,
        "sequence_update_delta_mean": trace_delta,
        "sequence_trace_delta_mean": trace_delta,
        "sequence_read_delta_mean": read_delta,
        "sequence_nonflat_score": route_delta + prim_delta + trace_delta + read_delta,
        "self_route_mass": route_extra["self_route_mass"].to(device),
        "useful_transition_mass": route_extra["useful_transition_mass"].to(device),
    }


def build_trace_feedback(rep, args, epoch):
    trace = _old_build(rep, args, epoch)
    routes = base._tensor_from_list(rep.get("route_matrix", []))
    read_group = base._tensor_from_list(rep.get("read_group_mass", []))
    route_delta = _step_delta(routes, args.tape_steps)
    read_delta = _step_delta(read_group, args.tape_steps)
    seq = trace.setdefault("sequence", {})
    route = trace.setdefault("route", {})
    route_extra = _route_extra_terms(routes, int(args.lanes))
    seq["route_delta_by_step"] = route_delta.tolist()[: int(args.tape_steps)] if route_delta.numel() else []
    seq["read_delta_by_step"] = read_delta.tolist()[: int(args.tape_steps)] if read_delta.numel() else []
    route["self_route_by_step"] = route_extra["self_route_by_step"]
    route["useful_transition_by_step"] = route_extra["useful_transition_by_step"]
    route["self_route_mass"] = float(route_extra["self_route_mass"])
    route["useful_transition_mass"] = float(route_extra["useful_transition_mass"])
    prim_delta = torch.tensor(seq.get("primitive_delta_by_step", []), dtype=torch.float32)
    trace_delta = torch.tensor(seq.get("trace_delta_by_step", []), dtype=torch.float32)
    boundary = torch.tensor(route.get("boundary_by_step", []), dtype=torch.float32)
    n = min(len(route_delta), len(prim_delta), len(trace_delta), len(read_delta))
    if n:
        change = route_delta[:n] + prim_delta[:n] + trace_delta[:n] + read_delta[:n]
        seq["sequence_change_by_step"] = change.tolist()
        seq["sequence_nonflat_score"] = float(change.mean())
        m = min(n, len(boundary))
        route["boundary_usefulness"] = (boundary[:m] * change[:m]).tolist() if m else []
    flags = []
    bm = float(route.get("boundary_mean", 0.0) or 0.0)
    bf = float(route.get("boundary_flatness", 0.0) or 0.0)
    ent = float(route.get("entropy_mean", 0.0) or 0.0)
    self_mass = float(route.get("self_route_mass", 0.0) or 0.0)
    useful_mass = float(route.get("useful_transition_mass", 0.0) or 0.0)
    head = trace.get("head", {}) or {}
    memory = trace.get("memory", {}) or {}
    detail = float(head.get("detail_attention_mass", 0.0) or 0.0)
    mem_w = float(memory.get("write_mean", 0.0) or 0.0)
    mem_c = float(memory.get("consumer_score", 0.0) or 0.0)
    if bm > 0.90 and bf < 0.05:
        flags.append("BOUNDARY_EXPLOIT")
    if bm < 0.05 or int(route.get("boundary_peak_count", 0) or 0) == 0:
        flags.append("BOUNDARY_DEAD")
    if ent > 1.30:
        flags.append("ROUTE_UNIFORM")
    if self_mass > 0.88 and useful_mass < 0.12:
        flags.append("ROUTE_IDENTITY_COLLAPSE")
    if detail > 0.55:
        flags.append("DETAIL_SHORTCUT")
    if mem_w < 0.03 and mem_c < 0.08:
        flags.append("MEMORY_DEAD")
    if mem_w > 0.30 and mem_c < 0.08:
        flags.append("MEMORY_JUNK")
    trace["collapse_flags"] = flags
    trace["version"] = "v4.3_min_heart_audit_fixed_fast"
    return trace


def generate_candidate_suggestions(trace, args, epoch):
    old = int(getattr(args, "max_candidates_per_epoch", 8))
    args.max_candidates_per_epoch = max(0, min(old, 12))
    out = _old_gen(trace, args, epoch)
    out.setdefault("diversity", {})["max_candidates_effective"] = int(args.max_candidates_per_epoch)
    out["diversity"]["attempted_after_filter"] = len(out.get("candidates", []))
    args.max_candidates_per_epoch = old
    for c in out.get("candidates", []):
        c["deploy"] = False
    route = trace.get("route", {}) or {}
    if float(route.get("self_route_mass", 0.0) or 0.0) > 0.88 and float(route.get("useful_transition_mass", 0.0) or 0.0) < 0.12:
        out.setdefault("candidates", []).append({
            "source": "route_identity_rule",
            "target_type": "route",
            "location": {"t": "all"},
            "action": "increase",
            "target": "useful_transition_mass",
            "reason": "route is too close to identity/self-loop while useful transitions are low",
            "evidence_metrics": {"self_route_mass": route.get("self_route_mass", 0.0), "useful_transition_mass": route.get("useful_transition_mass", 0.0)},
            "risk": "medium",
            "delta_scale": 0.03,
            "deploy": False,
        })
        out["candidates"] = out["candidates"][: int(getattr(args, "max_candidates_per_epoch", 8))]
    return out


def write_chatgpt_report(out_dir, analysis, trace, candidates, args):
    _old_report(out_dir, analysis, trace, candidates, args)
    route = trace.get("route", {}) or {}
    flags = trace.get("collapse_flags", []) or []
    path = out_dir / "REPORT_TO_CHATGPT.txt"
    with path.open("a", encoding="utf-8") as f:
        f.write("\nAudit-fixed diagnostics:\n")
        f.write(f"- self_route_mass: {float(route.get('self_route_mass', 0.0) or 0.0):.4f}\n")
        f.write(f"- useful_transition_mass: {float(route.get('useful_transition_mass', 0.0) or 0.0):.4f}\n")
        f.write(f"- collapse_flags: {','.join(flags) if flags else 'NONE'}\n")
        f.write("- sequence_change_by_step includes route_delta + primitive_delta + trace/update_delta + read_delta.\n")
        f.write("- boundary_usefulness includes route_delta + primitive_delta + trace/update_delta + read_delta.\n")
        f.write("- fast eval computes loss/accuracy on all validation batches, but builds heavy trace/report once from last validation batch.\n")


def train_epoch(model, loader, opt, scaler, device, dtype, args, epoch: int):
    model.train()
    use_amp = device.startswith("cuda") and dtype != torch.float32
    totals = {"loss": 0.0, "ce": 0.0, "correct": 0, "n": 0}
    aux_sum: Dict[str, float] = {}
    warmup = max(1, int(getattr(args, "memory_write_warmup_epochs", 2)))
    mem_write_lambda = float(args.lambda_memory_write_cost) * min(1.0, float(epoch) / float(warmup))
    mem_over_lambda = float(args.lambda_memory_overwrite) * min(1.0, float(epoch) / float(warmup))
    for step, (wav, y) in enumerate(loader, 1):
        if args.max_train_batches and step > args.max_train_batches:
            break
        wav = wav.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        opt.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.split(":")[0], dtype=dtype, enabled=use_amp):
            logits, baux, haux = model(wav)
            ce = F.cross_entropy(logits.float(), y)
            losses = base.aux_losses_v43(logits, baux, haux, args)
            loss = ce
            loss = loss + args.lambda_write_budget * losses["write_budget"]
            loss = loss + args.lambda_update_alive * losses["update_alive"]
            loss = loss + args.lambda_class_read_div * losses["class_read_div"]
            loss = loss + args.lambda_lane_balance * losses["lane_balance"]
            loss = loss + args.lambda_slot_div * losses["slot_div"]
            loss = loss + args.lambda_route_entropy * losses["route_entropy_band"]
            loss = loss + args.lambda_step_alive_budget * losses["step_alive_budget"]
            loss = loss + args.lambda_route_offdiag_outside_boundary * losses["route_offdiag_outside_boundary_cost"]
            loss = loss + args.lambda_boundary_budget * losses["boundary_budget_cost"]
            loss = loss + args.lambda_late_input_read * losses["late_input_read_cost"]
            loss = loss + mem_write_lambda * losses["memory_write_cost"]
            loss = loss + mem_over_lambda * losses["memory_overwrite_cost"]
            loss = loss + args.lambda_detail_head_shortcut * losses["detail_head_shortcut_cost"]
            loss = loss + args.lambda_skip_cost * losses["skip_cost"]
            loss = loss + args.lambda_operator_complexity * losses["operator_complexity_cost"]
            loss = loss + args.lambda_logit_norm * losses["logit_norm"]
        if not torch.isfinite(loss):
            print("NONFINITE_LOSS skip", flush=True)
            continue
        scaler.scale(loss).backward()
        if args.grad_clip > 0:
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        scaler.step(opt)
        scaler.update()
        bs = y.numel()
        totals["loss"] += float(loss.detach().cpu()) * bs
        totals["ce"] += float(ce.detach().cpu()) * bs
        totals["correct"] += int((logits.argmax(-1) == y).sum().detach().cpu())
        totals["n"] += bs
        for k, v in losses.items():
            aux_sum[k] = aux_sum.get(k, 0.0) + float(v.detach().cpu()) * bs
        if args.log_every and step % args.log_every == 0:
            print(f"epoch {epoch:03d} step {step:05d} loss={totals['loss']/max(1, totals['n']):.4f} ce={totals['ce']/max(1, totals['n']):.4f} acc={100*totals['correct']/max(1, totals['n']):.2f}%", flush=True)
    out = {k: v / max(1, totals["n"]) for k, v in totals.items() if k != "correct"}
    out["acc"] = totals["correct"] / max(1, totals["n"])
    for k, v in aux_sum.items():
        out[k] = v / max(1, totals["n"])
    out["memory_write_lambda_effective"] = mem_write_lambda
    out["memory_overwrite_lambda_effective"] = mem_over_lambda
    return out


def run(args):
    if getattr(args, "enable_counterfactual_screen", False):
        print("[v4.3 audit-fixed] counterfactual screen is not implemented in MVP0; suggestions only, no deploy.", flush=True)
    if not hasattr(args, "memory_write_warmup_epochs"):
        args.memory_write_warmup_epochs = int(os.environ.get("MEMORY_WRITE_WARMUP_EPOCHS", "2"))
    if not hasattr(args, "prefetch_factor"):
        args.prefetch_factor = int(os.environ.get("PREFETCH_FACTOR", "2"))
    _old_run(args)


base.v42.make_loaders = make_loaders_fast
base.v42.evaluate = evaluate_fast
base.sequence_terms = sequence_terms
base.build_trace_feedback = build_trace_feedback
base.generate_candidate_suggestions = generate_candidate_suggestions
base.write_chatgpt_report = write_chatgpt_report
base.train_epoch = train_epoch
base.run = run
base.run(base.parser().parse_args(sys.argv[1:]))
PY
