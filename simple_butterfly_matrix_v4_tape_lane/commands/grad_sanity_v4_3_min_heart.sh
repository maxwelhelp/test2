#!/usr/bin/env bash
set -euo pipefail

cd "$(git rev-parse --show-toplevel 2>/dev/null || pwd)"
OUT="${OUT:-simple_butterfly_matrix_v4_tape_lane/agent_reports/grad_sanity_v4_3_min_heart.json}"
mkdir -p "$(dirname "$OUT")"

python - "$OUT" <<'PY'
import json, sys, torch
import torch.nn.functional as F
from simple_butterfly_matrix_v4_tape_lane import tape_lane_transport_v4_3_min_heart as m

out = sys.argv[1]
args = m.parser().parse_args([
    "--device", "cpu",
    "--amp", "fp32",
    "--dim", "32",
    "--evidence-cells", "12",
    "--lanes", "4",
    "--cells-per-lane", "4",
    "--tape-steps", "3",
    "--pair-slots", "4",
    "--batch-size", "2",
    "--eval-batch-size", "2",
    "--no-save-checkpoints",
])
classes = 10
args.num_classes = classes
torch.manual_seed(0)

checks = {
    "route": ["backbone.route_logits"],
    "boundary": ["backbone.boundary_logit"],
    "read": ["backbone.read_group_logits", "backbone.read_query_w"],
    "write": ["backbone.write_gate_logit"],
    "head": ["head.class_lane_logits", "head.class_q", "head.key_w", "head.value_w"],
}

# detail_head_shortcut_cost is hinge-style and can be exactly zero on a tiny random smoke.
# For isolated grad sanity, use a direct differentiable proxy that always exercises head attention.
def detail_head_proxy(haux):
    attn = haux["class_slot_attention"].float()
    return (attn ** 2).mean()
loss_specs = {
    "full": lambda ce, losses, haux: ce + args.lambda_route_offdiag_outside_boundary * losses["route_offdiag_outside_boundary_cost"] + args.lambda_boundary_budget * losses["boundary_budget_cost"] + args.lambda_late_input_read * losses["late_input_read_cost"] + args.lambda_memory_write_cost * losses["memory_write_cost"] + args.lambda_memory_overwrite * losses["memory_overwrite_cost"] + args.lambda_detail_head_shortcut * losses["detail_head_shortcut_cost"],
    "route_only": lambda ce, losses, haux: losses["route_offdiag_outside_boundary_cost"] + losses["route_entropy_band"],
    "read_only": lambda ce, losses, haux: losses["late_input_read_cost"] + losses["sequence_read_delta_mean"],
    "memory_only": lambda ce, losses, haux: losses["memory_write_cost"] + losses["memory_overwrite_cost"],
    "detail_only": lambda ce, losses, haux: detail_head_proxy(haux),
}
report = {
    "main_source": "simple_butterfly_matrix_v4_tape_lane/tape_lane_transport_v4_3_min_heart.py",
    "canonicalizer_used": False,
    "note": "detail_only uses direct head-attention proxy because shortcut hinge can be inactive",
    "losses": {},
    "ok": True,
}
for spec_name, make_loss in loss_specs.items():
    model = m.TapeLaneRouterClassifierV43(classes, args)
    model.train()
    wav = torch.randn(2, int(args.sample_rate * args.seconds))
    y = torch.randint(0, classes, (2,))
    logits, baux, haux = model(wav)
    losses = m.aux_losses_v43(logits, baux, haux, args)
    ce = F.cross_entropy(logits.float(), y)
    loss = make_loss(ce, losses, haux)
    loss.backward()
    name_to_param = dict(model.named_parameters())
    spec_report = {"loss": float(loss.detach()), "checks": {}}
    for group, needles in checks.items():
        total = 0.0
        found = []
        for name, p in name_to_param.items():
            if any(n in name for n in needles):
                if p.grad is not None:
                    total += float(p.grad.detach().abs().sum())
                found.append(name)
        spec_report["checks"][group] = {"grad_abs_sum": total, "found": found, "ok": total > 0.0}
    report["losses"][spec_name] = spec_report

# Required checks: full touches everything; isolated losses must touch their intended subsystems.
required = {
    "full": ["route", "boundary", "read", "write", "head"],
    "route_only": ["route", "boundary"],
    "read_only": ["read"],
    "memory_only": ["write"],
    "detail_only": ["head"],
}
for spec, groups in required.items():
    for group in groups:
        if not report["losses"][spec]["checks"][group]["ok"]:
            report["ok"] = False
            report.setdefault("failed_required", []).append({"loss": spec, "group": group})
open(out, "w", encoding="utf-8").write(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
print(json.dumps(report, ensure_ascii=False, indent=2))
if not report["ok"]:
    raise SystemExit(1)
PY

echo "[grad sanity] wrote $OUT"
