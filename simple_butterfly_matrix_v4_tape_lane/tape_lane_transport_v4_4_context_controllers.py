#!/usr/bin/env python3
"""v4.4 context-controller entrypoint.

This file is the Python runtime for the v4.4 controller experiment.
It intentionally reuses the already-implemented v4.3 context-controller launcher logic
instead of duplicating a large backbone patch here.

Architecture rule:
    base v4.3/v4.2 behavior + small context-dependent controller deltas

This entrypoint keeps v4.3 canonical main intact and provides a real file for:
    commands/sync_run_5ep_v4_4_context_controllers_push_logs_v2.sh

No Actor/Critic, no EditorLoop deploy, no MatrixMemory, no FeedbackBias auto-deploy.

Important: this is a bridge implementation. It covers read/route/boundary/write/alive
controllers through the existing wrapper. A full primitive_controller inside the
transform unit is the next code step after this bridge smoke passes.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path


PROJECT_DIR = "simple_butterfly_matrix_v4_tape_lane"


def _repo_root() -> Path:
    # file = <repo>/simple_butterfly_matrix_v4_tape_lane/tape_lane_transport_v4_4_context_controllers.py
    return Path(__file__).resolve().parents[1]


def _translate_args(argv: list[str]) -> tuple[list[str], dict[str, str]]:
    """Translate v4.4 CLI aliases to the existing v4.3 context wrapper.

    The existing controller wrapper accepts CONTEXT_CONTROLLER_SCALE as an env var.
    The v4.4 sync passes --context-alpha-min/--context-alpha-max.
    For this bridge we use the midpoint as a fixed safe controller scale.
    """
    env = os.environ.copy()
    out: list[str] = []
    i = 0
    alpha_min = None
    alpha_max = None
    compare_to = None

    while i < len(argv):
        a = argv[i]
        if a == "--context-alpha-min" and i + 1 < len(argv):
            alpha_min = float(argv[i + 1])
            i += 2
            continue
        if a == "--context-alpha-max" and i + 1 < len(argv):
            alpha_max = float(argv[i + 1])
            i += 2
            continue
        if a == "--compare-to" and i + 1 < len(argv):
            compare_to = argv[i + 1]
            # v4.3 wrapper/base parser may not know this arg in all revisions.
            i += 2
            continue
        out.append(a)
        i += 1

    if alpha_min is None:
        alpha_min = float(env.get("CONTEXT_ALPHA_MIN", "0.01"))
    if alpha_max is None:
        alpha_max = float(env.get("CONTEXT_ALPHA_MAX", "0.10"))
    scale = 0.5 * (alpha_min + alpha_max)
    env["CONTEXT_CONTROLLER_SCALE"] = env.get("CONTEXT_CONTROLLER_SCALE", f"{scale:.6g}")
    env["V44_CONTEXT_ALPHA_MIN"] = f"{alpha_min:.6g}"
    env["V44_CONTEXT_ALPHA_MAX"] = f"{alpha_max:.6g}"
    if compare_to is not None:
        env["COMPARE_TO"] = compare_to
    return out, env


def main(argv: list[str] | None = None) -> int:
    if argv is None:
        argv = sys.argv[1:]

    repo = _repo_root()
    wrapper = repo / PROJECT_DIR / "commands" / "run_v4_3_context_controller.sh"
    if not wrapper.exists():
        print(f"[v4.4] ERROR: missing existing context wrapper: {wrapper}", file=sys.stderr)
        print("[v4.4] expected path is <repo>/simple_butterfly_matrix_v4_tape_lane/commands/run_v4_3_context_controller.sh", file=sys.stderr)
        return 2

    forwarded, env = _translate_args(argv)
    print(
        "[v4.4] bridge entrypoint -> simple_butterfly_matrix_v4_tape_lane/commands/run_v4_3_context_controller.sh "
        f"CONTEXT_CONTROLLER_SCALE={env.get('CONTEXT_CONTROLLER_SCALE')} "
        f"alpha_min={env.get('V44_CONTEXT_ALPHA_MIN')} alpha_max={env.get('V44_CONTEXT_ALPHA_MAX')}",
        flush=True,
    )
    cmd = ["bash", str(wrapper), *forwarded]
    return subprocess.call(cmd, cwd=str(repo), env=env)


if __name__ == "__main__":
    raise SystemExit(main())
