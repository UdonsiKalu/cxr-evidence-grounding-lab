#!/usr/bin/env python3
"""Phase-2: freeze Phase-1 A–D prompts; rerun same 8 cases across a model panel.

No per-model prompt tuning. Unseen cases come only after this panel is frozen.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from n2s_lab.ollama_client import DEFAULT_MODEL  # noqa: E402
from n2s_lab.phase1_diagnostic import (  # noqa: E402
    PHASE1_CASE_IDS,
    PHASE2_PANEL_MODELS,
    format_panel_table,
    format_transition_table,
    run_phase1,
    run_phase2_panel,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Phase-2 model panel: same A–D, same 8 cases, prompts frozen"
    )
    parser.add_argument(
        "--models",
        default=",".join(PHASE2_PANEL_MODELS),
        help="comma-separated Ollama model tags",
    )
    parser.add_argument(
        "--ids",
        default=",".join(PHASE1_CASE_IDS),
        help="comma-separated case ids (default: C1–C4,U1–U4)",
    )
    parser.add_argument(
        "--out",
        default="phase2-model-panel.json",
        help="panel summary filename under artifacts/",
    )
    parser.add_argument(
        "--no-reuse-control",
        action="store_true",
        help="rerun control model instead of reusing phase1-transition-c-u.json",
    )
    parser.add_argument(
        "--single-model",
        default="",
        help="run one model only (writes phase2-<model>.json); skip panel summary",
    )
    args = parser.parse_args()
    ids = [x.strip() for x in args.ids.split(",") if x.strip()]

    if args.single_model:
        model = args.single_model.strip()
        tag = model.replace(":", "_").replace("/", "_")
        report = run_phase1(ids=ids, out_name=f"phase2-{tag}.json", model=model)
        print(format_transition_table(report))
        print(f"\nwrote {report['_artifact']}")
        return

    models = [x.strip() for x in args.models.split(",") if x.strip()]
    panel = run_phase2_panel(
        models=models,
        ids=ids,
        out_name=args.out,
        reuse_control_artifact=None if args.no_reuse_control else "phase1-transition-c-u.json",
    )
    print(format_panel_table(panel))
    print(f"\ncontrol reference model: {DEFAULT_MODEL}")
    print(f"wrote {panel['_artifact']}")
    print("Next (after freeze): unseen held-out cases — not in this runner.")


if __name__ == "__main__":
    main()
