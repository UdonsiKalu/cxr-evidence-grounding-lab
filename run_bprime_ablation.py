#!/usr/bin/env python3
"""B′ ablation: remap Condition B verdict prompt; keep frozen Phase-1/2 B.

Reuses Phase-2 analyses. Does not overwrite phase1/phase2 artifacts.
Unseen cases are not in this runner.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from n2s_lab.bprime_ablation import format_bprime_table, run_bprime_panel  # noqa: E402
from n2s_lab.phase1_diagnostic import PHASE2_PANEL_MODELS  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(
        description="B′ ablation: frozen B vs mapping prompt, same 8 analyses"
    )
    parser.add_argument(
        "--models",
        default=",".join(PHASE2_PANEL_MODELS),
        help="comma-separated Ollama model tags",
    )
    parser.add_argument("--out", default="bprime-ablation.json")
    args = parser.parse_args()
    models = [x.strip() for x in args.models.split(",") if x.strip()]
    panel = run_bprime_panel(models=models, out_name=args.out)
    print(format_bprime_table(panel))
    print(f"\nwrote {panel['_artifact']}")
    print("Frozen Phase-1/2 artifacts were not overwritten.")


if __name__ == "__main__":
    main()
