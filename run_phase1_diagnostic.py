#!/usr/bin/env python3
"""Optional Phase-1 transition diagnostic (Conditions A–D).

Does not replace run_experiment.py. Safe to delete this file and
n2s_lab/phase1_diagnostic.py if the research direction is abandoned.
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
    format_transition_table,
    run_phase1,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Optional Phase-1 A–D transition diagnostic (default: C1–C4 + U1–U4)"
    )
    parser.add_argument(
        "--ids",
        default=",".join(PHASE1_CASE_IDS),
        help="comma-separated case ids",
    )
    parser.add_argument(
        "--out",
        default="phase1-transition-c-u.json",
        help="filename under artifacts/",
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help="Ollama model tag (prompts stay frozen across models)",
    )
    args = parser.parse_args()
    ids = [x.strip() for x in args.ids.split(",") if x.strip()]
    report = run_phase1(ids=ids, out_name=args.out, model=args.model)
    print(format_transition_table(report))
    print(f"\n{report['ollama_output_mechanism']}")
    print(f"\nwrote {report['_artifact']}")
    print("Undo: delete this runner, n2s_lab/phase1_diagnostic.py, and the artifact.")


if __name__ == "__main__":
    main()
