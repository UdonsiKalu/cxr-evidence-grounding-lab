#!/usr/bin/env python3
"""Phase-3: frozen A–D on held-out C5–C8/U5–U8.

Does not overwrite Phase-1/2 artifacts. Does not use B′ as default.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from n2s_lab.phase1_diagnostic import (  # noqa: E402
    PHASE2_PANEL_MODELS,
    format_panel_table,
    run_phase3_panel,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Phase-3 held-out A–D panel (C5–C8, U5–U8), prompts frozen"
    )
    parser.add_argument(
        "--models",
        default=",".join(PHASE2_PANEL_MODELS),
        help="comma-separated Ollama model tags",
    )
    parser.add_argument("--out", default="phase3-heldout-panel.json")
    args = parser.parse_args()
    models = [x.strip() for x in args.models.split(",") if x.strip()]
    panel = run_phase3_panel(models=models, out_name=args.out)
    print(
        format_panel_table(
            panel,
            title="Phase-3 held-out (C5–C8/U5–U8, frozen A–D, B′ not default)",
        )
    )
    print(f"\nwrote {panel['_artifact']}")
    print("Phase-1/2 artifacts were not overwritten.")


if __name__ == "__main__":
    main()
