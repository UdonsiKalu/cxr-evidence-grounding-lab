#!/usr/bin/env python3
"""Phase-4: frozen A–D on held-out C9–C11/U9–U12.

Does not overwrite Phase-1/2/3 artifacts. Does not use B′ as default.
No C4/C8 analog.
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
    run_phase4_panel,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Phase-4 held-out A–D panel (C9–C11, U9–U12), prompts frozen"
    )
    parser.add_argument(
        "--models",
        default=",".join(PHASE2_PANEL_MODELS),
        help="comma-separated Ollama model tags",
    )
    parser.add_argument("--out", default="phase4-heldout-panel.json")
    args = parser.parse_args()
    models = [x.strip() for x in args.models.split(",") if x.strip()]
    panel = run_phase4_panel(models=models, out_name=args.out)
    print(
        format_panel_table(
            panel,
            title="Phase-4 held-out (C9–C11/U9–U12, frozen A–D, B′ off, no C8 analog)",
        )
    )
    print("\nRegex screen (not the protocol headline; human confirmation still required):")
    for entry in panel["models"]:
        scr = entry.get("phase4_regex_screen") or {}
        b = scr.get("B") or {}
        d = scr.get("D") or {}
        print(
            f"  {entry['model']}: "
            f"B loss {b.get('loss', 0)}/{b.get('n_screen', 0)} "
            f"ids_loss={b.get('ids_loss')}; "
            f"D rep_loss {d.get('representation_loss', 0)}/{d.get('n_screen', 0)} "
            f"ids={d.get('ids_representation_loss')}"
        )
    print(f"\nwrote {panel['_artifact']}")
    print("Phase-1/2/3 artifacts were not overwritten.")


if __name__ == "__main__":
    main()
