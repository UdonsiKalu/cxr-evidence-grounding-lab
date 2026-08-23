#!/usr/bin/env python3
"""Phase-5: verify → regenerate once → REVIEW (not UNCERTAIN).

Does not overwrite Phase-1–4 artifacts. See docs/PHASE5-PROTOCOL.md.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from n2s_lab.phase1_diagnostic import PHASE2_PANEL_MODELS  # noqa: E402
from n2s_lab.phase5_verify import run_phase5_panel  # noqa: E402
from n2s_lab.verify import selftest_verify  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Phase-5 verify/REVIEW panel (frozen A–D + C_v/D_v)"
    )
    parser.add_argument(
        "--models",
        default=",".join(PHASE2_PANEL_MODELS),
        help="comma-separated Ollama model tags",
    )
    parser.add_argument(
        "--set",
        choices=("phase4", "phase3"),
        default="phase4",
        help="held-out case set",
    )
    parser.add_argument("--out", default="phase5-verify-panel.json")
    parser.add_argument(
        "--selftest",
        action="store_true",
        help="run verify unit selftest only (no Ollama)",
    )
    args = parser.parse_args()

    if args.selftest:
        selftest_verify()
        return

    models = [x.strip() for x in args.models.split(",") if x.strip()]
    panel = run_phase5_panel(models=models, case_set=args.set, out_name=args.out)
    print(f"wrote {panel['_artifact']}")
    print("Phase-1–4 artifacts were not overwritten.")
    print("\nPhase-5 metrics (per model):")
    for entry in panel["models"]:
        m = entry["phase5_metrics"]
        d = m["d_representation_loss"]
        dv = m["D_v"]
        print(
            f"  {entry['model']}: "
            f"D rep-loss {d['baseline_D']}→{d['D_v']} "
            f"(REVIEW on D_v stage {d['D_v_review']}); "
            f"D_v coverage_auto={dv['coverage_auto']:.2f} "
            f"safety_among_auto={dv['safety_among_auto']}"
        )
        for row in m["former_D_rep_loss_outcomes"]:
            print(f"    was D-rep {row['id']}: → {row['D_v']}")


if __name__ == "__main__":
    main()
