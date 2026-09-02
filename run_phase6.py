#!/usr/bin/env python3
"""Phase-6: L2b round-trip + L2c dual-path faithfulness experiments.

Does not overwrite Phase-1–5 artifacts. See docs/PHASE6-PROTOCOL.md.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from n2s_lab.phase1_diagnostic import PHASE2_PANEL_MODELS  # noqa: E402
from n2s_lab.phase6_faithfulness import run_phase6_panel  # noqa: E402
from n2s_lab.roundtrip import selftest_roundtrip  # noqa: E402
from n2s_lab.verify import selftest_verify  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Phase-6 faithfulness panel (round-trip + dual-path)"
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
    parser.add_argument("--out", default="phase6-faithfulness-panel.json")
    parser.add_argument(
        "--selftest",
        action="store_true",
        help="run round-trip/dual + verify selftests only (no Ollama)",
    )
    args = parser.parse_args()

    if args.selftest:
        selftest_verify()
        selftest_roundtrip()
        return

    models = [x.strip() for x in args.models.split(",") if x.strip()]
    panel = run_phase6_panel(models=models, case_set=args.set, out_name=args.out)
    print(f"wrote {panel['_artifact']}")
    print("Phase-1–5 artifacts were not overwritten.")
    print("\nPhase-6 metrics (per model):")
    for entry in panel["models"]:
        m = entry["phase6_metrics"]
        print(f"  {entry['model']}:")
        for key in ("C_v", "D_v", "C_rt", "D_rt", "Dual", "Dual_rt"):
            s = m[key]
            safety = s["safety_among_auto"]
            safety_s = f"{safety:.2f}" if safety is not None else "n/a"
            print(
                f"    {key}: coverage={s['coverage_auto']:.2f} "
                f"safety_auto={safety_s} review_n={s['review_n']}"
            )
        rt = m["roundtrip_new_reviews"]
        dd = m["dual_disagreements"]
        print(
            f"    new REVIEW from round-trip: C={rt['C']} D={rt['D']}; "
            f"dual disagreements: Dual={dd['Dual']} Dual_rt={dd['Dual_rt']}"
        )


if __name__ == "__main__":
    main()
