#!/usr/bin/env python3
"""Phase-7: evidence object + BMT/CAR-T cross-domain panel.

Does not overwrite Phase-1–6 artifacts. See docs/PHASE7-PROTOCOL.md.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from n2s_lab.evidence import selftest_evidence  # noqa: E402
from n2s_lab.phase1_diagnostic import PHASE2_PANEL_MODELS  # noqa: E402
from n2s_lab.phase7_crossdomain import run_phase7_panel  # noqa: E402
from n2s_lab.roundtrip import selftest_roundtrip  # noqa: E402
from n2s_lab.verify import selftest_verify  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Phase-7 panel (evidence gate + BMT/CAR-T held-out)"
    )
    parser.add_argument(
        "--models",
        default=",".join(PHASE2_PANEL_MODELS),
        help="comma-separated Ollama model tags",
    )
    parser.add_argument(
        "--set",
        choices=("bmtcart", "phase4"),
        default="bmtcart",
        help="held-out case set (default: BMT/CAR-T phase7)",
    )
    parser.add_argument("--out", default="")
    parser.add_argument(
        "--selftest",
        action="store_true",
        help="run evidence/verify/roundtrip selftests only (no Ollama)",
    )
    args = parser.parse_args()

    if args.selftest:
        selftest_verify()
        selftest_roundtrip()
        selftest_evidence()
        return

    models = [x.strip() for x in args.models.split(",") if x.strip()]
    out_name = args.out or (
        "phase7-bmtcart-panel.json"
        if args.set == "bmtcart"
        else "phase7-phase4-panel.json"
    )
    panel = run_phase7_panel(models=models, case_set=args.set, out_name=out_name)
    print(f"wrote {panel['_artifact']}")
    print("Phase-1–6 artifacts were not overwritten.")
    print("\nPhase-7 metrics (per model):")
    for entry in panel["models"]:
        m = entry["phase7_metrics"]
        print(f"  {entry['model']}:")
        for key in ("Dual_rt", "Dual_full"):
            s = m[key]
            safety = s["safety_among_auto"]
            safety_s = f"{safety:.2f}" if safety is not None else "n/a"
            print(
                f"    {key}: coverage={s['coverage_auto']:.2f} "
                f"safety_auto={safety_s} review_n={s['review_n']}"
            )
        ev = m["evidence_new_reviews"]
        print(f"    new REVIEW from evidence gate: C={ev['C']} D={ev['D']}")


if __name__ == "__main__":
    main()
