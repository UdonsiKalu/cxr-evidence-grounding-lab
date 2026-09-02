#!/usr/bin/env python3
"""Phase-8 model ladder: qwen2.5:14b + qwen2.5:32b (non-coder).

Does not overwrite Phase-1–7 artifacts. See docs/PHASE8-PROTOCOL.md.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from n2s_lab.evidence import selftest_evidence  # noqa: E402
from n2s_lab.phase8_ladder import PHASE8_LADDER_MODELS, run_phase8_ladder  # noqa: E402
from n2s_lab.roundtrip import selftest_roundtrip  # noqa: E402
from n2s_lab.verify import selftest_verify  # noqa: E402


def _print_panel(panel: dict) -> None:
    print(f"wrote {panel['_artifact']}")
    print("Phase-1–7 artifacts were not overwritten.")
    print("\nPhase-8 Dual_full (per model):")
    for entry in panel["models"]:
        m = entry.get("phase7_metrics") or {}
        s = m.get("Dual_full") or {}
        saf = s.get("safety_among_auto")
        saf_s = f"{saf:.2f}" if saf is not None else "n/a"
        cov = s.get("coverage_auto")
        cov_s = f"{cov:.2f}" if cov is not None else "n/a"
        ev = m.get("evidence_new_reviews") or {}
        print(
            f"  {entry['model']}: Dual_full cov={cov_s} safety_auto={saf_s} "
            f"review_n={s.get('review_n')} evidence+REV C={ev.get('C')} D={ev.get('D')}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Phase-8 model ladder (14B + non-coder 32B)"
    )
    parser.add_argument(
        "--models",
        default=",".join(PHASE8_LADDER_MODELS),
        help="comma-separated Ollama model tags",
    )
    parser.add_argument(
        "--set",
        choices=("phase4", "bmtcart", "both"),
        default="both",
        help="held-out set(s); default both",
    )
    parser.add_argument(
        "--selftest",
        action="store_true",
        help="run selftests only (no Ollama)",
    )
    args = parser.parse_args()

    if args.selftest:
        selftest_verify()
        selftest_roundtrip()
        selftest_evidence()
        print("phase8 selftest OK")
        return

    models = [x.strip() for x in args.models.split(",") if x.strip()]
    sets = ("phase4", "bmtcart") if args.set == "both" else (args.set,)
    for case_set in sets:
        print(f"\n=== Phase-8 ladder · {case_set} ===")
        panel = run_phase8_ladder(case_set=case_set, models=models)
        _print_panel(panel)


if __name__ == "__main__":
    main()
