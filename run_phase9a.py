#!/usr/bin/env python3
"""Phase-9A: HF behavioral reproduction of BC_E1 contradiction.present.

See docs/PHASE9-PROTOCOL.md. Does not overwrite Phase-1–8 artifacts.
Does not run 9B/9C.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from n2s_lab.phase9a_behavioral import (  # noqa: E402
    PHASE9A_DEFAULT_MODELS,
    load_bc_e1,
    run_phase9a,
)
from n2s_lab.verify import selftest_verify  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase-9A HF BC_E1 behavioral reproduce")
    parser.add_argument(
        "--models",
        default=",".join(PHASE9A_DEFAULT_MODELS),
        help="comma-separated HF model ids",
    )
    parser.add_argument(
        "--selftest",
        action="store_true",
        help="load BC_E1 + verify selftest only (no HF download)",
    )
    args = parser.parse_args()

    if args.selftest:
        case = load_bc_e1()
        assert case["id"] == "BC_E1"
        assert case["expected"] == "SATISFIED"
        selftest_verify()
        print("phase9a selftest OK")
        print(f"BC_E1 note: {case['evidence'][:80]}…")
        return

    models = [x.strip() for x in args.models.split(",") if x.strip()]
    panel = run_phase9a(models=models)
    gate = panel["gate_clean_x_divergence"]
    print(f"\nwrote {panel['_artifact']}")
    print("Phase-1–8 artifacts were not overwritten.")
    print("\nPer-model final contradiction.present:")
    for mid, x in gate["per_model_final_x"].items():
        stage = gate["per_model_stage_that_set_x"].get(mid)
        print(f"  {mid}: X={x}  stage_that_set_x={stage}")
    print(f"\n9A gate clean X divergence: {'YES' if gate['pass'] else 'NO'}")
    print(gate["note"])
    if not gate["pass"]:
        print("STOP — do not start 9B/9C.")
        sys.exit(2)


if __name__ == "__main__":
    main()
