#!/usr/bin/env python3
"""Phase-9B: BC_E1 repair-path activation localization.

Requires 9A gate YES. See docs/PHASE9-PROTOCOL.md.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from n2s_lab.phase9a_behavioral import PHASE9A_DEFAULT_MODELS, load_bc_e1  # noqa: E402
from n2s_lab.phase9b_localization import run_phase9b  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase-9B BC_E1 localization")
    parser.add_argument(
        "--models",
        default=",".join(PHASE9A_DEFAULT_MODELS),
        help="comma-separated HF model ids (fail,safe order)",
    )
    parser.add_argument(
        "--selftest",
        action="store_true",
        help="check 9A artifact exists",
    )
    args = parser.parse_args()

    if args.selftest:
        panel_path = ROOT / "artifacts" / "phase9a-bc-e1-panel.json"
        if not panel_path.exists():
            print("9A panel missing — run run_phase9a.py first")
            sys.exit(1)
        panel = json.loads(panel_path.read_text())
        if not panel.get("gate_clean_x_divergence", {}).get("pass"):
            print("9A gate not YES — stop")
            sys.exit(1)
        case = load_bc_e1()
        assert case["id"] == "BC_E1"
        print("phase9b selftest OK (9A gate YES)")
        return

    models = [x.strip() for x in args.models.split(",") if x.strip()]
    panel = run_phase9b(models=models)
    gate = panel["gate_probe_signal"]
    print(f"\nwrote {panel['_artifact']}")
    print(f"9B gate probe signal: {'YES' if gate['pass'] else 'NO'}")
    print(gate["note"])
    if not gate["pass"]:
        print("STOP — do not start 9C.")
        sys.exit(2)


if __name__ == "__main__":
    main()
