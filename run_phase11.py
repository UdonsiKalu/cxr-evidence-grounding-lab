#!/usr/bin/env python3
"""Phase-11: fixed Ph10 steering vector generalization on held-out cases."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from n2s_lab.phase11_generalization import run_phase11  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase-11 generalization panel")
    parser.add_argument(
        "--selftest",
        action="store_true",
        help="check Ph10 gate + held-out file",
    )
    args = parser.parse_args()

    if args.selftest:
        p10 = ROOT / "artifacts" / "phase10-bc-e1-panel.json"
        h11 = ROOT / "data" / "heldout-phase11.json"
        if not p10.exists():
            print("phase10 panel missing")
            sys.exit(1)
        if not json.loads(p10.read_text()).get("gate_activation_intervention", {}).get(
            "pass"
        ):
            print("phase10 gate not YES")
            sys.exit(1)
        if not h11.exists():
            print("heldout-phase11.json missing")
            sys.exit(1)
        print("phase11 selftest OK")
        return

    panel = run_phase11()
    gate = panel["gate_generalization"]
    print(f"\nwrote {panel['_artifact']}")
    print(f"11 gate: {'YES' if gate['pass'] else 'NO'}")
    print(gate["note"])
    print(
        f"targets baseline false-X: {gate['targets_baseline_false_x_count']}/"
        f"{gate['targets_n']} fixed: {gate['targets_fixed_count']}"
    )
    if not gate["pass"]:
        sys.exit(2)


if __name__ == "__main__":
    main()
