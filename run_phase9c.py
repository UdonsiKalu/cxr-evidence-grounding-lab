#!/usr/bin/env python3
"""Phase-9C: causal intervention at repair commit on BC_E1 (7B).

Requires 9B gate YES. See docs/PHASE9-PROTOCOL.md.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from n2s_lab.phase9c_intervention import run_phase9c  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase-9C BC_E1 causal intervention")
    parser.add_argument(
        "--selftest",
        action="store_true",
        help="check 9B artifact gate YES",
    )
    args = parser.parse_args()

    if args.selftest:
        path = ROOT / "artifacts" / "phase9b-bc-e1-panel.json"
        if not path.exists():
            print("9B panel missing")
            sys.exit(1)
        panel = json.loads(path.read_text())
        if not panel.get("gate_probe_signal", {}).get("pass"):
            print("9B gate not YES")
            sys.exit(1)
        print("phase9c selftest OK (9B gate YES)")
        return

    panel = run_phase9c()
    gate = panel["gate_causal_intervention"]
    print(f"\nwrote {panel['_artifact']}")
    print(f"9C gate: {'YES' if gate['pass'] else 'NO'}")
    print(gate["note"])
    bc = panel["BC_E1"]
    print(
        f"BC_E1: baseline X={bc['baseline']['repair_final_x']} "
        f"verdict={bc['verdict_baseline']} → "
        f"intervened X={bc['intervened']['repair_final_x'] if bc['intervened'] else None} "
        f"verdict={bc['verdict_intervened']}"
    )
    if not gate["pass"]:
        sys.exit(2)


if __name__ == "__main__":
    main()
