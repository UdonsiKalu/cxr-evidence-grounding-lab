#!/usr/bin/env python3
"""Phase-10: activation steering at repair commit (HF 7B, no logit forcing).

Requires Phase-9B traces. See docs/PHASE10-PROTOCOL.md.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from n2s_lab.phase10_activation import run_phase10  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase-10 BC_E1 activation steering")
    parser.add_argument(
        "--selftest",
        action="store_true",
        help="check 9B artifact + hf_intervene activation_steer import",
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
        from n2s_lab.hf_intervene import ActivationSteerSpec  # noqa: F401

        print("phase10 selftest OK (9B gate YES)")
        return

    panel = run_phase10()
    gate = panel["gate_activation_intervention"]
    print(f"\nwrote {panel['_artifact']}")
    print(f"10C gate: {'YES' if gate['pass'] else 'NO'}")
    print(gate["note"])
    bc = panel["BC_E1"]
    print(
        f"BC_E1: baseline X={bc['baseline_x']} margin={bc['baseline_margin']} "
        f"verdict={bc['verdict_baseline']} → "
        f"steered X={bc['intervened_x']} margin={bc['intervened_margin']} "
        f"alpha={bc['best_alpha']} verdict={bc['verdict_intervened']}"
    )
    if not gate["pass"]:
        sys.exit(2)


if __name__ == "__main__":
    main()
