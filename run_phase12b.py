#!/usr/bin/env python3
"""Phase-12B: second held-out margin-band panel."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from n2s_lab.phase12b_heldout import run_phase12b  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase-12B held-out panel")
    parser.add_argument("--selftest", action="store_true")
    args = parser.parse_args()

    if args.selftest:
        p10 = ROOT / "artifacts" / "phase10-bc-e1-panel.json"
        p12a = ROOT / "artifacts" / "phase12-margin-regime-panel.json"
        proto = ROOT / "docs" / "PHASE12B-PROTOCOL.md"
        held = ROOT / "data" / "heldout-phase12b.json"
        for path in (p10, p12a, proto, held):
            if not path.exists():
                print(f"missing {path}")
                sys.exit(1)
        if not json.loads(p10.read_text()).get("gate_activation_intervention", {}).get(
            "pass"
        ):
            print("phase10 gate not YES")
            sys.exit(1)
        data = json.loads(held.read_text())
        print(
            f"phase12B selftest OK — {len(data['cases'])} cases · "
            f"wording={data.get('wording_status')}"
        )
        return

    panel = run_phase12b()
    gate = panel["gate_heldout"]
    print(f"\nwrote {panel['_artifact']}")
    print(f"12B gate: {'YES' if gate['pass'] else 'NO'}")
    print(gate["note"])
    print(f"controls_ok: {gate['controls_ok']}")
    print(f"direction_ok: {gate['direction_ok']}")
    print(f"band_results: {gate['band_results']}")
    if not gate["pass"]:
        sys.exit(2)


if __name__ == "__main__":
    main()
