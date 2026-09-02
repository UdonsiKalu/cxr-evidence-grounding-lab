#!/usr/bin/env python3
"""Phase-14: gated activation-steer panel (cos@1.00 vs τ)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from n2s_lab.phase14_gated import TAU_COS_1_00, run_phase14  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase-14 gated steer panel")
    parser.add_argument("--selftest", action="store_true")
    args = parser.parse_args()

    if args.selftest:
        p10 = ROOT / "artifacts" / "phase10-bc-e1-panel.json"
        p13 = ROOT / "artifacts" / "phase13-geometry-panel.json"
        proto = ROOT / "docs" / "PHASE14-PROTOCOL.md"
        held = ROOT / "data" / "heldout-phase14.json"
        for path in (p10, p13, proto, held):
            if not path.exists():
                print(f"missing {path}")
                sys.exit(1)
        if not json.loads(p10.read_text()).get("gate_activation_intervention", {}).get(
            "pass"
        ):
            print("phase10 gate not YES")
            sys.exit(1)
        data = json.loads(held.read_text())
        if data.get("wording_status") != "accepted":
            print(f"held-out wording not accepted ({data.get('wording_status')})")
            sys.exit(1)
        print(
            f"phase14 selftest OK — {len(data['cases'])} cases · "
            f"τ={TAU_COS_1_00} · wording={data.get('wording_status')}"
        )
        return

    panel = run_phase14()
    gate = panel["gate_gated"]
    print(f"\nwrote {panel['_artifact']}")
    print(f"14 gate: {'YES' if gate['pass'] else 'NO'}")
    print(gate["note"])
    print(f"contra_all_steer_off: {gate['contra_all_steer_off']}")
    print(f"n1_ok: {gate['n1_ok']}")
    print(f"utility_flips: {gate['utility_flips']}")
    print(f"soft_agreement: {gate['soft_gate_gold_agreement']}")
    if not gate["pass"]:
        sys.exit(2)


if __name__ == "__main__":
    main()
