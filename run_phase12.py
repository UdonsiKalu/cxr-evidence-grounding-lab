#!/usr/bin/env python3
"""Phase-12A: margin regime — symmetric α panel on Ph11 false-X targets."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from n2s_lab.phase12_margin_regime import (  # noqa: E402
    ALPHA_PANEL_TARGETS,
    TARGET_IDS,
    run_phase12,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase-12A margin regime panel")
    parser.add_argument(
        "--selftest",
        action="store_true",
        help="check Ph10/11 prerequisites + protocol file",
    )
    args = parser.parse_args()

    if args.selftest:
        p10 = ROOT / "artifacts" / "phase10-bc-e1-panel.json"
        p11 = ROOT / "artifacts" / "phase11-generalization-panel.json"
        proto = ROOT / "docs" / "PHASE12-PROTOCOL.md"
        h11 = ROOT / "data" / "heldout-phase11.json"
        for path in (p10, p11, proto, h11):
            if not path.exists():
                print(f"missing {path}")
                sys.exit(1)
        if not json.loads(p10.read_text()).get("gate_activation_intervention", {}).get(
            "pass"
        ):
            print("phase10 gate not YES")
            sys.exit(1)
        print(
            f"phase12A selftest OK — targets {TARGET_IDS} α={list(ALPHA_PANEL_TARGETS)}"
        )
        return

    panel = run_phase12()
    gate = panel["gate_margin_regime"]
    print(f"\nwrote {panel['_artifact']}")
    print(f"12A mechanism gate: {'YES' if gate['pass'] else 'NO'}")
    print(gate["note"])
    print(f"controls_ok: {gate['controls_ok']}")
    print(f"margin_monotone: {gate['margin_monotone_by_case']}")
    print(f"flip_first_alpha: {gate['flip_first_alpha']}")
    print(f"baseline_margin: {gate['baseline_margin_by_target']}")
    if not gate["pass"]:
        sys.exit(2)


if __name__ == "__main__":
    main()
