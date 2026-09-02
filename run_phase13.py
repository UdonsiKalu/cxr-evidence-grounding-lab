#!/usr/bin/env python3
"""Phase-13: representation geometry panel (course false-X vs contradiction)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from n2s_lab.phase13_geometry import CONTRA_IDS, COURSE_IDS, run_phase13  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase-13 geometry panel")
    parser.add_argument("--selftest", action="store_true")
    args = parser.parse_args()

    if args.selftest:
        p10 = ROOT / "artifacts" / "phase10-bc-e1-panel.json"
        proto = ROOT / "docs" / "PHASE13-PROTOCOL.md"
        for path in (p10, proto):
            if not path.exists():
                print(f"missing {path}")
                sys.exit(1)
        if not json.loads(p10.read_text()).get("gate_activation_intervention", {}).get(
            "pass"
        ):
            print("phase10 gate not YES")
            sys.exit(1)
        print(
            f"phase13 selftest OK — course={list(COURSE_IDS)} "
            f"contra={list(CONTRA_IDS)}"
        )
        return

    panel = run_phase13()
    gate = panel["gate_geometry"]
    print(f"\nwrote {panel['_artifact']}")
    print(f"13 gate: {'YES' if gate['pass'] else 'NO'}")
    print(gate["note"])
    print(
        f"usable course/contra: {gate['n_course_usable']}/{gate['n_contradiction_usable']}"
    )
    print(f"cos@1.00 course: {gate['cosine_1_00_course']}")
    print(f"cos@1.00 contra: {gate['cosine_1_00_contradiction']}")
    print(f"ranges overlap: {gate['cosine_1_00_ranges_overlap']} gap={gate['cosine_1_00_gap']}")
    print(f"LOO probe acc: {gate['loo_probe_acc']}")
    if not gate["pass"]:
        sys.exit(2)


if __name__ == "__main__":
    main()
