"""Held-out score of the frozen Batch 1 quote-promote. Then stop.

Same `failure_evidence_from_extract_text`. No retune. Roles are gold labels
(not new P-* design). TFT excluded. Does not overwrite Batch 1 panel.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from typing import Any

from .n2s_batch0x_census import ANALOG_PATH, SCHEMA_PATH
from .n2s_batch1_quote_promote import (
    _arm_counts,
    _load_extracts,
    apply_row,
    selftest as batch1_selftest,
)
from .n2s_note_library import load_dual_library
from .paths import ARTIFACTS_DIR

OUT_PATH = ARTIFACTS_DIR / "n2s-batch1h-heldout-promote-panel.json"


def build_panel() -> dict[str, Any]:
    notes = [
        n for n in load_dual_library(include_heldout=True) if not n["phenotype_design"]
    ]
    analog_ex = _load_extracts(ANALOG_PATH)
    schema_ex = _load_extracts(SCHEMA_PATH)
    analog_rows = [
        apply_row(n, analog_ex.get(n["id"]) or {}, prompt="analog", role=f"gold_{n['expected']}")
        for n in notes
    ]
    schema_rows = [
        apply_row(n, schema_ex.get(n["id"]) or {}, prompt="schema", role=f"gold_{n['expected']}")
        for n in notes
    ]
    schema_c = _arm_counts(schema_rows)
    analog_c = _arm_counts(analog_rows)
    leak_repeats = (
        schema_c.get("gold_NOT_SATISFIED", {}).get("collateral_to_SATISFIED", 0) > 0
        or schema_c.get("gold_CONTRADICTION", {}).get("collateral_to_SATISFIED", 0) > 0
    )
    panel = {
        "ok": True,
        "kind": "n2s_batch1h_heldout_promote_v1",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "protocol": "docs/TRACKB-FAILURE-INTERVENTION-MAP.md",
        "intervention": "failure_evidence_from_extract_text",
        "rule_changed": False,
        "n": len(notes),
        "ids": [n["id"] for n in notes],
        "not": [
            "promote retune",
            "new P-* from held-out",
            "temporal-family-test",
            "hidden-state patch",
            "ground() rewrite",
            "CAA",
            "Dual_full rescore",
            "Batch 1 panel overwrite",
            ":8260 GUI",
        ],
        "counts": {"schema": schema_c, "analog": analog_c},
        "finding": {
            "analog_repaired": analog_c.get("gold_SATISFIED", {}).get("repaired", 0),
            "analog_fired": sum(1 for r in analog_rows if r["fired"]),
            "schema_fired": sum(1 for r in schema_rows if r["fired"]),
            "schema_sat_repaired": schema_c.get("gold_SATISFIED", {}).get("repaired", 0),
            "schema_sat_n": schema_c.get("gold_SATISFIED", {}).get("n", 0),
            "schema_neg_collateral": schema_c.get("gold_NOT_SATISFIED", {}).get(
                "collateral_to_SATISFIED", 0
            ),
            "schema_xspan_collateral": schema_c.get("gold_CONTRADICTION", {}).get(
                "collateral_to_SATISFIED", 0
            ),
            "leak_repeats": leak_repeats,
            "do_not_retune": True,
            "stop": True,
        },
        "rows": {"schema": schema_rows, "analog": analog_rows},
        "note": (
            "Frozen Batch 1 promote on held-out Dual notes. Gold roles only. "
            "Leak repeating means the design-set P-neg/C4 flips were not a 28-note accident."
        ),
    }
    OUT_PATH.write_text(json.dumps(panel, indent=2) + "\n", encoding="utf-8")
    return panel


def selftest() -> None:
    batch1_selftest()
    notes = [n for n in load_dual_library(include_heldout=True) if not n["phenotype_design"]]
    assert len(notes) == 44, len(notes)
    assert all(n["id"] != "TFT_E1" for n in notes)
    assert all(not n["phenotype_design"] for n in notes)


def main() -> None:
    parser = argparse.ArgumentParser(description="Held-out frozen quote-promote, then stop")
    parser.add_argument("--selftest", action="store_true")
    parser.add_argument("--panel", action="store_true")
    args = parser.parse_args()
    if args.selftest:
        selftest()
        print("selftest ok")
        return
    if args.panel:
        panel = build_panel()
        print(
            json.dumps(
                {
                    "n": panel["n"],
                    "counts": panel["counts"],
                    "finding": panel["finding"],
                    "artifact": str(OUT_PATH),
                },
                indent=2,
            )
        )
        return
    parser.print_help()


if __name__ == "__main__":
    main()
