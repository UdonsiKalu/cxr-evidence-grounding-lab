"""Goal 4 — Dual mixed-span / simultaneous 1L conflict flag.

Predeclared from P-xspan (two incompatible first-line claims), not from
held-out leak IDs. Composition:

  1. Goal 3 gated quote-promote (asserted, non-negated failure markers)
  2. If frozen ground()'s hard-simultaneous check is true and the extract
     omitted contradiction.present, set present=true

Does not rewrite ground(). Does not mutate Goal 3. CPU on Dual-fill extracts.
Held-out scored with the frozen rule (no retune).
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from typing import Any

from .n2s_batch1_quote_promote import (
    ANCHOR_IDS,
    BIND_IDS,
    NEG_IDS,
    ROLE_OF,
    XSPAN_IDS,
    _arm_counts,
    apply_row,
    selftest as batch1_selftest,
)
from .n2s_g2_analog_fill import EXTRACT_PATH as FILL_EXTRACT_PATH
from .n2s_note_library import load_dual_library
from .n2s_phase2_translate import (
    atoms_payload,
    relabel_failure_polarity_from_text_gated,
    relabel_failure_polarity_from_text_gated_mixed,
)
from .paths import ARTIFACTS_DIR
from .types import Extraction

OUT_PATH = ARTIFACTS_DIR / "n2s-g4-mixed-span-panel.json"
GATED = "failure_evidence_from_extract_text_gated"
MIXED = "failure_evidence_from_extract_text_gated_mixed"
PROMPT = "analog_schema_in_system"


def _load_fill_extracts() -> dict[str, Any]:
    if not FILL_EXTRACT_PATH.is_file():
        raise SystemExit(f"missing {FILL_EXTRACT_PATH} — run Goal 2 recover first")
    return json.loads(FILL_EXTRACT_PATH.read_text(encoding="utf-8")).get("extracts") or {}


def _score_ids(
    notes: dict[str, Any],
    extracts: dict[str, Any],
    ids: list[str],
    *,
    mixed: bool,
    role_fn,
) -> list[dict[str, Any]]:
    relabel = (
        relabel_failure_polarity_from_text_gated_mixed
        if mixed
        else relabel_failure_polarity_from_text_gated
    )
    intervention = MIXED if mixed else GATED
    return [
        apply_row(
            notes[i],
            extracts.get(i) or {},
            prompt=PROMPT,
            role=role_fn(notes[i]),
            relabel=relabel,
            intervention=intervention,
        )
        for i in ids
    ]


def build_panel() -> dict[str, Any]:
    notes = {n["id"]: n for n in load_dual_library(include_heldout=True)}
    extracts = _load_fill_extracts()
    design_ids = list(BIND_IDS) + list(XSPAN_IDS) + list(NEG_IDS)
    heldout_ids = [n["id"] for n in notes.values() if not n["phenotype_design"]]

    def design_role(n: dict[str, Any]) -> str:
        return ROLE_OF.get(n["id"]) or f"gold_{n['expected']}"

    def held_role(n: dict[str, Any]) -> str:
        return f"gold_{n['expected']}"

    gated_design = _score_ids(
        notes, extracts, design_ids, mixed=False, role_fn=design_role
    )
    mixed_design = _score_ids(
        notes, extracts, design_ids, mixed=True, role_fn=design_role
    )
    gated_held = _score_ids(
        notes, extracts, heldout_ids, mixed=False, role_fn=held_role
    )
    mixed_held = _score_ids(
        notes, extracts, heldout_ids, mixed=True, role_fn=held_role
    )

    g_des = _arm_counts(gated_design)
    m_des = _arm_counts(mixed_design)
    g_h = _arm_counts(gated_held)
    m_h = _arm_counts(mixed_held)
    g_bind = g_des.get("P-bind") or {}
    m_bind = m_des.get("P-bind") or {}
    g_xspan = g_des.get("P-xspan") or {}
    m_xspan = m_des.get("P-xspan") or {}
    g_neg = g_des.get("P-neg") or {}
    m_neg = m_des.get("P-neg") or {}

    kept_bind = sorted(set(g_bind.get("ids_repaired") or []) & set(m_bind.get("ids_repaired") or []))
    lost_bind = sorted(set(g_bind.get("ids_repaired") or []) - set(m_bind.get("ids_repaired") or []))

    held_neg_ids = list((m_h.get("gold_NOT_SATISFIED") or {}).get("ids_collateral") or [])
    held_x_ids = list((m_h.get("gold_CONTRADICTION") or {}).get("ids_collateral") or [])
    held_leak = bool(held_neg_ids or held_x_ids)
    design_specificity = (
        m_xspan.get("collateral_to_SATISFIED", 0) == 0
        and m_neg.get("collateral_to_SATISFIED", 0) == 0
        and m_bind.get("flipped_away", 0) == 0
    )
    map_cell = {
        "gated_pbind_k": g_bind.get("repaired", 0),
        "mixed_pbind_k": m_bind.get("repaired", 0),
        "pbind_n": m_bind.get("n", 0),
        "kept_gated_bind_repairs": kept_bind,
        "lost_gated_bind_repairs": lost_bind,
        "gated_neg_collateral": g_neg.get("collateral_to_SATISFIED", 0),
        "mixed_neg_collateral": m_neg.get("collateral_to_SATISFIED", 0),
        "gated_xspan_collateral": g_xspan.get("collateral_to_SATISFIED", 0),
        "mixed_xspan_collateral": m_xspan.get("collateral_to_SATISFIED", 0),
        "gated_xspan_match_repaired": g_xspan.get("match_repaired", 0),
        "mixed_xspan_match_repaired": m_xspan.get("match_repaired", 0),
        "mixed_xspan_match_ids": m_xspan.get("ids_match_repaired") or [],
        "specificity_holds": design_specificity,
        "heldout_leak": held_leak,
        "heldout_leak_ids": sorted(held_neg_ids + held_x_ids),
        "heldout_sat_repaired": m_h.get("gold_SATISFIED", {}).get("repaired", 0),
        "heldout_contra_match_repaired": (m_h.get("gold_CONTRADICTION") or {}).get(
            "match_repaired", 0
        ),
        "heldout_contra_match_ids": (m_h.get("gold_CONTRADICTION") or {}).get(
            "ids_match_repaired"
        )
        or [],
        "dual_analog_user_evidence": True,
        "schema_in_system": True,
        "rule_changed_after_heldout": False,
        "ground_rewritten": False,
        "locator_chose_repair_class": False,
        "is_selected_editor": False,
    }
    panel = {
        "ok": True,
        "kind": "n2s_g4_mixed_span_v1",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "protocol": "docs/TRACKB-FAILURE-INTERVENTION-MAP.md",
        "intervention": MIXED,
        "gate": "asserted_failure_marker+simultaneous_1L",
        "extracts": str(FILL_EXTRACT_PATH.name),
        "prompt": PROMPT,
        "n_design": len(design_ids),
        "n_heldout": len(heldout_ids),
        "ids": {
            "P-bind": list(BIND_IDS),
            "P-xspan": list(XSPAN_IDS),
            "P-neg": list(NEG_IDS),
            "anchors_score_only": list(ANCHOR_IDS),
            "heldout": heldout_ids,
        },
        "not": [
            "Goal 3 gate retune",
            "silent retune from C8",
            "ground() rewrite",
            "hidden-state patch",
            "frozen-d",
            "CAA",
            "Dual_full rescore",
            "G2/G3 artifact overwrite",
            ":8260 GUI",
            "auto-correct",
        ],
        "counts": {
            "gated_design": g_des,
            "mixed_design": m_des,
            "gated_heldout": g_h,
            "mixed_heldout": m_h,
        },
        "map_cell": map_cell,
        "rows": {
            "gated_design": gated_design,
            "mixed_design": mixed_design,
            "gated_heldout": gated_held,
            "mixed_heldout": mixed_held,
        },
        "note": (
            "Goal 3 gated promote plus simultaneous 1L contradiction flag. "
            "Uses frozen ground() hard-simultaneous check; does not rewrite ground()."
        ),
    }
    OUT_PATH.write_text(json.dumps(panel, indent=2) + "\n", encoding="utf-8")
    return panel


def selftest() -> None:
    batch1_selftest()
    mixed = Extraction.from_dict(
        {
            "stated_line": "first",
            "administration_status": "given",
            "quotes": [
                "progression on first-line regimen",
                "excellent ongoing response to first-line regimen; continue current treatment",
            ],
            "outcome_statements": [
                {"text": "excellent ongoing response", "polarity": "response"}
            ],
            "contradiction_present": False,
            "implicit_cues": ["response_then_progression"],
        }
    )
    after_m, meta_m = relabel_failure_polarity_from_text_gated_mixed(mixed)
    assert meta_m["kind"] == MIXED
    assert meta_m["flagged_simultaneous_1L"] is True
    assert after_m.contradiction_present is True
    assert atoms_payload(after_m)["verdict"] == "CONTRADICTION"

    seq = Extraction.from_dict(
        {
            "stated_line": "first",
            "administration_status": "given",
            "quotes": [
                "Responded to first-line carboplatin for eight months, then progressed"
            ],
            "outcome_statements": [
                {
                    "text": "responded to first-line carboplatin for eight months, then progressed",
                    "polarity": "response",
                }
            ],
            "contradiction_present": False,
            "implicit_cues": ["response_then_progression"],
        }
    )
    after_s, meta_s = relabel_failure_polarity_from_text_gated_mixed(seq)
    assert meta_s["flagged_simultaneous_1L"] is False
    assert after_s.contradiction_present is not True
    assert atoms_payload(after_s)["verdict"] == "SATISFIED"

    denied = Extraction.from_dict(
        {
            "stated_line": "first",
            "administration_status": "given",
            "quotes": ["ongoing response; no progression documented"],
            "outcome_statements": [
                {"text": "ongoing response", "polarity": "response"}
            ],
            "contradiction_present": False,
        }
    )
    after_d, meta_d = relabel_failure_polarity_from_text_gated_mixed(denied)
    assert meta_d["flagged_simultaneous_1L"] is False
    assert meta_d["n_polarities_relabeled"] == 0
    assert not meta_d["promoted_failure_from"]
    assert atoms_payload(after_d)["verdict"] != "SATISFIED"
    notes = load_dual_library(include_heldout=True)
    assert len(notes) == 108
    assert FILL_EXTRACT_PATH.is_file()


def main() -> None:
    parser = argparse.ArgumentParser(description="Goal 4 Dual mixed-span flag")
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
                    "counts": {
                        "gated_design": panel["counts"]["gated_design"],
                        "mixed_design": panel["counts"]["mixed_design"],
                        "gated_heldout": panel["counts"]["gated_heldout"],
                        "mixed_heldout": panel["counts"]["mixed_heldout"],
                    },
                    "map_cell": panel["map_cell"],
                    "artifact": str(OUT_PATH),
                },
                indent=2,
            )
        )
        return
    parser.print_help()


if __name__ == "__main__":
    main()
