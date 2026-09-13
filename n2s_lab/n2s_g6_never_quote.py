"""Goal 6 — Dual never↔given quote repair (S-never-quote).

Predeclared from Goal 5 shape tag, not leftover IDs. Composition:

  1. Frozen Goal 4 mixed-span (gated promote + simultaneous 1L flag)
  2. If quotes/notes claim never/no-prior therapy AND another span (or
     administration_status=given) claims cycles/completed/received, copy
     those spans into contradiction cues and set present=true

Frozen ground() conflict blob omits quotes; this does not rewrite ground().
Does not mutate Goal 3/4. CPU on Dual-fill extracts.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from typing import Any

from .auto_contract import classify_case
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
    relabel_failure_polarity_from_text_gated_mixed,
    relabel_failure_polarity_from_text_gated_mixed_never,
)
from .paths import ARTIFACTS_DIR
from .types import Extraction

OUT_PATH = ARTIFACTS_DIR / "n2s-g6-never-quote-panel.json"
MIXED = "failure_evidence_from_extract_text_gated_mixed"
NEVER = "failure_evidence_from_extract_text_gated_mixed_never"
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
    never: bool,
    role_fn,
) -> list[dict[str, Any]]:
    relabel = (
        relabel_failure_polarity_from_text_gated_mixed_never
        if never
        else relabel_failure_polarity_from_text_gated_mixed
    )
    intervention = NEVER if never else MIXED
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


def _n108_miss(
    notes: dict[str, Any],
    extracts: dict[str, Any],
    *,
    never: bool,
) -> dict[str, Any]:
    relabel = (
        relabel_failure_polarity_from_text_gated_mixed_never
        if never
        else relabel_failure_polarity_from_text_gated_mixed
    )
    miss: list[str] = []
    flipped: list[str] = []
    repaired: list[str] = []
    for n in notes.values():
        rec = extracts.get(n["id"]) or {}
        if not rec.get("extract"):
            continue
        gold = n["expected"]
        ex = Extraction.from_dict(rec["extract"])
        before = atoms_payload(ex)
        after_ex, _meta = relabel(ex)
        after = atoms_payload(after_ex)
        cls_b = classify_case(
            gold=gold, verdict=before["verdict"], disposition="AUTO"
        )
        cls_a = classify_case(
            gold=gold, verdict=after["verdict"], disposition="AUTO"
        )
        if cls_a["match_gold"] is not True:
            miss.append(n["id"])
        if cls_b["match_gold"] is True and cls_a["match_gold"] is not True:
            flipped.append(n["id"])
        if cls_b["match_gold"] is not True and cls_a["match_gold"] is True:
            repaired.append(n["id"])
    return {
        "n_miss": len(miss),
        "ids_miss": miss,
        "flipped_away": flipped,
        "match_repaired": repaired,
    }


def build_panel() -> dict[str, Any]:
    notes = {n["id"]: n for n in load_dual_library(include_heldout=True)}
    extracts = _load_fill_extracts()
    design_ids = list(BIND_IDS) + list(XSPAN_IDS) + list(NEG_IDS)
    heldout_ids = [n["id"] for n in notes.values() if not n["phenotype_design"]]

    def design_role(n: dict[str, Any]) -> str:
        return ROLE_OF.get(n["id"]) or f"gold_{n['expected']}"

    def held_role(n: dict[str, Any]) -> str:
        return f"gold_{n['expected']}"

    mixed_design = _score_ids(
        notes, extracts, design_ids, never=False, role_fn=design_role
    )
    never_design = _score_ids(
        notes, extracts, design_ids, never=True, role_fn=design_role
    )
    mixed_held = _score_ids(
        notes, extracts, heldout_ids, never=False, role_fn=held_role
    )
    never_held = _score_ids(
        notes, extracts, heldout_ids, never=True, role_fn=held_role
    )
    m_des = _arm_counts(mixed_design)
    n_des = _arm_counts(never_design)
    m_h = _arm_counts(mixed_held)
    n_h = _arm_counts(never_held)
    m_bind = m_des.get("P-bind") or {}
    n_bind = n_des.get("P-bind") or {}
    m_xspan = m_des.get("P-xspan") or {}
    n_xspan = n_des.get("P-xspan") or {}
    m_neg = m_des.get("P-neg") or {}
    n_neg = n_des.get("P-neg") or {}
    kept_bind = sorted(
        set(m_bind.get("ids_repaired") or []) & set(n_bind.get("ids_repaired") or [])
    )
    lost_bind = sorted(
        set(m_bind.get("ids_repaired") or []) - set(n_bind.get("ids_repaired") or [])
    )
    n108_mixed = _n108_miss(notes, extracts, never=False)
    n108_never = _n108_miss(notes, extracts, never=True)
    held_leak = bool(
        (n_h.get("gold_NOT_SATISFIED") or {}).get("collateral_to_SATISFIED", 0)
        or (n_h.get("gold_CONTRADICTION") or {}).get("collateral_to_SATISFIED", 0)
    )
    design_specificity = (
        n_xspan.get("collateral_to_SATISFIED", 0) == 0
        and n_neg.get("collateral_to_SATISFIED", 0) == 0
        and n_bind.get("flipped_away", 0) == 0
    )
    map_cell = {
        "mixed_pbind_k": m_bind.get("repaired", 0),
        "never_pbind_k": n_bind.get("repaired", 0),
        "pbind_n": n_bind.get("n", 0),
        "kept_mixed_bind_repairs": kept_bind,
        "lost_mixed_bind_repairs": lost_bind,
        "mixed_neg_collateral": m_neg.get("collateral_to_SATISFIED", 0),
        "never_neg_collateral": n_neg.get("collateral_to_SATISFIED", 0),
        "mixed_xspan_match_repaired": m_xspan.get("match_repaired", 0),
        "never_xspan_match_repaired": n_xspan.get("match_repaired", 0),
        "never_xspan_match_ids": n_xspan.get("ids_match_repaired") or [],
        "design_xspan_all_match": (
            n_xspan.get("correct_stay", 0) + n_xspan.get("match_repaired", 0)
            == n_xspan.get("n", 0)
        ),
        "specificity_holds": design_specificity,
        "heldout_leak": held_leak,
        "heldout_contra_match_repaired": (n_h.get("gold_CONTRADICTION") or {}).get(
            "match_repaired", 0
        ),
        "heldout_contra_match_ids": (n_h.get("gold_CONTRADICTION") or {}).get(
            "ids_match_repaired"
        )
        or [],
        "heldout_sat_repaired": (n_h.get("gold_SATISFIED") or {}).get("repaired", 0),
        "n108_miss_mixed": n108_mixed["n_miss"],
        "n108_miss_never": n108_never["n_miss"],
        "n108_flips_vs_raw": n108_never["flipped_away"],
        "n108_new_flips_vs_mixed": [
            i
            for i in n108_never["flipped_away"]
            if i not in n108_mixed["flipped_away"]
        ],
        "n108_new_repairs": [
            i
            for i in n108_never["match_repaired"]
            if i not in n108_mixed["match_repaired"]
        ],
        "ground_rewritten": False,
        "g3_g4_mutated": False,
        "locator_chose_repair_class": False,
        "is_selected_editor": False,
    }
    panel = {
        "ok": True,
        "kind": "n2s_g6_never_quote_v1",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "protocol": "docs/TRACKB-FAILURE-INTERVENTION-MAP.md",
        "intervention": NEVER,
        "gate": "asserted_failure_marker+simultaneous_1L+never_vs_given",
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
            "Goal 3/4 retune",
            "ground() rewrite",
            "CAA",
            "frozen-d",
            "Dual_full rescore",
            "G5 census overwrite",
            ":8260 GUI",
            "auto-correct",
        ],
        "counts": {
            "mixed_design": m_des,
            "never_design": n_des,
            "mixed_heldout": m_h,
            "never_heldout": n_h,
            "n108_mixed": n108_mixed,
            "n108_never": n108_never,
        },
        "map_cell": map_cell,
        "rows": {
            "mixed_design": mixed_design,
            "never_design": never_design,
            "mixed_heldout": mixed_held,
            "never_heldout": never_held,
        },
        "note": (
            "Never/no-prior in quotes copied into contradiction cues so frozen "
            "ground() can score never↔given. ground() itself is unchanged."
        ),
    }
    OUT_PATH.write_text(json.dumps(panel, indent=2) + "\n", encoding="utf-8")
    return panel


def selftest() -> None:
    batch1_selftest()
    never_ex = Extraction.from_dict(
        {
            "stated_line": "first",
            "administration_status": "given",
            "quotes": [
                "never received systemic therapy",
                "progressed after six cycles of first-line carboplatin",
            ],
            "outcome_statements": [{"text": "progressed", "polarity": "failure"}],
            "contradiction_present": False,
        }
    )
    after_n, meta_n = relabel_failure_polarity_from_text_gated_mixed_never(never_ex)
    assert meta_n["flagged_never_vs_given"] is True
    assert after_n.contradiction_present is True
    assert atoms_payload(after_n)["verdict"] == "CONTRADICTION"
    planned = Extraction.from_dict(
        {
            "stated_line": "first",
            "administration_status": "planned",
            "quotes": ["No prior systemic therapy. Starting first-line FOLFIRINOX next week."],
            "contradiction_present": False,
        }
    )
    after_p, meta_p = relabel_failure_polarity_from_text_gated_mixed_never(planned)
    assert meta_p.get("flagged_never_vs_given") is False
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
    after_s, meta_s = relabel_failure_polarity_from_text_gated_mixed_never(seq)
    assert meta_s.get("flagged_never_vs_given") is False
    assert atoms_payload(after_s)["verdict"] == "SATISFIED"
    notes = load_dual_library(include_heldout=True)
    assert len(notes) == 108
    assert FILL_EXTRACT_PATH.is_file()


def main() -> None:
    parser = argparse.ArgumentParser(description="Goal 6 Dual never-quote repair")
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
                    "map_cell": panel["map_cell"],
                    "never_design": panel["counts"]["never_design"],
                    "never_heldout": panel["counts"]["never_heldout"],
                    "artifact": str(OUT_PATH),
                },
                indent=2,
            )
        )
        return
    parser.print_help()


if __name__ == "__main__":
    main()
