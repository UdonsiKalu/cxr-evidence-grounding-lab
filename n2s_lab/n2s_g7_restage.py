"""Goal 7 — Dual restaging confirmation (hedge-clear + stable-seq undo).

Predeclared from Goal 5 leftover shapes S-hedge and S-g4-stable-seq.
After frozen Goal 6:

  1. If a later span confirms failure, clear extractor uncertainty.present
     (possible/pseudo resolved by restaging).
  2. If Goal 4 flagged simultaneous 1L only from stable disease then later
     progression, and the note does not say continue-current, revert
     contradiction.present (temporal restaging, not P-xspan).

Does not mutate Goal 3/4/6. Does not rewrite ground(). CPU on Dual-fill.
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
    relabel_failure_polarity_from_text_gated_mixed_never,
    relabel_failure_polarity_from_text_gated_mixed_never_restage,
)
from .paths import ARTIFACTS_DIR
from .types import Extraction

OUT_PATH = ARTIFACTS_DIR / "n2s-g7-restage-panel.json"
NEVER = "failure_evidence_from_extract_text_gated_mixed_never"
RESTAGE = "failure_evidence_from_extract_text_gated_mixed_never_restage"
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
    restage: bool,
    role_fn,
) -> list[dict[str, Any]]:
    relabel = (
        relabel_failure_polarity_from_text_gated_mixed_never_restage
        if restage
        else relabel_failure_polarity_from_text_gated_mixed_never
    )
    intervention = RESTAGE if restage else NEVER
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
    restage: bool,
) -> dict[str, Any]:
    relabel = (
        relabel_failure_polarity_from_text_gated_mixed_never_restage
        if restage
        else relabel_failure_polarity_from_text_gated_mixed_never
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
        cls_b = classify_case(gold=gold, verdict=before["verdict"], disposition="AUTO")
        cls_a = classify_case(gold=gold, verdict=after["verdict"], disposition="AUTO")
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

    never_design = _score_ids(
        notes, extracts, design_ids, restage=False, role_fn=design_role
    )
    rest_design = _score_ids(
        notes, extracts, design_ids, restage=True, role_fn=design_role
    )
    never_held = _score_ids(
        notes, extracts, heldout_ids, restage=False, role_fn=held_role
    )
    rest_held = _score_ids(
        notes, extracts, heldout_ids, restage=True, role_fn=held_role
    )
    n_des = _arm_counts(never_design)
    r_des = _arm_counts(rest_design)
    n_h = _arm_counts(never_held)
    r_h = _arm_counts(rest_held)
    n_bind = n_des.get("P-bind") or {}
    r_bind = r_des.get("P-bind") or {}
    n_xspan = n_des.get("P-xspan") or {}
    r_xspan = r_des.get("P-xspan") or {}
    n_neg = n_des.get("P-neg") or {}
    r_neg = r_des.get("P-neg") or {}
    kept_bind = sorted(
        set(n_bind.get("ids_repaired") or []) & set(r_bind.get("ids_repaired") or [])
    )
    lost_bind = sorted(
        set(n_bind.get("ids_repaired") or []) - set(r_bind.get("ids_repaired") or [])
    )
    n108_never = _n108_miss(notes, extracts, restage=False)
    n108_rest = _n108_miss(notes, extracts, restage=True)
    held_leak = bool(
        (r_h.get("gold_NOT_SATISFIED") or {}).get("collateral_to_SATISFIED", 0)
        or (r_h.get("gold_CONTRADICTION") or {}).get("collateral_to_SATISFIED", 0)
    )
    design_specificity = (
        r_xspan.get("collateral_to_SATISFIED", 0) == 0
        and r_neg.get("collateral_to_SATISFIED", 0) == 0
        and r_bind.get("flipped_away", 0) == 0
    )
    map_cell = {
        "never_pbind_k": n_bind.get("repaired", 0),
        "restage_pbind_k": r_bind.get("repaired", 0),
        "pbind_n": r_bind.get("n", 0),
        "kept_never_bind_repairs": kept_bind,
        "lost_never_bind_repairs": lost_bind,
        "restage_bind_ids": r_bind.get("ids_repaired") or [],
        "never_neg_collateral": n_neg.get("collateral_to_SATISFIED", 0),
        "restage_neg_collateral": r_neg.get("collateral_to_SATISFIED", 0),
        "never_xspan_match_repaired": n_xspan.get("match_repaired", 0),
        "restage_xspan_match_repaired": r_xspan.get("match_repaired", 0),
        "design_xspan_all_match": (
            r_xspan.get("correct_stay", 0) + r_xspan.get("match_repaired", 0)
            == r_xspan.get("n", 0)
        ),
        "specificity_holds": design_specificity,
        "heldout_leak": held_leak,
        "heldout_sat_repaired": (r_h.get("gold_SATISFIED") or {}).get("repaired", 0),
        "n108_miss_never": n108_never["n_miss"],
        "n108_miss_restage": n108_rest["n_miss"],
        "n108_new_flips_vs_never": [
            i
            for i in n108_rest["flipped_away"]
            if i not in n108_never["flipped_away"]
        ],
        "n108_new_repairs": [
            i
            for i in n108_rest["match_repaired"]
            if i not in n108_never["match_repaired"]
        ],
        "n108_restored_vs_never": [
            i
            for i in n108_never["flipped_away"]
            if i not in n108_rest["flipped_away"]
        ],
        "ground_rewritten": False,
        "g3_g4_g6_mutated": False,
        "locator_chose_repair_class": False,
        "is_selected_editor": False,
    }
    panel = {
        "ok": True,
        "kind": "n2s_g7_restage_v1",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "protocol": "docs/TRACKB-FAILURE-INTERVENTION-MAP.md",
        "intervention": RESTAGE,
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
            "Goal 3/4/6 retune in place",
            "ground() rewrite",
            "CAA",
            "frozen-d",
            "Dual_full rescore",
            ":8260 GUI",
            "auto-correct",
        ],
        "counts": {
            "never_design": n_des,
            "restage_design": r_des,
            "never_heldout": n_h,
            "restage_heldout": r_h,
            "n108_never": n108_never,
            "n108_restage": n108_rest,
        },
        "map_cell": map_cell,
        "rows": {
            "never_design": never_design,
            "restage_design": rest_design,
            "never_heldout": never_held,
            "restage_heldout": rest_held,
        },
        "note": (
            "Restaging confirmation: confirmed failure clears hedge uncertainty; "
            "stable-then-progress without continue-current undoes G4 X. "
            "ground() unchanged."
        ),
    }
    OUT_PATH.write_text(json.dumps(panel, indent=2) + "\n", encoding="utf-8")
    return panel


def selftest() -> None:
    batch1_selftest()
    hedge = Extraction.from_dict(
        {
            "stated_line": "first",
            "administration_status": "given",
            "quotes": ["possible progression", "clear progressive disease"],
            "outcome_statements": [
                {"text": "possible progression", "polarity": "failure"}
            ],
            "uncertainty_present": True,
            "uncertainty_cues": ["possible"],
            "contradiction_present": False,
        }
    )
    after_h, meta_h = relabel_failure_polarity_from_text_gated_mixed_never_restage(hedge)
    assert meta_h["cleared_hedge_uncertainty"] is True
    assert after_h.uncertainty_present is False
    assert atoms_payload(after_h)["verdict"] == "SATISFIED"
    stable = Extraction.from_dict(
        {
            "stated_line": "first",
            "administration_status": "given",
            "quotes": [
                "imaging after two cycles showed stable disease",
                "restaging at cycle 5 demonstrated progressive osseous metastases",
            ],
            "outcome_statements": [
                {"text": "stable disease", "polarity": "response"},
                {"text": "progressive osseous metastases", "polarity": "failure"},
            ],
            "implicit_cues": ["response_then_progression"],
            "contradiction_present": False,
        }
    )
    after_s, meta_s = relabel_failure_polarity_from_text_gated_mixed_never_restage(stable)
    assert meta_s["reverted_stable_seq"] is True
    assert after_s.contradiction_present is not True
    assert atoms_payload(after_s)["verdict"] == "SATISFIED"
    mixed = Extraction.from_dict(
        {
            "stated_line": "first",
            "administration_status": "given",
            "quotes": [
                "Progressed on carboplatin/paclitaxel",
                "excellent ongoing response to first-line carboplatin/paclitaxel; continue current treatment",
            ],
            "outcome_statements": [
                {"text": "progressed on carboplatin/paclitaxel", "polarity": "failure"},
                {
                    "text": "excellent ongoing response; continue current treatment",
                    "polarity": "response",
                },
            ],
            "implicit_cues": ["response_then_progression"],
            "contradiction_present": False,
        }
    )
    after_m, meta_m = relabel_failure_polarity_from_text_gated_mixed_never_restage(mixed)
    assert meta_m.get("reverted_stable_seq") is False
    assert atoms_payload(after_m)["verdict"] == "CONTRADICTION"
    notes = load_dual_library(include_heldout=True)
    assert len(notes) == 108
    assert FILL_EXTRACT_PATH.is_file()


def main() -> None:
    parser = argparse.ArgumentParser(description="Goal 7 Dual restaging confirmation")
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
                    "restage_design": panel["counts"]["restage_design"],
                    "artifact": str(OUT_PATH),
                },
                indent=2,
            )
        )
        return
    parser.print_help()


if __name__ == "__main__":
    main()
