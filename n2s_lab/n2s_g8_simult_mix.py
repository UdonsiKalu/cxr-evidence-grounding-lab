"""Goal 8 — Dual same-time mix vs sequenced override + uncertain notes undo.

Predeclared leftover shapes after frozen Goal 7:
  1. Uncertain extract + Goal 3 notes-promote (meta commentary, not a
     clinical failure span) → revert to original extract.
  2. Fail + continue-now parked as response polarity / quotes so frozen
     ground() sequenced override fires instead of hard-simultaneous.

Does not mutate Goal 3–7. Does not rewrite ground(). CPU on Dual-fill.
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
    relabel_failure_polarity_from_text_gated_mixed_never_restage,
    relabel_failure_polarity_from_text_gated_mixed_never_restage_simult,
)
from .paths import ARTIFACTS_DIR
from .types import Extraction

OUT_PATH = ARTIFACTS_DIR / "n2s-g8-simult-mix-panel.json"
RESTAGE = "failure_evidence_from_extract_text_gated_mixed_never_restage"
SIMULT = "failure_evidence_from_extract_text_gated_mixed_never_restage_simult"
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
    simult: bool,
    role_fn,
) -> list[dict[str, Any]]:
    relabel = (
        relabel_failure_polarity_from_text_gated_mixed_never_restage_simult
        if simult
        else relabel_failure_polarity_from_text_gated_mixed_never_restage
    )
    intervention = SIMULT if simult else RESTAGE
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
    simult: bool,
) -> dict[str, Any]:
    relabel = (
        relabel_failure_polarity_from_text_gated_mixed_never_restage_simult
        if simult
        else relabel_failure_polarity_from_text_gated_mixed_never_restage
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

    rest_design = _score_ids(
        notes, extracts, design_ids, simult=False, role_fn=design_role
    )
    sim_design = _score_ids(
        notes, extracts, design_ids, simult=True, role_fn=design_role
    )
    rest_held = _score_ids(
        notes, extracts, heldout_ids, simult=False, role_fn=held_role
    )
    sim_held = _score_ids(
        notes, extracts, heldout_ids, simult=True, role_fn=held_role
    )
    r_des = _arm_counts(rest_design)
    s_des = _arm_counts(sim_design)
    r_h = _arm_counts(rest_held)
    s_h = _arm_counts(sim_held)
    r_bind = r_des.get("P-bind") or {}
    s_bind = s_des.get("P-bind") or {}
    r_xspan = r_des.get("P-xspan") or {}
    s_xspan = s_des.get("P-xspan") or {}
    r_neg = r_des.get("P-neg") or {}
    s_neg = s_des.get("P-neg") or {}
    kept_bind = sorted(
        set(r_bind.get("ids_repaired") or []) & set(s_bind.get("ids_repaired") or [])
    )
    lost_bind = sorted(
        set(r_bind.get("ids_repaired") or []) - set(s_bind.get("ids_repaired") or [])
    )
    n108_rest = _n108_miss(notes, extracts, simult=False)
    n108_sim = _n108_miss(notes, extracts, simult=True)
    rest_miss = set(n108_rest["ids_miss"])
    sim_miss = set(n108_sim["ids_miss"])
    held_leak = bool(
        (s_h.get("gold_NOT_SATISFIED") or {}).get("collateral_to_SATISFIED", 0)
        or (s_h.get("gold_CONTRADICTION") or {}).get("collateral_to_SATISFIED", 0)
    )
    design_specificity = (
        s_xspan.get("collateral_to_SATISFIED", 0) == 0
        and s_neg.get("collateral_to_SATISFIED", 0) == 0
        and s_bind.get("flipped_away", 0) == 0
    )
    map_cell = {
        "restage_pbind_k": r_bind.get("repaired", 0),
        "simult_pbind_k": s_bind.get("repaired", 0),
        "pbind_n": s_bind.get("n", 0),
        "kept_restage_bind_repairs": kept_bind,
        "lost_restage_bind_repairs": lost_bind,
        "simult_bind_ids": s_bind.get("ids_repaired") or [],
        "restage_neg_collateral": r_neg.get("collateral_to_SATISFIED", 0),
        "simult_neg_collateral": s_neg.get("collateral_to_SATISFIED", 0),
        "restage_xspan_match_repaired": r_xspan.get("match_repaired", 0),
        "simult_xspan_match_repaired": s_xspan.get("match_repaired", 0),
        "design_xspan_all_match": (
            s_xspan.get("correct_stay", 0) + s_xspan.get("match_repaired", 0)
            == s_xspan.get("n", 0)
        ),
        "specificity_holds": design_specificity,
        "heldout_leak": held_leak,
        "heldout_sat_repaired": (s_h.get("gold_SATISFIED") or {}).get("repaired", 0),
        "n108_miss_restage": n108_rest["n_miss"],
        "n108_miss_simult": n108_sim["n_miss"],
        "n108_fixed_vs_restage": sorted(rest_miss - sim_miss),
        "n108_new_miss_vs_restage": sorted(sim_miss - rest_miss),
        "n108_new_flips_vs_restage": [
            i
            for i in n108_sim["flipped_away"]
            if i not in n108_rest["flipped_away"]
        ],
        "n108_new_repairs_vs_raw": [
            i
            for i in n108_sim["match_repaired"]
            if i not in n108_rest["match_repaired"]
        ],
        "n108_restored_vs_restage": [
            i
            for i in n108_rest["flipped_away"]
            if i not in n108_sim["flipped_away"]
        ],
        "ground_rewritten": False,
        "g3_g7_mutated": False,
        "locator_chose_repair_class": False,
        "is_selected_editor": False,
    }
    panel = {
        "ok": True,
        "kind": "n2s_g8_simult_mix_v1",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "protocol": "docs/TRACKB-FAILURE-INTERVENTION-MAP.md",
        "intervention": SIMULT,
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
            "Goal 3–7 retune in place",
            "ground() rewrite",
            "CAA",
            "frozen-d",
            "Dual_full rescore",
            ":8260 GUI",
            "auto-correct",
        ],
        "counts": {
            "restage_design": r_des,
            "simult_design": s_des,
            "restage_heldout": r_h,
            "simult_heldout": s_h,
            "n108_restage": n108_rest,
            "n108_simult": n108_sim,
        },
        "map_cell": map_cell,
        "rows": {
            "restage_design": rest_design,
            "simult_design": sim_design,
            "restage_heldout": rest_held,
            "simult_heldout": sim_held,
        },
        "note": (
            "Same-time fail vs continue: undo notes-promote on uncertain extracts; "
            "relabel continue-now response polarities to ongoing; copy continue "
            "quotes into contradiction cues. ground() unchanged."
        ),
    }
    OUT_PATH.write_text(json.dumps(panel, indent=2) + "\n", encoding="utf-8")
    return panel


def selftest() -> None:
    batch1_selftest()
    unc_notes = Extraction.from_dict(
        {
            "stated_line": "unknown",
            "administration_status": "unknown",
            "quotes": ["no clear progression documented"],
            "notes": (
                "The note expresses uncertainty about the presence of progression."
            ),
            "outcome_statements": [
                {"text": "no clear progression documented", "polarity": "ongoing"}
            ],
            "uncertainty_present": True,
            "uncertainty_cues": ["questionable"],
            "contradiction_present": False,
        }
    )
    after_g7, meta_g7 = relabel_failure_polarity_from_text_gated_mixed_never_restage(
        unc_notes
    )
    assert meta_g7.get("promoted_failure_from") == "notes"
    assert atoms_payload(after_g7)["verdict"] == "CONTRADICTION"
    after_u, meta_u = relabel_failure_polarity_from_text_gated_mixed_never_restage_simult(
        unc_notes
    )
    assert meta_u["undid_notes_promote_on_uncertain"] is True
    assert after_u.uncertainty_present is True
    assert atoms_payload(after_u)["verdict"] == "UNCERTAIN"

    still = Extraction.from_dict(
        {
            "stated_line": "first",
            "administration_status": "given",
            "quotes": [
                "failed after four cycles",
                "still responding to first-line",
            ],
            "outcome_statements": [
                {"text": "failed after four cycles", "polarity": "failure"},
                {"text": "still responding to first-line", "polarity": "response"},
            ],
            "implicit_cues": ["response_then_progression"],
            "contradiction_present": True,
            "contradiction": {
                "present": True,
                "span_a": "failed after four cycles",
                "span_b": "still responding to first-line",
            },
        }
    )
    after_st, meta_st = relabel_failure_polarity_from_text_gated_mixed_never_restage_simult(
        still
    )
    assert meta_st["relabeled_ongoing_now"] is True
    assert atoms_payload(after_st)["verdict"] == "CONTRADICTION"

    cont_q = Extraction.from_dict(
        {
            "stated_line": "first",
            "administration_status": "given",
            "quotes": [
                "failed first-line therapy",
                "continue present therapy",
            ],
            "outcome_statements": [
                {"text": "failed first-line therapy", "polarity": "failure"}
            ],
            "implicit_cues": ["response_then_progression"],
            "contradiction_present": False,
        }
    )
    after_cq, meta_cq = relabel_failure_polarity_from_text_gated_mixed_never_restage_simult(
        cont_q
    )
    assert meta_cq["copied_continue_quotes"] is True
    assert after_cq.contradiction_present is True
    assert atoms_payload(after_cq)["verdict"] == "CONTRADICTION"

    sequential = Extraction.from_dict(
        {
            "stated_line": "first",
            "administration_status": "given",
            "quotes": [
                "partial response after two cycles",
                "then progressed on first-line",
            ],
            "outcome_statements": [
                {"text": "partial response after two cycles", "polarity": "response"},
                {"text": "then progressed on first-line", "polarity": "failure"},
            ],
            "implicit_cues": ["response_then_progression"],
            "contradiction_present": False,
        }
    )
    after_seq, meta_seq = (
        relabel_failure_polarity_from_text_gated_mixed_never_restage_simult(sequential)
    )
    assert meta_seq.get("relabeled_ongoing_now") is False
    assert meta_seq.get("copied_continue_quotes") is False
    assert atoms_payload(after_seq)["verdict"] == "SATISFIED"

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
    after_m, _meta_m = relabel_failure_polarity_from_text_gated_mixed_never_restage_simult(
        mixed
    )
    assert atoms_payload(after_m)["verdict"] == "CONTRADICTION"
    notes = load_dual_library(include_heldout=True)
    assert len(notes) == 108
    assert FILL_EXTRACT_PATH.is_file()


def main() -> None:
    parser = argparse.ArgumentParser(description="Goal 8 Dual same-time mix")
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
                    "simult_design": panel["counts"]["simult_design"],
                    "artifact": str(OUT_PATH),
                },
                indent=2,
            )
        )
        return
    parser.print_help()


if __name__ == "__main__":
    main()
