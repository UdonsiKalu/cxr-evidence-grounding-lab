"""Goal 9 — Dual S-admin-blocked (planned-after-refractory + never-dispensed).

Predeclared from Goal 5 leftover shape S-admin-blocked. After frozen Goal 8:

  1. Extractor marked first-line planned, but the note is refractory and
     now considering a next line → administration_status=given.
  2. Fail vs never-dispensed quotes → copy into contradiction cues; clear
     U if the uncertainty cue is that never-dispensed span.

Does not mutate Goal 3–8. Does not rewrite ground(). CPU on Dual-fill.
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
    relabel_failure_polarity_from_text_gated_mixed_never_restage_simult,
    relabel_failure_polarity_from_text_gated_mixed_never_restage_simult_admin,
)
from .paths import ARTIFACTS_DIR
from .types import Extraction

OUT_PATH = ARTIFACTS_DIR / "n2s-g9-admin-blocked-panel.json"
SIMULT = "failure_evidence_from_extract_text_gated_mixed_never_restage_simult"
ADMIN = "failure_evidence_from_extract_text_gated_mixed_never_restage_simult_admin"
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
    admin: bool,
    role_fn,
) -> list[dict[str, Any]]:
    relabel = (
        relabel_failure_polarity_from_text_gated_mixed_never_restage_simult_admin
        if admin
        else relabel_failure_polarity_from_text_gated_mixed_never_restage_simult
    )
    intervention = ADMIN if admin else SIMULT
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
    admin: bool,
) -> dict[str, Any]:
    relabel = (
        relabel_failure_polarity_from_text_gated_mixed_never_restage_simult_admin
        if admin
        else relabel_failure_polarity_from_text_gated_mixed_never_restage_simult
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

    sim_design = _score_ids(
        notes, extracts, design_ids, admin=False, role_fn=design_role
    )
    adm_design = _score_ids(
        notes, extracts, design_ids, admin=True, role_fn=design_role
    )
    sim_held = _score_ids(
        notes, extracts, heldout_ids, admin=False, role_fn=held_role
    )
    adm_held = _score_ids(
        notes, extracts, heldout_ids, admin=True, role_fn=held_role
    )
    s_des = _arm_counts(sim_design)
    a_des = _arm_counts(adm_design)
    s_h = _arm_counts(sim_held)
    a_h = _arm_counts(adm_held)
    s_bind = s_des.get("P-bind") or {}
    a_bind = a_des.get("P-bind") or {}
    s_xspan = s_des.get("P-xspan") or {}
    a_xspan = a_des.get("P-xspan") or {}
    s_neg = s_des.get("P-neg") or {}
    a_neg = a_des.get("P-neg") or {}
    kept_bind = sorted(
        set(s_bind.get("ids_repaired") or []) & set(a_bind.get("ids_repaired") or [])
    )
    lost_bind = sorted(
        set(s_bind.get("ids_repaired") or []) - set(a_bind.get("ids_repaired") or [])
    )
    n108_sim = _n108_miss(notes, extracts, admin=False)
    n108_adm = _n108_miss(notes, extracts, admin=True)
    sim_miss = set(n108_sim["ids_miss"])
    adm_miss = set(n108_adm["ids_miss"])
    held_leak = bool(
        (a_h.get("gold_NOT_SATISFIED") or {}).get("collateral_to_SATISFIED", 0)
        or (a_h.get("gold_CONTRADICTION") or {}).get("collateral_to_SATISFIED", 0)
    )
    design_specificity = (
        a_xspan.get("collateral_to_SATISFIED", 0) == 0
        and a_neg.get("collateral_to_SATISFIED", 0) == 0
        and a_bind.get("flipped_away", 0) == 0
    )
    map_cell = {
        "simult_pbind_k": s_bind.get("repaired", 0),
        "admin_pbind_k": a_bind.get("repaired", 0),
        "pbind_n": a_bind.get("n", 0),
        "kept_simult_bind_repairs": kept_bind,
        "lost_simult_bind_repairs": lost_bind,
        "admin_bind_ids": a_bind.get("ids_repaired") or [],
        "simult_neg_collateral": s_neg.get("collateral_to_SATISFIED", 0),
        "admin_neg_collateral": a_neg.get("collateral_to_SATISFIED", 0),
        "design_xspan_all_match": (
            a_xspan.get("correct_stay", 0) + a_xspan.get("match_repaired", 0)
            == a_xspan.get("n", 0)
        ),
        "specificity_holds": design_specificity,
        "heldout_leak": held_leak,
        "heldout_sat_repaired": (a_h.get("gold_SATISFIED") or {}).get("repaired", 0),
        "n108_miss_simult": n108_sim["n_miss"],
        "n108_miss_admin": n108_adm["n_miss"],
        "n108_fixed_vs_simult": sorted(sim_miss - adm_miss),
        "n108_new_miss_vs_simult": sorted(adm_miss - sim_miss),
        "n108_new_flips_vs_simult": [
            i
            for i in n108_adm["flipped_away"]
            if i not in n108_sim["flipped_away"]
        ],
        "n108_new_repairs_vs_raw": [
            i
            for i in n108_adm["match_repaired"]
            if i not in n108_sim["match_repaired"]
        ],
        "n108_restored_vs_simult": [
            i
            for i in n108_sim["flipped_away"]
            if i not in n108_adm["flipped_away"]
        ],
        "ground_rewritten": False,
        "g3_g8_mutated": False,
        "locator_chose_repair_class": False,
        "is_selected_editor": False,
    }
    panel = {
        "ok": True,
        "kind": "n2s_g9_admin_blocked_v1",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "protocol": "docs/TRACKB-FAILURE-INTERVENTION-MAP.md",
        "intervention": ADMIN,
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
            "Goal 3–8 retune in place",
            "ground() rewrite",
            "CAA",
            "frozen-d",
            "Dual_full rescore",
            ":8260 GUI",
            "auto-correct",
        ],
        "counts": {
            "simult_design": s_des,
            "admin_design": a_des,
            "simult_heldout": s_h,
            "admin_heldout": a_h,
            "n108_simult": n108_sim,
            "n108_admin": n108_adm,
        },
        "map_cell": map_cell,
        "rows": {
            "simult_design": sim_design,
            "admin_design": adm_design,
            "simult_heldout": sim_held,
            "admin_heldout": adm_held,
        },
        "note": (
            "S-admin-blocked: planned next-line after refractory → given; "
            "fail vs never-dispensed quotes → contradiction cues + clear U "
            "parked on that span. ground() unchanged."
        ),
    }
    OUT_PATH.write_text(json.dumps(panel, indent=2) + "\n", encoding="utf-8")
    return panel


def selftest() -> None:
    batch1_selftest()
    planned_ref = Extraction.from_dict(
        {
            "stated_line": "unspecified",
            "administration_status": "planned",
            "quotes": [
                "Platinum-refractory disease. Now considering next line."
            ],
            "outcome_statements": [],
            "implicit_cues": ["platinum_refractory"],
            "contradiction_present": False,
        }
    )
    after_p, meta_p = (
        relabel_failure_polarity_from_text_gated_mixed_never_restage_simult_admin(
            planned_ref
        )
    )
    assert meta_p["relabeled_planned_next_after_refractory"] is True
    assert after_p.administration_status == "given"
    assert atoms_payload(after_p)["verdict"] == "SATISFIED"

    planned_start = Extraction.from_dict(
        {
            "stated_line": "first",
            "administration_status": "planned",
            "quotes": ["Starting first-line FOLFIRINOX next week."],
            "outcome_statements": [],
            "contradiction_present": False,
        }
    )
    after_s, meta_s = (
        relabel_failure_polarity_from_text_gated_mixed_never_restage_simult_admin(
            planned_start
        )
    )
    assert meta_s["relabeled_planned_next_after_refractory"] is False
    assert after_s.administration_status == "planned"
    assert atoms_payload(after_s)["verdict"] == "NOT_SATISFIED"

    dispensed = Extraction.from_dict(
        {
            "stated_line": "unknown",
            "administration_status": "not_given",
            "quotes": [
                "failed first-line systemic therapy",
                "no first-line systemic anticancer drug was ever dispensed",
            ],
            "outcome_statements": [
                {"text": "failed first-line systemic therapy", "polarity": "failure"}
            ],
            "uncertainty_present": True,
            "uncertainty_cues": [
                "no first-line systemic anticancer drug was ever dispensed"
            ],
            "uncertainty_cue": "no first-line systemic anticancer drug was ever dispensed",
            "uncertainty_why": "unresolved_line",
            "contradiction_present": False,
        }
    )
    after_d, meta_d = (
        relabel_failure_polarity_from_text_gated_mixed_never_restage_simult_admin(
            dispensed
        )
    )
    assert meta_d["copied_never_dispensed"] is True
    assert meta_d["cleared_dispense_uncertainty"] is True
    assert after_d.contradiction_present is True
    assert after_d.uncertainty_present is False
    assert atoms_payload(after_d)["verdict"] == "CONTRADICTION"

    hold = Extraction.from_dict(
        {
            "stated_line": "first",
            "administration_status": "planned",
            "quotes": [
                "Cycle 3 held for neutropenia. Plan to resume. No evidence of progression."
            ],
            "outcome_statements": [
                {"text": "No evidence of progression", "polarity": "response"}
            ],
            "implicit_cues": ["temporary_hold"],
            "contradiction_present": False,
        }
    )
    after_h, meta_h = (
        relabel_failure_polarity_from_text_gated_mixed_never_restage_simult_admin(hold)
    )
    assert meta_h["relabeled_planned_next_after_refractory"] is False
    assert atoms_payload(after_h)["verdict"] == "NOT_SATISFIED"

    notes = load_dual_library(include_heldout=True)
    assert len(notes) == 108
    assert FILL_EXTRACT_PATH.is_file()


def main() -> None:
    parser = argparse.ArgumentParser(description="Goal 9 Dual S-admin-blocked")
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
                    "admin_design": panel["counts"]["admin_design"],
                    "artifact": str(OUT_PATH),
                },
                indent=2,
            )
        )
        return
    parser.print_help()


if __name__ == "__main__":
    main()
