"""Batch 1 — quote-promote map cell (no hidden-state patch).

Applies Phase 2 `failure_evidence_from_extract_text` to saved Batch 0x
extracts. Does not rewrite ground(), rescore Dual_full, or touch :8260.
Does not overwrite Phase 2 or Batch 0/0x extract artifacts.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from collections.abc import Callable
from typing import Any

from .auto_contract import classify_case
from .n2s_batch0_phenotype import propose_phenotypes
from .n2s_batch0x_census import ANALOG_PATH, SCHEMA_PATH
from .n2s_note_library import load_dual_library
from .n2s_phase2_translate import relabel_failure_polarity_from_text
from .n2s_phase2_translate import atoms_payload
from .paths import ARTIFACTS_DIR
from .types import Extraction

OUT_PATH = ARTIFACTS_DIR / "n2s-batch1-quote-promote-panel.json"

# Schema-defined P-bind (design). Same IDs used on analog (expected no-op).
BIND_IDS = ("E2", "I2", "T3", "TF_E3", "TF_E4", "TX_E06", "TX_E19")
XSPAN_IDS = (
    "C1",
    "C2",
    "C3",
    "C4",
    "TF_C1",
    "BC_C1",
    "TX_C01",
    "TX_C02",
    "TX_C03",
    "TX_C04",
)
NEG_IDS = (
    "E3",
    "E4",
    "I3",
    "T2",
    "T4",
    "TF_T1",
    "TF_T2",
    "TF_N1",
    "BC_E2",
    "TX_N01",
    "TX_N02",
)
# Phase 2 anchors — score-only, not in design k/n.
ANCHOR_IDS = ("EX_TEMPORAL_FOLFOX", "EX_CONTRA")
ROLE_OF = {
    **{i: "P-bind" for i in BIND_IDS},
    **{i: "P-xspan" for i in XSPAN_IDS},
    **{i: "P-neg" for i in NEG_IDS},
    "EX_TEMPORAL_FOLFOX": "anchor_folfox",
    "EX_CONTRA": "anchor_contra",
}


def _load_extracts(path) -> dict[str, Any]:
    if not path.is_file():
        raise SystemExit(f"missing {path} — run Batch 0x recover first")
    return json.loads(path.read_text(encoding="utf-8")).get("extracts") or {}


def _fired(meta: dict[str, Any] | None) -> bool:
    if not meta:
        return False
    return bool(
        meta.get("n_polarities_relabeled")
        or meta.get("promoted_failure_from")
        or meta.get("added_response_then_progression")
        or meta.get("flagged_simultaneous_1L")
        or meta.get("flagged_never_vs_given")
        or meta.get("cleared_hedge_uncertainty")
        or meta.get("reverted_stable_seq")
        or meta.get("undid_notes_promote_on_uncertain")
        or meta.get("relabeled_ongoing_now")
        or meta.get("copied_continue_quotes")
        or meta.get("relabeled_planned_next_after_refractory")
        or meta.get("copied_never_dispensed")
        or meta.get("cleared_dispense_uncertainty")
        or meta.get("touched_contradiction_present")
    )


def apply_row(
    note: dict[str, Any],
    rec: dict[str, Any],
    *,
    prompt: str,
    role: str | None = None,
    relabel: Callable[..., tuple[Extraction, dict[str, Any]]] | None = None,
    intervention: str = "failure_evidence_from_extract_text",
) -> dict[str, Any]:
    gold = note["expected"]
    nid = note["id"]
    role = role or ROLE_OF.get(nid) or f"gold_{gold}"
    relabel_fn = relabel or relabel_failure_polarity_from_text
    if not rec.get("parse_ok") or not rec.get("extract"):
        return {
            "id": nid,
            "role": role,
            "prompt": prompt,
            "slice": note["slice"],
            "gold": gold,
            "parse_ok": False,
            "fired": False,
            "verdict_before": None,
            "verdict_after": None,
            "repaired": False,
            "match_repaired": False,
            "collateral_to_SATISFIED": False,
            "flipped_away": False,
            "correct_stay": False,
            "no_op": True,
            "hidden_state_patched": False,
            "intervention": intervention,
        }
    ex = Extraction.from_dict(rec["extract"])
    before = atoms_payload(ex)
    after_ex, meta = relabel_fn(ex)
    after = atoms_payload(after_ex)
    vb = before["verdict"]
    va = after["verdict"]
    cls_b = classify_case(gold=gold, verdict=vb, disposition="AUTO" if vb else "UNKNOWN")
    cls_a = classify_case(gold=gold, verdict=va, disposition="AUTO" if va else "UNKNOWN")
    repaired = gold == "SATISFIED" and vb != "SATISFIED" and va == "SATISFIED"
    match_repaired = cls_b["match_gold"] is not True and cls_a["match_gold"] is True
    collateral = gold != "SATISFIED" and vb != "SATISFIED" and va == "SATISFIED"
    flipped_away = cls_b["match_gold"] is True and cls_a["match_gold"] is not True
    correct_stay = cls_b["match_gold"] is True and cls_a["match_gold"] is True
    pheno = propose_phenotypes(note, ex=ex, grounded=before)
    return {
        "id": nid,
        "role": role,
        "prompt": prompt,
        "slice": note["slice"],
        "phenotype_design": note["phenotype_design"],
        "gold": gold,
        "parse_ok": True,
        "phenotypes_before": pheno,
        "fired": _fired(meta),
        "verdict_before": vb,
        "verdict_after": va,
        "wrong_AUTO_before": cls_b["wrong_AUTO"],
        "wrong_AUTO_after": cls_a["wrong_AUTO"],
        "match_gold_before": cls_b["match_gold"],
        "match_gold_after": cls_a["match_gold"],
        "repaired": repaired,
        "match_repaired": match_repaired,
        "collateral_to_SATISFIED": collateral,
        "flipped_away": flipped_away,
        "correct_stay": correct_stay,
        "no_op": vb == va,
        "atoms_before": before["atoms"],
        "atoms_after": after["atoms"],
        "intervention_meta": meta,
        "hidden_state_patched": False,
        "intervention": intervention,
        "ground_changed": False,
    }


def _arm_counts(rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_role: dict[str, list[dict[str, Any]]] = {}
    for r in rows:
        by_role.setdefault(r["role"], []).append(r)
    out: dict[str, Any] = {"n": len(rows)}
    for role, rs in sorted(by_role.items()):
        out[role] = {
            "n": len(rs),
            "fired": sum(1 for r in rs if r["fired"]),
            "repaired": sum(1 for r in rs if r["repaired"]),
            "match_repaired": sum(1 for r in rs if r.get("match_repaired")),
            "collateral_to_SATISFIED": sum(1 for r in rs if r["collateral_to_SATISFIED"]),
            "flipped_away": sum(1 for r in rs if r["flipped_away"]),
            "correct_stay": sum(1 for r in rs if r["correct_stay"]),
            "no_op": sum(1 for r in rs if r["no_op"]),
            "ids_repaired": [r["id"] for r in rs if r["repaired"]],
            "ids_match_repaired": [r["id"] for r in rs if r.get("match_repaired")],
            "ids_collateral": [r["id"] for r in rs if r["collateral_to_SATISFIED"]],
            "ids_flipped_away": [r["id"] for r in rs if r["flipped_away"]],
        }
    return out


def build_panel() -> dict[str, Any]:
    notes = {n["id"]: n for n in load_dual_library(include_heldout=True)}
    analog_ex = _load_extracts(ANALOG_PATH)
    schema_ex = _load_extracts(SCHEMA_PATH)
    design_ids = list(BIND_IDS) + list(XSPAN_IDS) + list(NEG_IDS)
    analog_rows = [
        apply_row(notes[i], analog_ex.get(i) or {}, prompt="analog") for i in design_ids
    ]
    schema_rows = [
        apply_row(notes[i], schema_ex.get(i) or {}, prompt="schema") for i in design_ids
    ]
    anchor_rows = []
    for i in ANCHOR_IDS:
        if i in notes:
            anchor_rows.append(
                apply_row(notes[i], schema_ex.get(i) or {}, prompt="schema")
            )
            anchor_rows.append(
                apply_row(notes[i], analog_ex.get(i) or {}, prompt="analog")
            )
    schema_bind = _arm_counts([r for r in schema_rows if r["role"] == "P-bind"])
    schema_xspan = _arm_counts([r for r in schema_rows if r["role"] == "P-xspan"])
    schema_neg = _arm_counts([r for r in schema_rows if r["role"] == "P-neg"])
    analog_all = _arm_counts(analog_rows)
    map_cell = {
        "schema_pbind_helps": schema_bind.get("P-bind", {}).get("repaired", 0) >= 1,
        "schema_pbind_k": schema_bind.get("P-bind", {}).get("repaired", 0),
        "schema_pbind_n": schema_bind.get("P-bind", {}).get("n", 0),
        "schema_xspan_collateral": schema_xspan.get("P-xspan", {}).get(
            "collateral_to_SATISFIED", 0
        ),
        "schema_neg_collateral": schema_neg.get("P-neg", {}).get(
            "collateral_to_SATISFIED", 0
        ),
        "analog_repaired": analog_all.get("P-bind", {}).get("repaired", 0),
        "specificity_holds": (
            schema_xspan.get("P-xspan", {}).get("collateral_to_SATISFIED", 0) == 0
            and schema_neg.get("P-neg", {}).get("collateral_to_SATISFIED", 0) == 0
        ),
        "schema_is_not_dual_editor": True,
        "locator_chose_repair_class": False,
    }
    panel = {
        "ok": True,
        "kind": "n2s_batch1_quote_promote_v1",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "protocol": "docs/TRACKB-FAILURE-INTERVENTION-MAP.md",
        "intervention": "failure_evidence_from_extract_text",
        "source_extracts": [str(ANALOG_PATH.name), str(SCHEMA_PATH.name)],
        "n_design": len(design_ids),
        "ids": {
            "P-bind": list(BIND_IDS),
            "P-xspan": list(XSPAN_IDS),
            "P-neg": list(NEG_IDS),
            "anchors_score_only": list(ANCHOR_IDS),
        },
        "not": [
            "hidden-state patch",
            "frozen-d",
            "ground() rewrite",
            "Dual_full rescore",
            "new note synthesis",
            "auto-correct",
            ":8260 GUI",
            "Phase 2 artifact overwrite",
        ],
        "counts": {
            "schema": _arm_counts(schema_rows),
            "analog": analog_all,
            "anchors_score_only": _arm_counts(anchor_rows),
        },
        "map_cell": map_cell,
        "rows": {
            "schema": schema_rows,
            "analog": analog_rows,
            "anchors": anchor_rows,
        },
        "note": (
            "Quote-promote on saved extracts. Schema P-bind is substrate, not Dual. "
            "Analog expected no-op. Held-out anchors are score-only."
        ),
    }
    OUT_PATH.write_text(json.dumps(panel, indent=2) + "\n", encoding="utf-8")
    return panel


def selftest() -> None:
    notes = {n["id"]: n for n in load_dual_library(include_heldout=True)}
    for i in list(BIND_IDS) + list(XSPAN_IDS) + list(NEG_IDS) + list(ANCHOR_IDS):
        assert i in notes, i
    empty = Extraction.from_dict({})
    after, meta = relabel_failure_polarity_from_text(empty)
    assert meta["kind"] == "failure_evidence_from_extract_text"
    assert after.outcome_statements == []
    rich = Extraction.from_dict(
        {
            "quotes": ["FOLFOX was discontinued for treatment failure"],
            "outcome_statements": [{"text": "partial response", "polarity": "response"}],
        }
    )
    after2, meta2 = relabel_failure_polarity_from_text(rich)
    assert meta2["promoted_failure_from"] == "quotes"
    assert any((o.polarity or "") == "failure" for o in after2.outcome_statements)


def main() -> None:
    parser = argparse.ArgumentParser(description="Batch 1 quote-promote map cell")
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
                    "counts": panel["counts"],
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
