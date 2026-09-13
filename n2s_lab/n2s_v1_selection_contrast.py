"""v1 close — extract-shape selection vs apply-all vs none.

Original Q, second half: could the Dual Translate map support intervention
selection? This scores frozen G3–G9 wrappers only. No new repair. No
ground() rewrite. Not a locator-causal claim.

Arms on Dual-fill n=108:
  none     — raw extract
  selector — deepest predeclared extract signature → that wrapper
  oracle   — shallowest wrapper that matches gold (upper bound)
  apply-all — always G9 (current v1 stack)
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from typing import Any, Callable

from .auto_contract import classify_case
from .n2s_batch1_quote_promote import BIND_IDS, NEG_IDS, ROLE_OF, XSPAN_IDS
from .n2s_g2_analog_fill import EXTRACT_PATH as FILL_EXTRACT_PATH
from .n2s_note_library import load_dual_library
from .n2s_phase2_translate import (
    CONFIRMED_FAILURE_PHRASES,
    CONTINUE_CURRENT_PHRASES,
    NEXT_LINE_PHRASES,
    ONGOING_NOW_PHRASES,
    REFRACTORY_PHRASES,
    _extract_blob,
    _has_asserted_failure_marker,
    _never_vs_given_spans,
    _span_has_never_dispensed,
    atoms_payload,
    relabel_failure_polarity_from_text_gated,
    relabel_failure_polarity_from_text_gated_mixed,
    relabel_failure_polarity_from_text_gated_mixed_never,
    relabel_failure_polarity_from_text_gated_mixed_never_restage,
    relabel_failure_polarity_from_text_gated_mixed_never_restage_simult,
    relabel_failure_polarity_from_text_gated_mixed_never_restage_simult_admin,
)
from .paths import ARTIFACTS_DIR
from .types import Extraction

OUT_PATH = ARTIFACTS_DIR / "n2s-v1-selection-contrast.json"

RelabelFn = Callable[[Extraction], tuple[Extraction, dict[str, Any]]]

CELLS: list[tuple[str, RelabelFn | None]] = [
    ("none", None),
    ("G3", relabel_failure_polarity_from_text_gated),
    ("G4", relabel_failure_polarity_from_text_gated_mixed),
    ("G6", relabel_failure_polarity_from_text_gated_mixed_never),
    ("G7", relabel_failure_polarity_from_text_gated_mixed_never_restage),
    ("G8", relabel_failure_polarity_from_text_gated_mixed_never_restage_simult),
    ("G9", relabel_failure_polarity_from_text_gated_mixed_never_restage_simult_admin),
]
CELL_INDEX = {name: i for i, (name, _) in enumerate(CELLS)}


def _load_fill_extracts() -> dict[str, Any]:
    if not FILL_EXTRACT_PATH.is_file():
        raise SystemExit(f"missing {FILL_EXTRACT_PATH} — run Goal 2 recover first")
    return json.loads(FILL_EXTRACT_PATH.read_text(encoding="utf-8")).get("extracts") or {}


def _apply(ex: Extraction, name: str) -> tuple[Extraction, dict[str, Any]]:
    fn = CELLS[CELL_INDEX[name]][1]
    if fn is None:
        return ex, {"kind": "none"}
    return fn(ex)


def _match(gold: str, ex: Extraction) -> bool:
    verdict = atoms_payload(ex)["verdict"]
    return classify_case(gold=gold, verdict=verdict, disposition="AUTO")["match_gold"] is True


def oracle_cell(ex: Extraction, gold: str) -> str:
    """Shallowest frozen wrapper that matches gold. unrepaired if none do."""
    for name, _fn in CELLS:
        after, _meta = _apply(ex, name)
        if _match(gold, after):
            return name
    return "unrepaired"


def select_cell(ex: Extraction) -> str:
    """Deepest predeclared extract signature. Not fitted to leftover IDs."""
    blob = _extract_blob(ex)
    admin = (ex.administration_status or "").lower().strip()
    cues = {c.lower() for c in ex.implicit_cues}
    has_fail_pol = any(
        (o.polarity or "").lower().strip() == "failure" for o in ex.outcome_statements
    )
    asserted = _has_asserted_failure_marker(blob)
    has_fail = has_fail_pol or asserted
    has_cont = any(p in blob for p in ONGOING_NOW_PHRASES)
    has_refractory = any(p in blob for p in REFRACTORY_PHRASES) or (
        "platinum_refractory" in cues
    )
    has_next = any(p in blob for p in NEXT_LINE_PHRASES)
    spans = list(ex.quotes) + [o.text for o in ex.outcome_statements] + (
        [ex.notes] if ex.notes else []
    )
    never_disp = next((s for s in spans if _span_has_never_dispensed(s)), None)
    fail_span = next(
        (
            s
            for s in spans
            if s != never_disp
            and any(h in s.lower() for h in ("failed", "progressed", "progression", "failure"))
            and not _span_has_never_dispensed(s)
        ),
        None,
    )
    need = "none"
    if not has_fail_pol and asserted:
        need = "G3"
    if has_fail and has_cont:
        need = "G4"
    if _never_vs_given_spans(ex):
        need = "G6"
    if (ex.uncertainty_present is True and any(p in blob for p in CONFIRMED_FAILURE_PHRASES)) or (
        "stable disease" in blob
        and has_fail
        and not any(p in blob for p in CONTINUE_CURRENT_PHRASES)
    ):
        need = "G7"
    notes_promote = (
        ex.uncertainty_present is True
        and _has_asserted_failure_marker(ex.notes)
        and not any(_has_asserted_failure_marker(q) for q in ex.quotes)
    )
    relabel_ongoing = any(
        (o.polarity or "").lower().strip() == "response"
        and any(p in o.text.lower() for p in ONGOING_NOW_PHRASES)
        for o in ex.outcome_statements
    )
    copy_continue = (
        has_fail
        and has_cont
        and ex.contradiction_present is not True
    )
    if notes_promote or relabel_ongoing or copy_continue:
        need = "G8"
    if (admin == "planned" and has_refractory and has_next) or (
        never_disp and fail_span
    ):
        need = "G9"
    return need


def build_panel() -> dict[str, Any]:
    notes = load_dual_library(include_heldout=True)
    extracts = _load_fill_extracts()
    rows: list[dict[str, Any]] = []
    for note in notes:
        nid = note["id"]
        rec = extracts.get(nid) or {}
        if not rec.get("parse_ok") or not rec.get("extract"):
            continue
        gold = note["expected"]
        ex = Extraction.from_dict(rec["extract"])
        chosen = select_cell(ex)
        ora = oracle_cell(ex, gold)
        none_ok = _match(gold, ex)
        sel_ex, _ = _apply(ex, chosen)
        all_ex, _ = _apply(ex, "G9")
        ora_ex, _ = _apply(ex, "G9" if ora == "unrepaired" else ora)
        sel_ok = _match(gold, sel_ex)
        all_ok = _match(gold, all_ex)
        ora_ok = _match(gold, ora_ex)
        rows.append(
            {
                "id": nid,
                "phenotype_design": note["phenotype_design"],
                "role": ROLE_OF.get(nid) or f"gold_{gold}",
                "gold": gold,
                "selector": chosen,
                "oracle": ora,
                "selector_eq_oracle": chosen == ora,
                "match_none": none_ok,
                "match_selector": sel_ok,
                "match_oracle": ora_ok,
                "match_apply_all": all_ok,
            }
        )
    n = len(rows)
    miss_none = [r["id"] for r in rows if not r["match_none"]]
    miss_sel = [r["id"] for r in rows if not r["match_selector"]]
    miss_all = [r["id"] for r in rows if not r["match_apply_all"]]
    miss_ora = [r["id"] for r in rows if not r["match_oracle"]]
    n_sel_g9 = sum(1 for r in rows if r["selector"] == "G9")
    agree = sum(1 for r in rows if r["selector_eq_oracle"])
    sel_lost = sorted(set(miss_sel) - set(miss_all))
    sel_gained = sorted(set(miss_all) - set(miss_sel))
    bind = [r for r in rows if r["id"] in BIND_IDS]
    neg = [r for r in rows if r["id"] in NEG_IDS]
    xspan = [r for r in rows if r["id"] in XSPAN_IDS]
    always_g9 = n_sel_g9 == n
    mapping_supports_selection = (
        len(miss_sel) <= len(miss_all)
        and not sel_lost
        and not always_g9
        and agree / n >= 0.5
        if n
        else False
    )
    finding = {
        "n_scored": n,
        "n_miss_none": len(miss_none),
        "n_miss_selector": len(miss_sel),
        "n_miss_apply_all": len(miss_all),
        "n_miss_oracle": len(miss_ora),
        "ids_miss_selector": miss_sel,
        "ids_miss_apply_all": miss_all,
        "selector_lost_vs_apply_all": sel_lost,
        "selector_gained_vs_apply_all": sel_gained,
        "selector_eq_oracle_n": agree,
        "selector_eq_oracle_frac": round(agree / n, 3) if n else 0,
        "n_selector_chose_G9": n_sel_g9,
        "selector_is_apply_all_in_disguise": always_g9,
        "design_pbind_selector_all_match": all(r["match_selector"] for r in bind),
        "design_pbind_apply_all_all_match": all(r["match_apply_all"] for r in bind),
        "design_neg_selector_all_match": all(r["match_selector"] for r in neg),
        "design_xspan_selector_all_match": all(r["match_selector"] for r in xspan),
        "original_q_mapping": "partial YES — six Dual Translate cells, controlled collateral",
        "original_q_selection": (
            "YES at extract-signature depth (selector ≤ apply-all miss, not always G9)"
            if mapping_supports_selection
            else "NO — extract signatures do not select among cells better than apply-all"
        ),
        "locator_chose_repair_class": False,
        "is_selected_editor": False,
        "mapping_supports_selection": mapping_supports_selection,
        "new_repair": False,
        "ground_rewritten": False,
        "original_q_closed": True,
    }
    by_sel = {}
    for name, _ in CELLS:
        by_sel[name] = sum(1 for r in rows if r["selector"] == name)
    ora_names = [name for name, _ in CELLS] + ["unrepaired"]
    by_ora = {name: sum(1 for r in rows if r["oracle"] == name) for name in ora_names}
    panel = {
        "ok": True,
        "kind": "n2s_v1_selection_contrast_v1",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "protocol": "docs/TRACKB-FAILURE-INTERVENTION-MAP.md",
        "extracts": str(FILL_EXTRACT_PATH.name),
        "n_scored": n,
        "cells": [name for name, _ in CELLS],
        "selector_counts": by_sel,
        "oracle_counts": by_ora,
        "finding": finding,
        "rows": rows,
        "not": [
            "new Translate cell / G11 leftover repair",
            "ground() rewrite",
            "CAA",
            "frozen-d",
            "Dual_full rescore",
            ":8260 GUI",
            "auto-correct",
            "locator-causal selection",
        ],
        "note": (
            "Closes the original mapping Q for this pass: recurring Dual "
            "Translate signatures → cells, and whether extract signatures "
            "can select among those cells vs apply-all G9."
        ),
    }
    OUT_PATH.write_text(json.dumps(panel, indent=2) + "\n", encoding="utf-8")
    return panel


def selftest() -> None:
    notes = load_dual_library(include_heldout=True)
    assert len(notes) == 108
    assert FILL_EXTRACT_PATH.is_file()
    planned_ref = Extraction.from_dict(
        {
            "stated_line": "unspecified",
            "administration_status": "planned",
            "quotes": ["Platinum-refractory disease. Now considering next line."],
            "implicit_cues": ["platinum_refractory"],
        }
    )
    assert select_cell(planned_ref) == "G9"
    planned_start = Extraction.from_dict(
        {
            "stated_line": "first",
            "administration_status": "planned",
            "quotes": ["Starting first-line FOLFIRINOX next week."],
        }
    )
    assert select_cell(planned_start) != "G9"
    never_q = Extraction.from_dict(
        {
            "stated_line": "first",
            "administration_status": "given",
            "quotes": ["never received systemic therapy", "completed six cycles"],
            "outcome_statements": [{"text": "progressed", "polarity": "failure"}],
        }
    )
    assert CELL_INDEX[select_cell(never_q)] >= CELL_INDEX["G6"]
    assert oracle_cell(planned_start, "NOT_SATISFIED") == "none"


def main() -> None:
    parser = argparse.ArgumentParser(description="v1 extract-shape selection contrast")
    parser.add_argument("--selftest", action="store_true")
    parser.add_argument("--panel", action="store_true")
    args = parser.parse_args()
    if args.selftest:
        selftest()
        print("selftest ok")
        return
    if args.panel:
        panel = build_panel()
        print(json.dumps({"finding": panel["finding"], "artifact": str(OUT_PATH)}, indent=2))
        return
    parser.print_help()


if __name__ == "__main__":
    main()
