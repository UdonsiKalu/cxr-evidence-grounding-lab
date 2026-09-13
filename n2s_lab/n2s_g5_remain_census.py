"""Goal 5 — remaining Dual-miss census after frozen Goal 4 (read-only).

Applies Goal 4 mixed-span to Dual-fill extracts and tags misses by extract
shape. Does not retune Goal 3/4, rewrite ground(), or start a new repair.

Shape tags are predeclared from extract fields, not leftover note IDs.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from datetime import datetime, timezone
from typing import Any

from .auto_contract import classify_case
from .n2s_batch1_quote_promote import ROLE_OF
from .n2s_g2_analog_fill import EXTRACT_PATH as FILL_EXTRACT_PATH
from .n2s_note_library import load_dual_library
from .n2s_phase2_translate import (
    _has_asserted_failure_marker,
    atoms_payload,
    relabel_failure_polarity_from_text_gated_mixed,
)
from .paths import ARTIFACTS_DIR
from .types import Extraction

OUT_PATH = ARTIFACTS_DIR / "n2s-g5-remain-census.json"

NEVER_PHRASES = (
    "never received",
    "never having",
    "never receiving",
    "no prior",
    "no induction",
    "systemic-therapy naive",
    "therapy naive",
    "not been administered",
    "was not administered",
    "no cytotoxic",
)
HEDGE_PHRASES = (
    "possible ",
    "questionable",
    "may have",
    "pseudoprogression",
    "versus true",
)
TOO_EARLY_PHRASES = (
    "no restaging",
    "initiated yesterday",
    "cycle 1",
    "day 2",
)
SHAPE_ORDER = (
    "S-x-override",
    "S-g4-stable-seq",
    "S-g4-notes-promote",
    "S-admin-blocked",
    "S-may-have",
    "S-never-quote",
    "S-hedge",
    "S-too-early",
    "S-response-only",
    "S-other",
)


def _load_fill_extracts() -> dict[str, Any]:
    if not FILL_EXTRACT_PATH.is_file():
        raise SystemExit(f"missing {FILL_EXTRACT_PATH} — run Goal 2 recover first")
    return json.loads(FILL_EXTRACT_PATH.read_text(encoding="utf-8")).get("extracts") or {}


def _blob(ex: Extraction) -> str:
    parts = list(ex.quotes) + [ex.notes] + [o.text for o in ex.outcome_statements]
    return " ".join(parts).lower()


def shape_tags(
    *,
    ex: Extraction,
    after: Extraction,
    meta: dict[str, Any],
    gold: str,
    atoms: dict[str, Any],
) -> list[str]:
    """Extract-shape tags. First match in SHAPE_ORDER is the primary tag."""
    blob = _blob(after)
    admin = (after.administration_status or "").lower().strip()
    pols = [(o.polarity or "").lower().strip() for o in after.outcome_statements]
    tags: list[str] = []
    if after.contradiction_present is True and atoms.get("X_contradiction") is False:
        tags.append("S-x-override")
    if meta.get("flagged_simultaneous_1L") and "stable disease" in blob:
        tags.append("S-g4-stable-seq")
    if meta.get("promoted_failure_from") == "notes":
        tags.append("S-g4-notes-promote")
    if admin in {"planned", "not_given"}:
        tags.append("S-admin-blocked")
    if admin == "may_have" or "may have received" in blob:
        tags.append("S-may-have")
    if any(p in blob for p in NEVER_PHRASES):
        tags.append("S-never-quote")
    if any(p in blob for p in HEDGE_PHRASES):
        tags.append("S-hedge")
    if not after.outcome_statements and any(p in blob for p in TOO_EARLY_PHRASES):
        tags.append("S-too-early")
    if gold == "CONTRADICTION" and pols and all(p in {"response", "ongoing"} for p in pols):
        tags.append("S-response-only")
    if not tags:
        tags.append("S-other")
    return [t for t in SHAPE_ORDER if t in tags]


def score_misses() -> dict[str, Any]:
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
        before = atoms_payload(ex)
        after_ex, meta = relabel_failure_polarity_from_text_gated_mixed(ex)
        after = atoms_payload(after_ex)
        cls_b = classify_case(
            gold=gold, verdict=before["verdict"], disposition="AUTO" if before["verdict"] else "UNKNOWN"
        )
        cls_a = classify_case(
            gold=gold, verdict=after["verdict"], disposition="AUTO" if after["verdict"] else "UNKNOWN"
        )
        if cls_a["match_gold"] is True:
            continue
        tags = shape_tags(
            ex=ex,
            after=after_ex,
            meta=meta,
            gold=gold,
            atoms=after["atoms"],
        )
        rows.append(
            {
                "id": nid,
                "phenotype_design": note["phenotype_design"],
                "role": ROLE_OF.get(nid) or f"gold_{gold}",
                "gold": gold,
                "verdict_before": before["verdict"],
                "verdict_after": after["verdict"],
                "match_gold_before": cls_b["match_gold"],
                "match_gold_after": cls_a["match_gold"],
                "flipped_away": cls_b["match_gold"] is True and cls_a["match_gold"] is not True,
                "stated_line": after_ex.stated_line,
                "administration_status": after_ex.administration_status,
                "contradiction_present_before": ex.contradiction_present,
                "contradiction_present_after": after_ex.contradiction_present,
                "atoms_after": after["atoms"],
                "asserted_failure": _has_asserted_failure_marker(_blob(after_ex)),
                "intervention_meta": meta,
                "shape_tags": tags,
                "shape_primary": tags[0],
                "quotes": list(after_ex.quotes),
                "outcome_statements": [
                    {"text": o.text, "polarity": o.polarity}
                    for o in after_ex.outcome_statements
                ],
            }
        )
    by_shape = Counter(r["shape_primary"] for r in rows)
    by_gold = Counter(r["gold"] for r in rows)
    design_rows = [r for r in rows if r["phenotype_design"]]
    held_rows = [r for r in rows if not r["phenotype_design"]]
    g4_collateral = [r["id"] for r in rows if r["flipped_away"]]
    panel = {
        "ok": True,
        "kind": "n2s_g5_remain_census_v1",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "protocol": "docs/TRACKB-FAILURE-INTERVENTION-MAP.md",
        "intervention": "failure_evidence_from_extract_text_gated_mixed",
        "extracts": str(FILL_EXTRACT_PATH.name),
        "n_library": len(notes),
        "n_miss": len(rows),
        "n_design_miss": len(design_rows),
        "n_heldout_miss": len(held_rows),
        "shape_order": list(SHAPE_ORDER),
        "counts": {
            "by_shape_primary": dict(by_shape),
            "by_gold": dict(by_gold),
            "design_by_shape": dict(Counter(r["shape_primary"] for r in design_rows)),
            "heldout_by_shape": dict(Counter(r["shape_primary"] for r in held_rows)),
            "g4_flipped_away": g4_collateral,
        },
        "finding": {
            "largest_shape": (by_shape.most_common(1)[0][0] if by_shape else None),
            "g4_not_fully_specific_on_n64": bool(g4_collateral),
            "never_quote_mechanism": (
                "ground()._clinical_blob_for_conflict omits quotes/notes, so "
                "never↔given in quotes is invisible to Goal 4's hard-simultaneous flag"
            ),
            "do_not_retune_g3_g4": True,
            "locator_chose_repair_class": False,
            "started_next_repair": False,
        },
        "rows": rows,
        "not": [
            "Goal 3/4 retune",
            "new repair",
            "ground() rewrite",
            "CAA",
            "frozen-d",
            "Dual_full rescore",
            ":8260 GUI",
            "G4 panel overwrite",
        ],
        "note": (
            "Read-only remaining-miss census after frozen Goal 4. "
            "Next repair, if any, must be predeclared from a shape tag, not from IDs."
        ),
    }
    OUT_PATH.write_text(json.dumps(panel, indent=2) + "\n", encoding="utf-8")
    return panel


def selftest() -> None:
    notes = load_dual_library(include_heldout=True)
    assert len(notes) == 108
    assert FILL_EXTRACT_PATH.is_file()
    empty = Extraction.from_dict(
        {
            "stated_line": "first",
            "administration_status": "given",
            "quotes": ["never received systemic therapy"],
            "outcome_statements": [{"text": "progressed", "polarity": "failure"}],
            "contradiction_present": False,
        }
    )
    after, meta = relabel_failure_polarity_from_text_gated_mixed(empty)
    tags = shape_tags(
        ex=empty,
        after=after,
        meta=meta,
        gold="CONTRADICTION",
        atoms=atoms_payload(after)["atoms"],
    )
    assert tags[0] == "S-never-quote", tags
    planned = Extraction.from_dict(
        {
            "stated_line": "unspecified",
            "administration_status": "planned",
            "quotes": ["Platinum-refractory disease. Considering next line."],
            "contradiction_present": False,
        }
    )
    after_p, meta_p = relabel_failure_polarity_from_text_gated_mixed(planned)
    tags_p = shape_tags(
        ex=planned,
        after=after_p,
        meta=meta_p,
        gold="SATISFIED",
        atoms=atoms_payload(after_p)["atoms"],
    )
    assert tags_p[0] == "S-admin-blocked", tags_p


def main() -> None:
    parser = argparse.ArgumentParser(description="Goal 5 remaining Dual-miss census")
    parser.add_argument("--selftest", action="store_true")
    parser.add_argument("--census", action="store_true")
    args = parser.parse_args()
    if args.selftest:
        selftest()
        print("selftest ok")
        return
    if args.census:
        panel = score_misses()
        print(
            json.dumps(
                {
                    "n_miss": panel["n_miss"],
                    "n_design_miss": panel["n_design_miss"],
                    "n_heldout_miss": panel["n_heldout_miss"],
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
