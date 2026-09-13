"""Goal 10 — post-G9 leftover-shape prevalence (read-only).

Scores frozen Goal 9 on Dual-fill n=108 and tags the six leftover
mechanisms on *every* note, including notes that already match gold.

Does not retune G3–G9, rewrite ground(), or start a new repair.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from datetime import datetime, timezone
from typing import Any

from .auto_contract import classify_case
from .n2s_batch1_quote_promote import BIND_IDS, ROLE_OF
from .n2s_g2_analog_fill import EXTRACT_PATH as FILL_EXTRACT_PATH
from .n2s_note_library import load_dual_library
from .n2s_phase2_translate import (
    atoms_payload,
    relabel_failure_polarity_from_text_gated_mixed_never_restage_simult_admin,
)
from .paths import ARTIFACTS_DIR
from .types import Extraction

OUT_PATH = ARTIFACTS_DIR / "n2s-g10-leftover-census.json"
ADMIN = "failure_evidence_from_extract_text_gated_mixed_never_restage_simult_admin"

TOO_EARLY_PHRASES = (
    "no restaging",
    "initiated yesterday",
    "cycle 1",
    "day 2",
)
GROUND_NEVER_PHRASES = ("never", "no prior", "not given", "not_given")
NAIVE_PHRASES = ("naive",)
HELD_PROGRESSION_PHRASES = ("held for progression",)
LEFTOVER_SHAPES = (
    "S-too-early",
    "S-may-have",
    "S-held-for-progression",
    "S-response-only",
    "S-naive-not-never",
)
KNOWN_MISS = ("T2", "U2", "U6", "TX_E03", "BC12_C1", "BC14_C1")


def _load_fill_extracts() -> dict[str, Any]:
    if not FILL_EXTRACT_PATH.is_file():
        raise SystemExit(f"missing {FILL_EXTRACT_PATH} — run Goal 2 recover first")
    return json.loads(FILL_EXTRACT_PATH.read_text(encoding="utf-8")).get("extracts") or {}


def _blob(ex: Extraction) -> str:
    parts = list(ex.quotes) + [ex.notes] + [o.text for o in ex.outcome_statements]
    return " ".join(parts).lower()


def leftover_tags(
    *,
    after: Extraction,
    gold: str,
) -> list[str]:
    """Leftover-mechanism tags. Order is LEFTOVER_SHAPES."""
    blob = _blob(after)
    admin = (after.administration_status or "").lower().strip()
    pols = [(o.polarity or "").lower().strip() for o in after.outcome_statements]
    tags: list[str] = []
    if not after.outcome_statements and any(p in blob for p in TOO_EARLY_PHRASES):
        tags.append("S-too-early")
    if admin == "may_have" or "may have" in blob:
        tags.append("S-may-have")
    if any(p in blob for p in HELD_PROGRESSION_PHRASES):
        tags.append("S-held-for-progression")
    if (
        gold == "CONTRADICTION"
        and pols
        and all(p in {"response", "ongoing"} for p in pols)
    ):
        tags.append("S-response-only")
    if any(p in blob for p in NAIVE_PHRASES) and not any(
        p in blob for p in GROUND_NEVER_PHRASES
    ):
        tags.append("S-naive-not-never")
    return [t for t in LEFTOVER_SHAPES if t in tags]


def _shape_table(rows: list[dict[str, Any]]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for shape in LEFTOVER_SHAPES:
        hit = [r for r in rows if shape in r["leftover_tags"]]
        miss = [r for r in hit if r["match_gold_after"] is not True]
        match = [r for r in hit if r["match_gold_after"] is True]
        out[shape] = {
            "n_library": len(hit),
            "n_miss": len(miss),
            "n_match": len(match),
            "ids_library": [r["id"] for r in hit],
            "ids_miss": [r["id"] for r in miss],
            "ids_match": [r["id"] for r in match],
            "one_off_miss": len(miss) <= 1,
            "recurs_in_library": len(hit) >= 2,
            "recurs_as_miss": len(miss) >= 2,
        }
    return out


def build_census() -> dict[str, Any]:
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
        after_ex, meta = (
            relabel_failure_polarity_from_text_gated_mixed_never_restage_simult_admin(ex)
        )
        after = atoms_payload(after_ex)
        cls_b = classify_case(
            gold=gold,
            verdict=before["verdict"],
            disposition="AUTO" if before["verdict"] else "UNKNOWN",
        )
        cls_a = classify_case(
            gold=gold,
            verdict=after["verdict"],
            disposition="AUTO" if after["verdict"] else "UNKNOWN",
        )
        tags = leftover_tags(after=after_ex, gold=gold)
        rows.append(
            {
                "id": nid,
                "phenotype_design": note["phenotype_design"],
                "role": ROLE_OF.get(nid) or f"gold_{gold}",
                "gold": gold,
                "verdict_before": before["verdict"],
                "verdict_after": after["verdict"],
                "match_gold_before": cls_b["match_gold"] is True,
                "match_gold_after": cls_a["match_gold"] is True,
                "leftover_tags": tags,
                "leftover_primary": tags[0] if tags else None,
                "administration_status": after_ex.administration_status,
                "n_outcomes": len(after_ex.outcome_statements),
            }
        )
    n_scored = len(rows)
    miss_rows = [r for r in rows if r["match_gold_after"] is not True]
    leftover_rows = [r for r in rows if r["leftover_tags"]]
    table = _shape_table(rows)
    bind_rows = [r for r in rows if r["id"] in BIND_IDS]
    bind_all_match = all(r["match_gold_after"] is True for r in bind_rows)
    miss_ge3 = [s for s, d in table.items() if d["n_miss"] >= 3]
    # A new Translate cell needs a Dual-wrong cluster, not a shape that
    # prior cells already repair on most notes.
    justifies_new_cell = [
        s
        for s, d in table.items()
        if d["n_miss"] >= 3 or (d["n_miss"] >= 2 and d["n_match"] == 0)
    ]
    freeze = not justifies_new_cell
    finding = {
        "n_miss": len(miss_rows),
        "ids_miss": [r["id"] for r in miss_rows],
        "known_miss_unchanged": {r["id"] for r in miss_rows} == set(KNOWN_MISS),
        "design_pbind_all_match": bind_all_match,
        "n_pbind": len(bind_rows),
        "shapes_miss_ge3": miss_ge3,
        "shapes_that_justify_new_translate_cell": justifies_new_cell,
        "any_leftover_shape_recurs_as_miss": any(
            d["recurs_as_miss"] for d in table.values()
        ),
        "v1_freeze_recommended": freeze,
        "v1_reason": (
            "Leftover Dual-wrong notes are one-offs (too-early, held-for-progression, "
            "naive≠never) or weak pairs that are not a new Translate cell: "
            "may-have is 2 miss + 1 already-correct (evaluate_rule/ground hole); "
            "response-only is 6 library / 4 already match via prior cells / 2 "
            "heterogeneous leftover (missing pole vs naive)."
            if freeze
            else "At least one leftover mechanism is a Dual-wrong cluster that "
            "could justify a new Translate cell."
        ),
        "do_not_retune_g3_g9": True,
        "new_repair": False,
        "ground_rewritten": False,
        "locator_chose_repair_class": False,
        "is_selected_editor": False,
    }
    panel = {
        "ok": True,
        "kind": "n2s_g10_leftover_census_v1",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "protocol": "docs/TRACKB-FAILURE-INTERVENTION-MAP.md",
        "intervention": ADMIN,
        "extracts": str(FILL_EXTRACT_PATH.name),
        "n_library": len(notes),
        "n_scored": n_scored,
        "n_miss": len(miss_rows),
        "n_with_leftover_shape": len(leftover_rows),
        "leftover_shapes": list(LEFTOVER_SHAPES),
        "shape_table": table,
        "counts": {
            "miss_by_leftover_primary": dict(
                Counter(r["leftover_primary"] for r in miss_rows)
            ),
            "library_by_leftover_primary": dict(
                Counter(r["leftover_primary"] for r in leftover_rows)
            ),
        },
        "finding": finding,
        "rows": rows,
        "not": [
            "Goal 3–9 retune",
            "new repair",
            "ground() rewrite",
            "CAA",
            "frozen-d",
            "Dual_full rescore",
            ":8260 GUI",
            "G5/G9 panel overwrite",
        ],
        "note": (
            "Read-only leftover-shape prevalence after frozen Goal 9. "
            "Tags every Dual-fill note, not only misses. No new repair."
        ),
    }
    OUT_PATH.write_text(json.dumps(panel, indent=2) + "\n", encoding="utf-8")
    return panel


def selftest() -> None:
    notes = load_dual_library(include_heldout=True)
    assert len(notes) == 108
    assert FILL_EXTRACT_PATH.is_file()
    too = Extraction.from_dict(
        {
            "stated_line": "first",
            "administration_status": "given",
            "quotes": ["initiated yesterday. No restaging yet. cycle 1 day 2."],
            "outcome_statements": [],
        }
    )
    assert leftover_tags(after=too, gold="NOT_SATISFIED") == ["S-too-early"]
    may = Extraction.from_dict(
        {
            "stated_line": "unknown",
            "administration_status": "may_have",
            "quotes": ["may have received platinum at another facility"],
            "outcome_statements": [],
        }
    )
    assert leftover_tags(after=may, gold="UNCERTAIN") == ["S-may-have"]
    held = Extraction.from_dict(
        {
            "stated_line": "first",
            "administration_status": "given",
            "quotes": ["platinum doublet held for progression"],
            "outcome_statements": [
                {"text": "shrinkage", "polarity": "response"}
            ],
        }
    )
    assert leftover_tags(after=held, gold="SATISFIED") == ["S-held-for-progression"]
    resp = Extraction.from_dict(
        {
            "stated_line": "first",
            "administration_status": "given",
            "quotes": ["CR documented on PET"],
            "outcome_statements": [{"text": "CR", "polarity": "response"}],
        }
    )
    assert leftover_tags(after=resp, gold="CONTRADICTION") == ["S-response-only"]
    naive = Extraction.from_dict(
        {
            "stated_line": "first",
            "administration_status": "given",
            "quotes": ["systemic-therapy naive", "completed four cycles"],
            "outcome_statements": [
                {"text": "documented responses", "polarity": "response"}
            ],
        }
    )
    assert leftover_tags(after=naive, gold="CONTRADICTION") == [
        "S-response-only",
        "S-naive-not-never",
    ]
    never_ok = Extraction.from_dict(
        {
            "stated_line": "first",
            "administration_status": "given",
            "quotes": ["never received systemic therapy", "completed six cycles"],
            "outcome_statements": [
                {"text": "progressed", "polarity": "failure"}
            ],
        }
    )
    assert leftover_tags(after=never_ok, gold="CONTRADICTION") == []


def main() -> None:
    parser = argparse.ArgumentParser(description="Goal 10 leftover-shape census")
    parser.add_argument("--selftest", action="store_true")
    parser.add_argument("--census", action="store_true")
    args = parser.parse_args()
    if args.selftest:
        selftest()
        print("selftest ok")
        return
    if args.census:
        panel = build_census()
        print(
            json.dumps(
                {
                    "n_scored": panel["n_scored"],
                    "n_miss": panel["n_miss"],
                    "shape_table": panel["shape_table"],
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
