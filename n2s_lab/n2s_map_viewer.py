"""V1 Dual Translate map viewer — frozen G3–G9 ladder per note.

Read-only. Shows nested approaches on Dual-fill extracts. Not a selected
editor. Does not expand :8260. Does not rewrite ground().
"""

from __future__ import annotations

import json
from typing import Any

from .n2s_batch1_quote_promote import BIND_IDS, NEG_IDS, ROLE_OF, XSPAN_IDS
from .n2s_g2_analog_fill import EXTRACT_PATH as FILL_EXTRACT_PATH
from .n2s_g10_leftover_census import leftover_tags
from .n2s_note_library import load_dual_library
from .n2s_phase2_translate import atoms_payload
from .n2s_v1_selection_contrast import (
    CELLS,
    _apply,
    _match,
    oracle_cell,
    select_cell,
)
from .paths import ARTIFACTS_DIR
from .types import Extraction

OUT_PATH = ARTIFACTS_DIR / "n2s-map-viewer-panel.json"

CELL_HELP = {
    "none": "Raw Dual analog extract. No Translate patch.",
    "G3": "Gated failure promote (asserted, non-negated markers).",
    "G4": "G3 + mixed-span: set contradiction.present on simultaneous 1L.",
    "G6": "G4 + copy never/no-prior quotes into contradiction cues.",
    "G7": "G6 + restage: clear confirmed-failure hedge; undo stable-then-progress X.",
    "G8": "G7 + same-time mix: undo notes-promote; continue-now → ongoing.",
    "G9": "G8 + admin-blocked: planned-after-refractory; never-dispensed.",
}


def _load_fill_extracts() -> dict[str, Any]:
    if not FILL_EXTRACT_PATH.is_file():
        raise SystemExit(f"missing {FILL_EXTRACT_PATH}")
    return json.loads(FILL_EXTRACT_PATH.read_text(encoding="utf-8")).get("extracts") or {}


def _snapshot(ex: Extraction, payload: dict[str, Any]) -> dict[str, Any]:
    atoms = payload["atoms"]
    return {
        "admin": ex.administration_status,
        "A": atoms["A_first_line_identified"],
        "B": atoms["B_first_line_administered"],
        "C": atoms["C_failure_event"],
        "D": atoms["D_failure_of_first_line"],
        "X": atoms["X_contradiction"],
        "outcomes": [o.polarity for o in ex.outcome_statements],
        "contradiction_present": ex.contradiction_present,
        "uncertainty_present": ex.uncertainty_present,
        "fired_rule": payload["fired_rule"],
        "notes": (ex.notes or "")[:240],
    }


def _delta(prev: dict[str, Any] | None, cur: dict[str, Any]) -> list[str]:
    if prev is None:
        return ["raw extract"]
    out: list[str] = []
    for key in (
        "admin",
        "A",
        "B",
        "C",
        "D",
        "X",
        "outcomes",
        "contradiction_present",
        "uncertainty_present",
        "fired_rule",
        "notes",
    ):
        if prev.get(key) != cur.get(key):
            out.append(f"{key}: {prev.get(key)} → {cur.get(key)}")
    return out


def _ladder(ex: Extraction, gold: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    prev_snap: dict[str, Any] | None = None
    for name, _fn in CELLS:
        after, meta = _apply(ex, name)
        payload = atoms_payload(after)
        snap = _snapshot(after, payload)
        delta = _delta(prev_snap, snap)
        rows.append(
            {
                "cell": name,
                "label": CELL_HELP[name],
                "verdict": payload["verdict"],
                "match": _match(gold, after),
                "nested": name != "none",
                "kind": (meta or {}).get("kind") or "none",
                "snapshot": snap,
                "delta": delta,
                "changed": bool(delta) and delta != ["raw extract"],
            }
        )
        prev_snap = snap
    return rows


def build_panel() -> dict[str, Any]:
    notes = load_dual_library(include_heldout=True)
    extracts = _load_fill_extracts()
    cases: list[dict[str, Any]] = []
    for note in notes:
        nid = note["id"]
        rec = extracts.get(nid) or {}
        if not rec.get("parse_ok") or not rec.get("extract"):
            continue
        gold = note["expected"]
        ex = Extraction.from_dict(rec["extract"])
        ladder = _ladder(ex, gold)
        by_cell = {r["cell"]: r for r in ladder}
        chosen = select_cell(ex)
        ora = oracle_cell(ex, gold)
        after_g9, _ = _apply(ex, "G9")
        leftover = leftover_tags(after=after_g9, gold=gold)
        match_sel = by_cell["none"]["match"] if chosen == "none" else by_cell[chosen]["match"]
        cases.append(
            {
                "id": nid,
                "gold": gold,
                "role": ROLE_OF.get(nid) or f"gold_{gold}",
                "slice": note["slice"],
                "phenotype_design": note["phenotype_design"],
                "category": note["category"],
                "why": note.get("why") or "",
                "evidence": note["evidence"],
                "quotes": list(ex.quotes),
                "selector": chosen,
                "oracle": ora,
                "match_none": by_cell["none"]["match"],
                "match_selector": match_sel,
                "match_apply_all": by_cell["G9"]["match"],
                "leftover_tags": leftover,
                "ladder": ladder,
            }
        )
    n = len(cases)
    panel = {
        "ok": True,
        "kind": "n2s_map_viewer_v1",
        "port": 8263,
        "protocol": "docs/TRACKB-FAILURE-INTERVENTION-MAP.md",
        "claim": (
            "V1 demonstrated mapping, not selection. Approaches are nested "
            "Translate wrappers (deeper includes earlier). Not independent "
            "A vs B vs C. locator_chose_repair_class=false. Not auto-correct."
        ),
        "n_cases": n,
        "n_miss_none": sum(1 for c in cases if not c["match_none"]),
        "n_miss_apply_all": sum(1 for c in cases if not c["match_apply_all"]),
        "n_design_bind": len(BIND_IDS),
        "ids": {
            "P-bind": list(BIND_IDS),
            "P-xspan": list(XSPAN_IDS),
            "P-neg": list(NEG_IDS),
        },
        "cell_help": CELL_HELP,
        "cases": cases,
        "not": [
            ":8260 expansion",
            "new Translate cell",
            "ground() rewrite",
            "Encode/Compute editor",
            "auto-correct",
            "locator-chosen repair class",
        ],
    }
    OUT_PATH.write_text(json.dumps(panel, indent=2) + "\n", encoding="utf-8")
    return panel


_CACHE: dict[str, Any] | None = None


def cached_panel() -> dict[str, Any]:
    global _CACHE
    if _CACHE is None:
        _CACHE = build_panel()
    return _CACHE


def selftest() -> None:
    notes = load_dual_library(include_heldout=True)
    assert len(notes) == 108
    assert FILL_EXTRACT_PATH.is_file()
    rec = json.loads(FILL_EXTRACT_PATH.read_text(encoding="utf-8"))["extracts"]["I2"]
    ex = Extraction.from_dict(rec["extract"])
    ladder = _ladder(ex, "SATISFIED")
    assert ladder[-1]["cell"] == "G9"
    assert ladder[-1]["match"] is True
    assert ladder[-2]["match"] is False
    assert select_cell(ex) == "G9"
    assert any(d.startswith("admin:") for d in ladder[-1]["delta"])


if __name__ == "__main__":
    selftest()
    print("selftest ok")
