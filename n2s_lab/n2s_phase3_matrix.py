"""Phase 3 — Encode × Compute × Translate matrix (read-only).

Assembles frozen locator probes + Phase 2 contrast results.
Does not invent an Encode editor, reopen frozen-d, rescore Dual_full, or expand :8260.

See docs/TRACKB-PHASE3-MATRIX.md.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from typing import Any

from .paths import ARTIFACTS_DIR

FOLFOX_ID = "EX_TEMPORAL_FOLFOX"
CONTRA_ID = "EX_CONTRA"

LOCATOR_PANEL = ARTIFACTS_DIR / "n2s-locator-panel.json"
CLEAN_TRANSLATE = ARTIFACTS_DIR / "n2s-phase2-translate-panel-clean.json"
REVERSE = ARTIFACTS_DIR / "n2s-phase2-compute-reverse-panel.json"
OUT_PATH = ARTIFACTS_DIR / "n2s-phase3-matrix-panel.json"


def _load(path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _surface(rec: dict[str, Any], name: str) -> dict[str, Any]:
    return rec.get(name) or {}


def _repair_row(nid: str, rec: dict[str, Any], *, clean: dict[str, Any], reverse: dict[str, Any]) -> dict[str, Any]:
    route = (rec.get("first_break") or {}).get("surface")
    encode_st = _surface(rec, "encode").get("status")
    compute_st = _surface(rec, "compute").get("status")
    translate_st = _surface(rec, "translate").get("status")
    contain = (rec.get("offered_response") or {}).get("action")
    if nid == FOLFOX_ID:
        tcase = (clean.get("cases") or {}).get(FOLFOX_ID) or {}
        return {
            "encode_editor": "none",
            "compute_editor": "none",
            "translate_intervention": "failure_evidence_from_extract_text",
            "translate_applied": True,
            "translate_after": tcase.get("verdict_after"),
            "translate_repaired": bool(tcase.get("repaired")),
            "compute_repairs_this_miss": False,
            "note": "Boundary Translate repair on Schema extract. Frozen-d / G5 do not repair this Dual miss.",
        }
    if nid == CONTRA_ID:
        tcase = (clean.get("cases") or {}).get(CONTRA_ID) or {}
        d = (reverse.get("frozen_d_l12l16l20_alpha2") or {}).get(CONTRA_ID) or {}
        return {
            "encode_editor": "none",
            "compute_editor": "CLOSED (frozen-d PARTIAL historical)",
            "translate_intervention": "failure_evidence_from_extract_text",
            "translate_applied": True,
            "translate_after": tcase.get("verdict_after"),
            "translate_repaired": False,
            "translate_noop": True,
            "n_lost_after_d_patch": d.get("n_lost_after"),
            "compute_clears_all_lost": False,
            "note": "Dual analog already gold. Translate no-op. G5 contains. frozen-d PARTIAL, editor CLOSED.",
        }
    if nid == "BC_E1":
        return {
            "encode_editor": "none",
            "compute_editor": "none",
            "translate_intervention": "not_applied",
            "translate_applied": False,
            "why": "Frozen Dual_full 32B snapshot; do not rescore Phase-7. Quote-promote not transferred.",
            "note": "Translate-routed wrong_AUTO; containment REVIEW only.",
        }
    return {
        "encode_editor": "none",
        "compute_editor": "none",
        "translate_intervention": "not_measured",
        "translate_applied": False,
        "why": "Locator incomplete — no encode/compute/translate snapshot.",
        "note": "Do not manufacture probes or editors to fill the row.",
    }


def build_panel() -> dict[str, Any]:
    loc = _load(LOCATOR_PANEL)
    clean = _load(CLEAN_TRANSLATE)
    reverse = _load(REVERSE)
    rows: list[dict[str, Any]] = []
    n_incomplete = 0
    n_located = 0
    n_translate_repair = 0
    for rec in loc.get("cases") or []:
        nid = rec.get("id")
        fb = rec.get("first_break") or {}
        enc = _surface(rec, "encode")
        comp = _surface(rec, "compute")
        tr = _surface(rec, "translate")
        status = fb.get("status")
        if status == "incomplete":
            n_incomplete += 1
        elif status == "located":
            n_located += 1
        repair = _repair_row(nid, rec, clean=clean, reverse=reverse)
        if repair.get("translate_repaired"):
            n_translate_repair += 1
        rows.append(
            {
                "id": nid,
                "gold": rec.get("gold"),
                "encode": {
                    "status": enc.get("status"),
                    "class_correct_L8": enc.get("class_correct"),
                    "editor": "none",
                },
                "compute": {
                    "status": comp.get("status"),
                    "n_lost_steps": comp.get("n_lost_steps"),
                    "editor": "CLOSED",
                },
                "translate": {
                    "status": tr.get("status"),
                    "verdict": tr.get("verdict"),
                    "bucket": tr.get("bucket"),
                    "probe_type": tr.get("probe_type"),
                },
                "route": {
                    "surface": fb.get("surface"),
                    "status": status,
                    "kind": "routing",
                    "causal": False,
                },
                "contain": {
                    "action": (rec.get("offered_response") or {}).get("action"),
                    "reason": (rec.get("offered_response") or {}).get("reason"),
                    "is_correction": False,
                },
                "repair": repair,
            }
        )
    thesis = {
        "matrix_assembled": True,
        "n_cases": len(rows),
        "n_located": n_located,
        "n_incomplete": n_incomplete,
        "n_boundary_translate_repair": n_translate_repair,
        "encode_editor_invented": False,
        "compute_editor_invented": False,
        "d_patch_reopened": False,
        "dual_full_rescored": False,
        "locator_chose_repair_class": False,
        "ground_changed": False,
        "gui_expanded": False,
        "phase2_forward_holds": bool((clean.get("thesis") or {}).get("contrast_holds")),
        "phase2_reverse_holds": bool((reverse.get("thesis") or {}).get("reverse_contrast_holds")),
        "note": (
            "Matrix is diagnostic routing + which interventions exist. "
            "Encode = detect only. Compute editor CLOSED. Translate boundary "
            "repair shown on FOLFOX Schema extract only. Not locator-causal."
        ),
    }
    panel = {
        "ok": True,
        "kind": "n2s_phase3_matrix_v1",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "protocol": "docs/TRACKB-PHASE3-MATRIX.md",
        "pairing": "read-only locator + Phase 2 artifacts; no GPU",
        "not": [
            "Encode editor",
            "Compute editor",
            "frozen-d reopen",
            "Dual_full Phase-7 rescore",
            "auto-correct pipeline",
            "REVIEW-as-correction",
            "locator-chose-repair-class",
            ":8260 GUI expansion",
            "ground() rewrite",
        ],
        "sources": {
            "locator": str(LOCATOR_PANEL.name),
            "translate_clean": str(CLEAN_TRANSLATE.name),
            "compute_reverse": str(REVERSE.name),
        },
        "rows": rows,
        "thesis": thesis,
    }
    OUT_PATH.write_text(json.dumps(panel, indent=2) + "\n", encoding="utf-8")
    return panel


def selftest() -> None:
    panel = build_panel()
    t = panel["thesis"]
    assert t["matrix_assembled"] is True
    assert t["encode_editor_invented"] is False
    assert t["compute_editor_invented"] is False
    assert t["d_patch_reopened"] is False
    assert t["dual_full_rescored"] is False
    assert t["locator_chose_repair_class"] is False
    assert t["n_cases"] == 7
    assert t["n_located"] == 3
    assert t["n_incomplete"] == 4
    assert t["n_boundary_translate_repair"] == 1
    by_id = {r["id"]: r for r in panel["rows"]}
    assert by_id[FOLFOX_ID]["route"]["surface"] == "translate"
    assert by_id[FOLFOX_ID]["repair"]["translate_repaired"] is True
    assert by_id[CONTRA_ID]["route"]["surface"] == "compute"
    assert by_id[CONTRA_ID]["repair"]["translate_noop"] is True
    assert by_id["BC_E1"]["route"]["surface"] == "translate"
    assert by_id["BC_E1"]["repair"]["translate_applied"] is False
    assert by_id["T1"]["route"]["status"] == "incomplete"
    assert all(r["encode"]["editor"] == "none" for r in panel["rows"])
    assert all(r["compute"]["editor"] == "CLOSED" for r in panel["rows"])


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 3 Encode×Compute×Translate matrix (read-only)")
    parser.add_argument("--selftest", action="store_true")
    parser.add_argument("--panel", action="store_true")
    args = parser.parse_args()
    if args.selftest:
        selftest()
        print("selftest ok")
        return
    if args.panel:
        panel = build_panel()
        print(json.dumps({"thesis": panel["thesis"], "artifact": str(OUT_PATH)}, indent=2))
        return
    parser.print_help()


if __name__ == "__main__":
    main()
