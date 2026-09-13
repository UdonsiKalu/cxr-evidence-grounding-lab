"""Goal 3 — predeclared Dual gated quote-promote (negation-aware).

Gate (frozen before scoring; not fitted to leak IDs):
  Same quote-promote as Phase 2, but a span is failure evidence only if a
  FAILURE_MARKER occurs that is not under local negation
  (no / not / without / never / n't / denies / denied / absent /
  no evidence of / negative for / lack of / lacking in the 28 characters
  before the marker, or the marker is followed by -free / " free").

Rationale: P-neg is denial of failure in extract text. Ungated promote
matches the substring "progression" inside "no progression documented".
Admin/adjuvant gates would miss first-line given + ongoing notes.

CPU on Goal 2 Dual-fill extracts. Does not overwrite G2 / Batch 1 artifacts.
Does not rewrite ground(), retune after held-out, run CAA, or patch hidden
states. locator_chose_repair_class stays false.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from typing import Any

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
    _has_asserted_failure_marker,
    _has_failure_marker,
    relabel_failure_polarity_from_text,
    relabel_failure_polarity_from_text_gated,
)
from .paths import ARTIFACTS_DIR
from .types import Extraction

OUT_PATH = ARTIFACTS_DIR / "n2s-g3-gated-promote-panel.json"
UNGATED = "failure_evidence_from_extract_text"
GATED = "failure_evidence_from_extract_text_gated"
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
    gated: bool,
    role_fn,
) -> list[dict[str, Any]]:
    relabel = (
        relabel_failure_polarity_from_text_gated
        if gated
        else relabel_failure_polarity_from_text
    )
    intervention = GATED if gated else UNGATED
    rows = []
    for i in ids:
        rec = extracts.get(i) or {}
        rows.append(
            apply_row(
                notes[i],
                rec,
                prompt=PROMPT,
                role=role_fn(notes[i]),
                relabel=relabel,
                intervention=intervention,
            )
        )
    return rows


def build_panel() -> dict[str, Any]:
    notes = {n["id"]: n for n in load_dual_library(include_heldout=True)}
    extracts = _load_fill_extracts()
    design_ids = list(BIND_IDS) + list(XSPAN_IDS) + list(NEG_IDS)
    heldout = [n for n in notes.values() if not n["phenotype_design"]]
    heldout_ids = [n["id"] for n in heldout]

    def design_role(n: dict[str, Any]) -> str:
        return ROLE_OF.get(n["id"]) or f"gold_{n['expected']}"

    def held_role(n: dict[str, Any]) -> str:
        return f"gold_{n['expected']}"

    ungated_design = _score_ids(
        notes, extracts, design_ids, gated=False, role_fn=design_role
    )
    gated_design = _score_ids(
        notes, extracts, design_ids, gated=True, role_fn=design_role
    )
    ungated_held = _score_ids(
        notes, extracts, heldout_ids, gated=False, role_fn=held_role
    )
    gated_held = _score_ids(
        notes, extracts, heldout_ids, gated=True, role_fn=held_role
    )

    u_des = _arm_counts(ungated_design)
    g_des = _arm_counts(gated_design)
    u_h = _arm_counts(ungated_held)
    g_h = _arm_counts(gated_held)
    u_bind = u_des.get("P-bind") or {}
    g_bind = g_des.get("P-bind") or {}
    u_xspan = u_des.get("P-xspan") or {}
    g_xspan = g_des.get("P-xspan") or {}
    u_neg = u_des.get("P-neg") or {}
    g_neg = g_des.get("P-neg") or {}

    ungated_bind_ids = set(u_bind.get("ids_repaired") or [])
    gated_bind_ids = set(g_bind.get("ids_repaired") or [])
    kept_bind = sorted(ungated_bind_ids & gated_bind_ids)
    lost_bind = sorted(ungated_bind_ids - gated_bind_ids)

    design_specificity = (
        g_xspan.get("collateral_to_SATISFIED", 0) == 0
        and g_neg.get("collateral_to_SATISFIED", 0) == 0
    )
    held_neg_ids = list(
        (g_h.get("gold_NOT_SATISFIED") or {}).get("ids_collateral") or []
    )
    held_x_ids = list(
        (g_h.get("gold_CONTRADICTION") or {}).get("ids_collateral") or []
    )
    held_leak = bool(held_neg_ids or held_x_ids)
    map_cell = {
        "ungated_pbind_k": u_bind.get("repaired", 0),
        "gated_pbind_k": g_bind.get("repaired", 0),
        "pbind_n": g_bind.get("n", 0),
        "ungated_neg_collateral": u_neg.get("collateral_to_SATISFIED", 0),
        "gated_neg_collateral": g_neg.get("collateral_to_SATISFIED", 0),
        "ungated_xspan_collateral": u_xspan.get("collateral_to_SATISFIED", 0),
        "gated_xspan_collateral": g_xspan.get("collateral_to_SATISFIED", 0),
        "kept_ungated_bind_repairs": kept_bind,
        "lost_ungated_bind_repairs": lost_bind,
        "specificity_holds": design_specificity,
        "heldout_leak": held_leak,
        "heldout_leak_ids": sorted(held_neg_ids + held_x_ids),
        "heldout_sat_repaired": g_h.get("gold_SATISFIED", {}).get("repaired", 0),
        "heldout_sat_n": g_h.get("gold_SATISFIED", {}).get("n", 0),
        "heldout_neg_collateral": g_h.get("gold_NOT_SATISFIED", {}).get(
            "collateral_to_SATISFIED", 0
        ),
        "heldout_xspan_collateral": g_h.get("gold_CONTRADICTION", {}).get(
            "collateral_to_SATISFIED", 0
        ),
        "dual_analog_user_evidence": True,
        "schema_in_system": True,
        "rule_changed_after_heldout": False,
        "locator_chose_repair_class": False,
        "is_selected_editor": False,
    }
    panel = {
        "ok": True,
        "kind": "n2s_g3_gated_promote_v1",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "protocol": "docs/TRACKB-FAILURE-INTERVENTION-MAP.md",
        "intervention": GATED,
        "gate": "asserted_failure_marker",
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
            "ungated promote overwrite",
            "silent retune from leak IDs",
            "hidden-state patch",
            "frozen-d",
            "ground() rewrite",
            "CAA",
            "Dual_full rescore",
            "G2 extract overwrite",
            "Batch 1 panel overwrite",
            ":8260 GUI",
            "auto-correct",
        ],
        "counts": {
            "ungated_design": u_des,
            "gated_design": g_des,
            "ungated_heldout": u_h,
            "gated_heldout": g_h,
        },
        "map_cell": map_cell,
        "rows": {
            "ungated_design": ungated_design,
            "gated_design": gated_design,
            "ungated_heldout": ungated_held,
            "gated_heldout": gated_held,
        },
        "note": (
            "Predeclared negation-aware quote-promote on Dual-fill extracts. "
            "Gate frozen before scoring. Held-out is score-only, not a retune set."
        ),
    }
    OUT_PATH.write_text(json.dumps(panel, indent=2) + "\n", encoding="utf-8")
    return panel


def selftest() -> None:
    batch1_selftest()
    denied = (
        "no progression documented",
        "without progression",
        "no new lesions",
        "progression-free",
        "never progressed",
        "no evidence of treatment failure",
    )
    asserted = (
        "then progressed",
        "discontinued for intolerance",
        "platinum-refractory disease",
        "new hepatic metastases",
        "treatment failure after FOLFOX",
    )
    for span in denied:
        assert not _has_asserted_failure_marker(span), span
        assert _has_failure_marker(span) or "free" in span, span
    for span in asserted:
        assert _has_asserted_failure_marker(span), span
    empty = Extraction.from_dict({})
    after, meta = relabel_failure_polarity_from_text_gated(empty)
    assert meta["kind"] == GATED
    assert after.outcome_statements == []
    leakish = Extraction.from_dict(
        {
            "quotes": ["ongoing response; no progression documented"],
            "outcome_statements": [
                {"text": "ongoing response", "polarity": "response"}
            ],
        }
    )
    after_l, meta_l = relabel_failure_polarity_from_text_gated(leakish)
    assert not meta_l["promoted_failure_from"]
    assert meta_l["n_polarities_relabeled"] == 0
    assert not meta_l["added_response_then_progression"]
    assert all((o.polarity or "") != "failure" for o in after_l.outcome_statements)
    rich = Extraction.from_dict(
        {
            "quotes": ["FOLFOX was discontinued for treatment failure"],
            "outcome_statements": [{"text": "partial response", "polarity": "response"}],
        }
    )
    after_r, meta_r = relabel_failure_polarity_from_text_gated(rich)
    assert meta_r["promoted_failure_from"] == "quotes"
    assert any((o.polarity or "") == "failure" for o in after_r.outcome_statements)
    notes = load_dual_library(include_heldout=True)
    assert len(notes) == 108
    assert FILL_EXTRACT_PATH.is_file()


def main() -> None:
    parser = argparse.ArgumentParser(description="Goal 3 Dual gated quote-promote")
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
                    "counts": {
                        "ungated_design": panel["counts"]["ungated_design"],
                        "gated_design": panel["counts"]["gated_design"],
                        "ungated_heldout": panel["counts"]["ungated_heldout"],
                        "gated_heldout": panel["counts"]["gated_heldout"],
                    },
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
