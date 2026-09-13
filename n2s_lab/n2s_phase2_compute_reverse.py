"""Phase 2 reverse contrast — frozen Compute tools vs Translate FOLFOX miss.

Read-only assembly of already-run artifacts. Does not invent a Compute editor,
does not re-run frozen-d GPU, does not change ground().

See docs/TRACKB-PHASE2-COMPUTE-REVERSE.md.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from typing import Any

from .paths import ARTIFACTS_DIR

FOLFOX_ID = "EX_TEMPORAL_FOLFOX"
CONTRA_ID = "EX_CONTRA"

CLEAN_PANEL = ARTIFACTS_DIR / "n2s-phase2-translate-panel-clean.json"
G5_PANEL = ARTIFACTS_DIR / "n2s-nn-lost-review-panel.json"
RESIDUAL_PATCH = ARTIFACTS_DIR / "n2s-nn-layer-patch-l12l16l20-panel.json"
OUT_PATH = ARTIFACTS_DIR / "n2s-phase2-compute-reverse-panel.json"


def _load(path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _g5_case(blob: dict[str, Any], nid: str) -> dict[str, Any]:
    for rec in blob.get("cases") or []:
        if rec.get("id") == nid:
            return rec
    return {}


def _patch_row(blob: dict[str, Any], nid: str, *, label: str, alpha: float) -> dict[str, Any]:
    for rec in blob.get("rows") or []:
        if (
            rec.get("case_id") == nid
            and rec.get("label") == label
            and float(rec.get("alpha") or 0) == alpha
        ):
            return rec
    return {}


def build_panel() -> dict[str, Any]:
    clean = _load(CLEAN_PANEL)
    g5 = _load(G5_PANEL)
    residual = _load(RESIDUAL_PATCH)

    g5_folfox = _g5_case(g5, FOLFOX_ID)
    g5_contra = _g5_case(g5, CONTRA_ID)
    d_folfox = _patch_row(residual, FOLFOX_ID, label="patch", alpha=2.0)
    d_contra = _patch_row(residual, CONTRA_ID, label="patch", alpha=2.0)
    d_folfox0 = _patch_row(residual, FOLFOX_ID, label="baseline", alpha=0.0)
    d_contra0 = _patch_row(residual, CONTRA_ID, label="baseline", alpha=0.0)

    translate = (clean.get("thesis") or {})
    translate_cases = clean.get("cases") or {}
    folfox_t = translate_cases.get(FOLFOX_ID) or {}
    contra_t = translate_cases.get(CONTRA_ID) or {}

    g5_folfox_fires = bool((g5_folfox.get("n2n_review") or {}).get("triggered"))
    g5_contra_fires = bool((g5_contra.get("n2n_review") or {}).get("triggered"))
    folfox_d_verdict = d_folfox.get("verdict") or d_folfox0.get("verdict")
    contra_d_verdict = d_contra.get("verdict") or d_contra0.get("verdict")
    contra_lost_after = d_contra.get("n_lost_steps")
    contra_lost_before = d_contra0.get("n_lost_steps")
    folfox_lost_after = d_folfox.get("n_lost_steps")

    compute_does_not_repair_folfox = (
        not g5_folfox_fires
        and folfox_d_verdict == "NOT_SATISFIED"
        and folfox_lost_after == 0
    )
    translate_does_not_clear_contra_loss = bool(
        contra_t.get("ok")
        and (contra_t.get("frozen_trace") or {}).get("n_lost_steps") == 2
        and contra_t.get("verdict_after") == "CONTRADICTION"
        and not contra_t.get("broke_control")
    )
    g5_contains_not_repair = bool(
        g5_contra_fires
        and (g5_contra.get("n2n_review") or {}).get("disposition") == "REVIEW"
        and (g5_contra.get("n_lost_steps") == 2)
    )
    d_patch_partial = bool(
        contra_lost_before == 2
        and contra_lost_after == 1
        and contra_d_verdict == "CONTRADICTION"
        and folfox_d_verdict == "NOT_SATISFIED"
    )
    reverse_holds = bool(
        compute_does_not_repair_folfox and translate_does_not_clear_contra_loss
    )

    thesis = {
        "reverse_contrast_holds": reverse_holds,
        "compute_does_not_repair_folfox_translate_miss": compute_does_not_repair_folfox,
        "translate_does_not_clear_contra_compute_loss": translate_does_not_clear_contra_loss,
        "g5_contains_contra_does_not_repair": g5_contains_not_repair,
        "d_patch_partial_not_editor": d_patch_partial,
        "compute_editor_invented": False,
        "d_patch_rerun": False,
        "locator_chose_repair_class": False,
        "ground_changed": False,
        "note": (
            "Reverse = existing Compute tools (G5 REVIEW, closed frozen-d) do not "
            "repair FOLFOX Dual miss; Translate quote-promote does not clear CONTRA "
            "lost-steps. Not a new Compute editor. Not locator-causal."
        ),
    }
    panel = {
        "ok": True,
        "kind": "n2s_phase2_compute_reverse_v1",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "protocol": "docs/TRACKB-PHASE2-COMPUTE-REVERSE.md",
        "pairing": "read-only frozen artifacts; no GPU",
        "not": [
            "new Compute editor",
            "frozen-d GPU rerun",
            "α-chase",
            "ground() rewrite",
            "Encode editor",
            "auto-correct pipeline",
            "REVIEW-as-correction",
            "locator-chose-repair-class",
            ":8260 GUI expansion",
        ],
        "sources": {
            "translate_clean": str(CLEAN_PANEL.name),
            "g5_review": str(G5_PANEL.name),
            "frozen_d_furthest": str(RESIDUAL_PATCH.name),
        },
        "forward_translate": {
            "folfox_repaired": bool(translate.get("folfox_repaired")),
            "contra_dual_held": bool(translate.get("contra_dual_held")),
            "contrast_holds": bool(translate.get("contrast_holds")),
            "folfox_before": folfox_t.get("verdict_before"),
            "folfox_after": folfox_t.get("verdict_after"),
            "contra_after": contra_t.get("verdict_after"),
            "contra_n_lost": (contra_t.get("frozen_trace") or {}).get("n_lost_steps"),
        },
        "g5_n2n_lost_review": {
            FOLFOX_ID: {
                "n_lost_steps": g5_folfox.get("n_lost_steps"),
                "triggered": g5_folfox_fires,
                "reason": (g5_folfox.get("n2n_review") or {}).get("reason"),
                "repairs_dual_miss": False,
            },
            CONTRA_ID: {
                "n_lost_steps": g5_contra.get("n_lost_steps"),
                "triggered": g5_contra_fires,
                "disposition": (g5_contra.get("n2n_review") or {}).get("disposition"),
                "reason": (g5_contra.get("n2n_review") or {}).get("reason"),
                "clears_lost_steps": False,
            },
        },
        "frozen_d_l12l16l20_alpha2": {
            FOLFOX_ID: {
                "n_lost_after": folfox_lost_after,
                "verdict": folfox_d_verdict,
                "repairs_translate_miss": False,
            },
            CONTRA_ID: {
                "n_lost_before": contra_lost_before,
                "n_lost_after": contra_lost_after,
                "verdict": contra_d_verdict,
                "status": "PARTIAL",
                "clears_all_lost": False,
            },
        },
        "thesis": thesis,
    }
    OUT_PATH.write_text(json.dumps(panel, indent=2) + "\n", encoding="utf-8")
    return panel


def selftest() -> None:
    panel = build_panel()
    t = panel["thesis"]
    assert t["compute_editor_invented"] is False
    assert t["d_patch_rerun"] is False
    assert t["locator_chose_repair_class"] is False
    assert t["ground_changed"] is False
    assert t["compute_does_not_repair_folfox_translate_miss"] is True
    assert t["translate_does_not_clear_contra_compute_loss"] is True
    assert t["reverse_contrast_holds"] is True
    assert panel["g5_n2n_lost_review"][FOLFOX_ID]["triggered"] is False
    assert panel["g5_n2n_lost_review"][CONTRA_ID]["triggered"] is True
    assert panel["frozen_d_l12l16l20_alpha2"][FOLFOX_ID]["verdict"] == "NOT_SATISFIED"
    assert panel["frozen_d_l12l16l20_alpha2"][CONTRA_ID]["n_lost_after"] == 1


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 2 Compute reverse contrast (read-only)")
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
