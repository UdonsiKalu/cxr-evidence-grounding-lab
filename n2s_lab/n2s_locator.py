"""Locator v1 — diagnostic routing across encode / compute / translate.

CPU. Frozen traces + Dual snapshots. Automated response = REVIEW / ABSTAIN
(containment). Not causal first-break. Not a neural repair.
See docs/TRACKB-LOCATOR-V1.md.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any

from .auto_contract import classify_case, score_row
from .n2s_nn_layer_loss import (
    LAYERS,
    apply_n2n_lost_review,
    build_panel,
    class_correct,
    first_lost_step,
)
from .paths import ARTIFACTS_DIR
from .types import Verdict

PANEL_PATH = ARTIFACTS_DIR / "n2s-locator-panel.json"
FIXES_PATH = ARTIFACTS_DIR / "n2s-locator-fixes.json"
PATCH_PANELS = (
    ARTIFACTS_DIR / "n2s-nn-layer-patch-l12l16l20-panel.json",
    ARTIFACTS_DIR / "n2s-nn-layer-patch-l12l16-panel.json",
    ARTIFACTS_DIR / "n2s-nn-layer-patch-panel.json",
)
BC_E1_DUAL_PATH = ARTIFACTS_DIR / "phase7-bmtcart-qwen2.5-coder_32b.json"

ENCODE_LAYER = 8
PROTOCOL = "docs/TRACKB-LOCATOR-V1.md"


def _load_json(path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _load_fixes() -> dict[str, Any]:
    if not FIXES_PATH.is_file():
        return {"ok": True, "kind": "n2s_locator_fixes_v1", "fixes": {}}
    return _load_json(FIXES_PATH)


def _save_fixes(blob: dict[str, Any]) -> None:
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    FIXES_PATH.write_text(json.dumps(blob, indent=2) + "\n", encoding="utf-8")


def _dual_analog_from_patch(case_id: str) -> dict[str, Any] | None:
    for path in PATCH_PANELS:
        if not path.is_file():
            continue
        panel = _load_json(path)
        for row in panel.get("rows") or []:
            if row.get("case_id") != case_id or row.get("label") != "baseline":
                continue
            gold = str(row.get("gold") or "")
            classified = classify_case(gold=gold, verdict=row.get("verdict"))
            return {
                "ok": True,
                **classified,
                "source": "dual_analog_7b_patch_baseline",
                "artifact": path.name,
                "extract_x": row.get("extract_x"),
                "not": "Dual_full 32B",
            }
    return None


def _dual_full_from_phase7(case_id: str) -> dict[str, Any] | None:
    if not BC_E1_DUAL_PATH.is_file():
        return None
    blob = _load_json(BC_E1_DUAL_PATH)
    for row in blob.get("rows") or []:
        if row.get("id") != case_id:
            continue
        scored = score_row(row, "Dual_full")
        if not scored:
            return None
        return {
            "ok": True,
            **scored,
            "source": "dual_full_phase7_frozen",
            "artifact": BC_E1_DUAL_PATH.name,
            "not": "rescored Dual_full",
        }
    return None


def translate_snapshot(case_id: str, gold: str) -> dict[str, Any]:
    analog = _dual_analog_from_patch(case_id)
    if analog:
        return analog
    full = _dual_full_from_phase7(case_id)
    if full:
        return full
    return {
        "ok": False,
        "source": None,
        "gold": gold,
        "verdict": None,
        "disposition": None,
        "bucket": "UNKNOWN",
        "wrong_AUTO": False,
        "status": "unknown",
        "note": "No frozen Dual analog / Dual_full snapshot for this note.",
    }


def encode_check(rec: dict[str, Any]) -> dict[str, Any]:
    scores = rec.get("mean_score_d") or {}
    key = f"L{ENCODE_LAYER}"
    if not rec.get("has_trace") or key not in scores:
        return {
            "surface": "encode",
            "status": "unknown",
            "layer": ENCODE_LAYER,
            "note": "No layer trace — encode not scored (same stick as :8258).",
            "pass": None,
            "probe_type": "none",
            "claim": "not measured",
            "evidence_strength": "none",
            "evidence_why": "No frozen U-A readout at L8.",
        }
    score = float(scores[key])
    side = rec.get("side") or "temporal"
    cc = class_correct(score, side)
    ok = cc > 0
    return {
        "surface": "encode",
        "status": "pass" if ok else "fail",
        "pass": ok,
        "layer": ENCODE_LAYER,
        "mean_score_d": score,
        "class_correct": round(cc, 3),
        "side": side,
        "probe_type": "correlational_readout",
        "claim": "encode probe/readout",
        "evidence_strength": "moderate",
        "evidence_why": (
            "L8 class-correct on frozen d. Correlational. "
            "Not causal proof the failure emerged at encode."
        ),
        "note": (
            f"Formation @ L{ENCODE_LAYER}: class-correct {cc:.2f} "
            f"({'right' if ok else 'wrong'} side of frozen d). Not L24 after compute."
        ),
        "deep_dive": "http://127.0.0.1:8258/",
    }


def compute_check(rec: dict[str, Any]) -> dict[str, Any]:
    if not rec.get("has_trace"):
        return {
            "surface": "compute",
            "status": "unknown",
            "n_lost_steps": None,
            "pass": None,
            "note": "No layer trace — G5 does not fire.",
            "n2n_review": rec.get("n2n_review"),
            "probe_type": "none",
            "claim": "not measured",
            "evidence_strength": "none",
            "evidence_why": "No layer-loss trace.",
            "deep_dive": "http://127.0.0.1:8259/",
        }
    n_lost = int(rec.get("n_lost_steps") or 0)
    lost = first_lost_step(rec.get("steps") or [])
    ok = n_lost == 0
    return {
        "surface": "compute",
        "status": "pass" if ok else "fail",
        "pass": ok,
        "n_lost_steps": n_lost,
        "first_lost": lost,
        "control_ok": rec.get("control_ok"),
        "n2n_review": rec.get("n2n_review")
        or apply_n2n_lost_review(n_lost_steps=n_lost, has_trace=True),
        "probe_type": "correlational_trace",
        "claim": "compute-associated routing signal",
        "evidence_strength": "moderate",
        "evidence_why": (
            "Lost-step pattern on frozen d. Correlational. "
            "Not causal localization of a layer write."
        ),
        "note": (
            "0 lost steps."
            if ok
            else (
                f"{n_lost} lost step(s); first "
                f"L{lost['from_layer']}→L{lost['to_layer']}"
                if lost
                else f"{n_lost} lost step(s)."
            )
        ),
        "deep_dive": "http://127.0.0.1:8259/",
    }


def translate_check(case_id: str, gold: str) -> dict[str, Any]:
    snap = translate_snapshot(case_id, gold)
    if snap.get("bucket") == "UNKNOWN" or not snap.get("ok"):
        return {
            "surface": "translate",
            "status": "unknown",
            "pass": None,
            **snap,
            "probe_type": "none",
            "claim": "not measured",
            "evidence_strength": "none",
            "evidence_why": "No Dual analog / Dual_full snapshot.",
            "deep_dive": "http://127.0.0.1:8253/",
        }
    wrong = bool(snap.get("wrong_AUTO"))
    already_review = snap.get("bucket") == "REVIEW" or snap.get("disposition") == Verdict.REVIEW.value
    src = str(snap.get("source") or "")
    if "dual_full" in src:
        probe_type = "dual_comparison"
        why = "Frozen Dual_full snapshot vs gold. Routing signal, not causal origin."
    else:
        probe_type = "dual_analog"
        why = "7B Dual analog snapshot vs gold. Not Dual_full 32B; not causal origin."
    if wrong:
        status, ok, note = "fail", False, "Dual would AUTO a wrong answer (wrong_AUTO)."
    elif already_review:
        status, ok, note = "review", True, "Already REVIEW — contained."
    else:
        status, ok, note = "pass", True, "Dual AUTO matches gold (this snapshot)."
    return {
        "surface": "translate",
        "status": status,
        "pass": ok if not wrong else False,
        **snap,
        "probe_type": probe_type,
        "claim": "translation-associated routing signal",
        "evidence_strength": "moderate",
        "evidence_why": why,
        "note": note,
        "deep_dive": "http://127.0.0.1:8253/",
    }


def locate_first_break(
    encode: dict[str, Any],
    compute: dict[str, Any],
    translate: dict[str, Any],
) -> dict[str, Any]:
    """Earliest failed *routing* surface. Not causal first-emergence."""
    if encode.get("status") == "fail":
        src = encode
        action = "ABSTAIN"
    elif compute.get("status") == "fail":
        src = compute
        action = "REVIEW"
    elif translate.get("status") == "fail":
        src = translate
        action = "REVIEW"
    else:
        src = None
        action = None
    if src is not None:
        return {
            "surface": src["surface"],
            "status": "located",
            "kind": "routing",
            "claim": src.get("claim"),
            "probe_type": src.get("probe_type"),
            "evidence_strength": src.get("evidence_strength"),
            "evidence_why": src.get("evidence_why"),
            "why": src.get("note"),
            "response": f"{action} (containment, not repair)",
            "not": "causal first emergence",
        }
    known = [
        s.get("status") not in {None, "unknown"}
        for s in (encode, compute, translate)
    ]
    if all(s.get("status") in {"pass", "review"} for s in (encode, compute, translate)):
        return {
            "surface": None,
            "status": "pass",
            "kind": "routing",
            "why": "No failed surface on measured checks.",
            "response": None,
            "not": "causal first emergence",
        }
    missing = [
        name
        for name, s in (
            ("encode", encode),
            ("compute", compute),
            ("translate", translate),
        )
        if s.get("status") == "unknown"
    ]
    return {
        "surface": None,
        "status": "incomplete",
        "kind": "routing",
        "why": "Need a trace and/or Dual snapshot: " + ", ".join(missing),
        "response": None,
        "evidence_strength": "none",
        "probe_type": "none",
        "missing": missing,
        "measured_ok": known,
        "not": "causal first emergence",
    }


def _trusted_response_for(first: dict[str, Any]) -> dict[str, Any] | None:
    """Containment only. Not a neural repair."""
    surface = first.get("surface")
    if first.get("status") != "located" or not surface:
        return None
    if surface == "encode":
        action, reason = "ABSTAIN", "encode_abstain"
    elif surface == "compute":
        action, reason = "REVIEW", "n2n_layer_loss"
    else:
        action, reason = "REVIEW", "dual_wrong_AUTO"
    return {
        "allowed": True,
        "class": "contain",
        "action": action,
        "disposition": action if action == "ABSTAIN" else Verdict.REVIEW.value,
        "reason": reason,
        "surface": surface,
        "do_not": [
            "call this a correction / neural repair",
            "n2n frozen-d patch",
            "α-chase",
            "U-C SAE / U-D",
            "silent Dual_full rewrite",
        ],
    }


def locate_note(rec: dict[str, Any], *, fixes: dict[str, Any] | None = None) -> dict[str, Any]:
    encode = encode_check(rec)
    compute = compute_check(rec)
    translate = translate_check(rec["id"], rec.get("gold") or "")
    first = locate_first_break(encode, compute, translate)
    offered = _trusted_response_for(first)
    applied = None
    blob = fixes if fixes is not None else _load_fixes()
    hit = (blob.get("fixes") or {}).get(rec["id"])
    if hit:
        applied = hit
    return {
        "ok": True,
        "id": rec["id"],
        "gold": rec.get("gold"),
        "side": rec.get("side"),
        "category": rec.get("category"),
        "evidence": rec.get("evidence") or "",
        "why": rec.get("why") or "",
        "has_trace": rec.get("has_trace"),
        "layers": list(LAYERS),
        "mean_score_d": rec.get("mean_score_d"),
        "encode": encode,
        "compute": compute,
        "translate": translate,
        "first_break": first,
        "offered_response": offered,
        "offered_fix": offered,
        "applied_response": applied,
        "applied_fix": applied,
        "invariants": {
            "status": "held",
            "hidden_state_patched": False,
            "dual_full_rewritten": False,
            "overlay_only": True,
            "note": (
                "Containment overlay only. No hidden-state patch, "
                "no Dual_full rewrite, no neural repair."
            ),
        },
        "protocol": PROTOCOL,
        "workbench": "diagnostic_routing",
    }


def build_locator_panel() -> dict[str, Any]:
    nn = build_panel()
    fixes = _load_fixes()
    cases = [locate_note(c, fixes=fixes) for c in nn["cases"]]
    n_located = sum(1 for c in cases if c["first_break"]["status"] == "located")
    n_pass = sum(1 for c in cases if c["first_break"]["status"] == "pass")
    n_incomplete = sum(1 for c in cases if c["first_break"]["status"] == "incomplete")
    panel = {
        "ok": True,
        "kind": "n2s_locator_v1",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "protocol": PROTOCOL,
        "workbench": "diagnostic_routing",
        "quest": (
            "Route known CXR cases to an encode-, compute-, or translate-associated "
            "first-break using frozen probes. Containment only. Not causal origin."
        ),
        "n_cases": len(cases),
        "n_located": n_located,
        "n_pass": n_pass,
        "n_incomplete": n_incomplete,
        "encode_layer": ENCODE_LAYER,
        "cases": cases,
        "claim_hygiene": {
            "say": (
                "diagnostic routing on frozen probes; automated response = "
                "REVIEW or ABSTAIN (containment)"
            ),
            "do_not_say": (
                "causal first-break; auto-correction; neural repair; "
                "n2n d-patch; Dual_full rescore; production CXR"
            ),
        },
        "not": [
            "n2n frozen-d editor (CLOSED)",
            "α=4/8",
            "U-C SAE / U-D",
        ],
    }
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    PANEL_PATH.write_text(json.dumps(panel, indent=2) + "\n", encoding="utf-8")
    return panel


def apply_gated_fix(case_id: str) -> dict[str, Any]:
    """Containment only: REVIEW or ABSTAIN overlay. Does not rewrite the net."""
    panel = build_locator_panel()
    rec = next((c for c in panel["cases"] if c["id"] == case_id), None)
    if rec is None:
        return {"ok": False, "error": f"unknown case {case_id}"}
    offered = rec.get("offered_response") or rec.get("offered_fix")
    if not offered:
        return {
            "ok": False,
            "error": (
                "No automated response: first-break is not a failed surface "
                f"({rec['first_break'].get('status')})."
            ),
            "first_break": rec["first_break"],
        }
    blob = _load_fixes()
    fixes = blob.setdefault("fixes", {})
    entry = {
        "case_id": case_id,
        "gold": rec.get("gold"),
        "first_break": rec["first_break"]["surface"],
        "class": "contain",
        "action": offered.get("action"),
        "disposition": offered["disposition"],
        "reason": offered["reason"],
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "protocol": PROTOCOL,
        "would_have_been": (rec.get("translate") or {}).get("bucket"),
        "not": "neural repair",
    }
    fixes[case_id] = entry
    blob["timestamp"] = entry["timestamp"]
    _save_fixes(blob)
    rec["applied_response"] = entry
    rec["applied_fix"] = entry
    return {"ok": True, "response": entry, "fix": entry, "case": rec}


def selftest() -> None:
    panel = build_locator_panel()
    by_id = {c["id"]: c for c in panel["cases"]}
    contra = by_id["EX_CONTRA"]
    folfox = by_id["EX_TEMPORAL_FOLFOX"]
    assert contra["encode"]["status"] == "pass", contra["encode"]
    assert contra["compute"]["status"] == "fail", contra["compute"]
    assert contra["first_break"]["surface"] == "compute"
    assert contra["first_break"]["kind"] == "routing"
    assert contra["compute"]["probe_type"] == "correlational_trace"
    assert contra["compute"]["evidence_strength"] == "moderate"
    assert contra["offered_fix"]["class"] == "contain"
    assert contra["offered_fix"]["action"] == "REVIEW"
    assert contra["invariants"]["status"] == "held"
    assert contra["invariants"]["hidden_state_patched"] is False
    assert folfox["encode"]["status"] == "pass", folfox["encode"]
    assert folfox["compute"]["status"] == "pass", folfox["compute"]
    assert folfox["translate"]["status"] == "fail", folfox["translate"]
    assert folfox["first_break"]["surface"] == "translate"
    assert folfox["translate"]["probe_type"] == "dual_analog"
    bce1 = by_id["BC_E1"]
    assert bce1["encode"]["status"] == "unknown"
    assert bce1["first_break"]["surface"] == "translate"
    assert bce1["translate"]["wrong_AUTO"] is True
    assert bce1["translate"]["probe_type"] == "dual_comparison"
    t1 = by_id["T1"]
    assert t1["first_break"]["status"] == "incomplete"
    assert apply_gated_fix("T1")["ok"] is False
    prev = _load_fixes()
    try:
        applied = apply_gated_fix("EX_CONTRA")
        assert applied["ok"] is True
        assert applied["fix"]["disposition"] == Verdict.REVIEW.value
        again = locate_note(
            next(c for c in build_panel()["cases"] if c["id"] == "EX_CONTRA")
        )
        assert again["applied_fix"]["reason"] == "n2n_layer_loss"
    finally:
        _save_fixes(prev)
