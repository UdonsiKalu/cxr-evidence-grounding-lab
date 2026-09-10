"""Neural-to-neural layer-loss v1 — frozen replay of U-A mean_score_d.

No GPU. No new d. Reads artifacts/n2s-upstream-ua-map.json plus coding-set notes.
"""

from __future__ import annotations

import json
from typing import Any

from .paths import ARTIFACTS_DIR, DATA_PATH, TEMPORAL_FAMILY_DEV_PATH
from .types import Verdict

# Inlined so this module does not import torch via n2s_upstream_prefill.
DEFAULT_NOTES: dict[str, dict[str, str]] = {
    "EX_TEMPORAL_FOLFOX": {
        "id": "EX_TEMPORAL_FOLFOX",
        "gold": "SATISFIED",
        "evidence": (
            "Metastatic colorectal cancer. First-line FOLFOX produced a partial response. "
            "At follow-up four months later, imaging demonstrated new hepatic lesions "
            "consistent with progression. FOLFOX was discontinued for treatment failure; "
            "second-line therapy discussed."
        ),
    },
    "EX_CONTRA": {
        "id": "EX_CONTRA",
        "gold": "CONTRADICTION",
        "evidence": (
            "Note A: Patient failed first-line FOLFOX after four cycles. "
            "Addendum same day: Disease remains responsive to FOLFOX; continue current regimen."
        ),
    },
}

UA_MAP_PATH = ARTIFACTS_DIR / "n2s-upstream-ua-map.json"
PANEL_PATH = ARTIFACTS_DIR / "n2s-nn-layer-loss-panel.json"
N2N_REVIEW_PATH = ARTIFACTS_DIR / "n2s-nn-lost-review-panel.json"

LAYERS = (8, 12, 16, 20, 24)
TAU = 5.0

# Coding-set ids from :8253 cases.json + Track A seed (same notes as N2S analysis).
CODING_IDS = ("T1", "T3", "C1", "C4")
FAMILY_IDS = ("BC_E1",)


def _load_json(path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def ground_rules() -> dict[str, Any]:
    return {
        "version": "nn_layer_loss_v1",
        "information": (
            "mean_score_d = mean note-body residual projected onto frozen U-A "
            "d = unit(μ_T−μ_C). + temporal-change side; − contradiction side."
        ),
        "where": list(LAYERS),
        "lost": (
            f"On step L→L′: drop if class-correct falls by τ≥{TAU}, "
            "or flip if sign(mean_score_d) changes. "
            "Temporal class-correct = mean_score_d; "
            "contradiction class-correct = −mean_score_d."
        ),
        "tau": TAU,
        "control": (
            "Contradiction note must stay below the temporal note’s mean_score_d "
            "at every listed layer."
        ),
        "not": [
            "Shannon bit-loss layer 1→N",
            "causal editor / α-chase / U-C SAE / U-D",
            "production CXR",
        ],
    }


def _side_for_gold(gold: str, category: str | None = None) -> str:
    if (gold or "").upper() == "CONTRADICTION":
        return "contra"
    cat = (category or "").lower()
    if cat in {"conflicting", "contradiction"}:
        return "contra"
    return "temporal"


def class_correct(score_d: float, side: str) -> float:
    if side == "contra":
        return -float(score_d)
    return float(score_d)


def apply_n2n_lost_review(*, n_lost_steps: int | None, has_trace: bool) -> dict[str, Any]:
    """Track A contain: lost n2n steps → Dual REVIEW. Not a G3 change.

    No layer trace → gate does not fire (cannot contain from missing evidence).
    """
    if not has_trace or n_lost_steps is None:
        return {
            "triggered": None,
            "disposition": None,
            "reason": "no_layer_trace",
        }
    if int(n_lost_steps) > 0:
        return {
            "triggered": True,
            "disposition": Verdict.REVIEW.value,
            "reason": "n2n_layer_loss",
            "n_lost_steps": int(n_lost_steps),
        }
    return {
        "triggered": False,
        "disposition": None,
        "reason": "no_lost_steps",
        "n_lost_steps": 0,
    }


def first_lost_step(steps: list[dict[str, Any]]) -> dict[str, Any] | None:
    """Earliest L→L′ step flagged lost (τ-drop or sign flip)."""
    for s in steps:
        if s.get("lost"):
            return s
    return None


def lost_step_endpoints(steps: list[dict[str, Any]]) -> list[int]:
    """Unique layer indices on lost steps, in listed order (e.g. 12, 16, 20)."""
    out: list[int] = []
    for s in steps:
        if not s.get("lost"):
            continue
        for L in (s.get("from_layer"), s.get("to_layer")):
            if L is None:
                continue
            Li = int(L)
            if Li not in out:
                out.append(Li)
    return out


def detect_patch_plan() -> dict[str, Any]:
    """CPU: first lost step on the frozen pair → Dual-shaped patch site.

    Act at the earlier of the two layers. Temporal push +α·d; contra −α·d.
    Dual analog: AUTO ∧ verdict ≠ gold → fail patch, fallback REVIEW.
    """
    panel = build_panel()
    by_id = {c["id"]: c for c in panel["cases"]}
    contra = by_id["EX_CONTRA"]
    temporal = by_id["EX_TEMPORAL_FOLFOX"]
    lost = first_lost_step(contra.get("steps") or [])
    patch_layer = int(lost["from_layer"]) if lost else None
    patch_layers = (
        [int(lost["from_layer"]), int(lost["to_layer"])] if lost else []
    )
    residual_layers = lost_step_endpoints(contra.get("steps") or [])
    return {
        "ok": True,
        "kind": "n2s_nn_layer_patch_detect_v1",
        "quest": (
            "Detect first n2n lost step, then a small frozen-d push at the "
            "earlier layer; Dual AUTO-wrong → fail, fallback REVIEW."
        ),
        "protocol": "docs/TRACKB-NN-LAYER-PATCH.md",
        "temporal_id": temporal["id"],
        "contra_id": contra["id"],
        "temporal_n_lost": temporal.get("n_lost_steps"),
        "contra_n_lost": contra.get("n_lost_steps"),
        "first_lost": lost,
        "patch_layer": patch_layer,
        "patch_layers": patch_layers,
        "residual_layers": residual_layers,
        "alphas": [1.0, 2.0],
        "push": "temporal +α·d; contradiction −α·d (class-correct)",
        "dual_analog": (
            "Patched 7B extract → ground → rule. "
            "If disposition AUTO and verdict ≠ gold → fail patch, fallback REVIEW. "
            "True-contra must not AUTO SATISFIED / flip extract_x to false. "
            "Not Dual_full 32B C+D."
        ),
        "not": [
            "α=8 Intervene / expand-v",
            "U-C SAE / U-D",
            "L24-only U-A causal (this is first-lost layer)",
        ],
    }


def _step_table(by_layer_scores: dict[int, float], side: str) -> list[dict[str, Any]]:
    steps: list[dict[str, Any]] = []
    for a, b in zip(LAYERS, LAYERS[1:]):
        sa = float(by_layer_scores[a])
        sb = float(by_layer_scores[b])
        ca = class_correct(sa, side)
        cb = class_correct(sb, side)
        drop = ca - cb
        sign_flip = (sa >= 0) != (sb >= 0)
        steps.append(
            {
                "from_layer": a,
                "to_layer": b,
                "mean_score_d_from": round(sa, 3),
                "mean_score_d_to": round(sb, 3),
                "class_correct_from": round(ca, 3),
                "class_correct_to": round(cb, 3),
                "drop": round(drop, 3),
                "sign_flip": sign_flip,
                "lost_drop": drop >= TAU,
                "lost": bool(sign_flip or drop >= TAU),
            }
        )
    return steps


def _scores_from_ua_map(ua: dict[str, Any], note_id: str) -> dict[int, float] | None:
    by_layer = ua.get("by_layer") or {}
    out: dict[int, float] = {}
    for L in LAYERS:
        row = by_layer.get(f"L{L}") or {}
        notes = row.get("notes") or {}
        rec = notes.get(note_id)
        if not rec or "mean_score_d" not in rec:
            return None
        out[L] = float(rec["mean_score_d"])
    return out


def _coding_cases() -> list[dict[str, Any]]:
    cases = _load_json(DATA_PATH).get("cases") or []
    want = {i: None for i in CODING_IDS}
    for c in cases:
        if c.get("id") in want:
            want[c["id"]] = c
    family = _load_json(TEMPORAL_FAMILY_DEV_PATH).get("cases") or []
    fam = {c["id"]: c for c in family if c.get("id") in FAMILY_IDS}
    ordered: list[dict[str, Any]] = []
    for nid in CODING_IDS:
        if want[nid]:
            ordered.append(want[nid])
    for nid in FAMILY_IDS:
        if nid in fam:
            ordered.append(fam[nid])
    return ordered


def _ua_pair_cases() -> list[dict[str, Any]]:
    out = []
    for nid, rec in DEFAULT_NOTES.items():
        gold = rec.get("gold") or "UNKNOWN"
        out.append(
            {
                "id": nid,
                "category": "ua_pair",
                "expected": gold,
                "why": "Frozen U-A map pair (has layer trace).",
                "evidence": rec.get("evidence") or "",
                "source_set": "DEFAULT_NOTES / n2s-upstream-ua-map.json",
            }
        )
    return out


def analyze_note(
    case: dict[str, Any],
    *,
    scores: dict[int, float] | None,
    temporal_scores: dict[int, float] | None,
) -> dict[str, Any]:
    gold = str(case.get("expected") or case.get("gold") or "UNKNOWN")
    side = _side_for_gold(gold, case.get("category"))
    has_trace = scores is not None
    steps: list[dict[str, Any]] = []
    control_ok: bool | None = None
    n_lost = 0
    peak_layer = None
    if scores is not None:
        steps = _step_table(scores, side)
        n_lost = sum(1 for s in steps if s["lost"])
        peak_layer = max(scores, key=lambda L: class_correct(scores[L], side))
        if side == "contra" and temporal_scores is not None:
            control_ok = all(scores[L] < temporal_scores[L] for L in LAYERS)
        elif side == "temporal" and temporal_scores is not None:
            control_ok = None
    return {
        "id": case["id"],
        "category": case.get("category"),
        "gold": gold,
        "side": side,
        "why": case.get("why") or "",
        "evidence": case.get("evidence") or "",
        "source_set": case.get("source_set") or "data/cases.json",
        "has_trace": has_trace,
        "layers": list(LAYERS),
        "mean_score_d": (
            {f"L{L}": round(scores[L], 3) for L in LAYERS} if scores else None
        ),
        "steps": steps,
        "n_lost_steps": n_lost,
        "peak_layer": peak_layer,
        "control_ok": control_ok,
        "tau": TAU,
        "n2n_review": apply_n2n_lost_review(n_lost_steps=n_lost, has_trace=has_trace),
    }


def build_panel() -> dict[str, Any]:
    ua = _load_json(UA_MAP_PATH)
    t_scores = _scores_from_ua_map(ua, "EX_TEMPORAL_FOLFOX")
    c_scores = _scores_from_ua_map(ua, "EX_CONTRA")
    traces = {
        "EX_TEMPORAL_FOLFOX": t_scores,
        "EX_CONTRA": c_scores,
    }
    cases = _ua_pair_cases() + _coding_cases()
    analyzed = [
        analyze_note(c, scores=traces.get(c["id"]), temporal_scores=t_scores)
        for c in cases
    ]
    panel = {
        "ok": True,
        "kind": "n2s_nn_layer_loss_v1",
        "quest": "Where does class-correct mean_score_d drop or flip across U-A layers?",
        "model_id": ua.get("model_id"),
        "ua_map": str(UA_MAP_PATH.name),
        "pair": ua.get("pair"),
        "ground_rules": ground_rules(),
        "n_cases": len(analyzed),
        "n_with_trace": sum(1 for a in analyzed if a["has_trace"]),
        "cases": analyzed,
        "claim_hygiene": (
            "Correlational replay of frozen U-A scores. Not a layer-1…N loss curve. "
            "Not a causal neural-to-neural editor."
        ),
    }
    PANEL_PATH.write_text(json.dumps(panel, indent=2) + "\n", encoding="utf-8")
    return panel


def scores_from_live_by_layer(by_layer: dict[str, Any]) -> dict[int, float] | None:
    out: dict[int, float] = {}
    for L in LAYERS:
        rec = (by_layer or {}).get(f"L{L}") or {}
        if "mean_score_d" not in rec:
            return None
        out[L] = float(rec["mean_score_d"])
    return out


def analyze_pasted_note(
    evidence: str,
    *,
    gold: str = "UNKNOWN",
    scores: dict[int, float] | None,
    temporal_scores: dict[int, float] | None,
    case_id: str = "PASTE",
) -> dict[str, Any]:
    case = {
        "id": case_id,
        "category": "paste",
        "expected": gold,
        "why": "User-pasted note vs frozen U-A d (same stick as :8258).",
        "evidence": (evidence or "").strip(),
        "source_set": "paste",
    }
    rec = analyze_note(case, scores=scores, temporal_scores=temporal_scores)
    rec["live"] = scores is not None
    return rec


def frozen_temporal_scores() -> dict[int, float] | None:
    ua = _load_json(UA_MAP_PATH)
    return _scores_from_ua_map(ua, "EX_TEMPORAL_FOLFOX")


def selftest() -> None:
    panel = build_panel()
    assert panel["n_with_trace"] >= 2, panel["n_with_trace"]
    by_id = {c["id"]: c for c in panel["cases"]}
    folfox = by_id["EX_TEMPORAL_FOLFOX"]
    contra = by_id["EX_CONTRA"]
    assert folfox["has_trace"] and contra["has_trace"]
    assert contra["control_ok"] is True
    assert by_id["T1"]["has_trace"] is False
    assert by_id["BC_E1"]["evidence"]
    # Frozen pair: temporal should not τ-drop; contra has a sign flip L12→L16.
    assert all(not s["lost_drop"] for s in folfox["steps"])
    assert any(s["sign_flip"] for s in contra["steps"])
    plan = detect_patch_plan()
    lost = plan["first_lost"]
    assert lost is not None
    assert lost["from_layer"] == 12 and lost["to_layer"] == 16
    assert plan["patch_layer"] == 12
    assert plan["patch_layers"] == [12, 16]
    assert plan["residual_layers"] == [12, 16, 20]
    assert contra["n2n_review"]["triggered"] is True
    assert contra["n2n_review"]["disposition"] == Verdict.REVIEW.value
    assert folfox["n2n_review"]["triggered"] is False
    assert by_id["T1"]["n2n_review"]["triggered"] is None
    rev = build_n2n_lost_review_panel()
    assert rev["n_review"] >= 1
    assert rev["n_pass"] >= 1


def build_n2n_lost_review_panel() -> dict[str, Any]:
    """CPU: Dual contain on frozen traces — lost steps → REVIEW. No GPU. Not G3."""
    panel = build_panel()
    rows = []
    for c in panel["cases"]:
        n2n = c.get("n2n_review") or apply_n2n_lost_review(
            n_lost_steps=c.get("n_lost_steps"), has_trace=bool(c.get("has_trace"))
        )
        rows.append(
            {
                "id": c["id"],
                "gold": c["gold"],
                "has_trace": c["has_trace"],
                "n_lost_steps": c.get("n_lost_steps"),
                "n2n_review": n2n,
            }
        )
    n_review = sum(1 for r in rows if r["n2n_review"].get("triggered") is True)
    n_pass = sum(1 for r in rows if r["n2n_review"].get("triggered") is False)
    out = {
        "ok": True,
        "kind": "n2s_nn_lost_review_v1",
        "quest": "If n2n layer-loss flags lost steps, Dual REVIEW (contain).",
        "protocol": "docs/TRACKB-NN-LOST-REVIEW.md",
        "n_cases": len(rows),
        "n_with_trace": sum(1 for r in rows if r["has_trace"]),
        "n_review": n_review,
        "n_pass": n_pass,
        "cases": rows,
        "claim_hygiene": (
            "Optional Track A gate when a layer trace exists. "
            "Does not change G1–G3. Does not rescore Dual_full Phase-7."
        ),
    }
    N2N_REVIEW_PATH.write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
    return out
