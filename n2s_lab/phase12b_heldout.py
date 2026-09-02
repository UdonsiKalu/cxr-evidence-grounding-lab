"""Phase-12B: second held-out — pre-declared LOW/HIGH margin bands @ α={4,8}."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any

from .hf_client import unload_model
from .paths import ARTIFACTS_DIR, HELDOUT_PHASE12B_PATH
from .phase10_activation import (
    FAIL_MODEL,
    build_steering_vectors,
    _summarize_repair_run,
)
from .phase11_generalization import _load_phase10_gate

ALPHA_PANEL_TARGETS = (4.0, 8.0)
ALPHA_CONTROLS = (4.0,)
BAND_LOW_MAX = 1.5
BAND_HIGH_MIN = 2.0


def _load_phase12b_cases() -> list[dict[str, Any]]:
    data = json.loads(HELDOUT_PHASE12B_PATH.read_text(encoding="utf-8"))
    return list(data["cases"])


def _band(baseline_margin: float | None) -> str | None:
    if baseline_margin is None:
        return None
    m = float(baseline_margin)
    if m <= BAND_LOW_MAX:
        return "LOW"
    if m >= BAND_HIGH_MIN:
        return "HIGH"
    return "MID"


def _flip_at(row: dict[str, Any]) -> float | None:
    if row.get("baseline_x") is not True:
        return None
    for r in sorted(row.get("alpha_runs") or [], key=lambda x: x["alpha"]):
        if r.get("steered_x") is False:
            return float(r["alpha"])
    return None


def _band_hit(row: dict[str, Any]) -> str | None:
    """Return 'hit' | 'miss' | None (not scored)."""
    if row.get("role") != "generalization_target":
        return None
    if row.get("baseline_x") is not True:
        return None
    band = row.get("band")
    if band not in ("LOW", "HIGH"):
        return None
    flip = _flip_at(row)
    if band == "LOW":
        return "hit" if flip is not None and flip <= 4.0 else "miss"
    # HIGH: no flip at 4; flip by 8
    run4 = next((a for a in row["alpha_runs"] if a["alpha"] == 4.0), None)
    flipped_at_4 = run4 is not None and run4.get("steered_x") is False
    flip8 = flip is not None and flip <= 8.0
    if not flipped_at_4 and flip8:
        return "hit"
    return "miss"


def _gate_12b(rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_id = {r["case_id"]: r for r in rows}

    def _ctrl(cid: str, want: bool) -> bool:
        row = by_id.get(cid)
        if not row:
            return False
        run = next((a for a in row["alpha_runs"] if a["alpha"] == 4.0), None)
        return run is not None and run.get("steered_x") is want

    controls_ok = _ctrl("BC12_C1", True) and _ctrl("BC12_N1", False)

    targets = [r for r in rows if r.get("role") == "generalization_target"]
    direction_ok = True
    for t in targets:
        if t.get("baseline_x") is not True:
            continue
        base_m = t.get("baseline_margin")
        run8 = next((a for a in t["alpha_runs"] if a["alpha"] == 8.0), None)
        if base_m is None or run8 is None or run8.get("steered_margin") is None:
            direction_ok = False
            continue
        if float(run8["steered_margin"]) > float(base_m) + 1e-6:
            direction_ok = False

    band_results = {t["case_id"]: _band_hit(t) for t in targets}
    scored = [v for v in band_results.values() if v in ("hit", "miss")]
    hits = sum(1 for v in scored if v == "hit")
    misses = sum(1 for v in scored if v == "miss")
    band_ok = len(scored) > 0 and hits >= 1 and misses == 0

    yes = controls_ok and direction_ok and band_ok
    return {
        "pass": yes,
        "controls_ok": controls_ok,
        "direction_ok": direction_ok,
        "band_results": band_results,
        "band_hits": hits,
        "band_misses": misses,
        "band_scored_n": len(scored),
        "note": (
            "12B YES — controls held; direction OK; band predictions hit with no misses"
            if yes
            else "12B NO — controls, direction, or band rule failed / nothing scored"
        ),
        "claim_scope": (
            "Second held-out pilot only — does not prove adaptive α or reopen Ph11"
        ),
    }


def run_phase12b(*, model_id: str = FAIL_MODEL) -> dict[str, Any]:
    ph10 = _load_phase10_gate()
    steer_meta = build_steering_vectors(model_id=model_id)
    vectors = steer_meta.pop("_vectors_by_layer")

    cases = _load_phase12b_cases()
    rows: list[dict[str, Any]] = []
    panel: dict[str, Any] = {
        "phase": "12B",
        "protocol": "docs/PHASE12B-PROTOCOL.md",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "model_id": model_id,
        "frozen_intervention": {
            "vector_source": "phase10 BC_E2−BC_E1 commit @ 0.75/1.00",
            "alpha_panel_targets": list(ALPHA_PANEL_TARGETS),
            "alpha_controls": list(ALPHA_CONTROLS),
            "bands": {"LOW_max": BAND_LOW_MAX, "HIGH_min": BAND_HIGH_MIN},
            "steering_vectors_meta": ph10["steering_source"],
            "per_layer": steer_meta.get("per_layer"),
        },
        "heldout": str(HELDOUT_PHASE12B_PATH.name),
        "rows": rows,
    }
    out = ARTIFACTS_DIR / "phase12b-heldout-panel.json"
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

    def _save() -> None:
        panel["gate_heldout"] = _gate_12b(rows)
        out.write_text(json.dumps(panel, indent=2), encoding="utf-8")

    for case in cases:
        cid = case["id"]
        role = case.get("role")
        alphas = (
            ALPHA_PANEL_TARGETS
            if role == "generalization_target"
            else ALPHA_CONTROLS
        )

        print(f"\n=== 12B · {cid} · baseline ===")
        base = _summarize_repair_run(case, model_id=model_id, intervention="none")
        unload_model()
        base_m = base.get("logit_margin_true_minus_false")
        row: dict[str, Any] = {
            "case_id": cid,
            "role": role,
            "gold": case["expected"],
            "baseline": base,
            "baseline_x": base.get("repair_final_x"),
            "baseline_margin": base_m,
            "baseline_verdict": base.get("verdict"),
            "band": _band(base_m) if role == "generalization_target" else None,
            "alpha_runs": [],
        }
        rows.append(row)
        _save()

        for alpha in alphas:
            print(f"\n=== 12B · {cid} · steered α={alpha} ===")
            steered = _summarize_repair_run(
                case,
                model_id=model_id,
                intervention="activation_steer",
                vectors_by_layer=vectors,
                alpha=alpha,
            )
            unload_model()
            entry = {
                "alpha": alpha,
                "steered": steered,
                "steered_x": steered.get("repair_final_x"),
                "steered_margin": steered.get("logit_margin_true_minus_false"),
                "steered_verdict": steered.get("verdict"),
            }
            row["alpha_runs"].append(entry)
            _save()
            print(
                f"  {cid} α={alpha}: X {row['baseline_x']}→{entry['steered_x']} "
                f"margin {row['baseline_margin']}→{entry['steered_margin']} "
                f"band={row.get('band')}"
            )

    gate = _gate_12b(rows)
    panel["gate_heldout"] = gate
    panel["note"] = (
        "Ph12B second held-out — α={4,8}; LOW/HIGH bands from Ph12A; soft claim."
    )
    out.write_text(json.dumps(panel, indent=2), encoding="utf-8")
    panel["_artifact"] = str(out)
    return panel
