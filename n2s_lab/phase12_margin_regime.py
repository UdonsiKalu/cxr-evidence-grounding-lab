"""Phase-12A: margin regime — symmetric α panel on Ph11 false-X targets."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any

from .hf_client import unload_model
from .paths import ARTIFACTS_DIR, HELDOUT_PHASE11_PATH
from .phase10_activation import (
    FAIL_MODEL,
    build_steering_vectors,
    _summarize_repair_run,
)
from .phase11_generalization import _load_phase10_gate, _load_phase11_cases

ALPHA_PANEL_TARGETS = (2.0, 4.0, 6.0, 8.0)
ALPHA_CONTROLS = (4.0,)
TARGET_IDS = ("BC11_E2", "BC11_E3")
CONTROL_IDS = ("BC11_C1", "BC11_C2", "BC11_N1")


def _gate_12a(rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_id = {r["case_id"]: r for r in rows}

    def _control_ok(cid: str, want_x: bool) -> bool:
        row = by_id.get(cid)
        if not row:
            return False
        run = next((a for a in row["alpha_runs"] if a["alpha"] == 4.0), None)
        return run is not None and run.get("steered_x") is want_x

    controls_ok = (
        _control_ok("BC11_C1", True)
        and _control_ok("BC11_C2", True)
        and _control_ok("BC11_N1", False)
    )

    monotonic: dict[str, bool] = {}
    flip_at: dict[str, float | None] = {}
    for cid in TARGET_IDS:
        row = by_id.get(cid)
        if not row:
            monotonic[cid] = False
            flip_at[cid] = None
            continue
        runs = sorted(row["alpha_runs"], key=lambda x: x["alpha"])
        margins = [r["steered_margin"] for r in runs if r.get("steered_margin") is not None]
        monotonic[cid] = len(margins) < 2 or all(
            margins[i] >= margins[i + 1] for i in range(len(margins) - 1)
        )
        flip_alpha = None
        if row.get("baseline_x") is True:
            for r in runs:
                if r.get("steered_x") is False:
                    flip_alpha = r["alpha"]
                    break
        flip_at[cid] = flip_alpha

    mechanism_ok = controls_ok and all(monotonic.get(cid, False) for cid in TARGET_IDS)
    baseline_margins = {
        cid: by_id[cid].get("baseline_margin")
        for cid in TARGET_IDS
        if cid in by_id
    }

    return {
        "pass": mechanism_ok,
        "controls_ok": controls_ok,
        "margin_monotone_by_case": monotonic,
        "flip_first_alpha": flip_at,
        "baseline_margin_by_target": baseline_margins,
        "note": (
            "12A YES — controls held; margin monotone with α on E2/E3"
            if mechanism_ok
            else "12A NO — controls or margin monotonicity failed"
        ),
        "claim_scope": (
            "Mechanism pilot only — does not reopen Ph11 universal generalization gate"
        ),
    }


def run_phase12(*, model_id: str = FAIL_MODEL) -> dict[str, Any]:
    ph10 = _load_phase10_gate()
    steer_meta = build_steering_vectors(model_id=model_id)
    vectors = steer_meta.pop("_vectors_by_layer")

    all_cases = {c["id"]: c for c in _load_phase11_cases()}
    order: list[tuple[str, tuple[float, ...]]] = [
        *((cid, ALPHA_PANEL_TARGETS) for cid in TARGET_IDS),
        *((cid, ALPHA_CONTROLS) for cid in CONTROL_IDS),
    ]

    rows: list[dict[str, Any]] = []
    panel: dict[str, Any] = {
        "phase": "12A",
        "protocol": "docs/PHASE12-PROTOCOL.md",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "model_id": model_id,
        "frozen_intervention": {
            "vector_source": "phase10 BC_E2−BC_E1 commit @ 0.75/1.00",
            "alpha_panel_targets": list(ALPHA_PANEL_TARGETS),
            "alpha_controls": list(ALPHA_CONTROLS),
            "steering_vectors_meta": ph10["steering_source"],
            "per_layer": steer_meta.get("per_layer"),
        },
        "heldout": str(HELDOUT_PHASE11_PATH.name),
        "rows": rows,
    }
    out = ARTIFACTS_DIR / "phase12-margin-regime-panel.json"
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

    def _save() -> None:
        panel["gate_margin_regime"] = _gate_12a(rows)
        out.write_text(json.dumps(panel, indent=2), encoding="utf-8")

    row_by_id: dict[str, dict[str, Any]] = {}

    for cid, alphas in order:
        case = all_cases[cid]
        if cid not in row_by_id:
            print(f"\n=== 12A · {cid} · baseline ===")
            base = _summarize_repair_run(case, model_id=model_id, intervention="none")
            unload_model()
            row_by_id[cid] = {
                "case_id": cid,
                "role": case.get("role"),
                "gold": case["expected"],
                "baseline": base,
                "baseline_x": base.get("repair_final_x"),
                "baseline_margin": base.get("logit_margin_true_minus_false"),
                "baseline_verdict": base.get("verdict"),
                "alpha_runs": [],
            }
            if cid not in {r["case_id"] for r in rows}:
                rows.append(row_by_id[cid])

        row = row_by_id[cid]
        for alpha in alphas:
            if any(a["alpha"] == alpha for a in row["alpha_runs"]):
                continue
            print(f"\n=== 12A · {cid} · steered α={alpha} ===")
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
                f"verdict {row['baseline_verdict']}→{entry['steered_verdict']}"
            )

    gate = _gate_12a(rows)
    panel["gate_margin_regime"] = gate
    panel["note"] = (
        "Ph12A margin regime — symmetric α panel on E2/E3; controls @ α=4; soft claim."
    )
    out.write_text(json.dumps(panel, indent=2), encoding="utf-8")
    panel["_artifact"] = str(out)
    return panel
