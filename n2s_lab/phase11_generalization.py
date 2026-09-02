"""Phase-11: fixed Ph10 steering vector generalization on held-out cases."""

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

FROZEN_ALPHA = 4.0


def _load_phase11_cases() -> list[dict[str, Any]]:
    data = json.loads(HELDOUT_PHASE11_PATH.read_text(encoding="utf-8"))
    return list(data["cases"])


def _load_phase10_gate() -> dict[str, Any]:
    path = ARTIFACTS_DIR / "phase10-bc-e1-panel.json"
    if not path.exists():
        raise FileNotFoundError(f"missing {path}")
    panel = json.loads(path.read_text(encoding="utf-8"))
    gate = panel.get("gate_activation_intervention", {})
    if not gate.get("pass"):
        raise RuntimeError("Phase-10 gate not YES — run phase10 first")
    alpha = panel.get("BC_E1", {}).get("best_alpha", FROZEN_ALPHA)
    return {"alpha": float(alpha), "steering_source": panel.get("steering_vectors")}


def _gate_11(rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_id = {r["case_id"]: r for r in rows}
    con_ok = all(
        by_id[cid]["steered"]["repair_final_x"] is True
        for cid in ("BC11_C1", "BC11_C2")
        if cid in by_id
    )
    n_ok = by_id.get("BC11_N1", {}).get("steered", {}).get("repair_final_x") is False

    targets = [r for r in rows if r.get("role") == "generalization_target"]
    baseline_fps = [t for t in targets if t["baseline"]["repair_final_x"] is True]
    fixed = [
        t
        for t in baseline_fps
        if t["steered"]["repair_final_x"] is False
        and t["steered"].get("verdict") != "CONTRADICTION"
    ]

    gen_ok = len(baseline_fps) > 0 and len(fixed) == len(baseline_fps)
    yes = con_ok and n_ok and gen_ok

    return {
        "pass": yes,
        "control_contradiction_ok": con_ok,
        "control_no_failure_ok": n_ok,
        "targets_n": len(targets),
        "targets_baseline_false_x_count": len(baseline_fps),
        "targets_fixed_count": len(fixed),
        "generalization_ok": gen_ok,
        "note": (
            "11C YES — fixed vector generalizes on held-out false-X targets; controls held"
            if yes
            else "11C NO — generalization or controls failed"
        ),
    }


def run_phase11(*, model_id: str = FAIL_MODEL) -> dict[str, Any]:
    ph10 = _load_phase10_gate()
    alpha = ph10["alpha"]

    steer_meta = build_steering_vectors(model_id=model_id)
    vectors = steer_meta.pop("_vectors_by_layer")

    cases = _load_phase11_cases()
    rows: list[dict[str, Any]] = []

    panel: dict[str, Any] = {
        "phase": "11",
        "protocol": "docs/PHASE11-PROTOCOL.md",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "model_id": model_id,
        "frozen_intervention": {
            "alpha": alpha,
            "vector_source": "phase10 BC_E2−BC_E1 commit @ 0.75/1.00",
            "steering_vectors_meta": ph10["steering_source"],
            "per_layer": steer_meta.get("per_layer"),
        },
        "heldout": str(HELDOUT_PHASE11_PATH.name),
        "rows": rows,
    }
    out = ARTIFACTS_DIR / "phase11-generalization-panel.json"
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

    def _save() -> None:
        panel["gate_generalization"] = _gate_11(rows)
        out.write_text(json.dumps(panel, indent=2), encoding="utf-8")

    for case in cases:
        cid = case["id"]
        print(f"\n=== 11 · {cid} · baseline ===")
        base = _summarize_repair_run(case, model_id=model_id, intervention="none")
        unload_model()
        print(f"\n=== 11 · {cid} · steered alpha={alpha} ===")
        steered = _summarize_repair_run(
            case,
            model_id=model_id,
            intervention="activation_steer",
            vectors_by_layer=vectors,
            alpha=alpha,
        )
        unload_model()
        row = {
            "case_id": cid,
            "role": case.get("role"),
            "gold": case["expected"],
            "baseline": base,
            "steered": steered,
            "baseline_x": base.get("repair_final_x"),
            "steered_x": steered.get("repair_final_x"),
            "baseline_margin": base.get("logit_margin_true_minus_false"),
            "steered_margin": steered.get("logit_margin_true_minus_false"),
            "baseline_verdict": base.get("verdict"),
            "steered_verdict": steered.get("verdict"),
        }
        rows.append(row)
        panel["rows"] = rows
        _save()
        print(
            f"  {cid}: X {row['baseline_x']}→{row['steered_x']} "
            f"margin {row['baseline_margin']}→{row['steered_margin']} "
            f"verdict {row['baseline_verdict']}→{row['steered_verdict']}"
        )

    gate = _gate_11(rows)
    panel["gate_generalization"] = gate
    panel["note"] = (
        f"Fixed Ph10 vector @ alpha={alpha}; n={len(cases)} held-out; soft claim."
    )
    out.write_text(json.dumps(panel, indent=2), encoding="utf-8")
    panel["_artifact"] = str(out)
    return panel
