"""Track B — α-sweep frozen Ph10 BC_E1 steering vector on false-X cluster (DEV).

Characterizes whether transfer failure at α=4 is magnitude vs qualitative.
Does not replace the vector.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any

from .hf_client import unload_model
from .paths import ARTIFACTS_DIR
from .phase10_activation import ALPHA_GRID, FAIL_MODEL, build_steering_vectors
from .trackb_falsex_cluster import (
    CONTRA_CONTROLS,
    NOFAIL_CONTROLS,
    TARGETS,
    _load_dev_case,
    _repair_run,
)

# Positive magnitudes only for transfer characterization (+ keep 0 as baseline)
SWEEP_ALPHAS = (0.0, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0)


def run_bce1_alpha_sweep(*, model_id: str = FAIL_MODEL) -> dict[str, Any]:
    steer_meta = build_steering_vectors(model_id=model_id)
    vectors = steer_meta.pop("_vectors_by_layer")

    # Baselines (α=0 ≡ no steer)
    baselines: dict[str, Any] = {}
    for cid in list(TARGETS) + list(CONTRA_CONTROLS) + list(NOFAIL_CONTROLS):
        print(f"\n=== BC_E1-α · baseline · {cid} ===")
        row = _repair_run(_load_dev_case(cid), model_id=model_id, intervention="none")
        unload_model()
        baselines[cid] = {
            "repair_final_x": row["repair_final_x"],
            "margin": row["logit_margin_true_minus_false"],
            "verdict": row["verdict"],
        }

    by_alpha: dict[str, Any] = {}
    for alpha in SWEEP_ALPHAS:
        if alpha == 0.0:
            by_alpha["0"] = {
                "alpha": 0.0,
                "results": {
                    cid: {
                        "baseline_x": baselines[cid]["repair_final_x"],
                        "steered_x": baselines[cid]["repair_final_x"],
                        "baseline_margin": baselines[cid]["margin"],
                        "steered_margin": baselines[cid]["margin"],
                        "verdict_steered": baselines[cid]["verdict"],
                        "x_flipped_true_to_false": False,
                    }
                    for cid in TARGETS
                },
                "controls": {
                    cid: {
                        "steered_x": baselines[cid]["repair_final_x"],
                        "steered_margin": baselines[cid]["margin"],
                    }
                    for cid in list(CONTRA_CONTROLS) + list(NOFAIL_CONTROLS)
                },
                "n_target_flips": 0,
                "contra_stay_true": True,
                "nofail_stay_false": True,
            }
            continue

        print(f"\n=== BC_E1-α · sweep α={alpha} ===")
        results = {}
        for cid in TARGETS:
            print(f"  · target {cid}")
            row = _repair_run(
                _load_dev_case(cid),
                model_id=model_id,
                intervention="activation_steer",
                vectors_by_layer=vectors,
                alpha=alpha,
            )
            unload_model()
            base = baselines[cid]
            results[cid] = {
                "baseline_x": base["repair_final_x"],
                "steered_x": row["repair_final_x"],
                "baseline_margin": base["margin"],
                "steered_margin": row["logit_margin_true_minus_false"],
                "verdict_steered": row["verdict"],
                "x_flipped_true_to_false": base["repair_final_x"] is True
                and row["repair_final_x"] is False,
                "margin_delta": (
                    None
                    if row["logit_margin_true_minus_false"] is None or base["margin"] is None
                    else row["logit_margin_true_minus_false"] - base["margin"]
                ),
            }

        controls = {}
        for cid in list(CONTRA_CONTROLS) + list(NOFAIL_CONTROLS):
            print(f"  · control {cid}")
            row = _repair_run(
                _load_dev_case(cid),
                model_id=model_id,
                intervention="activation_steer",
                vectors_by_layer=vectors,
                alpha=alpha,
            )
            unload_model()
            controls[cid] = {
                "baseline_x": baselines[cid]["repair_final_x"],
                "steered_x": row["repair_final_x"],
                "baseline_margin": baselines[cid]["margin"],
                "steered_margin": row["logit_margin_true_minus_false"],
                "verdict_steered": row["verdict"],
            }

        n_flip = sum(1 for r in results.values() if r["x_flipped_true_to_false"])
        contra_ok = all(controls[c]["steered_x"] is True for c in CONTRA_CONTROLS)
        # Prefer BC_E2 for nofail (TF_N1 was wrong_AUTO on Ollama Dual)
        nofail_ok = controls.get("BC_E2", {}).get("steered_x") is not True

        by_alpha[str(alpha)] = {
            "alpha": alpha,
            "results": results,
            "controls": controls,
            "n_target_flips": n_flip,
            "contra_stay_true": contra_ok,
            "nofail_BC_E2_stay_false": nofail_ok,
            "mean_target_margin_delta": (
                sum(
                    r["margin_delta"]
                    for r in results.values()
                    if r["margin_delta"] is not None
                )
                / max(
                    1,
                    sum(1 for r in results.values() if r["margin_delta"] is not None),
                )
            ),
        }

    # Characterization
    any_flip = any(by_alpha[str(a)]["n_target_flips"] > 0 for a in SWEEP_ALPHAS if a > 0)
    deltas = [
        by_alpha[str(a)]["mean_target_margin_delta"]
        for a in SWEEP_ALPHAS
        if a > 0 and "mean_target_margin_delta" in by_alpha[str(a)]
    ]
    monotonically_helpful = bool(deltas) and all(d < 0 for d in deltas)

    gate = {
        "pass": True,  # observational characterization always records
        "any_target_flip": any_flip,
        "margins_move_toward_false_at_all_alpha": monotonically_helpful,
        "transfer_failure_looks": (
            "magnitude_only_if_high_alpha_flips"
            if any_flip
            else (
                "qualitative_or_wrong_direction"
                if not any_flip and monotonically_helpful
                else "qualitative_no_flip"
            )
        ),
        "note": (
            "BC_E1 vector flips ≥1 target at some α — transfer failure was magnitude"
            if any_flip
            else (
                "BC_E1 vector lowers margins but never flips — shared signal, incomplete family direction"
                if monotonically_helpful
                else "BC_E1 vector does not reliably move or flip this cluster"
            )
        ),
    }

    panel = {
        "kind": "trackb_falsex_bce1_alpha_sweep",
        "protocol": "docs/TRACKB-FALSEX-CLUSTER.md",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "model_id": model_id,
        "case_source": "data/temporal-family-dev.json",
        "steering_vectors": steer_meta,
        "alpha_grid": list(SWEEP_ALPHAS),
        "phase10_alpha_grid_ref": list(ALPHA_GRID),
        "targets": list(TARGETS),
        "baselines": baselines,
        "by_alpha": by_alpha,
        "gate_bce1_alpha": gate,
        "note": (
            "Frozen Ph10 BC_E2−BC_E1 vector only — not a refit. "
            "TF_N1 reported but BC_E2 is the preferred nofail control integrity check."
        ),
    }

    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    out = ARTIFACTS_DIR / "trackb-falsex-bce1-alpha-panel.json"
    out.write_text(json.dumps(panel, indent=2) + "\n", encoding="utf-8")
    panel["_artifact"] = str(out)
    return panel
