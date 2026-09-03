"""Track B — family-level L20 steering vector (DEV only).

Semantic axis (ChatGPT order step 3):
  v = mean(h_clean temporal-change) − mean(h_true same-state contradiction)
Fit at L20 (earliest sufficient patch site). Do NOT include the three false-X
failures in the contrast. Do NOT use TF_N1 as a clean negative.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any

import torch

from .hf_client import unload_model
from .paths import ARTIFACTS_DIR
from .phase10_activation import FAIL_MODEL
from .trackb_falsex_cluster import (
    CONTRA_CONTROLS,
    TARGETS,
    _load_dev_case,
    _repair_run,
)

FAMILY_LAYER = 20
# Gold SATISFIED temporal-change; kept only if HF commit X=false (clean class A)
CLASS_A_CANDIDATES = ("TF_E1", "TF_E2", "TF_E5")
CLASS_B = CONTRA_CONTROLS  # TF_C1, BC_C1
NOFAIL_CONTROL = "BC_E2"  # integrity only — not in the contrast
EVAL_TARGETS = TARGETS
SWEEP_ALPHAS = (0.0, 1.0, 2.0, 4.0, 8.0, 16.0, 32.0)
EXCLUDED_FROM_CONTRAST = ("TF_N1", "TF_E3", "BC11_E3", "TF_E4")


def _unit(vec: torch.Tensor) -> torch.Tensor:
    n = float(vec.norm().item())
    if n < 1e-8:
        return vec
    return vec / n


def _hidden_l20(row: dict[str, Any]) -> torch.Tensor | None:
    h = row.get("commit_layer_hidden") or {}
    if "L20" not in h:
        return None
    return torch.tensor(h["L20"], dtype=torch.float32)


def _collect(cid: str, *, model_id: str) -> dict[str, Any]:
    print(f"\n=== family-L20 · collect · {cid} ===")
    row = _repair_run(
        _load_dev_case(cid),
        model_id=model_id,
        intervention="none",
        store_commit_hidden=True,
        layer_indices=(FAMILY_LAYER,),
    )
    unload_model()
    vec = _hidden_l20(row)
    return {
        "case_id": cid,
        "repair_final_x": row["repair_final_x"],
        "margin": row["logit_margin_true_minus_false"],
        "verdict": row["verdict"],
        "usable_hidden": vec is not None,
        "hidden_L20": vec.tolist() if vec is not None else None,
    }


def run_family_vector(*, model_id: str = FAIL_MODEL) -> dict[str, Any]:
    collect_ids = list(
        dict.fromkeys(
            [*CLASS_A_CANDIDATES, *CLASS_B, *EVAL_TARGETS, NOFAIL_CONTROL]
        )
    )
    collected: dict[str, Any] = {}
    for cid in collect_ids:
        collected[cid] = _collect(cid, model_id=model_id)

    class_a: list[str] = []
    for cid in CLASS_A_CANDIDATES:
        row = collected[cid]
        if row["repair_final_x"] is False and row["usable_hidden"]:
            class_a.append(cid)
        else:
            print(
                f"  skip class A {cid}: X={row['repair_final_x']} "
                f"usable={row['usable_hidden']}"
            )

    class_b: list[str] = []
    for cid in CLASS_B:
        row = collected[cid]
        if row["repair_final_x"] is True and row["usable_hidden"]:
            class_b.append(cid)
        else:
            print(
                f"  skip class B {cid}: X={row['repair_final_x']} "
                f"usable={row['usable_hidden']}"
            )

    construction = {
        "layer_idx": FAMILY_LAYER,
        "class_a_definition": (
            "gold SATISFIED temporal-change AND HF commit X=false "
            f"(candidates {list(CLASS_A_CANDIDATES)})"
        ),
        "class_b_definition": (
            "gold CONTRADICTION AND HF commit X=true "
            f"({list(CLASS_B)})"
        ),
        "class_a_kept": class_a,
        "class_b_kept": class_b,
        "excluded_from_contrast": list(EXCLUDED_FROM_CONTRAST),
        "formula": "unit(mean(A) − mean(B)) at L20",
        "n_class_a": len(class_a),
        "n_class_b": len(class_b),
    }

    can_fit = len(class_a) >= 2 and len(class_b) >= 2
    vectors_by_layer: dict[int, torch.Tensor] = {}
    if can_fit:
        mean_a = torch.stack(
            [torch.tensor(collected[c]["hidden_L20"], dtype=torch.float32) for c in class_a]
        ).mean(dim=0)
        mean_b = torch.stack(
            [torch.tensor(collected[c]["hidden_L20"], dtype=torch.float32) for c in class_b]
        ).mean(dim=0)
        vec = _unit(mean_a - mean_b)
        vectors_by_layer[FAMILY_LAYER] = vec
        construction["vector_norm_raw"] = float((mean_a - mean_b).norm().item())
        construction["mean_a_norm"] = float(mean_a.norm().item())
        construction["mean_b_norm"] = float(mean_b.norm().item())
    else:
        construction["error"] = "need ≥2 clean class A and ≥2 class B to fit"

    # Drop bulky hiddens from panel
    baselines = {
        cid: {
            "repair_final_x": collected[cid]["repair_final_x"],
            "margin": collected[cid]["margin"],
            "verdict": collected[cid]["verdict"],
            "in_class_a": cid in class_a,
            "in_class_b": cid in class_b,
        }
        for cid in collect_ids
    }

    by_alpha: dict[str, Any] = {}
    if can_fit:
        eval_ids = list(EVAL_TARGETS) + list(CLASS_B) + [NOFAIL_CONTROL]
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
                        for cid in EVAL_TARGETS
                    },
                    "controls": {
                        cid: {
                            "steered_x": baselines[cid]["repair_final_x"],
                            "steered_margin": baselines[cid]["margin"],
                        }
                        for cid in list(CLASS_B) + [NOFAIL_CONTROL]
                    },
                    "n_target_flips": 0,
                    "contra_stay_true": True,
                    "nofail_stay_false": True,
                }
                continue

            print(f"\n=== family-L20 · sweep α={alpha} ===")
            results = {}
            for cid in EVAL_TARGETS:
                print(f"  · target {cid}")
                row = _repair_run(
                    _load_dev_case(cid),
                    model_id=model_id,
                    intervention="activation_steer",
                    vectors_by_layer=vectors_by_layer,
                    alpha=alpha,
                    layer_indices=(FAMILY_LAYER,),
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
                        if row["logit_margin_true_minus_false"] is None
                        or base["margin"] is None
                        else row["logit_margin_true_minus_false"] - base["margin"]
                    ),
                }

            controls = {}
            for cid in list(CLASS_B) + [NOFAIL_CONTROL]:
                print(f"  · control {cid}")
                row = _repair_run(
                    _load_dev_case(cid),
                    model_id=model_id,
                    intervention="activation_steer",
                    vectors_by_layer=vectors_by_layer,
                    alpha=alpha,
                    layer_indices=(FAMILY_LAYER,),
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
            contra_ok = all(controls[c]["steered_x"] is True for c in CLASS_B)
            nofail_ok = controls[NOFAIL_CONTROL]["steered_x"] is not True
            deltas = [
                r["margin_delta"]
                for r in results.values()
                if r["margin_delta"] is not None
            ]
            by_alpha[str(alpha)] = {
                "alpha": alpha,
                "results": results,
                "controls": controls,
                "n_target_flips": n_flip,
                "contra_stay_true": contra_ok,
                "nofail_BC_E2_stay_false": nofail_ok,
                "mean_target_margin_delta": (
                    sum(deltas) / len(deltas) if deltas else None
                ),
                "safe_family_edit": n_flip >= 1 and contra_ok and nofail_ok,
            }

    best_safe = None
    for a in SWEEP_ALPHAS:
        if a == 0.0:
            continue
        block = by_alpha.get(str(a))
        if block and block.get("safe_family_edit"):
            best_safe = a

    any_flip = any(
        by_alpha.get(str(a), {}).get("n_target_flips", 0) > 0
        for a in SWEEP_ALPHAS
        if a > 0
    )
    gate = {
        "pass": can_fit,
        "fitted": can_fit,
        "any_target_flip": any_flip,
        "best_safe_alpha": best_safe,
        "note": (
            f"family L20 fitted A={class_a} B={class_b}; "
            + (
                f"best safe α={best_safe}"
                if best_safe is not None
                else (
                    "flips exist but none preserve contra+nofail"
                    if any_flip
                    else "no target flips on α grid"
                )
            )
            if can_fit
            else construction.get("error", "fit failed")
        ),
    }

    panel = {
        "kind": "trackb_falsex_family_vector_l20",
        "protocol": "docs/TRACKB-FALSEX-CLUSTER.md",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "model_id": model_id,
        "case_source": "data/temporal-family-dev.json",
        "test_set_not_used": "data/temporal-family-test.json",
        "construction": construction,
        "baselines": baselines,
        "alpha_grid": list(SWEEP_ALPHAS),
        "by_alpha": by_alpha,
        "gate_family_vector": gate,
        "note": (
            "Semantic family direction at L20. Failures TF_E3/BC11_E3/TF_E4 "
            "are eval-only, not in the contrast. TF_N1 excluded."
        ),
    }

    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    out = ARTIFACTS_DIR / "trackb-falsex-family-l20-panel.json"
    out.write_text(json.dumps(panel, indent=2) + "\n", encoding="utf-8")
    panel["_artifact"] = str(out)
    return panel
