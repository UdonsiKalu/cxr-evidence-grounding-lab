"""Track B — patch-depth sweep + specificity controls (DEV only).

Uses absolute layer indices (Qwen2.5-7B: 28 blocks). Never patches final layer alone
as the primary claim (that sets commit logits by construction).
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any

import torch

from .hf_client import load_model, unload_model
from .hf_trace import _layer_count
from .paths import ARTIFACTS_DIR
from .phase10_activation import FAIL_MODEL
from .trackb_falsex_cluster import (
    CONTRA_CONTROLS,
    TARGETS,
    _load_dev_case,
    _repair_run,
)

# Coarse sweep on 28-layer Qwen2.5-7B (0-indexed). Exclude final layer (27).
DEFAULT_SWEEP_LAYERS = (4, 8, 12, 16, 20, 24)
PATCH_DONOR = "BC_E2"  # clean no-failure on HF Track B; not TF_N1
REVERSE_SOURCE = "TF_E3"  # one false-X commit state for reverse patch
GAUSS_SEED = 20260903


def _hidden_at(commit_hidden: dict[str, list[float]], layer_idx: int) -> torch.Tensor:
    key = f"L{layer_idx}"
    if key not in commit_hidden:
        raise KeyError(f"missing {key} in commit_hidden keys={list(commit_hidden)}")
    return torch.tensor(commit_hidden[key], dtype=torch.float32)


def _collect(
    case_id: str,
    *,
    model_id: str,
    layer_indices: tuple[int, ...],
) -> dict[str, Any]:
    print(f"\n=== patch-depth · collect · {case_id} @ {layer_indices} ===")
    row = _repair_run(
        _load_dev_case(case_id),
        model_id=model_id,
        intervention="none",
        store_commit_hidden=True,
        layer_indices=layer_indices,
    )
    unload_model()
    return row


def _patch_one(
    recipient_id: str,
    *,
    model_id: str,
    layer_idx: int,
    donor_vec: torch.Tensor,
    donor_id: str,
    donor_kind: str,
    baseline: dict[str, Any] | None = None,
) -> dict[str, Any]:
    print(
        f"\n=== patch-depth · L{layer_idx} · {recipient_id} ← {donor_id} ({donor_kind}) ==="
    )
    if baseline is None:
        base = _repair_run(
            _load_dev_case(recipient_id),
            model_id=model_id,
            intervention="none",
            layer_indices=(layer_idx,),
        )
        unload_model()
        base_x = base["repair_final_x"]
        base_margin = base["logit_margin_true_minus_false"]
        base_verdict = base["verdict"]
    else:
        base_x = baseline["repair_final_x"]
        base_margin = baseline["logit_margin_true_minus_false"]
        base_verdict = baseline["verdict"]
    row = _repair_run(
        _load_dev_case(recipient_id),
        model_id=model_id,
        intervention="activation_patch",
        patch_by_layer={layer_idx: donor_vec},
        layer_indices=(layer_idx,),
    )
    unload_model()
    return {
        "recipient": recipient_id,
        "donor": donor_id,
        "donor_kind": donor_kind,
        "layer_idx": layer_idx,
        "baseline_x": base_x,
        "patched_x": row["repair_final_x"],
        "baseline_margin": base_margin,
        "patched_margin": row["logit_margin_true_minus_false"],
        "verdict_baseline": base_verdict,
        "verdict_patched": row["verdict"],
        "x_flipped_true_to_false": base_x is True and row["repair_final_x"] is False,
        "x_flipped_false_to_true": base_x is False and row["repair_final_x"] is True,
        "x_unchanged": base_x == row["repair_final_x"],
    }


def run_patch_depth_sweep(
    *,
    model_id: str = FAIL_MODEL,
    sweep_layers: tuple[int, ...] = DEFAULT_SWEEP_LAYERS,
) -> dict[str, Any]:
    model, _ = load_model(model_id)
    n_layers = _layer_count(model)
    del model
    unload_model()

    layers = tuple(i for i in sweep_layers if 0 <= i < n_layers - 1)  # never final alone
    if not layers:
        raise RuntimeError("no valid sweep layers")

    collect_ids = list(
        dict.fromkeys(
            [PATCH_DONOR, REVERSE_SOURCE, *TARGETS, *CONTRA_CONTROLS]
        )
    )
    collected: dict[str, Any] = {}
    for cid in collect_ids:
        collected[cid] = _collect(cid, model_id=model_id, layer_indices=layers)

    donor_margin = collected[PATCH_DONOR]["logit_margin_true_minus_false"]

    # --- primary depth sweep: BC_E2 → false-X targets ---
    depth_rows: list[dict[str, Any]] = []
    by_layer: dict[str, Any] = {}
    for L in layers:
        donor_vec = _hidden_at(collected[PATCH_DONOR]["commit_layer_hidden"], L)
        layer_results = []
        for tid in TARGETS:
            r = _patch_one(
                tid,
                model_id=model_id,
                layer_idx=L,
                donor_vec=donor_vec,
                donor_id=PATCH_DONOR,
                donor_kind="clean_nofail_donor",
                baseline=collected[tid],
            )
            layer_results.append(r)
            depth_rows.append(r)
        margins = [r["patched_margin"] for r in layer_results if r["patched_margin"] is not None]
        n_flip = sum(1 for r in layer_results if r["x_flipped_true_to_false"])
        n_base = sum(1 for r in layer_results if r["baseline_x"] is True)
        tautology = bool(margins) and all(
            abs(m - donor_margin) < 1e-3 for m in margins if donor_margin is not None
        )
        by_layer[f"L{L}"] = {
            "layer_idx": L,
            "n_baseline_false_x": n_base,
            "n_flips": n_flip,
            "margins_equal_donor_tautology": tautology,
            "patched_margins": margins,
            "donor_margin": donor_margin,
            "results": layer_results,
        }

    # Earliest layer with ≥2/3 flips and not tautology
    earliest = None
    for L in layers:
        block = by_layer[f"L{L}"]
        if block["n_flips"] >= 2 and not block["margins_equal_donor_tautology"]:
            earliest = L
            break
    if earliest is None:
        for L in layers:
            block = by_layer[f"L{L}"]
            if block["n_flips"] >= 1 and not block["margins_equal_donor_tautology"]:
                earliest = L
                break

    control_layers = []
    if earliest is not None:
        control_layers.append(earliest)
    if 20 in layers and 20 not in control_layers:
        control_layers.append(20)

    # --- specificity controls at earliest + L20 ---
    controls: dict[str, Any] = {}
    for L in control_layers:
        donor_vec = _hidden_at(collected[PATCH_DONOR]["commit_layer_hidden"], L)
        fx_vec = _hidden_at(collected[REVERSE_SOURCE]["commit_layer_hidden"], L)
        contra_vec = _hidden_at(collected["TF_C1"]["commit_layer_hidden"], L)

        # Unrelated: Gaussian matched-norm
        g = torch.Generator()
        g.manual_seed(GAUSS_SEED + L)
        noise = torch.randn(donor_vec.shape, generator=g, dtype=torch.float32)
        noise = noise * (donor_vec.norm() / (noise.norm() + 1e-8))
        gauss_rows = [
            _patch_one(
                tid,
                model_id=model_id,
                layer_idx=L,
                donor_vec=noise,
                donor_id="gaussian_matched_norm",
                donor_kind="unrelated_gaussian",
                baseline=collected[tid],
            )
            for tid in TARGETS
        ]

        # Unrelated semantic: true-contradiction hidden into false-X
        contra_donor_rows = [
            _patch_one(
                tid,
                model_id=model_id,
                layer_idx=L,
                donor_vec=contra_vec,
                donor_id="TF_C1",
                donor_kind="unrelated_true_contradiction_donor",
                baseline=collected[tid],
            )
            for tid in TARGETS
        ]

        # Reverse: false-X state into already-correct no-failure
        reverse_row = _patch_one(
            PATCH_DONOR,
            model_id=model_id,
            layer_idx=L,
            donor_vec=fx_vec,
            donor_id=REVERSE_SOURCE,
            donor_kind="reverse_falsex_into_nofail",
            baseline=collected[PATCH_DONOR],
        )

        # Already-correct recipients: clean donor into true contradictions (must stay true)
        contra_recipients = [
            _patch_one(
                cid,
                model_id=model_id,
                layer_idx=L,
                donor_vec=donor_vec,
                donor_id=PATCH_DONOR,
                donor_kind="donor_into_true_contradiction",
                baseline=collected[cid],
            )
            for cid in CONTRA_CONTROLS
        ]

        # Already-correct: donor into itself
        self_row = _patch_one(
            PATCH_DONOR,
            model_id=model_id,
            layer_idx=L,
            donor_vec=donor_vec,
            donor_id=PATCH_DONOR,
            donor_kind="donor_into_self",
            baseline=collected[PATCH_DONOR],
        )

        controls[f"L{L}"] = {
            "layer_idx": L,
            "gaussian_into_falsex": gauss_rows,
            "gaussian_n_flips": sum(1 for r in gauss_rows if r["x_flipped_true_to_false"]),
            "contradiction_donor_into_falsex": contra_donor_rows,
            "contradiction_donor_n_flips": sum(
                1 for r in contra_donor_rows if r["x_flipped_true_to_false"]
            ),
            "reverse_falsex_into_nofail": reverse_row,
            "reverse_induced_false_x": reverse_row["x_flipped_false_to_true"],
            "donor_into_contradiction": contra_recipients,
            "contradiction_stayed_true": all(
                r["patched_x"] is True for r in contra_recipients
            ),
            "donor_into_self": self_row,
            "self_stayed_false": self_row["patched_x"] is False,
        }

    # Gates
    any_depth = any(
        by_layer[f"L{L}"]["n_flips"] >= 2
        and not by_layer[f"L{L}"]["margins_equal_donor_tautology"]
        for L in layers
    )
    ctrl_ok = False
    if earliest is not None and f"L{earliest}" in controls:
        c = controls[f"L{earliest}"]
        # Specificity soft: primary rescue > gaussian; contradictions not destroyed
        ctrl_ok = (
            c["gaussian_n_flips"] < by_layer[f"L{earliest}"]["n_flips"]
            and c["contradiction_stayed_true"]
            and c["self_stayed_false"]
        )

    gate = {
        "pass": bool(any_depth),
        "earliest_sufficient_layer": earliest,
        "n_layers_model": n_layers,
        "specificity_soft_pass": ctrl_ok,
        "note": (
            f"depth YES — earliest sufficient L{earliest}; specificity "
            f"{'soft-pass' if ctrl_ok else 'soft-fail/incomplete'}"
            if earliest is not None
            else "depth NO — no non-tautological rescue at swept layers"
        ),
    }

    panel = {
        "kind": "trackb_falsex_patch_depth_sweep",
        "protocol": "docs/TRACKB-FALSEX-CLUSTER.md",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "model_id": model_id,
        "case_source": "data/temporal-family-dev.json",
        "test_set_not_used": "data/temporal-family-test.json",
        "n_layers": n_layers,
        "sweep_layers": list(layers),
        "patch_donor": PATCH_DONOR,
        "targets": list(TARGETS),
        "baselines": {
            cid: {
                "repair_final_x": collected[cid]["repair_final_x"],
                "margin": collected[cid]["logit_margin_true_minus_false"],
                "verdict": collected[cid]["verdict"],
            }
            for cid in collect_ids
        },
        "by_layer": by_layer,
        "controls": controls,
        "gate_patch_depth_sweep": gate,
        "note": (
            "Absolute-layer patch sweep + specificity controls. "
            "TF_N1 not used as donor (Qwen Dual wrong_AUTO). "
            "Final layer excluded from primary sweep."
        ),
    }

    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    out = ARTIFACTS_DIR / "trackb-falsex-patch-depth-panel.json"
    out.write_text(json.dumps(panel, indent=2) + "\n", encoding="utf-8")
    panel["_artifact"] = str(out)
    return panel
