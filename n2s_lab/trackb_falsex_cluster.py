"""Track B — false-X cluster on temporal-family-dev (probe → localize → patch → steer → ablate).

Uses DEV cases only. Do not load temporal-family-test evidence while designing.
"""

from __future__ import annotations

import json
import math
from datetime import datetime, timezone
from typing import Any

import torch

from .ground import ground
from .hf_client import load_model, unload_model
from .hf_intervene import ActivationPatchSpec, ActivationSteerSpec, generate_intervened
from .hf_trace import _fraction_layers, find_present_commit_step
from .neural import EXTRACT_SYSTEM
from .ollama_client import parse_json_object
from .paths import ARTIFACTS_DIR, TEMPORAL_FAMILY_DEV_PATH
from .phase10_activation import FAIL_MODEL, STEER_FRACTIONS, build_steering_vectors
from .phase9a_behavioral import _analyze_hf, _extract_hf
from .phase9b_localization import _repair_user_prompt
from .predicate import evaluate_rule
from .types import Extraction
from .verify import verify_formalization

# Qwen Dual_full false-contradiction cluster from temporal-dev baseline
TARGETS = ("TF_E3", "BC11_E3", "TF_E4")
CONTRA_CONTROLS = ("TF_C1", "BC_C1")
NOFAIL_CONTROLS = ("TF_N1", "BC_E2")
STEER_ALPHA = 4.0
ABLATE_ALPHA = -4.0
PATCH_DONOR = "BC_E2"


def _load_dev_case(case_id: str) -> dict[str, Any]:
    data = json.loads(TEMPORAL_FAMILY_DEV_PATH.read_text(encoding="utf-8"))
    for case in data["cases"]:
        if case["id"] == case_id:
            return case
    raise KeyError(f"{case_id} not in temporal-family-dev")


def _cosine(a: list[float], b: torch.Tensor) -> float:
    ta = torch.tensor(a, dtype=torch.float32)
    tb = b.detach().float().cpu()
    denom = float(ta.norm().item() * tb.norm().item())
    if denom < 1e-12:
        return 0.0
    return float(torch.dot(ta, tb).item() / denom)


def _norm(a: list[float]) -> float:
    return float(math.sqrt(sum(x * x for x in a)))


def _loo_probe_acc(rows: list[dict[str, Any]]) -> float | None:
    """Leave-one-out 1-NN on [cos_0.75, cos_1.00]; false_x=0, contradiction=1."""
    pts: list[tuple[list[float], int]] = []
    for r in rows:
        if r.get("class") not in ("false_x", "contradiction"):
            continue
        if not r.get("usable"):
            continue
        c075 = r.get("cosine_by_frac", {}).get("0.75")
        c100 = r.get("cosine_by_frac", {}).get("1.00")
        if c075 is None or c100 is None:
            continue
        label = 0 if r["class"] == "false_x" else 1
        pts.append(([float(c075), float(c100)], label))
    if len(pts) < 4:
        return None
    correct = 0
    for i, (xi, yi) in enumerate(pts):
        best_d = None
        pred = None
        for j, (xj, yj) in enumerate(pts):
            if i == j:
                continue
            d = (xi[0] - xj[0]) ** 2 + (xi[1] - xj[1]) ** 2
            if best_d is None or d < best_d:
                best_d = d
                pred = yj
        if pred == yi:
            correct += 1
    return correct / len(pts)


def _repair_run(
    case: dict[str, Any],
    *,
    model_id: str,
    intervention: str = "none",
    vectors_by_layer: dict[int, torch.Tensor] | None = None,
    alpha: float = 1.0,
    patch_by_layer: dict[int, torch.Tensor] | None = None,
    store_commit_hidden: bool = False,
) -> dict[str, Any]:
    evidence = case["evidence"]
    gold = case["expected"]
    case_id = case["id"]

    raw = _extract_hf(evidence, model_id=model_id)
    analysis = _analyze_hf(evidence, model_id=model_id)
    v1 = verify_formalization(
        analysis=analysis,
        extraction=raw,
        grounding=ground(raw).to_dict(),
        case_id=case_id,
        gold=gold,
    )
    user = _repair_user_prompt(evidence, raw, v1.reasons)

    kwargs: dict[str, Any] = {
        "system": EXTRACT_SYSTEM,
        "user": user,
        "model_id": model_id,
        "max_new_tokens": 400,
        "layer_fractions": STEER_FRACTIONS,
    }
    if intervention == "activation_steer" and vectors_by_layer is not None:
        kwargs["intervention"] = "activation_steer"
        kwargs["activation_steer"] = ActivationSteerSpec(
            vectors_by_layer=vectors_by_layer, alpha=alpha
        )
    elif intervention == "activation_patch" and patch_by_layer is not None:
        kwargs["intervention"] = "activation_patch"
        kwargs["activation_patch"] = ActivationPatchSpec(vectors_by_layer=patch_by_layer)
    else:
        kwargs["intervention"] = "none"

    trace, intervened_steps = generate_intervened(**kwargs)
    gen_only = "".join(s.token_text for s in trace.steps)
    try:
        ex = Extraction.from_dict(parse_json_object(gen_only))
        final_x = bool(ex.contradiction_present)
        parse_ok = True
    except ValueError:
        ex = None
        final_x = None
        parse_ok = False

    g = ground(ex) if ex else None
    rule = evaluate_rule(g) if g else None
    commit_step, commit_val = find_present_commit_step(trace.steps)

    margin = None
    commit_hidden: dict[str, list[float]] | None = None
    if commit_step is not None:
        st = trace.steps[commit_step]
        if st.logit_true is not None and st.logit_false is not None:
            margin = st.logit_true - st.logit_false
        if store_commit_hidden:
            commit_hidden = dict(st.layer_hidden)

    return {
        "case_id": case_id,
        "gold": gold,
        "model_id": model_id,
        "intervention": intervention,
        "alpha": alpha if intervention == "activation_steer" else None,
        "raw_x": bool(raw.contradiction_present),
        "verify_ok": v1.ok,
        "repair_final_x": final_x,
        "parse_ok": parse_ok,
        "commit_step": commit_step,
        "commit_value": commit_val,
        "logit_margin_true_minus_false": margin,
        "intervened_steps": intervened_steps,
        "verdict": rule.verdict.value if rule else None,
        "commit_layer_hidden": commit_hidden,
        "usable": commit_step is not None and commit_hidden is not None
        if store_commit_hidden
        else commit_step is not None,
    }


def run_patch_layer_subset(
    *,
    model_id: str = FAIL_MODEL,
    patch_fracs: tuple[float, ...] = (0.75,),
) -> dict[str, Any]:
    """Patch a SUBSET of layers only.

    Patching fraction 1.00 (final layer) sets the commit-token logits to the
    donor's by construction, so a full-depth patch cannot separate sufficiency
    from token forcing. Restricting to 0.75 leaves the remaining blocks free to
    recompute, which is the claim we actually want to test.
    """
    model, _ = load_model(model_id)
    layer_idxs = _fraction_layers(model, STEER_FRACTIONS)
    frac_by_idx = {
        idx: f"{f:.2f}" for idx, f in zip(layer_idxs, STEER_FRACTIONS, strict=False)
    }
    del model
    unload_model()
    keep = {f"{f:.2f}" for f in patch_fracs}

    print(f"\n=== TrackB-patchL · donor collect · {PATCH_DONOR} ===")
    donor_row = _repair_run(
        _load_dev_case(PATCH_DONOR),
        model_id=model_id,
        intervention="none",
        store_commit_hidden=True,
    )
    unload_model()
    donor_h = donor_row.get("commit_layer_hidden") or {}
    patch_by_layer: dict[int, torch.Tensor] = {
        idx: torch.tensor(donor_h[frac], dtype=torch.float32)
        for idx, frac in frac_by_idx.items()
        if frac in donor_h and frac in keep
    }
    if not patch_by_layer:
        raise RuntimeError(f"no donor hidden for fracs {sorted(keep)}")

    results: dict[str, Any] = {}
    for cid in TARGETS:
        print(f"\n=== TrackB-patchL · patch@{sorted(keep)} · {cid} ← {PATCH_DONOR} ===")
        base = _repair_run(_load_dev_case(cid), model_id=model_id, intervention="none")
        unload_model()
        row = _repair_run(
            _load_dev_case(cid),
            model_id=model_id,
            intervention="activation_patch",
            patch_by_layer=patch_by_layer,
        )
        unload_model()
        results[cid] = {
            "baseline_x": base["repair_final_x"],
            "patched_x": row["repair_final_x"],
            "baseline_margin": base["logit_margin_true_minus_false"],
            "patched_margin": row["logit_margin_true_minus_false"],
            "verdict_baseline": base["verdict"],
            "verdict_patched": row["verdict"],
            "x_flipped_true_to_false": base["repair_final_x"] is True
            and row["repair_final_x"] is False,
        }

    donor_margin = donor_row.get("logit_margin_true_minus_false")
    margins = [
        r["patched_margin"] for r in results.values() if r["patched_margin"] is not None
    ]
    n_flip = sum(1 for r in results.values() if r["x_flipped_true_to_false"])
    n_base_fx = sum(1 for r in results.values() if r["baseline_x"] is True)
    tautology = bool(margins) and all(m == donor_margin for m in margins)

    gate = {
        "pass": bool(n_base_fx >= 1 and n_flip >= 1 and not tautology),
        "patched_fracs": sorted(keep),
        "n_targets_baseline_false_x": n_base_fx,
        "n_patch_flips": n_flip,
        "donor_margin": donor_margin,
        "patched_margins": margins,
        "margins_equal_donor_tautology": tautology,
        "note": (
            "patch NO — margins identical to donor; final-layer overwrite, not sufficiency"
            if tautology
            else (
                "patch YES — mid-depth donor activation flips commit downstream (soft)"
                if n_flip >= 1
                else "patch NO — no target flipped at this depth"
            )
        ),
    }

    panel = {
        "kind": "trackb_falsex_patch_layer_subset",
        "protocol": "docs/TRACKB-FALSEX-CLUSTER.md",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "model_id": model_id,
        "case_source": "data/temporal-family-dev.json",
        "patch_donor": PATCH_DONOR,
        "targets": list(TARGETS),
        "results": results,
        "gate_patch_depth": gate,
        "note": (
            "Depth-restricted patch. Full-depth (incl. 1.00) patching is "
            "confounded with token forcing; this arm isolates mid-depth sufficiency."
        ),
    }

    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    frac_tag = "-".join(f"{f:.2f}".replace(".", "_") for f in sorted(patch_fracs))
    out = ARTIFACTS_DIR / f"trackb-falsex-patch-{frac_tag}-panel.json"
    out.write_text(json.dumps(panel, indent=2) + "\n", encoding="utf-8")
    panel["_artifact"] = str(out)
    return panel


def run_trackb_falsex(*, model_id: str = FAIL_MODEL) -> dict[str, Any]:
    """behavior → probe/localize → patch → steer → ablate on DEV false-X cluster."""
    steer_meta = build_steering_vectors(model_id=model_id)
    vectors = steer_meta.pop("_vectors_by_layer")
    model, _ = load_model(model_id)
    layer_idxs = _fraction_layers(model, STEER_FRACTIONS)
    frac_by_idx = {
        idx: f"{f:.2f}" for idx, f in zip(layer_idxs, STEER_FRACTIONS, strict=False)
    }
    del model
    unload_model()

    all_ids = list(TARGETS) + list(CONTRA_CONTROLS) + list(NOFAIL_CONTROLS)
    baselines: dict[str, Any] = {}
    probe_rows: list[dict[str, Any]] = []

    # --- A/B: behavior + localize/probe features ---
    for cid in all_ids:
        case = _load_dev_case(cid)
        print(f"\n=== TrackB · baseline · {cid} ===")
        row = _repair_run(
            case,
            model_id=model_id,
            intervention="none",
            store_commit_hidden=True,
        )
        unload_model()
        cosine_by_frac: dict[str, float] = {}
        norm_by_frac: dict[str, float] = {}
        hidden = row.get("commit_layer_hidden") or {}
        for idx, frac in frac_by_idx.items():
            if frac not in hidden or idx not in vectors:
                continue
            cosine_by_frac[frac] = _cosine(hidden[frac], vectors[idx])
            norm_by_frac[frac] = _norm(hidden[frac])
        if cid in TARGETS:
            cls = "false_x"
        elif cid in CONTRA_CONTROLS:
            cls = "contradiction"
        else:
            cls = "nofail"
        probe_row = {
            "case_id": cid,
            "class": cls,
            "usable": bool(row.get("usable") and cosine_by_frac),
            "baseline_x": row["repair_final_x"],
            "margin": row["logit_margin_true_minus_false"],
            "verdict": row["verdict"],
            "cosine_by_frac": cosine_by_frac,
            "norm_by_frac": norm_by_frac,
        }
        probe_rows.append(probe_row)
        baselines[cid] = {**row, "cosine_by_frac": cosine_by_frac, "class": cls}
        # drop bulky hidden from panel
        baselines[cid].pop("commit_layer_hidden", None)

    loo = _loo_probe_acc(probe_rows)
    fx = [r for r in probe_rows if r["class"] == "false_x" and r["usable"]]
    cx = [r for r in probe_rows if r["class"] == "contradiction" and r["usable"]]

    def _cos100(rs: list[dict[str, Any]]) -> list[float]:
        return [
            float(r["cosine_by_frac"]["1.00"])
            for r in rs
            if r.get("cosine_by_frac", {}).get("1.00") is not None
        ]

    fx_c = _cos100(fx)
    cx_c = _cos100(cx)
    overlap = True
    gap = None
    if fx_c and cx_c:
        flo, fhi = min(fx_c), max(fx_c)
        clo, chi = min(cx_c), max(cx_c)
        overlap = not (fhi < clo or chi < flo)
        if fhi < clo:
            gap = clo - fhi
        elif chi < flo:
            gap = flo - chi
        else:
            gap = 0.0

    probe_pass = bool(
        len(fx) >= 2
        and len(cx) >= 2
        and ((not overlap) or (loo is not None and loo >= 0.75))
    )
    gate_probe = {
        "pass": probe_pass,
        "n_false_x_usable": len(fx),
        "n_contradiction_usable": len(cx),
        "cosine_1_00_false_x": fx_c,
        "cosine_1_00_contradiction": cx_c,
        "cosine_1_00_ranges_overlap": overlap,
        "cosine_1_00_gap": gap,
        "loo_probe_acc": loo,
        "note": (
            "probe YES — false-X vs contradiction separable at commit (soft)"
            if probe_pass
            else "probe NO — insufficient separability / usable commits"
        ),
    }

    # --- C: activation patch (donor BC_E2 false-commit hiddens) into false-X targets ---
    # Re-collect donor commit tensors (baselines drop bulky hidden state).
    print(f"\n=== TrackB · patch donor collect · {PATCH_DONOR} ===")
    donor_case = _load_dev_case(PATCH_DONOR)
    donor_row = _repair_run(
        donor_case,
        model_id=model_id,
        intervention="none",
        store_commit_hidden=True,
    )
    unload_model()
    donor_h = donor_row.get("commit_layer_hidden") or {}
    patch_by_layer: dict[int, torch.Tensor] = {}
    for idx, frac in frac_by_idx.items():
        if frac in donor_h:
            patch_by_layer[idx] = torch.tensor(donor_h[frac], dtype=torch.float32)

    patch_results: dict[str, Any] = {}
    for cid in TARGETS:
        if not patch_by_layer:
            patch_results[cid] = {"error": "no donor hidden"}
            continue
        case = _load_dev_case(cid)
        print(f"\n=== TrackB · activation_patch · {cid} ← {PATCH_DONOR} ===")
        row = _repair_run(
            case,
            model_id=model_id,
            intervention="activation_patch",
            patch_by_layer=patch_by_layer,
        )
        unload_model()
        base_x = baselines[cid]["repair_final_x"]
        patch_results[cid] = {
            "baseline_x": base_x,
            "patched_x": row["repair_final_x"],
            "baseline_margin": baselines[cid]["logit_margin_true_minus_false"],
            "patched_margin": row["logit_margin_true_minus_false"],
            "verdict_patched": row["verdict"],
            "x_flipped_true_to_false": base_x is True and row["repair_final_x"] is False,
        }

    # --- D: steer transfer (Ph10 α=4) ---
    steer_results: dict[str, Any] = {}
    for cid in list(TARGETS) + list(CONTRA_CONTROLS) + list(NOFAIL_CONTROLS):
        case = _load_dev_case(cid)
        print(f"\n=== TrackB · steer α={STEER_ALPHA} · {cid} ===")
        row = _repair_run(
            case,
            model_id=model_id,
            intervention="activation_steer",
            vectors_by_layer=vectors,
            alpha=STEER_ALPHA,
        )
        unload_model()
        base_x = baselines[cid]["repair_final_x"]
        steer_results[cid] = {
            "class": baselines[cid]["class"],
            "baseline_x": base_x,
            "steered_x": row["repair_final_x"],
            "baseline_margin": baselines[cid]["logit_margin_true_minus_false"],
            "steered_margin": row["logit_margin_true_minus_false"],
            "verdict_steered": row["verdict"],
            "x_flipped_true_to_false": base_x is True and row["repair_final_x"] is False,
        }

    # --- E: ablate (α=-4) on targets that baseline false-X — expect X to stay/return ---
    ablate_results: dict[str, Any] = {}
    for cid in TARGETS:
        case = _load_dev_case(cid)
        print(f"\n=== TrackB · ablate α={ABLATE_ALPHA} · {cid} ===")
        row = _repair_run(
            case,
            model_id=model_id,
            intervention="activation_steer",
            vectors_by_layer=vectors,
            alpha=ABLATE_ALPHA,
        )
        unload_model()
        ablate_results[cid] = {
            "baseline_x": baselines[cid]["repair_final_x"],
            "ablated_x": row["repair_final_x"],
            "baseline_margin": baselines[cid]["logit_margin_true_minus_false"],
            "ablated_margin": row["logit_margin_true_minus_false"],
            "verdict_ablated": row["verdict"],
            "note": "negative α — if direction causal, may worsen margin toward true",
        }

    # Control integrity under positive steer
    contra_ok = all(
        steer_results[c].get("steered_x") is True for c in CONTRA_CONTROLS if c in steer_results
    )
    nofail_ok = all(
        steer_results[c].get("steered_x") is not True
        for c in NOFAIL_CONTROLS
        if c in steer_results
    )
    n_flip = sum(
        1
        for c in TARGETS
        if steer_results.get(c, {}).get("x_flipped_true_to_false")
    )
    n_baseline_fx = sum(1 for c in TARGETS if baselines[c]["repair_final_x"] is True)
    n_patch_flip = sum(
        1 for c in TARGETS if patch_results.get(c, {}).get("x_flipped_true_to_false")
    )

    gate_causal = {
        "pass": bool(n_baseline_fx >= 1 and (n_flip >= 1 or n_patch_flip >= 1)),
        "n_targets_baseline_false_x": n_baseline_fx,
        "n_steer_flips": n_flip,
        "n_patch_flips": n_patch_flip,
        "contra_controls_stay_true_under_steer": contra_ok,
        "nofail_controls_not_true_under_steer": nofail_ok,
        "note": (
            "causal YES — at least one false-X commit flipped by patch or steer (soft)"
            if (n_flip >= 1 or n_patch_flip >= 1)
            else "causal NO — no target flipped; check HF reproduction of false-X"
        ),
    }

    panel = {
        "kind": "trackb_falsex_cluster",
        "protocol": "docs/TRACKB-FALSEX-CLUSTER.md",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "model_id": model_id,
        "case_source": "data/temporal-family-dev.json",
        "test_set_not_used": "data/temporal-family-test.json",
        "targets": list(TARGETS),
        "contra_controls": list(CONTRA_CONTROLS),
        "nofail_controls": list(NOFAIL_CONTROLS),
        "steering_vectors": steer_meta,
        "steer_alpha": STEER_ALPHA,
        "ablate_alpha": ABLATE_ALPHA,
        "patch_donor": PATCH_DONOR,
        "baselines": baselines,
        "probe_rows": probe_rows,
        "gate_probe": gate_probe,
        "patch": patch_results,
        "steer": steer_results,
        "ablate": ablate_results,
        "gate_causal": gate_causal,
        "note": (
            "Track B ladder on DEV false-X cluster only. "
            "G3 unchanged. temporal-family-test sealed — do not peek."
        ),
    }

    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    out = ARTIFACTS_DIR / "trackb-falsex-cluster-panel.json"
    out.write_text(json.dumps(panel, indent=2) + "\n", encoding="utf-8")
    panel["_artifact"] = str(out)
    return panel
