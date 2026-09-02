"""Phase-13: commit-hidden geometry — course false-X vs true contradiction."""

from __future__ import annotations

import json
import math
from datetime import datetime, timezone
from typing import Any

import torch

from .ground import ground
from .hf_client import unload_model
from .hf_trace import find_present_commit_step, generate_trace
from .neural import EXTRACT_SYSTEM
from .paths import (
    ARTIFACTS_DIR,
    HELDOUT_BMTCART_PHASE7_PATH,
    HELDOUT_PHASE11_PATH,
    HELDOUT_PHASE12B_PATH,
)
from .phase10_activation import FAIL_MODEL, STEER_FRACTIONS, build_steering_vectors
from .phase11_generalization import _load_phase10_gate
from .phase9a_behavioral import _analyze_hf, _extract_hf
from .phase9b_localization import _repair_user_prompt
from .verify import verify_formalization

COURSE_IDS = ("BC11_E2", "BC11_E3", "BC12_E1", "BC12_E2", "BC12_E3")
CONTRA_IDS = ("BC11_C1", "BC11_C2", "BC12_C1", "BC_C1")
LIGHT_FRACTIONS = (0.75, 1.0)


def _load_case(case_id: str) -> dict[str, Any]:
    for path in (
        HELDOUT_PHASE11_PATH,
        HELDOUT_PHASE12B_PATH,
        HELDOUT_BMTCART_PHASE7_PATH,
    ):
        if not path.exists():
            continue
        data = json.loads(path.read_text(encoding="utf-8"))
        for c in data.get("cases", []):
            if c["id"] == case_id:
                return c
    raise ValueError(f"unknown case_id: {case_id}")


def _cosine(a: list[float], b: torch.Tensor) -> float:
    ta = torch.tensor(a, dtype=torch.float32)
    tb = b.detach().float().cpu()
    if ta.numel() != tb.numel():
        raise ValueError("vector dim mismatch")
    denom = float(ta.norm().item() * tb.norm().item())
    if denom < 1e-12:
        return 0.0
    return float(torch.dot(ta, tb).item() / denom)


def _norm(a: list[float]) -> float:
    return float(math.sqrt(sum(x * x for x in a)))


def _loo_probe_acc(rows: list[dict[str, Any]]) -> float | None:
    """Leave-one-out 1-NN on [cos_0.75, cos_1.00]; labels course=0 contra=1."""
    pts: list[tuple[list[float], int, str]] = []
    for r in rows:
        if not r.get("usable"):
            continue
        c075 = r.get("cosine_by_frac", {}).get("0.75")
        c100 = r.get("cosine_by_frac", {}).get("1.00")
        if c075 is None or c100 is None:
            continue
        label = 0 if r["class"] == "course_false_x" else 1
        pts.append(([float(c075), float(c100)], label, r["case_id"]))
    if len(pts) < 6:
        return None
    correct = 0
    for i, (xi, yi, _) in enumerate(pts):
        best_d = None
        pred = None
        for j, (xj, yj, _) in enumerate(pts):
            if i == j:
                continue
            d = (xi[0] - xj[0]) ** 2 + (xi[1] - xj[1]) ** 2
            if best_d is None or d < best_d:
                best_d = d
                pred = yj
        if pred == yi:
            correct += 1
    return correct / len(pts)


def _gate_13(rows: list[dict[str, Any]]) -> dict[str, Any]:
    course = [r for r in rows if r.get("class") == "course_false_x" and r.get("usable")]
    contra = [r for r in rows if r.get("class") == "true_contradiction" and r.get("usable")]
    n_ok = len(course) >= 3 and len(contra) >= 3

    def _cos100(rs: list[dict[str, Any]]) -> list[float]:
        out = []
        for r in rs:
            v = r.get("cosine_by_frac", {}).get("1.00")
            if v is not None:
                out.append(float(v))
        return out

    c_vals = _cos100(course)
    k_vals = _cos100(contra)
    overlap = True
    gap = None
    if c_vals and k_vals:
        c_lo, c_hi = min(c_vals), max(c_vals)
        k_lo, k_hi = min(k_vals), max(k_vals)
        overlap = not (c_hi < k_lo or k_hi < c_lo)
        if c_hi < k_lo:
            gap = k_lo - c_hi
        elif k_hi < c_lo:
            gap = c_lo - k_hi
        else:
            gap = 0.0

    probe_acc = _loo_probe_acc(rows)
    sep = (not overlap and bool(c_vals) and bool(k_vals)) or (
        probe_acc is not None and probe_acc >= 0.75
    )
    yes = bool(n_ok and sep)
    return {
        "pass": yes,
        "n_course_usable": len(course),
        "n_contradiction_usable": len(contra),
        "n_ok": n_ok,
        "cosine_1_00_course": c_vals,
        "cosine_1_00_contradiction": k_vals,
        "cosine_1_00_ranges_overlap": overlap,
        "cosine_1_00_gap": gap,
        "loo_probe_acc": probe_acc,
        "separability_heuristic": sep,
        "note": (
            "13 YES — enough commits; soft separability on cosine@1.00 or LOO probe"
            if yes
            else "13 NO — insufficient commits or no soft separability"
        ),
        "claim_scope": (
            "Observational geometry pilot — not adaptive α; not 'found contradiction rep'"
        ),
    }


def run_phase13(*, model_id: str = FAIL_MODEL) -> dict[str, Any]:
    _load_phase10_gate()
    steer_meta = build_steering_vectors(model_id=model_id)
    vectors = steer_meta.pop("_vectors_by_layer")
    unload_model()

    # Map layer idx → fraction key used in traces
    from .hf_client import load_model
    from .hf_trace import _fraction_layers

    model, _ = load_model(model_id)
    idxs = _fraction_layers(model, STEER_FRACTIONS)
    frac_for_idx = {
        idx: f"{f:.2f}" for idx, f in zip(idxs, STEER_FRACTIONS, strict=False)
    }
    del model
    unload_model()

    schedule: list[tuple[str, str]] = [
        *((cid, "course_false_x") for cid in COURSE_IDS),
        *((cid, "true_contradiction") for cid in CONTRA_IDS),
    ]

    rows: list[dict[str, Any]] = []
    panel: dict[str, Any] = {
        "phase": "13",
        "protocol": "docs/PHASE13-PROTOCOL.md",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "model_id": model_id,
        "frozen": {
            "layers": list(LIGHT_FRACTIONS),
            "vector": "Ph10 BC_E2−BC_E1 unit direction (cosine only; no steer)",
            "intervention": "none",
        },
        "steering_meta": {
            k: v for k, v in steer_meta.items() if not str(k).startswith("_")
        },
        "rows": rows,
    }
    out = ARTIFACTS_DIR / "phase13-geometry-panel.json"
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

    def _save() -> None:
        panel["gate_geometry"] = _gate_13(rows)
        out.write_text(json.dumps(panel, indent=2), encoding="utf-8")

    for cid, cls in schedule:
        case = _load_case(cid)
        print(f"\n=== 13 · {cid} ({cls}) · repair trace ===")
        evidence = case["evidence"]
        raw = _extract_hf(evidence, model_id=model_id)
        analysis = _analyze_hf(evidence, model_id=model_id)
        v1 = verify_formalization(
            analysis=analysis,
            extraction=raw,
            grounding=ground(raw).to_dict(),
            case_id=cid,
            gold=case["expected"],
        )
        user = _repair_user_prompt(evidence, raw, v1.reasons)
        # Fresh load for long repair trace (avoid residual KV / fragmentation).
        unload_model()
        trace = generate_trace(
            system=EXTRACT_SYSTEM,
            user=user,
            model_id=model_id,
            max_new_tokens=400,
            layer_fractions=LIGHT_FRACTIONS,
            stop_on_contradiction_commit=True,
        )
        unload_model()

        commit_step, commit_val = find_present_commit_step(trace.steps)
        row: dict[str, Any] = {
            "case_id": cid,
            "class": cls,
            "gold": case["expected"],
            "raw_x": bool(raw.contradiction_present),
            "verify_ok": v1.ok,
            "commit_step": commit_step,
            "commit_value": commit_val,
            "usable": False,
            "logit_margin_true_minus_false": None,
            "layer_norm": {},
            "cosine_by_frac": {},
        }
        if commit_step is not None:
            st = trace.steps[commit_step]
            if st.logit_true is not None and st.logit_false is not None:
                row["logit_margin_true_minus_false"] = st.logit_true - st.logit_false
            for frac, vec in (st.layer_hidden or {}).items():
                row["layer_norm"][frac] = _norm(vec)
            for idx, unit in vectors.items():
                frac = frac_for_idx.get(idx)
                if frac is None:
                    continue
                # traces key fractions as "0.75" / "1.00" from zip with STEER
                h = st.layer_hidden.get(frac) or st.layer_hidden.get(f"{float(frac):.2f}")
                if h is None:
                    # try raw keys from generate_trace
                    for k, v in (st.layer_hidden or {}).items():
                        if abs(float(k) - float(frac)) < 1e-6:
                            h = v
                            frac = k
                            break
                if h is None:
                    continue
                row["cosine_by_frac"][frac if isinstance(frac, str) else f"{float(frac):.2f}"] = (
                    _cosine(h, unit)
                )
            row["usable"] = bool(row["cosine_by_frac"])
        rows.append(row)
        _save()
        print(
            f"  usable={row['usable']} commit={commit_val} "
            f"cos={row['cosine_by_frac']} margin={row['logit_margin_true_minus_false']}"
        )

    gate = _gate_13(rows)
    panel["gate_geometry"] = gate
    panel["note"] = (
        "Ph13 geometry — cosine to Ph10 direction @ commit; no steer; soft claim."
    )
    out.write_text(json.dumps(panel, indent=2), encoding="utf-8")
    panel["_artifact"] = str(out)
    return panel
