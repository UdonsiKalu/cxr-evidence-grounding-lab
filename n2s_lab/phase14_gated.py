"""Phase-14: conditional / gated activation steer (cos@1.00 vs τ)."""

from __future__ import annotations

import json
import math
from datetime import datetime, timezone
from typing import Any

import torch

from .ground import ground
from .hf_client import unload_model
from .hf_intervene import ActivationSteerSpec, generate_intervened
from .hf_trace import find_present_commit_step
from .neural import EXTRACT_SYSTEM
from .ollama_client import parse_json_object
from .paths import ARTIFACTS_DIR, HELDOUT_PHASE14_PATH
from .phase10_activation import FAIL_MODEL, STEER_FRACTIONS, build_steering_vectors
from .phase11_generalization import _load_phase10_gate
from .phase9a_behavioral import _analyze_hf, _extract_hf
from .phase9b_localization import _repair_user_prompt
from .predicate import evaluate_rule
from .types import Extraction
from .verify import verify_formalization

# Frozen from Ph13 mid-gap (docs/PHASE14-PROTOCOL.md).
TAU_COS_1_00 = -0.690
FROZEN_ALPHA = 4.0


def _cosine(a: list[float], b: torch.Tensor) -> float:
    ta = torch.tensor(a, dtype=torch.float32)
    tb = b.detach().float().cpu()
    if ta.numel() != tb.numel():
        raise ValueError("vector dim mismatch")
    denom = float(ta.norm().item() * tb.norm().item())
    if denom < 1e-12:
        return 0.0
    return float(torch.dot(ta, tb).item() / denom)


def _load_phase14_cases() -> list[dict[str, Any]]:
    data = json.loads(HELDOUT_PHASE14_PATH.read_text(encoding="utf-8"))
    return list(data["cases"])


def _cos_at_commit(
    steps: list[Any],
    *,
    vectors: dict[int, torch.Tensor],
    frac_for_idx: dict[int, str],
) -> tuple[float | None, float | None, float | None]:
    """Return (cos_0.75, cos_1.00, margin) at contradiction.present commit."""
    commit_step, _ = find_present_commit_step(steps)
    if commit_step is None:
        return None, None, None
    st = steps[commit_step]
    margin = None
    if st.logit_true is not None and st.logit_false is not None:
        margin = float(st.logit_true - st.logit_false)
    cos_by: dict[str, float] = {}
    for idx, unit in vectors.items():
        frac = frac_for_idx.get(idx)
        if frac is None:
            continue
        h = (st.layer_hidden or {}).get(frac)
        if h is None:
            for k, v in (st.layer_hidden or {}).items():
                if abs(float(k) - float(frac)) < 1e-6:
                    h = v
                    frac = k
                    break
        if h is None:
            continue
        key = frac if isinstance(frac, str) else f"{float(frac):.2f}"
        cos_by[key] = _cosine(h, unit)
    return cos_by.get("0.75"), cos_by.get("1.00"), margin


def _summarize_from_trace(
    *,
    case: dict[str, Any],
    model_id: str,
    raw: Extraction,
    v1_ok: bool,
    trace: Any,
    intervened_steps: list[int],
    intervention: str,
    alpha: float | None,
    vectors: dict[int, torch.Tensor],
    frac_for_idx: dict[int, str],
) -> dict[str, Any]:
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
    c075, c100, margin = _cos_at_commit(
        trace.steps, vectors=vectors, frac_for_idx=frac_for_idx
    )
    return {
        "case_id": case["id"],
        "gold": case["expected"],
        "model_id": model_id,
        "intervention": intervention,
        "alpha": alpha,
        "raw_x": bool(raw.contradiction_present),
        "verify_ok": v1_ok,
        "repair_final_x": final_x,
        "parse_ok": parse_ok,
        "commit_step": commit_step,
        "commit_value": commit_val,
        "logit_margin_true_minus_false": margin,
        "cosine_0_75": c075,
        "cosine_1_00": c100,
        "intervened_steps": intervened_steps,
        "verdict": rule.verdict.value if rule else None,
    }


def _gate_14(rows: list[dict[str, Any]]) -> dict[str, Any]:
    contras = [r for r in rows if r.get("role") == "control_contradiction"]
    n1 = next((r for r in rows if r.get("role") == "control_no_failure"), None)
    targets = [r for r in rows if r.get("role") == "generalization_target"]

    contra_all_off = bool(contras) and all(r.get("steer_on") is False for r in contras)
    n1_ok = True
    if n1 is not None:
        final_x = n1.get("final_x")
        n1_ok = final_x is False

    flips = [
        t
        for t in targets
        if t.get("baseline_x") is True
        and t.get("steer_on") is True
        and t.get("final_x") is False
    ]
    utility_ok = len(flips) >= 1
    yes = contra_all_off and n1_ok and utility_ok

    course_on = sum(1 for t in targets if t.get("steer_on") is True)
    course_off = sum(1 for t in targets if t.get("steer_on") is False)
    contra_off = sum(1 for c in contras if c.get("steer_on") is False)

    return {
        "pass": yes,
        "contra_all_steer_off": contra_all_off,
        "n1_ok": n1_ok,
        "utility_flips": len(flips),
        "utility_ok": utility_ok,
        "soft_gate_gold_agreement": {
            "course_steer_on": course_on,
            "course_steer_off": course_off,
            "contra_steer_off": contra_off,
            "contra_n": len(contras),
        },
        "note": (
            "14 YES — geometry gate withheld steer from contradictions; ≥1 course flip @ α=4"
            if yes
            else "14 NO — control entered STEER ON, N1 failed, or no course flip under gate"
        ),
        "claim_scope": (
            "Soft pilot gated steer — not adaptive α; τ fixed from Ph13 before held-out"
        ),
    }


def run_phase14(*, model_id: str = FAIL_MODEL) -> dict[str, Any]:
    _load_phase10_gate()
    steer_meta = build_steering_vectors(model_id=model_id)
    vectors = steer_meta.pop("_vectors_by_layer")
    unload_model()

    from .hf_client import load_model
    from .hf_trace import _fraction_layers

    model, _ = load_model(model_id)
    idxs = _fraction_layers(model, STEER_FRACTIONS)
    frac_for_idx = {
        idx: f"{f:.2f}" for idx, f in zip(idxs, STEER_FRACTIONS, strict=False)
    }
    del model
    unload_model()

    cases = _load_phase14_cases()
    rows: list[dict[str, Any]] = []
    panel: dict[str, Any] = {
        "phase": "14",
        "protocol": "docs/PHASE14-PROTOCOL.md",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "model_id": model_id,
        "frozen": {
            "tau_cos_1_00": TAU_COS_1_00,
            "alpha_when_on": FROZEN_ALPHA,
            "vector": "Ph10 BC_E2−BC_E1 @ 0.75/1.00",
            "rule": "STEER ON iff baseline cos@1.00 ≤ τ else OFF",
        },
        "heldout": HELDOUT_PHASE14_PATH.name,
        "steering_meta": {
            k: v for k, v in steer_meta.items() if not str(k).startswith("_")
        },
        "rows": rows,
    }
    out = ARTIFACTS_DIR / "phase14-gated-panel.json"
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

    def _save() -> None:
        panel["gate_gated"] = _gate_14(rows)
        out.write_text(json.dumps(panel, indent=2), encoding="utf-8")

    for case in cases:
        cid = case["id"]
        role = case.get("role")
        print(f"\n=== 14 · {cid} ({role}) · baseline + gate ===")
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
        unload_model()

        base_trace, base_iv = generate_intervened(
            system=EXTRACT_SYSTEM,
            user=user,
            model_id=model_id,
            max_new_tokens=400,
            intervention="none",
            layer_fractions=STEER_FRACTIONS,
        )
        baseline = _summarize_from_trace(
            case=case,
            model_id=model_id,
            raw=raw,
            v1_ok=v1.ok,
            trace=base_trace,
            intervened_steps=base_iv,
            intervention="none",
            alpha=None,
            vectors=vectors,
            frac_for_idx=frac_for_idx,
        )
        unload_model()

        cos100 = baseline.get("cosine_1_00")
        usable = cos100 is not None and math.isfinite(float(cos100))
        steer_on = bool(usable and float(cos100) <= TAU_COS_1_00)
        print(
            f"  baseline X={baseline.get('repair_final_x')} "
            f"cos@1.00={cos100} → STEER {'ON' if steer_on else 'OFF'}"
        )

        steered = None
        if steer_on:
            print(f"=== 14 · {cid} · steered alpha={FROZEN_ALPHA} ===")
            steered_trace, steered_iv = generate_intervened(
                system=EXTRACT_SYSTEM,
                user=user,
                model_id=model_id,
                max_new_tokens=400,
                intervention="activation_steer",
                layer_fractions=STEER_FRACTIONS,
                activation_steer=ActivationSteerSpec(
                    vectors_by_layer=vectors, alpha=FROZEN_ALPHA
                ),
            )
            steered = _summarize_from_trace(
                case=case,
                model_id=model_id,
                raw=raw,
                v1_ok=v1.ok,
                trace=steered_trace,
                intervened_steps=steered_iv,
                intervention="activation_steer",
                alpha=FROZEN_ALPHA,
                vectors=vectors,
                frac_for_idx=frac_for_idx,
            )
            unload_model()
            final_x = steered.get("repair_final_x")
            final_margin = steered.get("logit_margin_true_minus_false")
            final_verdict = steered.get("verdict")
        else:
            final_x = baseline.get("repair_final_x")
            final_margin = baseline.get("logit_margin_true_minus_false")
            final_verdict = baseline.get("verdict")

        row = {
            "case_id": cid,
            "role": role,
            "gold": case["expected"],
            "baseline_x": baseline.get("repair_final_x"),
            "baseline_margin": baseline.get("logit_margin_true_minus_false"),
            "baseline_verdict": baseline.get("verdict"),
            "cosine_0_75": baseline.get("cosine_0_75"),
            "cosine_1_00": cos100,
            "gate_usable": usable,
            "steer_on": steer_on,
            "steered_x": None if steered is None else steered.get("repair_final_x"),
            "steered_margin": (
                None if steered is None else steered.get("logit_margin_true_minus_false")
            ),
            "steered_verdict": None if steered is None else steered.get("verdict"),
            "final_x": final_x,
            "final_margin": final_margin,
            "final_verdict": final_verdict,
            "baseline": baseline,
            "steered": steered,
        }
        rows.append(row)
        _save()
        print(
            f"  final X={final_x} margin={final_margin} verdict={final_verdict} "
            f"steer_on={steer_on}"
        )

    gate = _gate_14(rows)
    panel["gate_gated"] = gate
    panel["note"] = (
        f"Ph14 gated steer τ={TAU_COS_1_00} α={FROZEN_ALPHA}; soft claim; not 12C."
    )
    out.write_text(json.dumps(panel, indent=2), encoding="utf-8")
    panel["_artifact"] = str(out)
    return panel
