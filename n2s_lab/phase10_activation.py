"""Phase-10: hidden-activation steering at repair commit (no logit forcing)."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any

import torch

from .ground import ground
from .hf_client import load_model, unload_model
from .hf_intervene import ActivationSteerSpec, generate_intervened
from .hf_trace import _fraction_layers, find_present_commit_step, generate_trace
from .neural import EXTRACT_SYSTEM
from .ollama_client import parse_json_object
from .paths import ARTIFACTS_DIR, HELDOUT_BMTCART_PHASE7_PATH
from .phase9a_behavioral import BC_E1_ID, _analyze_hf, _extract_hf
from .phase9b_localization import _repair_user_prompt
from .predicate import evaluate_rule
from .types import Extraction
from .verify import verify_formalization

FAIL_MODEL = "Qwen/Qwen2.5-7B-Instruct"
STEER_FRACTIONS = (0.75, 1.0)
ALPHA_GRID = (1.0, 2.0, 4.0, 8.0, 16.0, 32.0, -4.0, -8.0)
CONTROL_CASES = ("BC_C1", "BC_E2")


def _load_case(case_id: str) -> dict[str, Any]:
    data = json.loads(HELDOUT_BMTCART_PHASE7_PATH.read_text(encoding="utf-8"))
    for case in data["cases"]:
        if case["id"] == case_id:
            return case
    raise KeyError(case_id)


def _model_slug(model_id: str) -> str:
    return model_id.replace("/", "_").replace(":", "_")


def _commit_hidden_from_trace(trace_path: str) -> dict[str, list[float]]:
    data = json.loads((ARTIFACTS_DIR / trace_path).read_text(encoding="utf-8"))
    hidden = data.get("commit_layer_hidden")
    if not hidden:
        raise ValueError(f"no commit_layer_hidden in {trace_path}")
    return hidden


def _repair_trace_for_case(
    case: dict[str, Any], *, model_id: str
) -> tuple[Any, bool | None]:
    evidence = case["evidence"]
    case_id = case["id"]
    raw = _extract_hf(evidence, model_id=model_id)
    analysis = _analyze_hf(evidence, model_id=model_id)
    v1 = verify_formalization(
        analysis=analysis,
        extraction=raw,
        grounding=ground(raw).to_dict(),
        case_id=case_id,
        gold=case["expected"],
    )
    user = _repair_user_prompt(evidence, raw, v1.reasons)
    trace = generate_trace(
        system=EXTRACT_SYSTEM,
        user=user,
        model_id=model_id,
        max_new_tokens=400,
        layer_fractions=STEER_FRACTIONS,
    )
    gen_only = "".join(s.token_text for s in trace.steps)
    try:
        ex = Extraction.from_dict(parse_json_object(gen_only))
        final_x = bool(ex.contradiction_present)
    except ValueError:
        final_x = None
    return trace, final_x


def _ensure_e2_trace(*, model_id: str = FAIL_MODEL) -> dict[str, list[float]]:
    path = ARTIFACTS_DIR / f"phase10-bc-e2-{_model_slug(model_id)}-trace.json"
    if path.exists():
        return _commit_hidden_from_trace(path.name)

    print("\n=== 10A · collect BC_E2 repair commit hidden (7B) ===")
    case = _load_case("BC_E2")
    trace, final_x = _repair_trace_for_case(case, model_id=model_id)
    commit_step, commit_val = find_present_commit_step(trace.steps)
    if commit_step is None:
        raise RuntimeError("BC_E2 repair trace missing contradiction.present commit")

    payload = {
        "case_id": "BC_E2",
        "commit_step": commit_step,
        "commit_value": commit_val,
        "commit_layer_hidden": trace.steps[commit_step].layer_hidden,
        "repair_final_x": final_x,
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    unload_model()
    print(f"  BC_E2 repair X={final_x} commit={commit_val} step={commit_step}")
    return payload["commit_layer_hidden"]


def _unit_vec(a: list[float], b: list[float]) -> torch.Tensor:
    ta = torch.tensor(a, dtype=torch.float32)
    tb = torch.tensor(b, dtype=torch.float32)
    diff = tb - ta
    norm = diff.norm()
    if norm < 1e-8:
        return diff
    return diff / norm


def build_steering_vectors(*, model_id: str = FAIL_MODEL) -> dict[str, Any]:
    """10A: contrast BC_E1 (true commit) vs BC_E2 (false commit) at 0.75 / 1.00."""
    e1_path = ARTIFACTS_DIR / f"phase9b-bc-e1-{_model_slug(model_id)}-trace.json"
    if not e1_path.exists():
        raise FileNotFoundError(f"missing 9B trace: {e1_path}")

    h_true = _commit_hidden_from_trace(e1_path.name)
    h_false = _ensure_e2_trace(model_id=model_id)

    model, _ = load_model(model_id)
    layer_idxs = _fraction_layers(model, STEER_FRACTIONS)
    frac_by_idx = {
        idx: f"{f:.2f}" for idx, f in zip(layer_idxs, STEER_FRACTIONS, strict=False)
    }
    del model
    unload_model()

    vectors_by_layer: dict[int, torch.Tensor] = {}
    per_layer: dict[str, Any] = {}
    for idx in layer_idxs:
        frac = frac_by_idx[idx]
        if frac not in h_true or frac not in h_false:
            continue
        vec = _unit_vec(h_true[frac], h_false[frac])
        vectors_by_layer[idx] = vec
        per_layer[frac] = {
            "layer_idx": idx,
            "norm_delta_raw": float(
                torch.norm(
                    torch.tensor(h_false[frac]) - torch.tensor(h_true[frac])
                ).item()
            ),
            "direction": "BC_E2_false_commit_minus_BC_E1_true_commit",
        }

    return {
        "model_id": model_id,
        "source_cases": {"true_commit": BC_E1_ID, "false_commit": "BC_E2"},
        "layer_fractions": list(STEER_FRACTIONS),
        "vectors_by_layer_idx": list(vectors_by_layer.keys()),
        "per_layer": per_layer,
        "_vectors_by_layer": vectors_by_layer,
    }


def _summarize_repair_run(
    case: dict[str, Any],
    *,
    model_id: str,
    intervention: str = "none",
    vectors_by_layer: dict[int, torch.Tensor] | None = None,
    alpha: float = 1.0,
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

    if intervention == "activation_steer" and vectors_by_layer is not None:
        steer = ActivationSteerSpec(vectors_by_layer=vectors_by_layer, alpha=alpha)
        trace, intervened_steps = generate_intervened(
            system=EXTRACT_SYSTEM,
            user=user,
            model_id=model_id,
            max_new_tokens=400,
            intervention="activation_steer",
            layer_fractions=STEER_FRACTIONS,
            activation_steer=steer,
        )
    else:
        trace, intervened_steps = generate_intervened(
            system=EXTRACT_SYSTEM,
            user=user,
            model_id=model_id,
            max_new_tokens=400,
            intervention="none",
            layer_fractions=STEER_FRACTIONS,
        )

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
    if commit_step is not None:
        st = trace.steps[commit_step]
        if st.logit_true is not None and st.logit_false is not None:
            margin = st.logit_true - st.logit_false

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
    }


def _gate_10c(bc_e1_block: dict[str, Any], controls: dict[str, Any]) -> dict[str, Any]:
    baseline_x = bc_e1_block.get("baseline_x")
    intervened_x = bc_e1_block.get("intervened_x")
    margin = bc_e1_block.get("intervened_margin")
    baseline_margin = bc_e1_block.get("baseline_margin")

    x_flipped = baseline_x is True and intervened_x is False
    margin_crossed = (
        baseline_margin is not None
        and margin is not None
        and baseline_margin > 0
        and margin < 0
    )
    verdict_ok = bc_e1_block.get("verdict_intervened") in ("SATISFIED", "REVIEW")
    c1_ok = controls.get("BC_C1", {}).get("x_unchanged_or_correct") is True
    e2_ok = controls.get("BC_E2", {}).get("x_unchanged_or_correct") is True

    yes = baseline_x is True and (x_flipped or margin_crossed) and verdict_ok and c1_ok and e2_ok
    return {
        "pass": yes,
        "baseline_x_true": baseline_x is True,
        "x_flipped_to_false": x_flipped,
        "margin_crossed_below_zero": margin_crossed,
        "verdict_acceptable": verdict_ok,
        "control_BC_C1_ok": c1_ok,
        "control_BC_E2_ok": e2_ok,
        "note": (
            "10C YES — activation steer at repair commit (no logit forcing)"
            if yes
            else "10C NO — activation steer did not pass strict gate"
        ),
    }


def run_phase10(*, model_id: str = FAIL_MODEL) -> dict[str, Any]:
    steer_meta = build_steering_vectors(model_id=model_id)
    vectors = steer_meta.pop("_vectors_by_layer")

    bc_e1 = _load_case(BC_E1_ID)
    print("\n=== 10B · BC_E1 · baseline (no steer) ===")
    baseline = _summarize_repair_run(bc_e1, model_id=model_id, intervention="none")
    unload_model()

    best: dict[str, Any] | None = None
    sweep: list[dict[str, Any]] = []
    for alpha in ALPHA_GRID:
        print(f"\n=== 10B · BC_E1 · activation_steer alpha={alpha} ===")
        row = _summarize_repair_run(
            bc_e1,
            model_id=model_id,
            intervention="activation_steer",
            vectors_by_layer=vectors,
            alpha=alpha,
        )
        sweep.append(
            {
                "alpha": alpha,
                "repair_final_x": row["repair_final_x"],
                "commit_value": row["commit_value"],
                "margin": row["logit_margin_true_minus_false"],
                "verdict": row["verdict"],
            }
        )
        unload_model()
        if row["repair_final_x"] is False and best is None:
            best = row

    if best is None:
        candidates = [
            r
            for r in sweep
            if r.get("margin") is not None
            and baseline.get("logit_margin_true_minus_false") is not None
            and r["margin"] < baseline["logit_margin_true_minus_false"]
        ]
        if candidates:
            pick = min(candidates, key=lambda r: r["margin"])
            print(f"\n=== 10B · best-margin retry alpha={pick['alpha']} ===")
            best = _summarize_repair_run(
                bc_e1,
                model_id=model_id,
                intervention="activation_steer",
                vectors_by_layer=vectors,
                alpha=pick["alpha"],
            )
            unload_model()

    best_alpha = best["alpha"] if best else ALPHA_GRID[0]
    controls_out: dict[str, Any] = {}
    for cid in CONTROL_CASES:
        print(f"\n=== 10C · control {cid} · alpha={best_alpha} ===")
        case = _load_case(cid)
        base = _summarize_repair_run(case, model_id=model_id, intervention="none")
        unload_model()
        iv = _summarize_repair_run(
            case,
            model_id=model_id,
            intervention="activation_steer",
            vectors_by_layer=vectors,
            alpha=best_alpha,
        )
        unload_model()
        if cid == "BC_C1":
            ok = iv["repair_final_x"] is True
        else:
            ok = iv["repair_final_x"] is False
        controls_out[cid] = {
            "baseline_x": base["repair_final_x"],
            "intervened_x": iv["repair_final_x"],
            "baseline_verdict": base["verdict"],
            "intervened_verdict": iv["verdict"],
            "x_unchanged_or_correct": ok,
        }

    bc_e1_block = {
        "baseline_x": baseline.get("repair_final_x"),
        "baseline_margin": baseline.get("logit_margin_true_minus_false"),
        "intervened_x": best.get("repair_final_x") if best else None,
        "intervened_margin": best.get("logit_margin_true_minus_false") if best else None,
        "verdict_baseline": baseline.get("verdict"),
        "verdict_intervened": best.get("verdict") if best else None,
        "best_alpha": best_alpha,
        "baseline": baseline,
        "intervened": best,
        "sweep": sweep,
    }

    gate = _gate_10c(bc_e1_block, controls_out)

    panel = {
        "phase": "10",
        "protocol": "docs/PHASE10-PROTOCOL.md",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "model_id": model_id,
        "steering_vectors": steer_meta,
        "BC_E1": bc_e1_block,
        "controls": controls_out,
        "gate_activation_intervention": gate,
        "note": "Hidden activation steer at repair commit; no logit bias; soft claim n=1.",
    }

    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    out = ARTIFACTS_DIR / "phase10-bc-e1-panel.json"
    out.write_text(json.dumps(panel, indent=2), encoding="utf-8")
    panel["_artifact"] = str(out)
    return panel
