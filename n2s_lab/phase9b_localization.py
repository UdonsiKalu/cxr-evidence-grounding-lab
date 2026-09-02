"""Phase-9B: layer contrast + logit attribution on BC_E1 repair path."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any

import torch

from .ground import ground
from .hf_client import unload_model
from .hf_trace import (
    GenerationTrace,
    find_present_commit_step,
    find_step_indices,
    generate_trace,
)
from .neural import EXTRACT_SCHEMA, EXTRACT_SYSTEM, REPAIR_EXTRACT_SUFFIX
from .ollama_client import parse_json_object
from .paths import ARTIFACTS_DIR
from .phase9a_behavioral import (
    PHASE9A_DEFAULT_MODELS,
    _analyze_hf,
    _extract_hf,
    load_bc_e1,
)
from .types import Extraction
from .verify import verify_formalization


def _repair_user_prompt(
    evidence: str, prior: Extraction, verify_reasons: list[str]
) -> str:
    reasons = "; ".join(verify_reasons) or "structure/analysis mismatch"
    return (
        f"Schema:\n{EXTRACT_SCHEMA}\n\nNote:\n{evidence}\n\n"
        f"Previous JSON:\n{prior.to_dict()}\n\nVerify reasons: {reasons}\n"
        f"{REPAIR_EXTRACT_SUFFIX}\n\nReturn a single JSON object only."
    )


def _cosine(a: list[float], b: list[float]) -> float:
    """Unused cross-dim helper; kept for possible same-model comparisons."""
    ta = torch.tensor(a, dtype=torch.float32)
    tb = torch.tensor(b, dtype=torch.float32)
    if ta.shape != tb.shape or ta.norm() == 0 or tb.norm() == 0:
        return 0.0
    return float(torch.nn.functional.cosine_similarity(ta, tb, dim=0).item())


def _hidden_scalars(vec: list[float]) -> dict[str, float]:
    t = torch.tensor(vec, dtype=torch.float32)
    return {
        "norm": float(t.norm().item()),
        "mean": float(t.mean().item()),
        "std": float(t.std(unbiased=False).item()),
    }


def _layer_divergence(
    trace_a: GenerationTrace,
    trace_b: GenerationTrace,
    step_a: int,
    step_b: int,
) -> dict[str, Any]:
    """Compare scalar stats at matched fractional depths (hidden dim may differ)."""
    sa = trace_a.steps[step_a].layer_hidden
    sb = trace_b.steps[step_b].layer_hidden
    common = sorted(set(sa.keys()) & set(sb.keys()))
    per_layer = {}
    norm_deltas: list[float] = []
    for frac in common:
        pa = _hidden_scalars(sa[frac])
        pb = _hidden_scalars(sb[frac])
        nd = abs(pa["norm"] - pb["norm"])
        norm_deltas.append(nd)
        per_layer[frac] = {
            "fail_norm": pa["norm"],
            "safe_norm": pb["norm"],
            "norm_delta": nd,
            "fail_mean": pa["mean"],
            "safe_mean": pb["mean"],
        }
    return {
        "step_a": step_a,
        "step_b": step_b,
        "per_layer": per_layer,
        "mean_norm_delta": sum(norm_deltas) / len(norm_deltas) if norm_deltas else None,
        "max_norm_delta": max(norm_deltas) if norm_deltas else None,
    }


def _summarize_trace(label: str, trace: GenerationTrace) -> dict[str, Any]:
    gen_text = "".join(s.token_text for s in trace.steps)
    commit_step, commit_val = find_present_commit_step(trace.steps)

    logit_true = logit_false = None
    if commit_step is not None:
        st = trace.steps[commit_step]
        logit_true = st.logit_true
        logit_false = st.logit_false

    layer_at_commit: dict[str, float] = {}
    if commit_step is not None:
        for frac, vec in trace.steps[commit_step].layer_hidden.items():
            layer_at_commit[frac] = float(torch.norm(torch.tensor(vec)).item())

    return {
        "label": label,
        "generated_chars": len(gen_text),
        "n_steps": len(trace.steps),
        "contradiction_steps": find_step_indices(trace.steps, "contradiction")[:5],
        "present_steps": find_step_indices(trace.steps, "present")[:5],
        "span_related_steps": find_step_indices(
            trace.steps, "initial response", "progression", "span_a"
        )[:8],
        "commit_step": commit_step,
        "commit_value": commit_val,
        "logit_true_at_commit": logit_true,
        "logit_false_at_commit": logit_false,
        "logit_margin_true_minus_false": (
            (logit_true - logit_false)
            if logit_true is not None and logit_false is not None
            else None
        ),
        "layer_norm_at_commit": layer_at_commit,
    }


def run_one_model_repair_trace(
    case: dict[str, Any], *, model_id: str
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
    trace = generate_trace(
        system=EXTRACT_SYSTEM,
        user=user,
        model_id=model_id,
        max_new_tokens=400,
    )

    summary = _summarize_trace("repair", trace)

    gen_only = "".join(s.token_text for s in trace.steps)
    try:
        data = parse_json_object(gen_only)
        ex = Extraction.from_dict(data)
        final_x = bool(ex.contradiction_present)
    except ValueError:
        final_x = None

    return {
        "model_id": model_id,
        "raw_x": bool(raw.contradiction_present),
        "verify_ok": v1.ok,
        "verify_reasons": v1.reasons,
        "repair_final_x": final_x,
        "trace_summary": summary,
        "_trace": trace,
    }


def _gate_probe_signal(rows: list[dict[str, Any]], cross: dict[str, Any]) -> dict[str, Any]:
    by_model = {r["model_id"]: r for r in rows}
    fail_id = PHASE9A_DEFAULT_MODELS[0]
    safe_id = PHASE9A_DEFAULT_MODELS[1]

    fail = by_model.get(fail_id, {})
    safe = by_model.get(safe_id, {})

    fail_margin = (fail.get("trace_summary") or {}).get("logit_margin_true_minus_false")
    safe_margin = (safe.get("trace_summary") or {}).get("logit_margin_true_minus_false")
    fail_x = fail.get("repair_final_x")
    safe_x = safe.get("repair_final_x")

    max_norm_delta = cross.get("commit_step_divergence", {}).get("max_norm_delta")
    layer_signal = max_norm_delta is not None and max_norm_delta > 5.0
    logit_signal = (
        fail_margin is not None
        and safe_margin is not None
        and fail_margin > safe_margin + 0.5
    )
    behavior_signal = fail_x is True and safe_x is False

    yes = behavior_signal and (layer_signal or logit_signal)
    return {
        "pass": yes,
        "behavior_signal": behavior_signal,
        "layer_signal": layer_signal,
        "logit_signal": logit_signal,
        "fail_model": fail_id,
        "safe_model": safe_id,
        "fail_repair_x": fail_x,
        "safe_repair_x": safe_x,
        "fail_logit_margin": fail_margin,
        "safe_logit_margin": safe_margin,
        "max_norm_delta_at_commit": max_norm_delta,
        "note": (
            "9B YES — proceed only if user asks for 9C"
            if yes
            else "9B NO — stop; do not start 9C"
        ),
    }


def _trace_to_json(trace: GenerationTrace) -> dict[str, Any]:
    return {
        "prompt_token_count": trace.prompt_token_count,
        "n_steps": len(trace.steps),
        "steps": [
            {
                "i": i,
                "token": s.token_text,
                "logit_true": s.logit_true,
                "logit_false": s.logit_false,
                "logits_top5": s.logits_top5,
            }
            for i, s in enumerate(trace.steps)
        ],
    }


def run_phase9b(*, models: list[str] | None = None) -> dict[str, Any]:
    case = load_bc_e1()
    model_ids = list(models or PHASE9A_DEFAULT_MODELS)
    rows: list[dict[str, Any]] = []
    traces: dict[str, GenerationTrace] = {}

    try:
        for mid in model_ids:
            print(f"\n=== 9B · {mid} · repair trace ===")
            row = run_one_model_repair_trace(case, model_id=mid)
            traces[mid] = row.pop("_trace")
            rows.append(row)
            unload_model()
    finally:
        unload_model()

    fail_id, safe_id = model_ids[0], model_ids[1]
    fail_sum = rows[0]["trace_summary"]
    safe_sum = rows[1]["trace_summary"] if len(rows) > 1 else {}
    cross: dict[str, Any] = {}
    if fail_sum.get("commit_step") is not None and safe_sum.get("commit_step") is not None:
        cross["commit_step_divergence"] = _layer_divergence(
            traces[fail_id],
            traces[safe_id],
            fail_sum["commit_step"],
            safe_sum["commit_step"],
        )

    gate = _gate_probe_signal(rows, cross)

    panel = {
        "phase": "9B",
        "protocol": "docs/PHASE9-PROTOCOL.md",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "case_id": case["id"],
        "models": model_ids,
        "rows": rows,
        "cross_model": cross,
        "gate_probe_signal": gate,
        "note": "Repair-path activation contrast; soft claim n=1 case × 2 models.",
    }

    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    out = ARTIFACTS_DIR / "phase9b-bc-e1-panel.json"
    out.write_text(json.dumps(panel, indent=2), encoding="utf-8")
    panel["_artifact"] = str(out)

    for mid, trace in traces.items():
        slug = mid.replace("/", "_").replace(":", "_")
        trace_path = ARTIFACTS_DIR / f"phase9b-bc-e1-{slug}-trace.json"
        row = next(r for r in rows if r["model_id"] == mid)
        commit = (row.get("trace_summary") or {}).get("commit_step")
        compact = _trace_to_json(trace)
        if commit is not None and commit < len(trace.steps):
            compact["commit_step"] = commit
            compact["commit_value"] = row["trace_summary"].get("commit_value")
            compact["commit_layer_hidden"] = trace.steps[commit].layer_hidden
        trace_path.write_text(json.dumps(compact, indent=2), encoding="utf-8")

    return panel
