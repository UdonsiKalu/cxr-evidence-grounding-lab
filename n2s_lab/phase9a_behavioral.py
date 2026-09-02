"""Phase-9A: HF behavioral reproduction of BC_E1 contradiction.present divergence."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any

from .ground import ground
from .hf_client import chat_json, chat_text, unload_model
from .neural import (
    ANALYSIS_SYSTEM,
    EXTRACT_SCHEMA,
    EXTRACT_SYSTEM,
    REPAIR_EXTRACT_SUFFIX,
)
from .paths import ARTIFACTS_DIR, HELDOUT_BMTCART_PHASE7_PATH
from .phase1_diagnostic import analysis_flags
from .predicate import evaluate_rule
from .types import Extraction
from .verify import verify_formalization

# Default 9A panel (VRAM-realistic on 3090).
PHASE9A_DEFAULT_MODELS = (
    "Qwen/Qwen2.5-7B-Instruct",
    "Qwen/Qwen2.5-14B-Instruct",
)

BC_E1_ID = "BC_E1"


def load_bc_e1() -> dict[str, Any]:
    data = json.loads(HELDOUT_BMTCART_PHASE7_PATH.read_text(encoding="utf-8"))
    for case in data["cases"]:
        if case["id"] == BC_E1_ID:
            return case
    raise KeyError(f"{BC_E1_ID} not found in {HELDOUT_BMTCART_PHASE7_PATH}")


def _extract_hf(evidence: str, *, model_id: str) -> Extraction:
    data = chat_json(
        f"Schema:\n{EXTRACT_SCHEMA}\n\nNote:\n{evidence}",
        system=EXTRACT_SYSTEM,
        model_id=model_id,
        max_new_tokens=600,
    )
    extraction = Extraction.from_dict(data)
    extraction.backend = f"hf:{model_id}"
    return extraction


def _extract_hf_repair(
    evidence: str,
    *,
    model_id: str,
    prior: Extraction,
    verify_reasons: list[str],
) -> Extraction:
    reasons = "; ".join(verify_reasons) or "structure/analysis mismatch"
    data = chat_json(
        f"Schema:\n{EXTRACT_SCHEMA}\n\nNote:\n{evidence}\n\n"
        f"Previous JSON:\n{prior.to_dict()}\n\nVerify reasons: {reasons}\n"
        f"{REPAIR_EXTRACT_SUFFIX}",
        system=EXTRACT_SYSTEM,
        model_id=model_id,
        max_new_tokens=600,
    )
    extraction = Extraction.from_dict(data)
    extraction.backend = f"hf:{model_id}:repair"
    return extraction


def _analyze_hf(evidence: str, *, model_id: str) -> str:
    return chat_text(
        f"Note:\n{evidence}",
        system=ANALYSIS_SYSTEM,
        model_id=model_id,
        max_new_tokens=500,
    )


def _x_summary(extraction: Extraction) -> dict[str, Any]:
    cues = []
    if extraction.contradiction_cues:
        cues = [
            {"a": c.a, "b": c.b} for c in extraction.contradiction_cues
        ]
    return {
        "contradiction_present": bool(extraction.contradiction_present),
        "spans": cues,
        "backend": extraction.backend,
    }


def run_one_model(case: dict[str, Any], *, model_id: str) -> dict[str, Any]:
    evidence = case["evidence"]
    gold = case["expected"]
    case_id = case["id"]

    print(f"\n=== 9A · {model_id} · {case_id} ===")

    # Stage 1: raw extract
    raw = _extract_hf(evidence, model_id=model_id)
    raw_g = ground(raw)
    raw_rule = evaluate_rule(raw_g)

    # Stage 2: analysis (verify reference)
    analysis = _analyze_hf(evidence, model_id=model_id)
    flags = analysis_flags(analysis)

    # Stage 3: verify + optional repair (C_verified path)
    v1 = verify_formalization(
        analysis=analysis,
        extraction=raw,
        grounding=raw_g.to_dict(),
        case_id=case_id,
        gold=gold,
    )
    repaired: Extraction | None = None
    v2 = None
    final_ex = raw
    if not v1.ok:
        repaired = _extract_hf_repair(
            evidence,
            model_id=model_id,
            prior=raw,
            verify_reasons=v1.reasons,
        )
        repaired_g = ground(repaired)
        v2 = verify_formalization(
            analysis=analysis,
            extraction=repaired,
            grounding=repaired_g.to_dict(),
            case_id=case_id,
            gold=gold,
        )
        final_ex = repaired

    final_g = ground(final_ex)
    if v1.ok or (v2 is not None and v2.ok):
        final_rule = evaluate_rule(final_g)
        disposition = "AUTO"
        verdict = final_rule.verdict.value
        rule_dict = final_rule.to_dict()
    else:
        disposition = "REVIEW"
        verdict = "REVIEW"
        rule_dict = None

    stage_that_set_x = None
    if raw.contradiction_present:
        stage_that_set_x = "raw_extract"
    elif repaired is not None and repaired.contradiction_present:
        stage_that_set_x = "repair"

    return {
        "model_id": model_id,
        "case_id": case_id,
        "gold": gold,
        "raw_extract": {
            **_x_summary(raw),
            "extraction": raw.to_dict(),
            "verdict": raw_rule.verdict.value,
        },
        "analysis": analysis,
        "analysis_flags": flags,
        "verify": {
            "first": v1.to_dict(),
            "second": v2.to_dict() if v2 is not None else None,
        },
        "repaired": (
            {**_x_summary(repaired), "extraction": repaired.to_dict()}
            if repaired is not None
            else None
        ),
        "final": {
            **_x_summary(final_ex),
            "verdict": verdict,
            "disposition": disposition,
            "rule": rule_dict,
            "stage_that_set_x": stage_that_set_x,
        },
    }


def _gate_clean_divergence(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """YES if at least one final X=true and one final X=false among models."""
    xs = [bool(r["final"]["contradiction_present"]) for r in rows]
    has_true = any(xs)
    has_false = any(not x for x in xs)
    yes = has_true and has_false and len(rows) >= 2
    return {
        "pass": yes,
        "has_final_x_true": has_true,
        "has_final_x_false": has_false,
        "n_models": len(rows),
        "per_model_final_x": {
            r["model_id"]: r["final"]["contradiction_present"] for r in rows
        },
        "per_model_stage_that_set_x": {
            r["model_id"]: r["final"]["stage_that_set_x"] for r in rows
        },
        "note": (
            "9A YES — proceed only if user asks for 9B"
            if yes
            else "9A NO — stop; do not start 9B/9C"
        ),
    }


def run_phase9a(
    *,
    models: list[str] | None = None,
) -> dict[str, Any]:
    case = load_bc_e1()
    model_ids = list(models or PHASE9A_DEFAULT_MODELS)
    rows: list[dict[str, Any]] = []
    try:
        for mid in model_ids:
            row = run_one_model(case, model_id=mid)
            rows.append(row)
            # Persist immediately so a later-model failure does not lose earlier rows.
            ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
            slug = mid.replace("/", "_").replace(":", "_")
            per = ARTIFACTS_DIR / f"phase9a-bc-e1-{slug}.json"
            per.write_text(json.dumps(row, indent=2), encoding="utf-8")
            row["_artifact"] = str(per)
            print(f"[9A] wrote {per}")
            unload_model()
    finally:
        unload_model()

    gate = _gate_clean_divergence(rows)
    panel = {
        "phase": "9A",
        "protocol": "docs/PHASE9-PROTOCOL.md",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "case": {
            "id": case["id"],
            "evidence": case["evidence"],
            "gold": case["expected"],
            "why": case.get("why"),
        },
        "frozen_prompts": {
            "extract": "n2s_lab.neural.EXTRACT_SYSTEM + EXTRACT_SCHEMA",
            "repair": "n2s_lab.neural.REPAIR_EXTRACT_SUFFIX",
            "analysis": "n2s_lab.neural.ANALYSIS_SYSTEM",
        },
        "models": model_ids,
        "rows": rows,
        "gate_clean_x_divergence": gate,
        "citation_only_ollama": {
            "coder_32b": "artifacts/phase7-bmtcart-qwen2.5-coder_32b.json",
            "14b": "artifacts/phase8-bmtcart-qwen2.5_14b.json",
            "instruct_32b": "artifacts/phase8-bmtcart-qwen2.5_32b.json",
        },
        "note": "Does not overwrite phase1–8 artifacts. Not Dual_full replay.",
    }

    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    out = ARTIFACTS_DIR / "phase9a-bc-e1-panel.json"
    out.write_text(json.dumps(panel, indent=2), encoding="utf-8")
    panel["_artifact"] = str(out)

    return panel
