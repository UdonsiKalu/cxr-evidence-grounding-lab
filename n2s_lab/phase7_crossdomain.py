"""Phase-7: evidence-preservation object + BMT/CAR-T cross-domain held-out."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any

from .evidence import gate_verdict_with_evidence
from .experiment import (
    load_heldout_bmtcart_cases,
    load_heldout_phase4_cases,
    load_temporal_family_dev_cases,
    load_temporal_family_test_cases,
)
from .ollama_client import DEFAULT_MODEL, ollama_reachable
from .paths import ARTIFACTS_DIR
from .phase1_diagnostic import PHASE2_PANEL_MODELS, PHASE4_CASE_IDS
from .phase6_faithfulness import phase6_metrics, run_one_case_phase6
from .roundtrip import dual_path_verdict
from .types import Extraction, Verdict

BMTCART_CASE_IDS = [
    "BC_C1",
    "BC_C2",
    "BC_U1",
    "BC_U2",
    "BC_E1",
    "BC_E2",
]


def _apply_evidence_gate(
    *,
    block: dict[str, Any],
    source_text: str,
    source_id: str,
) -> dict[str, Any]:
    extraction = Extraction.from_dict(block["extraction"])
    grounding = block.get("grounding") or {}
    gated = gate_verdict_with_evidence(
        verdict=block["verdict"],
        extraction=extraction,
        grounding=grounding,
        source_text=source_text,
        source_id=source_id,
    )
    return {**block, **gated}


def run_one_case_phase7(
    case: dict[str, Any],
    *,
    model: str = DEFAULT_MODEL,
) -> dict[str, Any]:
    base = run_one_case_phase6(case, model=model)
    gold = case["expected"]
    evidence = case["evidence"]
    analysis = base["conditions"]["B_analysis_then_verdict"]["analysis"]

    c_full = _apply_evidence_gate(
        block=base["conditions"]["C_roundtrip"],
        source_text=evidence,
        source_id="note",
    )
    d_full = _apply_evidence_gate(
        block=base["conditions"]["D_roundtrip"],
        source_text=analysis,
        source_id="analysis",
    )
    dual_full = dual_path_verdict(c_full["verdict"], d_full["verdict"])

    def _auto_ok(verdict: str) -> bool | None:
        if verdict == Verdict.REVIEW.value:
            return None
        return verdict == gold

    match = dict(base["match_gold"])
    match["C_full"] = _auto_ok(c_full["verdict"])
    match["D_full"] = _auto_ok(d_full["verdict"])
    match["Dual_full"] = _auto_ok(dual_full["verdict"])

    review = dict(base["review"])
    review["C_full"] = c_full["verdict"] == Verdict.REVIEW.value
    review["D_full"] = d_full["verdict"] == Verdict.REVIEW.value
    review["Dual_full"] = dual_full["verdict"] == Verdict.REVIEW.value

    stages = dict(base["loss_stage"])
    for key, prior, gated in (
        ("C_full", "C_rt", c_full),
        ("D_full", "D_rt", d_full),
    ):
        if gated["verdict"] == Verdict.REVIEW.value and not base["review"].get(prior):
            stages[key] = "review_after_evidence"
        elif gated["verdict"] == Verdict.REVIEW.value:
            stages[key] = "review_after_verify_or_roundtrip"
        else:
            stages[key] = stages.get(prior)
    stages["Dual_full"] = (
        "review_dual_disagree" if review["Dual_full"] else "none"
    )

    evidence_new_reviews = {
        "C": sum(
            1
            for _ in [c_full]
            if c_full["verdict"] == Verdict.REVIEW.value
            and not base["review"]["C_rt"]
        ),
        "D": sum(
            1
            for _ in [d_full]
            if d_full["verdict"] == Verdict.REVIEW.value
            and not base["review"]["D_rt"]
        ),
    }

    return {
        **base,
        "conditions": {
            **base["conditions"],
            "C_full": c_full,
            "D_full": d_full,
            "Dual_full": dual_full,
        },
        "match_gold": match,
        "loss_stage": stages,
        "review": review,
        "phase7": True,
        "domain": case.get("domain_tag") or base.get("domain"),
        "evidence_new_reviews": evidence_new_reviews,
    }


def phase7_metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    def path_stats(key: str, review_key: str) -> dict[str, Any]:
        n = len(rows)
        auto = [r for r in rows if not r["review"][review_key]]
        review_n = sum(1 for r in rows if r["review"][review_key])
        correct = sum(1 for r in auto if r["match_gold"][key] is True)
        return {
            "n": n,
            "review_n": review_n,
            "coverage_auto": (n - review_n) / n if n else 0.0,
            "safety_among_auto": correct / len(auto) if auto else None,
            "gold_match_including_null": sum(
                1 for r in rows if r["match_gold"][key] is True
            ),
        }

    p6 = phase6_metrics(rows)
    return {
        **p6,
        "C_full": path_stats("C_full", "C_full"),
        "D_full": path_stats("D_full", "D_full"),
        "Dual_full": path_stats("Dual_full", "Dual_full"),
        "evidence_new_reviews": {
            "C": sum(r.get("evidence_new_reviews", {}).get("C", 0) for r in rows),
            "D": sum(r.get("evidence_new_reviews", {}).get("D", 0) for r in rows),
        },
    }


def run_phase7_panel(
    *,
    models: list[str] | None = None,
    case_set: str = "bmtcart",
    out_name: str = "phase7-bmtcart-panel.json",
    artifact_prefix: str = "phase7",
    experiment: str | None = None,
) -> dict[str, Any]:
    if not ollama_reachable():
        raise RuntimeError("Ollama is not reachable at 127.0.0.1:11434")
    models = models or list(PHASE2_PANEL_MODELS)
    if case_set == "bmtcart":
        cases = load_heldout_bmtcart_cases()
        case_ids = list(BMTCART_CASE_IDS)
        experiment = experiment or "phase7_bmtcart"
    elif case_set == "phase4":
        cases = load_heldout_phase4_cases()
        case_ids = list(PHASE4_CASE_IDS)
        experiment = experiment or "phase7_phase4_evidence"
        if out_name == "phase7-bmtcart-panel.json":
            out_name = "phase7-phase4-panel.json"
    elif case_set == "temporal-dev":
        cases = load_temporal_family_dev_cases()
        case_ids = [c["id"] for c in cases]
        experiment = experiment or "phase7_temporal_family_dev"
        if out_name == "phase7-bmtcart-panel.json":
            out_name = "phase7-temporal-dev-panel.json"
    elif case_set == "temporal-test":
        # Score only after intervention frozen on temporal-dev — do not peek while designing.
        cases = load_temporal_family_test_cases()
        case_ids = [c["id"] for c in cases]
        experiment = experiment or "phase7_temporal_family_test"
        if out_name == "phase7-bmtcart-panel.json":
            out_name = "phase7-temporal-test-panel.json"
    else:
        raise ValueError(case_set)

    model_entries = []
    for model in models:
        rows = [run_one_case_phase7(case, model=model) for case in cases]
        safe_name = model.replace(":", "_").replace("/", "_")
        per_name = f"{artifact_prefix}-{case_set}-{safe_name}.json"
        per = {
            "experiment": experiment,
            "optional": True,
            "prompts_frozen_abcd": True,
            "bprime_not_default": True,
            "model": model,
            "case_set": case_set,
            "case_ids": case_ids,
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "summary": summarize_phase7(rows),
            "phase6_metrics": phase6_metrics(rows),
            "phase7_metrics": phase7_metrics(rows),
            "rows": rows,
            "note": (
                "Phase-6 stack + L2d evidence-preservation gate. "
                "Does not prove faithfulness. REVIEW on fail."
            ),
        }
        path = ARTIFACTS_DIR / per_name
        path.write_text(json.dumps(per, indent=2) + "\n", encoding="utf-8")
        model_entries.append(
            {
                "model": model,
                "artifact": str(path),
                "summary": per["summary"],
                "phase7_metrics": per["phase7_metrics"],
            }
        )

    panel = {
        "experiment": experiment,
        "optional": True,
        "prompts_frozen_abcd": True,
        "bprime_not_default": True,
        "case_set": case_set,
        "case_ids": case_ids,
        "models": model_entries,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "note": "See docs/PHASE7-PROTOCOL.md",
    }
    out = ARTIFACTS_DIR / out_name
    out.write_text(json.dumps(panel, indent=2) + "\n", encoding="utf-8")
    panel["_artifact"] = str(out)
    return panel


def summarize_phase7(rows: list[dict[str, Any]]) -> dict[str, Any]:
    from .phase1_diagnostic import summarize

    base = summarize(rows)
    base["match_gold"]["C_full"] = sum(
        1 for r in rows if r["match_gold"].get("C_full") is True
    )
    base["match_gold"]["D_full"] = sum(
        1 for r in rows if r["match_gold"].get("D_full") is True
    )
    base["match_gold"]["Dual_full"] = sum(
        1 for r in rows if r["match_gold"].get("Dual_full") is True
    )
    return base
