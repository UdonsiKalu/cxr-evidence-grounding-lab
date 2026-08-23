"""Phase-5: L2 verify + one repair + L3 REVIEW (not UNCERTAIN on verify-fail)."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any

from .experiment import load_heldout_cases, load_heldout_phase4_cases
from .ground import ground
from .neural import (
    analyze_free_text,
    extract_from_analysis,
    extract_from_analysis_repair,
    extract_live,
    extract_live_repair,
)
from .ollama_client import DEFAULT_MODEL, ollama_reachable
from .paths import ARTIFACTS_DIR
from .phase1_diagnostic import (
    PHASE2_PANEL_MODELS,
    PHASE3_CASE_IDS,
    PHASE4_CASE_IDS,
    _condition_a,
    _condition_b,
    _condition_c,
    _condition_d,
    classify_condition_a,
    classify_condition_b,
    classify_condition_c,
    classify_condition_d,
    expected_distinction,
    summarize,
)
from .predicate import evaluate_rule
from .types import Extraction, Verdict
from .verify import verify_formalization


def _apply_rule_or_review(
    *,
    extraction: Extraction,
    verify_ok: bool,
    verify_payload: dict[str, Any],
    attempts: int,
) -> dict[str, Any]:
    grounding = ground(extraction)
    if not verify_ok:
        return {
            "extraction": extraction.to_dict(),
            "grounding": grounding.to_dict(),
            "verify": verify_payload,
            "verify_attempts": attempts,
            "rule": None,
            "verdict": Verdict.REVIEW.value,
            "disposition": "REVIEW",
            "note": "verification failed after repair — REVIEW, not UNCERTAIN",
        }
    rule = evaluate_rule(grounding)
    return {
        "extraction": extraction.to_dict(),
        "grounding": grounding.to_dict(),
        "verify": verify_payload,
        "verify_attempts": attempts,
        "rule": rule.to_dict(),
        "verdict": rule.verdict.value,
        "disposition": "AUTO",
    }


def _condition_c_verified(
    evidence: str,
    *,
    model: str,
    case_id: str,
    gold: str,
    analysis_for_verify: str,
) -> dict[str, Any]:
    """C with verify gate. Uses B/D analysis text as the semantic reference when available."""
    extraction = extract_live(evidence, model=model)
    grounding = ground(extraction)
    v1 = verify_formalization(
        analysis=analysis_for_verify,
        extraction=extraction,
        grounding=grounding.to_dict(),
        case_id=case_id,
        gold=gold,
    )
    attempts = 1
    if not v1.ok:
        extraction = extract_live_repair(
            evidence,
            model=model,
            prior=extraction,
            verify_reasons=v1.reasons,
        )
        grounding = ground(extraction)
        v2 = verify_formalization(
            analysis=analysis_for_verify,
            extraction=extraction,
            grounding=grounding.to_dict(),
            case_id=case_id,
            gold=gold,
        )
        attempts = 2
        return _apply_rule_or_review(
            extraction=extraction,
            verify_ok=v2.ok,
            verify_payload={"first": v1.to_dict(), "second": v2.to_dict()},
            attempts=attempts,
        )
    return _apply_rule_or_review(
        extraction=extraction,
        verify_ok=True,
        verify_payload={"first": v1.to_dict()},
        attempts=attempts,
    )


def _condition_d_verified(
    evidence: str,
    *,
    model: str,
    case_id: str,
    gold: str,
    analysis: str,
) -> dict[str, Any]:
    extraction = extract_from_analysis(analysis, model=model)
    grounding = ground(extraction)
    v1 = verify_formalization(
        analysis=analysis,
        extraction=extraction,
        grounding=grounding.to_dict(),
        case_id=case_id,
        gold=gold,
    )
    attempts = 1
    if not v1.ok:
        extraction = extract_from_analysis_repair(
            analysis,
            model=model,
            prior=extraction,
            verify_reasons=v1.reasons,
        )
        grounding = ground(extraction)
        v2 = verify_formalization(
            analysis=analysis,
            extraction=extraction,
            grounding=grounding.to_dict(),
            case_id=case_id,
            gold=gold,
        )
        attempts = 2
        return _apply_rule_or_review(
            extraction=extraction,
            verify_ok=v2.ok,
            verify_payload={"first": v1.to_dict(), "second": v2.to_dict()},
            attempts=attempts,
        )
    return _apply_rule_or_review(
        extraction=extraction,
        verify_ok=True,
        verify_payload={"first": v1.to_dict()},
        attempts=attempts,
    )


def run_one_case_phase5(
    case: dict[str, Any],
    *,
    model: str = DEFAULT_MODEL,
) -> dict[str, Any]:
    evidence = case["evidence"]
    gold = case["expected"]
    distinction = expected_distinction(case["id"], gold)

    a = _condition_a(evidence, model=model)
    b = _condition_b(evidence, model=model)
    c = _condition_c(evidence, model=model)
    d = _condition_d(evidence, model=model, analysis=b["analysis"])
    c_v = _condition_c_verified(
        evidence,
        model=model,
        case_id=case["id"],
        gold=gold,
        analysis_for_verify=b["analysis"],
    )
    d_v = _condition_d_verified(
        evidence,
        model=model,
        case_id=case["id"],
        gold=gold,
        analysis=b["analysis"],
    )

    c_ex = Extraction.from_dict(c["extraction"])
    d_ex = Extraction.from_dict(d["extraction"])
    c_v_ex = Extraction.from_dict(c_v["extraction"])
    d_v_ex = Extraction.from_dict(d_v["extraction"])

    def _auto_ok(verdict: str) -> bool | None:
        if verdict == Verdict.REVIEW.value:
            return None  # abstention — not scored as gold match
        return verdict == gold

    a_ok = a["verdict"] == gold
    b_ok = b["verdict"] == gold
    c_ok = c["verdict"] == gold
    d_ok = d["verdict"] == gold
    c_v_ok = _auto_ok(c_v["verdict"])
    d_v_ok = _auto_ok(d_v["verdict"])

    stages = {
        "A": classify_condition_a(verdict_ok=a_ok),
        "B": classify_condition_b(distinction=distinction, analysis=b["analysis"], verdict_ok=b_ok),
        "C": classify_condition_c(
            distinction=distinction,
            extraction=c_ex,
            grounding=c["grounding"],
            verdict_ok=c_ok,
        ),
        "D": classify_condition_d(
            distinction=distinction,
            analysis=d["analysis"],
            extraction=d_ex,
            grounding=d["grounding"],
            verdict_ok=d_ok,
        ),
        "C_v": (
            "review_after_verify"
            if c_v["verdict"] == Verdict.REVIEW.value
            else classify_condition_c(
                distinction=distinction,
                extraction=c_v_ex,
                grounding=c_v["grounding"],
                verdict_ok=bool(c_v_ok),
            )
        ),
        "D_v": (
            "review_after_verify"
            if d_v["verdict"] == Verdict.REVIEW.value
            else classify_condition_d(
                distinction=distinction,
                analysis=b["analysis"],
                extraction=d_v_ex,
                grounding=d_v["grounding"],
                verdict_ok=bool(d_v_ok),
            )
        ),
    }

    return {
        "id": case["id"],
        "category": case["category"],
        "evidence": evidence,
        "why": case.get("why"),
        "gold": gold,
        "expected_distinction": distinction,
        "model": model,
        "conditions": {
            "A_direct_verdict": a,
            "B_analysis_then_verdict": b,
            "C_structured_pipeline": c,
            "D_analysis_then_structure": d,
            "C_verified": c_v,
            "D_verified": d_v,
        },
        "match_gold": {
            "A": a_ok,
            "B": b_ok,
            "C": c_ok,
            "D": d_ok,
            "C_v": c_v_ok,
            "D_v": d_v_ok,
        },
        "loss_stage": stages,
        "review": {
            "C_v": c_v["verdict"] == Verdict.REVIEW.value,
            "D_v": d_v["verdict"] == Verdict.REVIEW.value,
        },
    }


def phase5_metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Safety / coverage / D-loss before vs after (protocol §6)."""

    def path_stats(key: str, review_key: str | None = None) -> dict[str, Any]:
        n = len(rows)
        if review_key:
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
        correct = sum(1 for r in rows if r["match_gold"][key] is True)
        return {"n": n, "gold_match": correct, "rate": correct / n if n else 0.0}

    d_rep = sum(1 for r in rows if r["loss_stage"]["D"] == "representation_loss")
    d_v_rep = sum(1 for r in rows if r["loss_stage"]["D_v"] == "representation_loss")
    d_v_review = sum(1 for r in rows if r["loss_stage"]["D_v"] == "review_after_verify")

    rep_rows = [r for r in rows if r["loss_stage"]["D"] == "representation_loss"]
    rep_outcomes = []
    for r in rep_rows:
        if r["review"]["D_v"]:
            rep_outcomes.append({"id": r["id"], "D_v": "REVIEW"})
        elif r["match_gold"]["D_v"] is True:
            rep_outcomes.append({"id": r["id"], "D_v": "fixed_gold_match"})
        else:
            rep_outcomes.append(
                {
                    "id": r["id"],
                    "D_v": "still_wrong_auto",
                    "verdict": r["conditions"]["D_verified"]["verdict"],
                }
            )

    return {
        "C": path_stats("C"),
        "D": path_stats("D"),
        "C_v": path_stats("C_v", "C_v"),
        "D_v": path_stats("D_v", "D_v"),
        "d_representation_loss": {"baseline_D": d_rep, "D_v": d_v_rep, "D_v_review": d_v_review},
        "former_D_rep_loss_outcomes": rep_outcomes,
    }


def run_phase5_panel(
    *,
    models: list[str] | None = None,
    case_set: str = "phase4",
    out_name: str = "phase5-verify-panel.json",
) -> dict[str, Any]:
    if not ollama_reachable():
        raise RuntimeError("Ollama is not reachable at 127.0.0.1:11434")
    models = models or list(PHASE2_PANEL_MODELS)
    if case_set == "phase4":
        cases = load_heldout_phase4_cases()
        case_ids = list(PHASE4_CASE_IDS)
    elif case_set == "phase3":
        cases = load_heldout_cases()
        case_ids = list(PHASE3_CASE_IDS)
    else:
        raise ValueError(case_set)

    model_entries = []
    for model in models:
        rows = [run_one_case_phase5(case, model=model) for case in cases]
        safe_name = model.replace(":", "_").replace("/", "_")
        per_name = f"phase5-{case_set}-{safe_name}.json"
        per = {
            "experiment": "phase5_verify_review",
            "optional": True,
            "prompts_frozen_abcd": True,
            "bprime_not_default": True,
            "model": model,
            "case_set": case_set,
            "case_ids": case_ids,
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "summary": summarize(rows),
            "phase5_metrics": phase5_metrics(rows),
            "rows": rows,
            "note": (
                "L2 verify + one repair; L3 REVIEW on verify-fail. "
                "UNCERTAIN only from symbolic rule / uncertainty field — never forced on verify-fail."
            ),
        }
        path = ARTIFACTS_DIR / per_name
        path.write_text(json.dumps(per, indent=2) + "\n", encoding="utf-8")
        model_entries.append(
            {
                "model": model,
                "artifact": str(path),
                "summary": per["summary"],
                "match_gold": per["summary"].get("match_gold"),
                "phase5_metrics": per["phase5_metrics"],
            }
        )

    panel = {
        "experiment": "phase5_verify_review",
        "optional": True,
        "prompts_frozen_abcd": True,
        "bprime_not_default": True,
        "case_set": case_set,
        "case_ids": case_ids,
        "models": model_entries,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "note": "See docs/PHASE5-PROTOCOL.md and docs/ARCHITECTURE-DIRECTION.md",
    }
    out = ARTIFACTS_DIR / out_name
    out.write_text(json.dumps(panel, indent=2) + "\n", encoding="utf-8")
    panel["_artifact"] = str(out)
    return panel
