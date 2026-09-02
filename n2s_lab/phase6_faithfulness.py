"""Phase-6: L2b round-trip + L2c dual-path on top of Phase-5 C_v / D_v."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any

from .experiment import load_heldout_cases, load_heldout_phase4_cases
from .neural import paraphrase_from_extraction
from .ollama_client import DEFAULT_MODEL, ollama_reachable
from .paths import ARTIFACTS_DIR
from .phase1_diagnostic import PHASE2_PANEL_MODELS, PHASE3_CASE_IDS, PHASE4_CASE_IDS, summarize
from .phase5_verify import phase5_metrics, run_one_case_phase5
from .roundtrip import dual_path_verdict, verify_roundtrip
from .types import Extraction, Verdict


def _gate_roundtrip(
    *,
    verified_block: dict[str, Any],
    source_text: str,
    model: str,
    case_id: str,
    gold: str,
) -> dict[str, Any]:
    """If L2a already REVIEW, stay REVIEW. Else paraphrase and L2b-check."""
    base_verdict = verified_block["verdict"]
    if base_verdict == Verdict.REVIEW.value:
        return {
            "verdict": Verdict.REVIEW.value,
            "disposition": "REVIEW",
            "roundtrip": None,
            "paraphrase": None,
            "skipped": "already_REVIEW_after_L2a",
            "base_verdict": base_verdict,
            "extraction": verified_block.get("extraction"),
            "grounding": verified_block.get("grounding"),
            "verify": verified_block.get("verify"),
        }

    extraction = Extraction.from_dict(verified_block["extraction"])
    paraphrase = paraphrase_from_extraction(extraction, model=model)
    rt = verify_roundtrip(
        source_text=source_text,
        paraphrase=paraphrase,
        extraction=extraction,
        case_id=case_id,
        gold=gold,
    )
    if not rt.ok:
        return {
            "verdict": Verdict.REVIEW.value,
            "disposition": "REVIEW",
            "roundtrip": rt.to_dict(),
            "paraphrase": paraphrase,
            "skipped": None,
            "base_verdict": base_verdict,
            "extraction": verified_block.get("extraction"),
            "grounding": verified_block.get("grounding"),
            "verify": verified_block.get("verify"),
            "note": "L2b round-trip failed — REVIEW, not UNCERTAIN",
        }
    return {
        "verdict": base_verdict,
        "disposition": "AUTO",
        "roundtrip": rt.to_dict(),
        "paraphrase": paraphrase,
        "skipped": None,
        "base_verdict": base_verdict,
        "extraction": verified_block.get("extraction"),
        "grounding": verified_block.get("grounding"),
        "verify": verified_block.get("verify"),
    }


def run_one_case_phase6(
    case: dict[str, Any],
    *,
    model: str = DEFAULT_MODEL,
) -> dict[str, Any]:
    base = run_one_case_phase5(case, model=model)
    gold = case["expected"]
    evidence = case["evidence"]
    analysis = base["conditions"]["B_analysis_then_verdict"]["analysis"]
    c_v = base["conditions"]["C_verified"]
    d_v = base["conditions"]["D_verified"]

    c_rt = _gate_roundtrip(
        verified_block=c_v,
        source_text=evidence,
        model=model,
        case_id=case["id"],
        gold=gold,
    )
    d_rt = _gate_roundtrip(
        verified_block=d_v,
        source_text=analysis,
        model=model,
        case_id=case["id"],
        gold=gold,
    )

    dual = dual_path_verdict(c_v["verdict"], d_v["verdict"])
    dual_rt = dual_path_verdict(c_rt["verdict"], d_rt["verdict"])

    def _auto_ok(verdict: str) -> bool | None:
        if verdict == Verdict.REVIEW.value:
            return None
        return verdict == gold

    match = dict(base["match_gold"])
    match["C_rt"] = _auto_ok(c_rt["verdict"])
    match["D_rt"] = _auto_ok(d_rt["verdict"])
    match["Dual"] = _auto_ok(dual["verdict"])
    match["Dual_rt"] = _auto_ok(dual_rt["verdict"])

    review = dict(base["review"])
    review["C_rt"] = c_rt["verdict"] == Verdict.REVIEW.value
    review["D_rt"] = d_rt["verdict"] == Verdict.REVIEW.value
    review["Dual"] = dual["verdict"] == Verdict.REVIEW.value
    review["Dual_rt"] = dual_rt["verdict"] == Verdict.REVIEW.value

    stages = dict(base["loss_stage"])
    stages["C_rt"] = (
        "review_after_roundtrip"
        if c_rt["verdict"] == Verdict.REVIEW.value and c_v["verdict"] != Verdict.REVIEW.value
        else (
            "review_after_verify"
            if c_rt["verdict"] == Verdict.REVIEW.value
            else stages.get("C_v")
        )
    )
    stages["D_rt"] = (
        "review_after_roundtrip"
        if d_rt["verdict"] == Verdict.REVIEW.value and d_v["verdict"] != Verdict.REVIEW.value
        else (
            "review_after_verify"
            if d_rt["verdict"] == Verdict.REVIEW.value
            else stages.get("D_v")
        )
    )
    stages["Dual"] = "review_dual_disagree" if review["Dual"] else "none"
    stages["Dual_rt"] = "review_dual_disagree" if review["Dual_rt"] else "none"

    return {
        **base,
        "conditions": {
            **base["conditions"],
            "C_roundtrip": c_rt,
            "D_roundtrip": d_rt,
            "Dual": dual,
            "Dual_rt": dual_rt,
        },
        "match_gold": match,
        "loss_stage": stages,
        "review": review,
        "phase6": True,
    }


def phase6_metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Safety / coverage for Phase-6 paths + deltas vs Phase-5 C_v/D_v."""

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
            "gold_match_including_null": sum(1 for r in rows if r["match_gold"][key] is True),
        }

    p5 = phase5_metrics(rows)
    out = {
        **{k: p5[k] for k in ("C", "D", "C_v", "D_v", "d_representation_loss", "former_D_rep_loss_outcomes")},
        "C_rt": path_stats("C_rt", "C_rt"),
        "D_rt": path_stats("D_rt", "D_rt"),
        "Dual": path_stats("Dual", "Dual"),
        "Dual_rt": path_stats("Dual_rt", "Dual_rt"),
        "roundtrip_new_reviews": {
            "C": sum(
                1
                for r in rows
                if r["review"]["C_rt"] and not r["review"]["C_v"]
            ),
            "D": sum(
                1
                for r in rows
                if r["review"]["D_rt"] and not r["review"]["D_v"]
            ),
        },
        "dual_disagreements": {
            "Dual": sum(1 for r in rows if r["conditions"]["Dual"].get("reason") == "paths_disagree"),
            "Dual_rt": sum(
                1 for r in rows if r["conditions"]["Dual_rt"].get("reason") == "paths_disagree"
            ),
        },
    }
    return out


def run_phase6_panel(
    *,
    models: list[str] | None = None,
    case_set: str = "phase4",
    out_name: str = "phase6-faithfulness-panel.json",
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
        rows = [run_one_case_phase6(case, model=model) for case in cases]
        safe_name = model.replace(":", "_").replace("/", "_")
        per_name = f"phase6-{case_set}-{safe_name}.json"
        per = {
            "experiment": "phase6_faithfulness",
            "optional": True,
            "prompts_frozen_abcd": True,
            "bprime_not_default": True,
            "model": model,
            "case_set": case_set,
            "case_ids": case_ids,
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "summary": summarize(rows),
            "phase6_metrics": phase6_metrics(rows),
            "rows": rows,
            "note": (
                "L2b round-trip + L2c dual-path on Phase-5 C_v/D_v. "
                "Does not prove faithfulness. REVIEW on fail — never force UNCERTAIN."
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
                "phase6_metrics": per["phase6_metrics"],
            }
        )

    panel = {
        "experiment": "phase6_faithfulness",
        "optional": True,
        "prompts_frozen_abcd": True,
        "bprime_not_default": True,
        "case_set": case_set,
        "case_ids": case_ids,
        "models": model_entries,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "note": "See docs/PHASE6-PROTOCOL.md and docs/ARCHITECTURE-DIRECTION.md",
    }
    out = ARTIFACTS_DIR / out_name
    out.write_text(json.dumps(panel, indent=2) + "\n", encoding="utf-8")
    panel["_artifact"] = str(out)
    return panel
