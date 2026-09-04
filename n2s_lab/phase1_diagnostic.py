"""Optional Phase-1 transition diagnostic — Conditions A–D.

Backward compatible: does not change M1–M3 experiment APIs or gold labels.
To abandon this research direction, delete this module, run_phase1_diagnostic.py,
and artifacts/phase1-*.json; existing lab behavior is unchanged.
"""

from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from typing import Any

from .experiment import load_cases, load_heldout_cases, load_heldout_phase4_cases
from .ground import ground
from .neural import (
    analyze_free_text,
    baseline_live,
    extract_from_analysis,
    extract_live,
    verdict_from_analysis,
)
from .ollama_client import DEFAULT_MODEL, ollama_reachable
from .paths import ARTIFACTS_DIR
from .predicate import PREDICATE_FORMULA, PREDICATE_ID, PREDICATE_TEXT, evaluate_rule
from .types import Atom, Extraction, Verdict

# Frozen Phase-1/2 panel defaults (do not retune prompts per model).
PHASE1_CASE_IDS = ("C1", "C2", "C3", "C4", "U1", "U2", "U3", "U4")
PHASE3_CASE_IDS = ("C5", "C6", "C7", "C8", "U5", "U6", "U7", "U8")
PHASE4_CASE_IDS = ("C9", "C10", "C11", "U9", "U10", "U11", "U12")
PHASE2_PANEL_MODELS = (
    "llama3:8b-instruct-q4_0",  # control (M1–M3 / Phase-1 reference)
    "mistral:instruct",  # different family, similar size
    "qwen2.5-coder:32b",  # stronger local (only clearly larger model available)
)

LOSS_STAGES = (
    "none",
    "model_interpretation_failure",
    "analysis_output_misalignment",
    "representation_loss",
    "symbolic_failure",
    "direct_verdict_miss",  # Condition A only: no separate analysis to inspect
)

CONTRA_MARKERS = re.compile(
    r"\b(contradict(?:ion|s|ory)?|conflict(?:ing|s)?|incompatible|"
    r"mutually exclusive|cannot both|disagree|inconsistent|vs\.?)\b",
    re.I,
)
UNCERT_MARKERS = re.compile(
    r"\b(uncertain|uncertainty|unclear|unknown|possible|questionable|"
    r"may have|awaiting|pending|mixed|insufficient|incomplete|not clear|"
    r"cannot determine|ambiguous)\b",
    re.I,
)
# "no conflicting claims" / "without contradiction" must not count as a contradiction cue.
_CONTRA_NEGATION_PREFIX = re.compile(
    r"(?:\bno|not|without|lack of)\s+(?:any\s+|clear\s+|explicit\s+)?$",
    re.I,
)


def _mentions_marker(text: str, pattern: re.Pattern[str]) -> bool:
    if not text:
        return False
    for m in pattern.finditer(text):
        prefix = text[max(0, m.start() - 28) : m.start()]
        if _CONTRA_NEGATION_PREFIX.search(prefix):
            continue
        # also: "does not mention … contradictions|conflicting"
        window = text[max(0, m.start() - 48) : m.start()].lower()
        if re.search(
            r"(?:does not|do not|didn't|did not)\s+(?:mention|describe|contain|include)\b",
            window,
        ):
            continue
        return True
    return False

OLLAMA_OUTPUT_NOTE = (
    "Current Ollama path uses POST /api/chat with format=json (when structured), "
    "a schema described only in the prompt, temperature=0, then parse_json_object. "
    "This is NOT field-level constrained decoding / JSON Schema / grammar enforcement. "
    "Condition B free-text analysis uses chat_text with no format=json."
)


def expected_distinction(case_id: str, gold: str) -> str:
    if case_id.startswith("C") or gold == Verdict.CONTRADICTION.value:
        return "contradiction"
    if case_id.startswith("U") or gold == Verdict.UNCERTAIN.value:
        return "uncertainty"
    return "other"


def analysis_flags(text: str) -> dict[str, bool]:
    return {
        "mentions_contradiction": _mentions_marker(text or "", CONTRA_MARKERS),
        "mentions_uncertainty": _mentions_marker(text or "", UNCERT_MARKERS),
    }


def analysis_has_distinction(text: str, distinction: str) -> bool:
    flags = analysis_flags(text)
    if distinction == "contradiction":
        return flags["mentions_contradiction"]
    if distinction == "uncertainty":
        return flags["mentions_uncertainty"]
    return False


def structure_preserves(extraction: Extraction, grounding: dict[str, Any], distinction: str) -> bool:
    if distinction == "contradiction":
        return bool(extraction.contradiction_present) or bool(grounding.get("contradiction"))
    if distinction == "uncertainty":
        if extraction.uncertainty_present is True or grounding.get("uncertainty"):
            return True
        # Unknown atoms that keep the rule from closing are also preservation of uncertainty.
        atoms = [
            grounding.get("first_line_identified"),
            grounding.get("first_line_administered"),
            grounding.get("failure_event"),
            grounding.get("failure_of_first_line"),
        ]
        return Atom.UNKNOWN.value in atoms and not grounding.get("contradiction")
    return False


def classify_condition_a(*, verdict_ok: bool) -> str:
    return "none" if verdict_ok else "direct_verdict_miss"


def classify_condition_b(
    *,
    distinction: str,
    analysis: str,
    verdict_ok: bool,
) -> str:
    has = analysis_has_distinction(analysis, distinction)
    if not has and not verdict_ok:
        return "model_interpretation_failure"
    if has and not verdict_ok:
        return "analysis_output_misalignment"
    if not has and verdict_ok:
        # Right answer without explicit cue in prose — still record interpretation gap.
        return "model_interpretation_failure"
    return "none"


def classify_condition_c(
    *,
    distinction: str,
    extraction: Extraction,
    grounding: dict[str, Any],
    verdict_ok: bool,
) -> str:
    preserved = structure_preserves(extraction, grounding, distinction)
    if not preserved and not verdict_ok:
        return "model_interpretation_failure"
    if preserved and not verdict_ok:
        return "symbolic_failure"
    if not preserved and verdict_ok:
        return "model_interpretation_failure"
    return "none"


def classify_condition_d(
    *,
    distinction: str,
    analysis: str,
    extraction: Extraction,
    grounding: dict[str, Any],
    verdict_ok: bool,
) -> str:
    has = analysis_has_distinction(analysis, distinction)
    preserved = structure_preserves(extraction, grounding, distinction)
    if not has:
        return "model_interpretation_failure"
    if has and not preserved:
        return "representation_loss"
    if preserved and not verdict_ok:
        return "symbolic_failure"
    return "none"


def _condition_a(evidence: str, *, model: str) -> dict[str, Any]:
    baseline = baseline_live(evidence, model=model)
    return {
        "verdict": baseline.verdict.value,
        "rationale": baseline.rationale,
        "backend": baseline.backend,
    }


def _condition_b(evidence: str, *, model: str) -> dict[str, Any]:
    analysis = analyze_free_text(evidence, model=model)
    verdict = verdict_from_analysis(analysis, model=model)
    return {
        "analysis": analysis,
        "analysis_flags": analysis_flags(analysis),
        "verdict": verdict.verdict.value,
        "rationale": verdict.rationale,
        "backend": verdict.backend,
    }


def _condition_c(evidence: str, *, model: str) -> dict[str, Any]:
    extraction = extract_live(evidence, model=model)
    grounding = ground(extraction)
    rule = evaluate_rule(grounding)
    return {
        "extraction": extraction.to_dict(),
        "grounding": grounding.to_dict(),
        "rule": rule.to_dict(),
        "verdict": rule.verdict.value,
    }


def _condition_d(evidence: str, *, model: str, analysis: str | None = None) -> dict[str, Any]:
    if analysis is None:
        analysis = analyze_free_text(evidence, model=model)
    extraction = extract_from_analysis(analysis, model=model)
    grounding = ground(extraction)
    rule = evaluate_rule(grounding)
    return {
        "analysis": analysis,
        "analysis_flags": analysis_flags(analysis),
        "extraction": extraction.to_dict(),
        "grounding": grounding.to_dict(),
        "rule": rule.to_dict(),
        "verdict": rule.verdict.value,
    }


def run_one_case(
    case: dict[str, Any],
    *,
    model: str = DEFAULT_MODEL,
    reuse_analysis_for_d: bool = True,
) -> dict[str, Any]:
    evidence = case["evidence"]
    gold = case["expected"]
    distinction = expected_distinction(case["id"], gold)

    a = _condition_a(evidence, model=model)
    b = _condition_b(evidence, model=model)
    c = _condition_c(evidence, model=model)
    d = _condition_d(
        evidence,
        model=model,
        analysis=b["analysis"] if reuse_analysis_for_d else None,
    )

    c_ex = Extraction.from_dict(c["extraction"])
    d_ex = Extraction.from_dict(d["extraction"])

    a_ok = a["verdict"] == gold
    b_ok = b["verdict"] == gold
    c_ok = c["verdict"] == gold
    d_ok = d["verdict"] == gold

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
        },
        "match_gold": {"A": a_ok, "B": b_ok, "C": c_ok, "D": d_ok},
        "loss_stage": stages,
    }


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_stage: dict[str, dict[str, int]] = {c: {s: 0 for s in LOSS_STAGES} for c in "ABCD"}
    error_stages: dict[str, dict[str, int]] = {c: {} for c in "ABCD"}
    match = {c: 0 for c in "ABCD"}
    for row in rows:
        for cond in "ABCD":
            stage = row["loss_stage"][cond]
            by_stage[cond][stage] = by_stage[cond].get(stage, 0) + 1
            if row["match_gold"][cond]:
                match[cond] += 1
            else:
                error_stages[cond][stage] = error_stages[cond].get(stage, 0) + 1
    n = len(rows)
    return {
        "n": n,
        "match_gold": match,
        "loss_stage_counts_all_rows": by_stage,
        "error_stage_counts_when_verdict_misses_gold": error_stages,
        "note": (
            "Headline research numbers are error_stage_counts_when_verdict_misses_gold. "
            "C4 left unresolved by design; do not treat its miss as a bug to fix here."
        ),
    }


def format_transition_table(report: dict[str, Any]) -> str:
    lines = [
        f"Phase-1 transition diagnostic  predicate={report['predicate_id']}  model={report['model']}",
        f"{'id':<4} {'gold':<16} {'A':<16} {'B':<16} {'C':<16} {'D':<16}",
        f"{'':4} {'':16} {'stage':<16} {'stage':<16} {'stage':<16} {'stage':<16}",
        "-" * 100,
    ]
    for row in report["rows"]:
        lines.append(
            f"{row['id']:<4} {row['gold']:<16} "
            f"{row['conditions']['A_direct_verdict']['verdict']:<16} "
            f"{row['conditions']['B_analysis_then_verdict']['verdict']:<16} "
            f"{row['conditions']['C_structured_pipeline']['verdict']:<16} "
            f"{row['conditions']['D_analysis_then_structure']['verdict']:<16}"
        )
        lines.append(
            f"{'':4} {'':16} "
            f"{row['loss_stage']['A']:<16} "
            f"{row['loss_stage']['B']:<16} "
            f"{row['loss_stage']['C']:<16} "
            f"{row['loss_stage']['D']:<16}"
        )
    s = report["summary"]
    lines.append("-" * 100)
    lines.append(
        f"match gold: A={s['match_gold']['A']}/{s['n']}  "
        f"B={s['match_gold']['B']}/{s['n']}  "
        f"C={s['match_gold']['C']}/{s['n']}  "
        f"D={s['match_gold']['D']}/{s['n']}"
    )
    lines.append("error-stage counts (only when verdict ≠ gold):")
    for cond in "ABCD":
        parts = [
            f"{stage}={count}"
            for stage, count in s["error_stage_counts_when_verdict_misses_gold"][cond].items()
            if count
        ]
        lines.append(f"  {cond}: " + (", ".join(parts) if parts else "(no misses)"))
    return "\n".join(lines)


def run_phase1(
    *,
    ids: list[str] | None = None,
    out_name: str = "phase1-transition-c-u.json",
    model: str = DEFAULT_MODEL,
    cases: list[dict[str, Any]] | None = None,
    experiment: str = "phase1_transition_diagnostic",
) -> dict[str, Any]:
    if not ollama_reachable():
        raise RuntimeError("Ollama is not reachable at 127.0.0.1:11434")

    pool = cases if cases is not None else load_cases()
    if ids is not None:
        wanted = list(ids)
    elif cases is not None:
        wanted = [c["id"] for c in pool]
    else:
        wanted = list(PHASE1_CASE_IDS)
    cases_by_id = {c["id"]: c for c in pool}
    missing = set(wanted) - set(cases_by_id)
    if missing:
        raise ValueError("unknown case ids: " + ", ".join(sorted(missing)))
    selected = [cases_by_id[i] for i in wanted]

    rows = []
    for i, case in enumerate(selected, 1):
        print(f"[{i}/{len(selected)}] {case['id']} A–D model={model}…", flush=True)
        rows.append(run_one_case(case, model=model))

    report = {
        "experiment": experiment,
        "optional": True,
        "prompts_frozen": True,
        "bprime_not_default": True,
        "undo": "Delete this artifact, n2s_lab/phase1_diagnostic.py, and run_phase1_diagnostic.py; M1–M3 unchanged.",
        "predicate_id": PREDICATE_ID,
        "predicate_text": PREDICATE_TEXT,
        "predicate_formula": PREDICATE_FORMULA,
        "model": model,
        "ollama_output_mechanism": OLLAMA_OUTPUT_NOTE,
        "case_ids": [c["id"] for c in selected],
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "summary": summarize(rows),
        "rows": rows,
    }
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    out = ARTIFACTS_DIR / out_name
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    report["_artifact"] = str(out)
    return report


def _safe_model_tag(model: str) -> str:
    return model.replace(":", "_").replace("/", "_")


def format_panel_table(panel: dict[str, Any], *, title: str | None = None) -> str:
    heading = title or "Phase-2 model panel (same A–D, same 8 cases, prompts frozen)"
    lines = [
        heading,
        f"{'model':<28} {'A':>5} {'B':>5} {'C':>5} {'D':>5}  D-error representation_loss",
        "-" * 88,
    ]
    for entry in panel["models"]:
        s = entry["summary"]
        n = s["n"]
        d_err = s["error_stage_counts_when_verdict_misses_gold"].get("D", {})
        rep = d_err.get("representation_loss", 0)
        lines.append(
            f"{entry['model']:<28} "
            f"{s['match_gold']['A']:>2}/{n} "
            f"{s['match_gold']['B']:>2}/{n} "
            f"{s['match_gold']['C']:>2}/{n} "
            f"{s['match_gold']['D']:>2}/{n}  "
            f"rep_loss={rep}"
        )
    lines.append("-" * 88)
    lines.append(
        "Question: does 'analysis has distinction → formalization drops it' (D representation_loss) "
        "repeat across models?"
    )
    return "\n".join(lines)


def run_phase2_panel(
    *,
    models: list[str] | None = None,
    ids: list[str] | None = None,
    out_name: str = "phase2-model-panel.json",
    reuse_control_artifact: str | None = "phase1-transition-c-u.json",
) -> dict[str, Any]:
    """Rerun frozen A–D across models. Reuses Phase-1 control artifact when model matches."""
    if not ollama_reachable():
        raise RuntimeError("Ollama is not reachable at 127.0.0.1:11434")

    panel_models = list(models) if models else list(PHASE2_PANEL_MODELS)
    wanted = list(ids) if ids else list(PHASE1_CASE_IDS)
    control = DEFAULT_MODEL
    entries: list[dict[str, Any]] = []

    for model in panel_models:
        reused = False
        report: dict[str, Any] | None = None
        if (
            reuse_control_artifact
            and model == control
            and (ARTIFACTS_DIR / reuse_control_artifact).exists()
        ):
            report = json.loads((ARTIFACTS_DIR / reuse_control_artifact).read_text(encoding="utf-8"))
            if report.get("model") == control and set(report.get("case_ids") or []) == set(wanted):
                reused = True
                print(f"reusing control artifact {reuse_control_artifact} for {model}", flush=True)
            else:
                report = None
        if report is None:
            per_out = f"phase2-{_safe_model_tag(model)}.json"
            report = run_phase1(ids=wanted, out_name=per_out, model=model)
        entries.append(
            {
                "model": model,
                "reused_phase1_artifact": reused,
                "artifact": report.get("_artifact")
                or str(ARTIFACTS_DIR / (reuse_control_artifact or "")),
                "summary": report["summary"],
                "match_gold": report["summary"]["match_gold"],
                "error_stage_counts_when_verdict_misses_gold": report["summary"][
                    "error_stage_counts_when_verdict_misses_gold"
                ],
            }
        )

    panel = {
        "experiment": "phase2_model_panel",
        "optional": True,
        "prompts_frozen": True,
        "case_ids": wanted,
        "models": entries,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "note": (
            "Same A–D conditions and gold as Phase-1. No per-model prompt tuning. "
            "No external/API model in this local panel. Unseen cases come after this freeze."
        ),
    }
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    out = ARTIFACTS_DIR / out_name
    out.write_text(json.dumps(panel, indent=2), encoding="utf-8")
    panel["_artifact"] = str(out)
    return panel


def run_phase3_panel(
    *,
    models: list[str] | None = None,
    out_name: str = "phase3-heldout-panel.json",
) -> dict[str, Any]:
    """Frozen A–D on held-out C5–C8/U5–U8. Never reuses Phase-1/2 artifacts. B′ not default."""
    if not ollama_reachable():
        raise RuntimeError("Ollama is not reachable at 127.0.0.1:11434")

    heldout = load_heldout_cases()
    wanted = list(PHASE3_CASE_IDS)
    panel_models = list(models) if models else list(PHASE2_PANEL_MODELS)
    entries: list[dict[str, Any]] = []

    for model in panel_models:
        per_out = f"phase3-{_safe_model_tag(model)}.json"
        report = run_phase1(
            ids=wanted,
            out_name=per_out,
            model=model,
            cases=heldout,
            experiment="phase3_heldout_diagnostic",
        )
        entries.append(
            {
                "model": model,
                "reused_phase1_artifact": False,
                "artifact": report["_artifact"],
                "summary": report["summary"],
                "match_gold": report["summary"]["match_gold"],
                "error_stage_counts_when_verdict_misses_gold": report["summary"][
                    "error_stage_counts_when_verdict_misses_gold"
                ],
            }
        )

    panel = {
        "experiment": "phase3_heldout_panel",
        "optional": True,
        "prompts_frozen": True,
        "bprime_not_default": True,
        "case_ids": wanted,
        "models": entries,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "note": (
            "Held-out C5–C8/U5–U8. Frozen A–D. B′ not used. "
            "Does not overwrite phase1-transition-c-u.json or phase2-model-panel.json."
        ),
    }
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    out = ARTIFACTS_DIR / out_name
    out.write_text(json.dumps(panel, indent=2), encoding="utf-8")
    panel["_artifact"] = str(out)
    return panel


def phase4_regex_screen(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Regex screen only — not the Phase-4 headline denominator (that is human-confirmed)."""
    b_hits: list[dict[str, Any]] = []
    d_hits: list[dict[str, Any]] = []
    for row in rows:
        distinction = row["expected_distinction"]
        b_text = row["conditions"]["B_analysis_then_verdict"].get("analysis") or ""
        d_text = row["conditions"]["D_analysis_then_structure"].get("analysis") or ""
        if analysis_has_distinction(b_text, distinction):
            b_hits.append(
                {
                    "id": row["id"],
                    "match": row["match_gold"]["B"],
                    "loss_stage": row["loss_stage"]["B"],
                }
            )
        if analysis_has_distinction(d_text, distinction):
            d_hits.append(
                {
                    "id": row["id"],
                    "match": row["match_gold"]["D"],
                    "loss_stage": row["loss_stage"]["D"],
                    "representation_loss": row["loss_stage"]["D"] == "representation_loss",
                }
            )
    return {
        "note": (
            "Regex analysis_has_distinction screen only. "
            "Protocol headline uses human-confirmed analyses (A_human)."
        ),
        "B": {
            "n_screen": len(b_hits),
            "loss": sum(1 for x in b_hits if not x["match"]),
            "ids_with_cue": [x["id"] for x in b_hits],
            "ids_loss": [x["id"] for x in b_hits if not x["match"]],
        },
        "D": {
            "n_screen": len(d_hits),
            "representation_loss": sum(1 for x in d_hits if x["representation_loss"]),
            "ids_with_cue": [x["id"] for x in d_hits],
            "ids_representation_loss": [x["id"] for x in d_hits if x["representation_loss"]],
        },
    }


def run_phase4_panel(
    *,
    models: list[str] | None = None,
    out_name: str = "phase4-heldout-panel.json",
) -> dict[str, Any]:
    """Frozen A–D on held-out C9–C11/U9–U12. Never reuses Phase-1/2/3 artifacts. B′ off."""
    if not ollama_reachable():
        raise RuntimeError("Ollama is not reachable at 127.0.0.1:11434")

    heldout = load_heldout_phase4_cases()
    wanted = list(PHASE4_CASE_IDS)
    panel_models = list(models) if models else list(PHASE2_PANEL_MODELS)
    entries: list[dict[str, Any]] = []

    for model in panel_models:
        per_out = f"phase4-{_safe_model_tag(model)}.json"
        report = run_phase1(
            ids=wanted,
            out_name=per_out,
            model=model,
            cases=heldout,
            experiment="phase4_heldout_diagnostic",
        )
        screen = phase4_regex_screen(report["rows"])
        report["phase4_regex_screen"] = screen
        (ARTIFACTS_DIR / per_out).write_text(json.dumps(report, indent=2), encoding="utf-8")
        entries.append(
            {
                "model": model,
                "reused_phase1_artifact": False,
                "artifact": report["_artifact"],
                "summary": report["summary"],
                "match_gold": report["summary"]["match_gold"],
                "error_stage_counts_when_verdict_misses_gold": report["summary"][
                    "error_stage_counts_when_verdict_misses_gold"
                ],
                "phase4_regex_screen": screen,
            }
        )

    panel = {
        "experiment": "phase4_heldout_panel",
        "optional": True,
        "prompts_frozen": True,
        "bprime_not_default": True,
        "case_ids": wanted,
        "models": entries,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "note": (
            "Held-out C9–C11/U9–U12. Frozen A–D. B′ not used. No C4/C8 analog. "
            "Does not overwrite phase1 / phase2 / phase3 artifacts. "
            "Headline B/D loss rates need human confirmation of analyses; "
            "phase4_regex_screen is a cue filter only."
        ),
    }
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    out = ARTIFACTS_DIR / out_name
    out.write_text(json.dumps(panel, indent=2), encoding="utf-8")
    panel["_artifact"] = str(out)
    return panel