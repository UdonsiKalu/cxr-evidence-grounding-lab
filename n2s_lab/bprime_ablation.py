"""B′ ablation: remap Condition B verdict prompt; freeze Phase-1/2 B.

Reuses Phase-2 free-text analyses. Only the verdict-from-analysis step is rerun.
Does not overwrite phase1-transition-c-u.json or phase2-model-panel.json.
Does not change Conditions A/C/D or M1–M3.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any

from .neural import VERDICT_FROM_ANALYSIS_SYSTEM_B_PRIME, verdict_from_analysis
from .ollama_client import DEFAULT_MODEL, ollama_reachable
from .paths import ARTIFACTS_DIR
from .phase1_diagnostic import (
    PHASE1_CASE_IDS,
    PHASE2_PANEL_MODELS,
    analysis_flags,
    classify_condition_b,
    expected_distinction,
)

CONTROL_ARTIFACTS = {
    "llama3:8b-instruct-q4_0": "phase1-transition-c-u.json",
    "mistral:instruct": "phase2-mistral_instruct.json",
    "qwen2.5-coder:32b": "phase2-qwen2.5-coder_32b.json",
}


def _load_control(model: str) -> dict[str, Any]:
    name = CONTROL_ARTIFACTS.get(model)
    if not name:
        raise ValueError(f"no frozen Phase-2 artifact mapped for {model}")
    path = ARTIFACTS_DIR / name
    if not path.exists():
        raise FileNotFoundError(f"missing control artifact {path}")
    report = json.loads(path.read_text(encoding="utf-8"))
    if report.get("model") != model:
        raise ValueError(f"{path} model={report.get('model')!r} != {model!r}")
    return report


def _b_from_row(row: dict[str, Any]) -> dict[str, Any]:
    return row["conditions"]["B_analysis_then_verdict"]


def run_bprime_for_model(model: str) -> dict[str, Any]:
    control = _load_control(model)
    rows_out: list[dict[str, Any]] = []
    wanted = list(PHASE1_CASE_IDS)
    by_id = {r["id"]: r for r in control["rows"]}
    missing = [i for i in wanted if i not in by_id]
    if missing:
        raise ValueError(f"{model} control missing cases: {missing}")

    for i, case_id in enumerate(wanted, 1):
        row = by_id[case_id]
        b = _b_from_row(row)
        analysis = b["analysis"]
        gold = row["gold"]
        distinction = row.get("expected_distinction") or expected_distinction(case_id, gold)
        print(f"[{i}/{len(wanted)}] {case_id} B′ model={model}…", flush=True)
        primed = verdict_from_analysis(
            analysis,
            model=model,
            system=VERDICT_FROM_ANALYSIS_SYSTEM_B_PRIME,
            variant="B_prime",
        )
        b_ok = b["verdict"] == gold
        bp_ok = primed.verdict.value == gold
        bp_stage = classify_condition_b(
            distinction=distinction,
            analysis=analysis,
            verdict_ok=bp_ok,
        )
        rows_out.append(
            {
                "id": case_id,
                "gold": gold,
                "expected_distinction": distinction,
                "analysis": analysis,
                "analysis_flags": analysis_flags(analysis),
                "B_frozen": {
                    "verdict": b["verdict"],
                    "rationale": b.get("rationale"),
                    "match_gold": b_ok,
                    "loss_stage": row["loss_stage"]["B"],
                },
                "B_prime": {
                    "verdict": primed.verdict.value,
                    "rationale": primed.rationale,
                    "backend": primed.backend,
                    "match_gold": bp_ok,
                    "loss_stage": bp_stage,
                },
            }
        )

    n = len(rows_out)
    b_match = sum(1 for r in rows_out if r["B_frozen"]["match_gold"])
    bp_match = sum(1 for r in rows_out if r["B_prime"]["match_gold"])
    b_err: dict[str, int] = {}
    bp_err: dict[str, int] = {}
    b_mis = bp_mis = 0
    for r in rows_out:
        if not r["B_frozen"]["match_gold"]:
            st = r["B_frozen"]["loss_stage"]
            b_err[st] = b_err.get(st, 0) + 1
            if st == "analysis_output_misalignment":
                b_mis += 1
        if not r["B_prime"]["match_gold"]:
            st = r["B_prime"]["loss_stage"]
            bp_err[st] = bp_err.get(st, 0) + 1
            if st == "analysis_output_misalignment":
                bp_mis += 1

    report = {
        "experiment": "bprime_ablation",
        "optional": True,
        "prompts_frozen_control": True,
        "reused_analyses_from": CONTROL_ARTIFACTS[model],
        "model": model,
        "case_ids": wanted,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "note": (
            "Same analyses as Phase-2 Condition B. Only the verdict-mapping prompt changed. "
            "B′ does not touch A/C/D or overwrite Phase-1/2 artifacts."
        ),
        "summary": {
            "n": n,
            "B_frozen_match_gold": b_match,
            "B_prime_match_gold": bp_match,
            "B_frozen_analysis_output_misalignment": b_mis,
            "B_prime_analysis_output_misalignment": bp_mis,
            "B_frozen_error_stages": b_err,
            "B_prime_error_stages": bp_err,
        },
        "rows": rows_out,
    }
    tag = model.replace(":", "_").replace("/", "_")
    out = ARTIFACTS_DIR / f"bprime-{tag}.json"
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    report["_artifact"] = str(out)
    return report


def format_bprime_table(panel: dict[str, Any]) -> str:
    lines = [
        "B′ ablation (same 8 cases, reused Phase-2 analyses, frozen B vs mapping prompt)",
        f"{'model':<28} {'B':>5} {'B′':>5}  B mismatch  B′ mismatch",
        "-" * 72,
    ]
    for entry in panel["models"]:
        s = entry["summary"]
        n = s["n"]
        lines.append(
            f"{entry['model']:<28} "
            f"{s['B_frozen_match_gold']:>2}/{n} "
            f"{s['B_prime_match_gold']:>2}/{n}  "
            f"mis={s['B_frozen_analysis_output_misalignment']} "
            f"mis={s['B_prime_analysis_output_misalignment']}"
        )
    lines.append("-" * 72)
    lines.append("Question: does a mapping constraint cut analysis→answer mismatch without retuning A/C/D?")
    return "\n".join(lines)


def run_bprime_panel(
    *,
    models: list[str] | None = None,
    out_name: str = "bprime-ablation.json",
) -> dict[str, Any]:
    if not ollama_reachable():
        raise RuntimeError("Ollama is not reachable at 127.0.0.1:11434")
    panel_models = list(models) if models else list(PHASE2_PANEL_MODELS)
    entries = []
    for model in panel_models:
        report = run_bprime_for_model(model)
        entries.append(
            {
                "model": model,
                "artifact": report["_artifact"],
                "summary": report["summary"],
            }
        )
    panel = {
        "experiment": "bprime_ablation_panel",
        "optional": True,
        "control_default_model": DEFAULT_MODEL,
        "models": entries,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "note": (
            "Frozen B prompt unchanged. Unseen cases not in this runner. "
            "Do not treat a B′ win on n=8 as Phase-3."
        ),
    }
    out = ARTIFACTS_DIR / out_name
    out.write_text(json.dumps(panel, indent=2), encoding="utf-8")
    panel["_artifact"] = str(out)
    return panel
