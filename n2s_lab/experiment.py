from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any

from .ground import ground, selftest_ground
from .neural import baseline_live, extract_live, extract_mock
from .ollama_client import DEFAULT_MODEL, ollama_reachable
from .paths import (
    ARTIFACTS_DIR,
    DATA_PATH,
    HELDOUT_BMTCART_PHASE7_PATH,
    HELDOUT_PATH,
    HELDOUT_PHASE4_PATH,
    TEMPORAL_FAMILY_DEV_PATH,
)
from .predicate import PREDICATE_FORMULA, PREDICATE_ID, PREDICATE_TEXT, evaluate_rule, selftest
from .types import BaselineResult, Verdict


def load_cases(path=None) -> list[dict[str, Any]]:
    payload = json.loads((path or DATA_PATH).read_text(encoding="utf-8"))
    return list(payload["cases"])


def load_heldout_cases() -> list[dict[str, Any]]:
    return load_cases(HELDOUT_PATH)


def load_heldout_phase4_cases() -> list[dict[str, Any]]:
    return load_cases(HELDOUT_PHASE4_PATH)


def load_heldout_bmtcart_cases() -> list[dict[str, Any]]:
    return load_cases(HELDOUT_BMTCART_PHASE7_PATH)


def load_temporal_family_dev_cases() -> list[dict[str, Any]]:
    """Track A/B development family — NOT a held-out test set."""
    return load_cases(TEMPORAL_FAMILY_DEV_PATH)


def run_pipeline(evidence: str, *, mode: str) -> dict[str, Any]:
    extraction = extract_mock(evidence) if mode == "mock" else extract_live(evidence)
    grounding = ground(extraction)
    rule = evaluate_rule(grounding)
    return {
        "extraction": extraction.to_dict(),
        "grounding": grounding.to_dict(),
        "rule": rule.to_dict(),
        "pipeline_verdict": rule.verdict.value,
    }


def run_one(case: dict[str, Any], *, mode: str) -> dict[str, Any]:
    evidence = case["evidence"]
    expected = Verdict(case["expected"])
    pipeline = run_pipeline(evidence, mode=mode)

    baseline: BaselineResult | None = None
    if mode == "live":
        baseline = baseline_live(evidence)

    pipeline_verdict = Verdict(pipeline["pipeline_verdict"])
    baseline_verdict = baseline.verdict if baseline else None

    row = {
        "id": case["id"],
        "category": case["category"],
        "evidence": evidence,
        "why": case.get("why"),
        "expected": expected.value,
        "pipeline": pipeline,
        "baseline": baseline.to_dict() if baseline else None,
        "pipeline_match": pipeline_verdict is expected,
        "baseline_match": (baseline_verdict is expected) if baseline else None,
        "pipeline_vs_baseline": (
            None
            if baseline is None
            else pipeline_verdict is baseline_verdict
        ),
        "disagree": (
            False
            if baseline is None
            else pipeline_verdict is not baseline_verdict
        ),
    }
    return row


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    n = len(rows)
    p_ok = sum(1 for r in rows if r["pipeline_match"])
    b_present = [r for r in rows if r["baseline_match"] is not None]
    b_ok = sum(1 for r in b_present if r["baseline_match"])
    disagree = [r["id"] for r in rows if r.get("disagree")]
    by_cat: dict[str, dict[str, int]] = {}
    for row in rows:
        cat = row["category"]
        bucket = by_cat.setdefault(cat, {"n": 0, "pipeline_ok": 0, "baseline_ok": 0})
        bucket["n"] += 1
        if row["pipeline_match"]:
            bucket["pipeline_ok"] += 1
        if row["baseline_match"]:
            bucket["baseline_ok"] += 1
    return {
        "n": n,
        "pipeline_match": p_ok,
        "pipeline_accuracy": None if not n else round(p_ok / n, 3),
        "baseline_match": b_ok if b_present else None,
        "baseline_accuracy": None if not b_present else round(b_ok / len(b_present), 3),
        "disagreements": disagree,
        "by_category": by_cat,
    }


def run_experiment(
    mode: str = "mock",
    *,
    ids: list[str] | None = None,
    out_name: str | None = None,
) -> dict[str, Any]:
    selftest()
    selftest_ground()
    if mode not in {"mock", "live"}:
        raise ValueError("mode must be mock or live")
    if mode == "live" and not ollama_reachable():
        raise RuntimeError("Ollama is not reachable at 127.0.0.1:11434")

    rows = []
    cases = load_cases()
    if ids:
        wanted = {x.strip() for x in ids if x.strip()}
        cases = [c for c in cases if c["id"] in wanted]
        missing = wanted - {c["id"] for c in cases}
        if missing:
            raise ValueError("unknown case ids: " + ", ".join(sorted(missing)))
    for i, case in enumerate(cases, 1):
        print(f"[{i}/{len(cases)}] {case['id']} {mode}…", flush=True)
        rows.append(run_one(case, mode=mode))
    report = {
        "predicate_id": PREDICATE_ID,
        "predicate_text": PREDICATE_TEXT,
        "predicate_formula": PREDICATE_FORMULA,
        "mode": mode,
        "model": None if mode == "mock" else DEFAULT_MODEL,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "case_ids": [c["id"] for c in cases],
        "summary": summarize(rows),
        "rows": rows,
    }
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    if out_name:
        out = ARTIFACTS_DIR / out_name
    elif ids:
        out = ARTIFACTS_DIR / f"{mode}-subset.json"
    else:
        out = ARTIFACTS_DIR / f"{mode}-latest.json"
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    report["_artifact"] = str(out)
    return report


def format_table(report: dict[str, Any]) -> str:
    lines = [
        f"predicate: {report['predicate_id']}  mode: {report['mode']}",
        f"{'id':<4} {'cat':<12} {'expected':<16} {'pipeline':<16} {'baseline':<16} {'P':<2} {'B':<2} {'Δ'}",
        "-" * 88,
    ]
    for row in report["rows"]:
        b = (row.get("baseline") or {}).get("verdict") if row.get("baseline") else "—"
        p = row["pipeline"]["pipeline_verdict"]
        p_ok = "Y" if row["pipeline_match"] else "n"
        b_ok = "—" if row["baseline_match"] is None else ("Y" if row["baseline_match"] else "n")
        delta = "DIFF" if row.get("disagree") else ""
        lines.append(
            f"{row['id']:<4} {row['category']:<12} {row['expected']:<16} {p:<16} {str(b):<16} {p_ok:<2} {b_ok:<2} {delta}"
        )
    s = report["summary"]
    lines.append("-" * 88)
    lines.append(
        f"pipeline {s['pipeline_match']}/{s['n']}"
        + ("" if s["baseline_match"] is None else f"   baseline {s['baseline_match']}/{s['n']}")
    )
    if s["disagreements"]:
        lines.append("disagreements: " + ", ".join(s["disagreements"]))
    return "\n".join(lines)
