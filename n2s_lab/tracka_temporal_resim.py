"""Track A reconnect — resim Dual_full after grounding temporal-change fix.

Re-grounds frozen C_full/D_full extractions from Phase-7 temporal-dev (or sealed
temporal-test) panels (no LLM). Does not modify G3. Test resim is score-only —
do not redesign from results.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .auto_contract import classify_case, summarize_scores
from .ground import ground
from .paths import ARTIFACTS_DIR
from .predicate import evaluate_rule
from .roundtrip import dual_path_verdict
from .types import Extraction, Verdict

TEMPORAL_DEV_REPORTS = (
    "phase7-temporal-dev-qwen2.5-coder_32b.json",
    "phase7-temporal-dev-llama3_8b-instruct-q4_0.json",
    "phase7-temporal-dev-mistral_instruct.json",
)

# Sealed test — Qwen Dual only (score re-open; no redesign from results).
TEMPORAL_TEST_REPORTS = (
    "phase7-temporal-test-qwen2.5-coder_32b.json",
)

PATH_KEYS = ("C_full", "D_full")


def _recompute_path(block: dict[str, Any]) -> dict[str, Any]:
    """Re-ground + rule from stored extraction; keep REVIEW if already REVIEW."""
    out = dict(block)
    if block.get("verdict") == Verdict.REVIEW.value or block.get("disposition") == "REVIEW":
        return out
    ex_raw = block.get("extraction")
    if not isinstance(ex_raw, dict):
        return out
    try:
        ex = Extraction.from_dict(ex_raw)
    except Exception:
        return out
    g = ground(ex)
    rule = evaluate_rule(g)
    out["grounding"] = g.to_dict()
    out["verdict"] = rule.verdict.value
    out["disposition"] = "AUTO"
    out["rule"] = rule.to_dict()
    out["resim_ground_trace"] = list(g.trace)
    return out


def resim_report(path: Path) -> dict[str, Any]:
    report = json.loads(path.read_text(encoding="utf-8"))
    rows_out = []
    for row in report.get("rows") or []:
        cond = dict(row.get("conditions") or {})
        for pk in PATH_KEYS:
            if pk in cond and isinstance(cond[pk], dict):
                cond[pk] = _recompute_path(cond[pk])
        c_v = (cond.get("C_full") or {}).get("verdict")
        d_v = (cond.get("D_full") or {}).get("verdict")
        if c_v and d_v:
            dual = dual_path_verdict(c_v, d_v)
            cond["Dual_full"] = dual
        new_row = dict(row)
        new_row["conditions"] = cond
        # refresh match_gold Dual_full if present
        mg = dict(new_row.get("match_gold") or {})
        gold = new_row.get("gold")
        dv = (cond.get("Dual_full") or {}).get("verdict")
        if gold and dv:
            if dv == Verdict.REVIEW.value:
                mg["Dual_full"] = None
            else:
                mg["Dual_full"] = dv == gold
        new_row["match_gold"] = mg
        rows_out.append(new_row)

    # score Dual_full under AUTO contract
    classified = []
    for row in rows_out:
        dual = (row.get("conditions") or {}).get("Dual_full") or {}
        classified.append(
            {
                "id": row.get("id"),
                **classify_case(
                    gold=str(row.get("gold") or ""),
                    verdict=dual.get("verdict"),
                    disposition=dual.get("disposition"),
                ),
            }
        )
    summary = summarize_scores(classified)
    return {
        "source": path.name,
        "model": report.get("model") or rows_out[0].get("model") if rows_out else None,
        "n": len(rows_out),
        "Dual_full": {"summary": summary, "cases": classified},
        "rows": rows_out,
    }


def _write_resim_panel(
    *,
    kind: str,
    note: str,
    reports: tuple[str, ...],
    out_name: str,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    results = []
    for name in reports:
        path = ARTIFACTS_DIR / name
        if not path.is_file():
            results.append({"source": name, "error": "missing"})
            continue
        results.append(resim_report(path))

    panel: dict[str, Any] = {
        "kind": kind,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "note": note,
        "g3_untouched": True,
        "results": [
            {
                "source": r.get("source"),
                "model": r.get("model"),
                "error": r.get("error"),
                "Dual_full_summary": (r.get("Dual_full") or {}).get("summary"),
            }
            for r in results
        ],
        "detail": results,
    }
    if extra:
        panel.update(extra)
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    out = ARTIFACTS_DIR / out_name
    slim = dict(panel)
    slim["detail"] = [
        {
            "source": r.get("source"),
            "model": r.get("model"),
            "error": r.get("error"),
            "Dual_full": r.get("Dual_full"),
        }
        for r in results
        if "error" not in r or not r.get("error")
    ]
    out.write_text(json.dumps(slim, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {out}")
    for r in slim["results"]:
        s = r.get("Dual_full_summary") or {}
        print(
            f"  {r.get('model')}: wrong_AUTO={s.get('wrong_AUTO')} "
            f"correct_AUTO={s.get('correct_AUTO')} REVIEW={s.get('REVIEW')} "
            f"wrong_ids={s.get('wrong_AUTO_ids')}"
        )
    return slim


def run_tracka_temporal_resim() -> dict[str, Any]:
    return _write_resim_panel(
        kind="tracka_temporal_dev_grounding_resim",
        note=(
            "Re-ground C_full/D_full from frozen Phase-7 temporal-dev extractions "
            "after sequenced temporal-change ≠ contradiction. G3 untouched. "
            "Test set not used."
        ),
        reports=TEMPORAL_DEV_REPORTS,
        out_name="tracka-temporal-dev-grounding-resim.json",
        extra={"test_set_not_used": "temporal-family-test.json"},
    )


def run_tracka_temporal_test_resim() -> dict[str, Any]:
    """Re-score sealed temporal-test Dual under current grounding (no LLM, no redesign)."""
    return _write_resim_panel(
        kind="tracka_temporal_test_grounding_resim",
        note=(
            "Score re-open only: re-ground frozen Phase-7 temporal-test extractions "
            "under current ground.py (incl. meta/predicate coverage). "
            "Do not redesign from this result. G3 untouched."
        ),
        reports=TEMPORAL_TEST_REPORTS,
        out_name="tracka-temporal-test-grounding-resim.json",
        extra={
            "held_out": True,
            "redesign_forbidden": True,
            "source_set": "temporal-family-test.json",
        },
    )
