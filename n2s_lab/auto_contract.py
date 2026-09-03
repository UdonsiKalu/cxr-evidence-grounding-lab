"""Track A AUTO contract — classify AUTO vs REVIEW and score wrong_AUTO.

See docs/AUTO-CONTRACT.md. Does not overwrite Phase 1–14 panels.
"""

from __future__ import annotations

import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .paths import ARTIFACTS_DIR, LAB_ROOT
from .types import Verdict

REVIEW = Verdict.REVIEW.value

# Paths we score by default when present on a row's conditions / match_gold.
DEFAULT_PATHS = (
    "Dual_full",
    "D_full",
    "C_full",
    "Dual",
    "D_v",
    "C_v",
    "D_rt",
    "C_rt",
)


def disposition_of(block: dict[str, Any] | None, verdict: str | None = None) -> str:
    """Normalize disposition from a condition block."""
    if not block:
        return "UNKNOWN"
    disp = block.get("disposition")
    v = verdict if verdict is not None else block.get("verdict")
    if disp in ("AUTO", "REVIEW"):
        return str(disp)
    if v == REVIEW:
        return "REVIEW"
    if v in {e.value for e in Verdict if e != Verdict.REVIEW}:
        return "AUTO"
    return "UNKNOWN"


def classify_case(
    *,
    gold: str,
    verdict: str | None,
    disposition: str | None = None,
) -> dict[str, Any]:
    """Bucket one case under the AUTO contract."""
    v = verdict or ""
    disp = disposition or ("REVIEW" if v == REVIEW else "AUTO" if v else "UNKNOWN")
    if disp == "REVIEW" or v == REVIEW:
        bucket = "REVIEW"
        wrong_auto = False
        correct_auto = False
    elif disp == "AUTO":
        match = v == gold
        bucket = "correct_AUTO" if match else "wrong_AUTO"
        wrong_auto = not match
        correct_auto = match
    else:
        bucket = "UNKNOWN"
        wrong_auto = False
        correct_auto = False
    return {
        "gold": gold,
        "verdict": v,
        "disposition": disp,
        "bucket": bucket,
        "wrong_AUTO": wrong_auto,
        "correct_AUTO": correct_auto,
        "is_REVIEW": bucket == "REVIEW",
        "match_gold": (v == gold) if disp == "AUTO" else None,
    }


def apply_gates(
    *,
    verify_ok: bool,
    dual_agreed: bool | None = None,
    contradiction_supported: bool | None = None,
    proposed_verdict: str | None = None,
) -> dict[str, Any]:
    """Decide AUTO vs REVIEW from gate booleans (live path helper)."""
    reasons: list[str] = []
    if not verify_ok:
        reasons.append("G1_verify_fail")
    if dual_agreed is False:
        reasons.append("G2_dual_disagree")
    if (
        proposed_verdict == Verdict.CONTRADICTION.value
        and contradiction_supported is False
    ):
        reasons.append("G3_contradiction_unsupported")
    ok = len(reasons) == 0
    return {
        "ok": ok,
        "disposition": "AUTO" if ok else "REVIEW",
        "reasons": reasons,
        "gates": {
            "G1_verify": verify_ok,
            "G2_dual_agree": dual_agreed,
            "G3_contradiction_supported": contradiction_supported,
        },
    }


def _condition_block(row: dict[str, Any], path: str) -> dict[str, Any] | None:
    cond = row.get("conditions") or {}
    if path in cond and isinstance(cond[path], dict):
        return cond[path]
    # Phase-5 naming variants
    aliases = {
        "C_v": ("C_verified", "C_v"),
        "D_v": ("D_verified", "D_v"),
        "Dual_full": ("Dual_full", "Dual"),
    }
    for name in aliases.get(path, (path,)):
        block = cond.get(name)
        if isinstance(block, dict):
            return block
    return None


def score_row(row: dict[str, Any], path: str) -> dict[str, Any] | None:
    gold = row.get("gold") or row.get("expected")
    if not gold:
        return None
    block = _condition_block(row, path)
    if block is None:
        # Fall back to match_gold / review flags when block missing
        match = (row.get("match_gold") or {}).get(path)
        review = (row.get("review") or {}).get(path)
        if match is None and review is None:
            return None
        if review is True:
            return {
                "id": row.get("id"),
                "path": path,
                **classify_case(gold=str(gold), verdict=REVIEW, disposition="REVIEW"),
                "source": "review_flag",
            }
        if match is True:
            return {
                "id": row.get("id"),
                "path": path,
                **classify_case(
                    gold=str(gold),
                    verdict=str(gold),
                    disposition="AUTO",
                ),
                "source": "match_gold_flag",
            }
        if match is False:
            return {
                "id": row.get("id"),
                "path": path,
                "gold": gold,
                "verdict": "≠gold",
                "disposition": "AUTO",
                "bucket": "wrong_AUTO",
                "wrong_AUTO": True,
                "correct_AUTO": False,
                "is_REVIEW": False,
                "match_gold": False,
                "source": "match_gold_flag",
            }
        return None

    verdict = block.get("verdict")
    disp = disposition_of(block, verdict)
    out = classify_case(gold=str(gold), verdict=verdict, disposition=disp)
    out["id"] = row.get("id")
    out["path"] = path
    out["model"] = row.get("model")
    out["source"] = "condition_block"
    return out


def summarize_scores(rows: list[dict[str, Any]]) -> dict[str, Any]:
    n = len(rows)
    counts = Counter(r["bucket"] for r in rows)
    wrong = counts.get("wrong_AUTO", 0)
    correct = counts.get("correct_AUTO", 0)
    review = counts.get("REVIEW", 0)
    auto_n = wrong + correct
    return {
        "n": n,
        "wrong_AUTO": wrong,
        "correct_AUTO": correct,
        "REVIEW": review,
        "UNKNOWN": counts.get("UNKNOWN", 0),
        "wrong_AUTO_rate": (wrong / n) if n else None,
        "REVIEW_rate": (review / n) if n else None,
        "safety_among_auto": (correct / auto_n) if auto_n else None,
        "coverage_auto": (auto_n / n) if n else None,
        "wrong_AUTO_ids": [r["id"] for r in rows if r.get("wrong_AUTO")],
        "REVIEW_ids": [r["id"] for r in rows if r.get("is_REVIEW")],
    }


def score_report(
    report: dict[str, Any],
    *,
    paths: tuple[str, ...] = DEFAULT_PATHS,
) -> dict[str, Any]:
    """Score a single-model experiment report (has rows[])."""
    rows_in = report.get("rows") or []
    by_path: dict[str, list[dict[str, Any]]] = {}
    for path in paths:
        scored: list[dict[str, Any]] = []
        for row in rows_in:
            s = score_row(row, path)
            if s is not None:
                scored.append(s)
        if scored:
            by_path[path] = scored
    return {
        "model": report.get("model"),
        "paths": {p: {"rows": by_path[p], "summary": summarize_scores(by_path[p])} for p in by_path},
    }


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def resolve_artifact_entry(entry: dict[str, Any], panel_dir: Path) -> Path | None:
    art = entry.get("artifact")
    if not art:
        return None
    p = Path(art)
    if p.is_file():
        return p
    name = p.name
    cand = panel_dir / name
    if cand.is_file():
        return cand
    cand2 = ARTIFACTS_DIR / name
    return cand2 if cand2.is_file() else None


def score_panel_or_report(
    path: Path,
    *,
    paths: tuple[str, ...] = DEFAULT_PATHS,
) -> dict[str, Any]:
    data = load_json(path)
    if isinstance(data.get("models"), list):
        models_out = []
        for entry in data["models"]:
            art = resolve_artifact_entry(entry, path.parent)
            if art is None:
                models_out.append(
                    {
                        "model": entry.get("model"),
                        "error": f"missing artifact {entry.get('artifact')}",
                    }
                )
                continue
            report = load_json(art)
            scored = score_report(report, paths=paths)
            scored["artifact"] = str(art.relative_to(LAB_ROOT)) if art.is_relative_to(LAB_ROOT) else str(art)
            models_out.append(scored)
        return {
            "kind": "panel",
            "source": str(path.relative_to(LAB_ROOT)) if path.is_relative_to(LAB_ROOT) else str(path),
            "models": models_out,
        }
    scored = score_report(data, paths=paths)
    scored["kind"] = "report"
    scored["source"] = str(path.relative_to(LAB_ROOT)) if path.is_relative_to(LAB_ROOT) else str(path)
    return scored


ARTIFACT_PRESETS: dict[str, list[str]] = {
    "phase5": ["phase5-verify-panel.json"],
    "phase6": ["phase6-faithfulness-panel.json"],
    "phase7": ["phase7-bmtcart-panel.json"],
    "phase7-onc": ["phase7-phase4-panel.json"] if False else [],  # may not exist
}


def score_preset(name: str, *, paths: tuple[str, ...] | None = None) -> dict[str, Any]:
    use_paths = paths or DEFAULT_PATHS
    if name == "phase7":
        files = ["phase7-bmtcart-panel.json"]
    elif name == "phase6":
        files = ["phase6-faithfulness-panel.json"]
    elif name == "phase5":
        files = ["phase5-verify-panel.json"]
    else:
        raise ValueError(f"unknown preset: {name}")

    results = []
    for fname in files:
        p = ARTIFACTS_DIR / fname
        if not p.is_file():
            results.append({"error": f"missing {fname}"})
            continue
        results.append(score_panel_or_report(p, paths=use_paths))
    return {
        "contract": "docs/AUTO-CONTRACT.md",
        "preset": name,
        "scored_at": datetime.now(timezone.utc).isoformat(),
        "primary_objective": "wrong_AUTO → 0",
        "results": results,
    }


def write_score_artifact(payload: dict[str, Any], out_name: str) -> Path:
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    out = ARTIFACTS_DIR / out_name
    out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return out


def selftest() -> None:
    # wrong AUTO
    w = classify_case(gold="SATISFIED", verdict="CONTRADICTION", disposition="AUTO")
    assert w["bucket"] == "wrong_AUTO" and w["wrong_AUTO"]
    # correct AUTO
    c = classify_case(gold="SATISFIED", verdict="SATISFIED", disposition="AUTO")
    assert c["bucket"] == "correct_AUTO"
    # REVIEW
    r = classify_case(gold="SATISFIED", verdict=REVIEW, disposition="REVIEW")
    assert r["bucket"] == "REVIEW" and not r["wrong_AUTO"]
    # gates
    g = apply_gates(verify_ok=False, dual_agreed=True)
    assert g["disposition"] == "REVIEW" and "G1_verify_fail" in g["reasons"]
    g2 = apply_gates(verify_ok=True, dual_agreed=True, contradiction_supported=True)
    assert g2["disposition"] == "AUTO"
    # BC_E1-shaped row
    row = {
        "id": "BC_E1",
        "gold": "SATISFIED",
        "conditions": {
            "Dual_full": {"verdict": "CONTRADICTION", "disposition": "AUTO"},
        },
    }
    s = score_row(row, "Dual_full")
    assert s and s["wrong_AUTO"] and s["id"] == "BC_E1"
    print("auto_contract selftest OK")


def load_temporal_family() -> dict[str, Any]:
    path = LAB_ROOT / "data" / "heldout-temporal-family.json"
    return load_json(path)
