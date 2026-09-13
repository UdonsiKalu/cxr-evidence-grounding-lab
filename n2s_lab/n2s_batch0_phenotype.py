"""Batch 0 — Dual analog phenotype census (no intervention).

7B analog extract (user=evidence, intervention=none) on data/cases.json.
Does not patch hidden states, rewrite ground(), rescore Dual_full, or touch :8260.

See docs/TRACKB-FAILURE-INTERVENTION-MAP.md.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from datetime import datetime, timezone
from typing import Any

from .auto_contract import classify_case
from .n2s_phase2_translate import FAILURE_MARKERS, atoms_payload
from .paths import ARTIFACTS_DIR, DATA_PATH
from .types import Extraction

EXTRACT_PATH = ARTIFACTS_DIR / "n2s-batch0-analog-extracts.json"
OUT_PATH = ARTIFACTS_DIR / "n2s-batch0-phenotype-census.json"
LOCATOR_INCOMPLETE = frozenset({"T1", "T3", "C1", "C4"})


def load_coding_cases() -> list[dict[str, Any]]:
    blob = json.loads(DATA_PATH.read_text(encoding="utf-8"))
    return list(blob.get("cases") or [])


def _has_failure_marker(text: str) -> bool:
    blob = (text or "").lower()
    return any(m in blob for m in FAILURE_MARKERS)


def _thin_extract(ex: Extraction | None) -> bool:
    if ex is None:
        return True
    return not bool(
        ex.quotes or ex.outcome_statements or ex.regimen_names or ex.contradiction_present
    )


def propose_phenotypes(
    case: dict[str, Any],
    *,
    ex: Extraction | None,
    grounded: dict[str, Any] | None,
) -> list[str]:
    """Proposed P-* tags. Gold/category first; extract overlays when parse_ok."""
    nid = str(case.get("id") or "")
    gold = str(case.get("expected") or "")
    cat = str(case.get("category") or "")
    tags: list[str] = []

    if gold == "CONTRADICTION" or cat == "conflicting":
        tags.append("P-xspan")
    if gold == "UNCERTAIN" or cat == "uncertain":
        tags.append("P-unc")
    if nid == "I4" or "adjuvant" in (case.get("why") or "").lower():
        tags.append("P-adj")
    if cat == "implicit" and gold == "SATISFIED":
        tags.append("P-impl")
    if cat == "temporal" and gold == "SATISFIED":
        tags.append("P-seq")
    if gold == "NOT_SATISFIED" and nid != "I4":
        tags.append("P-neg")
    if cat == "explicit" and gold == "SATISFIED":
        tags.append("P-expl")

    if ex is None or grounded is None:
        tags.append("P-unk")
        return list(dict.fromkeys(tags))

    if _thin_extract(ex):
        tags.append("P-unk")

    pols = [(o.polarity or "").lower() for o in ex.outcome_statements]
    has_fail_pol = any(p == "failure" for p in pols)
    quote_fail = any(_has_failure_marker(q) for q in ex.quotes) or _has_failure_marker(
        ex.notes
    )
    atoms = grounded.get("atoms") or {}
    c_fail = str(atoms.get("C_failure_event") or "")
    if (
        gold == "SATISFIED"
        and quote_fail
        and not has_fail_pol
        and c_fail in {"false", "unknown"}
    ):
        tags.append("P-bind")

    cues = {c.lower() for c in ex.implicit_cues}
    if (
        ex.contradiction_present
        and gold != "CONTRADICTION"
        and atoms.get("X_contradiction") is False
        and "response_then_progression" in cues
    ):
        tags.append("P-recon")

    return list(dict.fromkeys(tags))


def _probes(nid: str) -> dict[str, str]:
    if nid in LOCATOR_INCOMPLETE:
        return {
            "enc_L8": "unknown",
            "cmp_lost": "unknown",
            "trn_bucket": "unknown",
            "route_8260": "incomplete",
        }
    return {
        "enc_L8": "unknown",
        "cmp_lost": "unknown",
        "trn_bucket": "pending",
        "route_8260": "none",
    }


def score_row(case: dict[str, Any], *, extract_rec: dict[str, Any]) -> dict[str, Any]:
    gold = str(case.get("expected") or "")
    nid = str(case.get("id") or "")
    ex = None
    if extract_rec.get("parse_ok") and extract_rec.get("extract"):
        ex = Extraction.from_dict(extract_rec["extract"])
    grounded = extract_rec.get("grounded")
    if grounded is None and ex is not None:
        grounded = {
            k: atoms_payload(ex)[k]
            for k in ("atoms", "verdict", "ground_trace")
        }
    verdict = (grounded or {}).get("verdict")
    classified = classify_case(
        gold=gold,
        verdict=verdict,
        disposition="AUTO" if verdict else "UNKNOWN",
    )
    phenotypes = propose_phenotypes(case, ex=ex, grounded=grounded)
    probes = _probes(nid)
    if classified["bucket"] != "UNKNOWN":
        probes["trn_bucket"] = classified["bucket"]
    return {
        "id": nid,
        "gold": gold,
        "category": case.get("category"),
        "why": case.get("why"),
        "parse_ok": bool(extract_rec.get("parse_ok")),
        "thin_extract": _thin_extract(ex),
        "verdict": verdict,
        "disposition": classified["disposition"],
        "bucket": classified["bucket"],
        "wrong_AUTO": classified["wrong_AUTO"],
        "match_gold": classified["match_gold"],
        "atoms": None if grounded is None else grounded.get("atoms"),
        "phenotypes": phenotypes,
        "probes": probes,
        "hidden_state_patched": False,
        "intervention": "none",
        "prompt": "analog",
    }


def recover_coding_analog(*, model_id: str = "") -> dict[str, Any]:
    from .hf_client import unload_model
    from .hf_intervene import generate_intervened
    from .neural import EXTRACT_SYSTEM
    from .ollama_client import parse_json_object
    from .phase10_activation import FAIL_MODEL

    mid = model_id or FAIL_MODEL
    cases = load_coding_cases()
    extracts: dict[str, Any] = {}
    blob = {
        "ok": True,
        "kind": "n2s_batch0_analog_extracts_v1",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "model_id": mid,
        "prompt": "analog",
        "extracts": extracts,
    }

    def _persist() -> None:
        EXTRACT_PATH.write_text(json.dumps(blob, indent=2) + "\n", encoding="utf-8")

    try:
        for case in cases:
            nid = case["id"]
            print(f"[batch0] analog extract {nid}", flush=True)
            trace, _iv = generate_intervened(
                system=EXTRACT_SYSTEM,
                user=case["evidence"],
                model_id=mid,
                max_new_tokens=400,
                intervention="none",
                layer_indices=(12,),
            )
            gen_only = "".join(s.token_text for s in trace.steps)
            try:
                data = parse_json_object(gen_only)
                ex = Extraction.from_dict(data)
                parse_ok = True
            except ValueError:
                ex = None
                parse_ok = False
            scored = atoms_payload(ex) if ex is not None else None
            extracts[nid] = {
                "id": nid,
                "gold": case["expected"],
                "parse_ok": parse_ok,
                "prompt": "analog",
                "extract": None if ex is None else ex.to_dict(),
                "grounded": None
                if scored is None
                else {
                    "atoms": scored["atoms"],
                    "verdict": scored["verdict"],
                    "ground_trace": scored["ground_trace"],
                },
                "backend": f"hf:{mid}",
                "intervention": "none",
            }
            _persist()
    finally:
        unload_model()
        _persist()
    return {"model_id": mid, "extracts": extracts}


def build_census(*, extracts: dict[str, Any], model_id: str) -> dict[str, Any]:
    cases = load_coding_cases()
    rows: list[dict[str, Any]] = []
    for case in cases:
        rec = extracts.get(case["id"]) or {}
        rows.append(score_row(case, extract_rec=rec))
    pheno = Counter(p for r in rows for p in r["phenotypes"])
    buckets = Counter(r["bucket"] for r in rows)
    panel = {
        "ok": True,
        "kind": "n2s_batch0_phenotype_census_v1",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "protocol": "docs/TRACKB-FAILURE-INTERVENTION-MAP.md",
        "set": "data/cases.json",
        "n": len(rows),
        "prompt": "analog user=evidence",
        "model_id": model_id,
        "intervention": "none",
        "not": [
            "hidden-state patch",
            "frozen-d",
            "ground() rewrite",
            "Dual_full rescore",
            "quote-promote",
            "auto-correct",
            ":8260 GUI",
        ],
        "rows": rows,
        "counts": {
            "phenotypes": dict(pheno),
            "buckets": dict(buckets),
            "parse_ok": sum(1 for r in rows if r["parse_ok"]),
            "thin_extract": sum(1 for r in rows if r["thin_extract"]),
            "wrong_AUTO": sum(1 for r in rows if r["wrong_AUTO"]),
            "match_gold": sum(1 for r in rows if r["match_gold"] is True),
        },
        "note": (
            "P-* are proposed tags (gold/category + extract overlays). "
            "Analog prompt is Dual-matched (user=evidence); thin extracts are P-unk."
        ),
    }
    OUT_PATH.write_text(json.dumps(panel, indent=2) + "\n", encoding="utf-8")
    return panel


def selftest() -> None:
    folfox = {
        "id": "E1",
        "category": "explicit",
        "expected": "SATISFIED",
        "why": "Explicit first-line regimen plus explicit progression.",
    }
    ex = Extraction.from_dict(
        {
            "stated_line": "first",
            "regimen_names": ["FOLFOX"],
            "administration_status": "given",
            "outcome_statements": [
                {"text": "partial response", "polarity": "response"}
            ],
            "quotes": ["FOLFOX was discontinued for treatment failure"],
            "implicit_cues": ["response_then_progression"],
            "contradiction_present": False,
        }
    )
    g = atoms_payload(ex)
    tags = propose_phenotypes(folfox, ex=ex, grounded=g)
    assert "P-expl" in tags, tags
    assert "P-bind" in tags, tags
    assert "P-unk" not in tags, tags
    c1 = {
        "id": "C1",
        "category": "conflicting",
        "expected": "CONTRADICTION",
        "why": "Assessment vs addendum",
    }
    assert "P-xspan" in propose_phenotypes(c1, ex=None, grounded=None)
    i4 = {
        "id": "I4",
        "category": "implicit",
        "expected": "NOT_SATISFIED",
        "why": "Completed adjuvant therapy with NED",
    }
    assert "P-adj" in propose_phenotypes(i4, ex=None, grounded=None)


def main() -> None:
    parser = argparse.ArgumentParser(description="Batch 0 phenotype census")
    parser.add_argument("--selftest", action="store_true")
    parser.add_argument("--recover-hf", action="store_true")
    parser.add_argument("--census", action="store_true")
    parser.add_argument("--model-id", default="")
    args = parser.parse_args()
    if args.selftest:
        selftest()
        print("selftest ok")
        return
    extracts: dict[str, Any] | None = None
    model_id = args.model_id
    if args.recover_hf:
        blob = recover_coding_analog(model_id=args.model_id)
        extracts = blob["extracts"]
        model_id = blob["model_id"]
        print(json.dumps({k: v.get("parse_ok") for k, v in extracts.items()}))
    elif args.census and EXTRACT_PATH.is_file():
        saved = json.loads(EXTRACT_PATH.read_text(encoding="utf-8"))
        extracts = saved.get("extracts") or {}
        model_id = model_id or saved.get("model_id") or ""
    if args.census or args.recover_hf:
        if extracts is None:
            raise SystemExit("run --recover-hf (no saved extracts yet)")
        panel = build_census(extracts=extracts, model_id=model_id or "unknown")
        print(json.dumps({"counts": panel["counts"], "artifact": str(OUT_PATH)}, indent=2))
        return
    parser.print_help()


if __name__ == "__main__":
    main()
