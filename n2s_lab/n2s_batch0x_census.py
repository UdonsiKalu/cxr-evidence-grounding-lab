"""Batch 0x — expanded Dual analog + Schema census (no intervention).

Same frozen Batch 0 protocol on the existing Dual-shaped library.
Does not patch hidden states, rewrite ground(), rescore Dual_full, or touch :8260.
Does not overwrite Batch 0 n=20 artifacts.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from datetime import datetime, timezone
from typing import Any

from .n2s_batch0_phenotype import score_row
from .n2s_note_library import build_manifest, load_dual_library
from .n2s_phase2_translate import atoms_payload
from .neural import EXTRACT_SCHEMA, EXTRACT_SYSTEM
from .paths import ARTIFACTS_DIR
from .types import Extraction

ANALOG_PATH = ARTIFACTS_DIR / "n2s-batch0x-analog-extracts.json"
SCHEMA_PATH = ARTIFACTS_DIR / "n2s-batch0x-schema-extracts.json"
OUT_PATH = ARTIFACTS_DIR / "n2s-batch0x-census.json"
PROMPTS = ("analog", "schema")


def _user_text(evidence: str, prompt: str) -> str:
    if prompt == "analog":
        return evidence
    if prompt == "schema":
        return f"Schema:\n{EXTRACT_SCHEMA}\n\nNote:\n{evidence}"
    raise ValueError(prompt)


def _extract_dest(prompt: str) -> Any:
    return ANALOG_PATH if prompt == "analog" else SCHEMA_PATH


def recover_prompt(
    notes: list[dict[str, Any]],
    *,
    prompt: str,
    model_id: str,
    unload: bool = True,
) -> dict[str, Any]:
    from .hf_client import unload_model
    from .hf_intervene import generate_intervened
    from .ollama_client import parse_json_object
    from .phase10_activation import FAIL_MODEL

    mid = model_id or FAIL_MODEL
    dest = _extract_dest(prompt)
    extracts: dict[str, Any] = {}
    if dest.is_file():
        saved = json.loads(dest.read_text(encoding="utf-8"))
        if saved.get("prompt") == prompt:
            extracts = dict(saved.get("extracts") or {})
    blob = {
        "ok": True,
        "kind": f"n2s_batch0x_{prompt}_extracts_v1",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "model_id": mid,
        "prompt": prompt,
        "extracts": extracts,
    }

    def _persist() -> None:
        dest.write_text(json.dumps(blob, indent=2) + "\n", encoding="utf-8")

    try:
        for note in notes:
            nid = note["id"]
            if nid in extracts and "parse_ok" in extracts[nid]:
                print(f"[batch0x] skip {prompt} {nid}", flush=True)
                continue
            print(f"[batch0x] {prompt} extract {nid}", flush=True)
            trace, _iv = generate_intervened(
                system=EXTRACT_SYSTEM,
                user=_user_text(note["evidence"], prompt),
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
                "gold": note["expected"],
                "parse_ok": parse_ok,
                "prompt": prompt,
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
                "slice": note["slice"],
                "source": note["source"],
            }
            _persist()
    finally:
        if unload:
            unload_model()
        _persist()
    return {"model_id": mid, "extracts": extracts, "prompt": prompt}


def _score_prompt(
    notes: list[dict[str, Any]],
    extracts: dict[str, Any],
    *,
    prompt: str,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    by_id = {n["id"]: n for n in notes}
    for note in notes:
        rec = extracts.get(note["id"]) or {}
        row = score_row(note, extract_rec=rec)
        row["prompt"] = prompt
        row["slice"] = note["slice"]
        row["source"] = note["source"]
        row["phenotype_design"] = note["phenotype_design"]
        rows.append(row)
    missing = [nid for nid in by_id if nid not in extracts]
    if missing:
        print(f"[batch0x] missing {prompt} extracts: {missing[:8]}…", flush=True)
    return rows


def _counts(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "n": len(rows),
        "phenotypes": dict(Counter(p for r in rows for p in r["phenotypes"])),
        "buckets": dict(Counter(r["bucket"] for r in rows)),
        "parse_ok": sum(1 for r in rows if r["parse_ok"]),
        "thin_extract": sum(1 for r in rows if r["thin_extract"]),
        "wrong_AUTO": sum(1 for r in rows if r["wrong_AUTO"]),
        "match_gold": sum(1 for r in rows if r["match_gold"] is True),
        "p_bind": sum(1 for r in rows if "P-bind" in r["phenotypes"]),
        "slice": dict(Counter(r["slice"] for r in rows)),
    }


def build_census(
    *,
    notes: list[dict[str, Any]],
    analog: dict[str, Any],
    schema: dict[str, Any],
    model_id: str,
) -> dict[str, Any]:
    analog_rows = _score_prompt(notes, analog, prompt="analog")
    schema_rows = _score_prompt(notes, schema, prompt="schema")
    design_ids = {n["id"] for n in notes if n["phenotype_design"]}
    panel = {
        "ok": True,
        "kind": "n2s_batch0x_census_v1",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "protocol": "docs/TRACKB-FAILURE-INTERVENTION-MAP.md",
        "model_id": model_id,
        "intervention": "none",
        "n": len(notes),
        "n_design": sum(1 for n in notes if n["phenotype_design"]),
        "n_heldout_score_only": sum(1 for n in notes if not n["phenotype_design"]),
        "not": [
            "hidden-state patch",
            "frozen-d",
            "ground() rewrite",
            "Dual_full rescore",
            "quote-promote",
            "auto-correct",
            ":8260 GUI",
            "new note synthesis",
            "temporal-family-test",
        ],
        "counts": {
            "analog": _counts(analog_rows),
            "schema": _counts(schema_rows),
            "analog_design": _counts([r for r in analog_rows if r["id"] in design_ids]),
            "schema_design": _counts([r for r in schema_rows if r["id"] in design_ids]),
        },
        "rows": {
            "analog": analog_rows,
            "schema": schema_rows,
        },
        "note": (
            "Same notes, analog vs Schema. Held-out rows are score-only — do not "
            "invent new P-* from them. Frozen TFT excluded. P-bind needs extract text."
        ),
    }
    OUT_PATH.write_text(json.dumps(panel, indent=2) + "\n", encoding="utf-8")
    return panel


def selftest() -> None:
    notes = load_dual_library(include_heldout=True)
    ids = [n["id"] for n in notes]
    assert len(ids) == len(set(ids)), "duplicate ids"
    assert all(n["id"] != "TFT_E1" for n in notes)
    design = [n for n in notes if n["phenotype_design"]]
    assert len(design) == 64, len(design)
    assert any(n["id"] == "E1" for n in design)
    assert any(n["id"] == "TX_E01" for n in design)
    assert any(n["id"] == "EX_TEMPORAL_FOLFOX" for n in notes)
    assert _user_text("note", "schema").startswith("Schema:")
    empty = Extraction.from_dict({})
    rec = {
        "parse_ok": True,
        "extract": empty.to_dict(),
        "grounded": atoms_payload(empty),
    }
    row = score_row(design[0], extract_rec=rec)
    assert "phenotypes" in row
    manifest = build_manifest(include_heldout=True)
    assert manifest["n_design"] == 64
    assert manifest["n_tft_excluded"] == 12


def main() -> None:
    parser = argparse.ArgumentParser(description="Batch 0x analog+schema census")
    parser.add_argument("--selftest", action="store_true")
    parser.add_argument("--manifest", action="store_true")
    parser.add_argument("--recover-hf", action="store_true")
    parser.add_argument("--census", action="store_true")
    parser.add_argument("--design-only", action="store_true")
    parser.add_argument("--model-id", default="")
    args = parser.parse_args()
    if args.selftest:
        selftest()
        print("selftest ok")
        return
    notes = load_dual_library(include_heldout=not args.design_only)
    if args.manifest or args.recover_hf or args.census:
        manifest = build_manifest(include_heldout=not args.design_only)
        print(json.dumps({
            "n": manifest["n"],
            "n_design": manifest["n_design"],
            "n_heldout_score_only": manifest["n_heldout_score_only"],
            "n_tft_excluded": manifest["n_tft_excluded"],
            "honest_cap": manifest["honest_cap"],
        }, indent=2))
    analog: dict[str, Any] | None = None
    schema: dict[str, Any] | None = None
    model_id = args.model_id
    if args.recover_hf:
        from .phase10_activation import FAIL_MODEL

        mid = args.model_id or FAIL_MODEL
        analog = recover_prompt(notes, prompt="analog", model_id=mid, unload=False)[
            "extracts"
        ]
        schema = recover_prompt(notes, prompt="schema", model_id=mid, unload=True)[
            "extracts"
        ]
        model_id = mid
    if args.census or args.recover_hf:
        if analog is None and ANALOG_PATH.is_file():
            analog = json.loads(ANALOG_PATH.read_text(encoding="utf-8")).get("extracts")
            model_id = model_id or json.loads(ANALOG_PATH.read_text(encoding="utf-8")).get(
                "model_id", ""
            )
        if schema is None and SCHEMA_PATH.is_file():
            schema = json.loads(SCHEMA_PATH.read_text(encoding="utf-8")).get("extracts")
        if not analog or not schema:
            raise SystemExit("need analog + schema extracts (--recover-hf)")
        panel = build_census(
            notes=notes, analog=analog, schema=schema, model_id=model_id or "unknown"
        )
        print(json.dumps({"counts": panel["counts"], "artifact": str(OUT_PATH)}, indent=2))
        return
    if args.manifest:
        return
    parser.print_help()


if __name__ == "__main__":
    main()
