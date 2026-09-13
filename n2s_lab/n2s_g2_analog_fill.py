"""Goal 2 — Dual analog extract fill (schema in system, user=evidence).

Frozen analog EXTRACT_SYSTEM asks for “schema in the user message” but analog
puts only the note in user — extracts come out empty. This run keeps
user=evidence (Dual analog) and puts EXTRACT_SCHEMA in the system prompt.

Not Schema+note. Not quote-promote. Not ground() rewrite. Not CAA.
Does not overwrite Batch 0x analog extracts.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from datetime import datetime, timezone
from typing import Any

from .n2s_batch0_phenotype import score_row
from .n2s_batch0x_census import ANALOG_PATH as FROZEN_ANALOG_PATH
from .n2s_note_library import load_dual_library
from .n2s_phase2_translate import atoms_payload
from .neural import EXTRACT_SCHEMA, EXTRACT_SYSTEM
from .paths import ARTIFACTS_DIR
from .types import Extraction

PROMOTE_PATH = ARTIFACTS_DIR / "n2s-g2-analog-fill-promote-panel.json"
EXTRACT_PATH = ARTIFACTS_DIR / "n2s-g2-analog-fill-extracts.json"
OUT_PATH = ARTIFACTS_DIR / "n2s-g2-analog-fill-census.json"

FILL_SYSTEM = EXTRACT_SYSTEM.replace(
    "Return JSON only, matching the schema in the user message.",
    "Return JSON only, matching this schema:\n" + EXTRACT_SCHEMA,
)


def recover_fill(*, model_id: str = "") -> dict[str, Any]:
    from .hf_client import unload_model
    from .hf_intervene import generate_intervened
    from .ollama_client import parse_json_object
    from .phase10_activation import FAIL_MODEL

    mid = model_id or FAIL_MODEL
    notes = load_dual_library(include_heldout=True)
    extracts: dict[str, Any] = {}
    if EXTRACT_PATH.is_file():
        saved = json.loads(EXTRACT_PATH.read_text(encoding="utf-8"))
        if saved.get("kind") == "n2s_g2_analog_fill_extracts_v1":
            extracts = dict(saved.get("extracts") or {})
    blob = {
        "ok": True,
        "kind": "n2s_g2_analog_fill_extracts_v1",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "model_id": mid,
        "prompt": "analog_schema_in_system",
        "user": "evidence",
        "extracts": extracts,
    }

    def _persist() -> None:
        EXTRACT_PATH.write_text(json.dumps(blob, indent=2) + "\n", encoding="utf-8")

    try:
        for note in notes:
            nid = note["id"]
            if nid in extracts and "parse_ok" in extracts[nid]:
                print(f"[g2] skip {nid}", flush=True)
                continue
            print(f"[g2] analog-fill extract {nid}", flush=True)
            trace, _iv = generate_intervened(
                system=FILL_SYSTEM,
                user=note["evidence"],
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
                "prompt": "analog_schema_in_system",
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
            }
            _persist()
    finally:
        unload_model()
        _persist()
    return {"model_id": mid, "extracts": extracts}


def _counts(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "n": len(rows),
        "parse_ok": sum(1 for r in rows if r["parse_ok"]),
        "thin_extract": sum(1 for r in rows if r["thin_extract"]),
        "p_bind": sum(1 for r in rows if "P-bind" in r["phenotypes"]),
        "p_unk": sum(1 for r in rows if "P-unk" in r["phenotypes"]),
        "wrong_AUTO": sum(1 for r in rows if r["wrong_AUTO"]),
        "match_gold": sum(1 for r in rows if r["match_gold"] is True),
        "phenotypes": dict(Counter(p for r in rows for p in r["phenotypes"])),
    }


def build_census(*, extracts: dict[str, Any], model_id: str) -> dict[str, Any]:
    notes = load_dual_library(include_heldout=True)
    frozen = {}
    if FROZEN_ANALOG_PATH.is_file():
        frozen = json.loads(FROZEN_ANALOG_PATH.read_text(encoding="utf-8")).get(
            "extracts"
        ) or {}
    fill_rows = []
    frozen_rows = []
    for note in notes:
        rec = extracts.get(note["id"]) or {}
        row = score_row(note, extract_rec=rec)
        row["prompt"] = "analog_schema_in_system"
        row["slice"] = note["slice"]
        fill_rows.append(row)
        fr = score_row(note, extract_rec=frozen.get(note["id"]) or {})
        fr["prompt"] = "analog"
        frozen_rows.append(fr)
    design_ids = {n["id"] for n in notes if n["phenotype_design"]}
    fill_c = _counts(fill_rows)
    frozen_c = _counts(frozen_rows)
    fill_design = _counts([r for r in fill_rows if r["id"] in design_ids])
    frozen_design = _counts([r for r in frozen_rows if r["id"] in design_ids])
    thin_drop = frozen_c["thin_extract"] - fill_c["thin_extract"]
    panel = {
        "ok": True,
        "kind": "n2s_g2_analog_fill_census_v1",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "goal": "Dual analog extracts contain note evidence (user=evidence, schema in system)",
        "model_id": model_id,
        "n": len(fill_rows),
        "not": [
            "Schema+note user prompt",
            "quote-promote",
            "ground() rewrite",
            "CAA",
            "frozen-d",
            "Batch 0x overwrite",
        ],
        "counts": {
            "frozen_analog": frozen_c,
            "fill": fill_c,
            "frozen_analog_design": frozen_design,
            "fill_design": fill_design,
        },
        "thin_drop": thin_drop,
        "fill_worked": fill_c["thin_extract"] < frozen_c["thin_extract"],
        "rows": fill_rows,
        "note": (
            "Compare frozen analog (schema promised in user, not sent) vs "
            "schema-in-system. Still Dual analog: user=evidence only."
        ),
    }
    OUT_PATH.write_text(json.dumps(panel, indent=2) + "\n", encoding="utf-8")
    return panel


def build_promote() -> dict[str, Any]:
    from .n2s_batch1_quote_promote import (
        BIND_IDS,
        NEG_IDS,
        XSPAN_IDS,
        _arm_counts,
        apply_row,
    )

    if not EXTRACT_PATH.is_file():
        raise SystemExit("run --recover-hf first")
    extracts = json.loads(EXTRACT_PATH.read_text(encoding="utf-8")).get("extracts") or {}
    notes = {n["id"]: n for n in load_dual_library(include_heldout=True)}
    design_ids = list(BIND_IDS) + list(XSPAN_IDS) + list(NEG_IDS)
    rows = [
        apply_row(
            notes[i],
            extracts.get(i) or {},
            prompt="analog_schema_in_system",
        )
        for i in design_ids
    ]
    counts = _arm_counts(rows)
    bind = counts.get("P-bind") or {}
    xspan = counts.get("P-xspan") or {}
    neg = counts.get("P-neg") or {}
    panel = {
        "ok": True,
        "kind": "n2s_g2_analog_fill_promote_v1",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "intervention": "failure_evidence_from_extract_text",
        "extracts": str(EXTRACT_PATH.name),
        "prompt": "analog_schema_in_system",
        "n_design": len(design_ids),
        "counts": counts,
        "map_cell": {
            "pbind_k": bind.get("repaired", 0),
            "pbind_n": bind.get("n", 0),
            "xspan_collateral": xspan.get("collateral_to_SATISFIED", 0),
            "neg_collateral": neg.get("collateral_to_SATISFIED", 0),
            "specificity_holds": (
                xspan.get("collateral_to_SATISFIED", 0) == 0
                and neg.get("collateral_to_SATISFIED", 0) == 0
            ),
            "dual_analog_user_evidence": True,
            "schema_in_system": True,
        },
        "rows": rows,
        "not": ["Schema+note", "ground() rewrite", "CAA", "promote retune"],
    }
    PROMOTE_PATH.write_text(json.dumps(panel, indent=2) + "\n", encoding="utf-8")
    return panel


def selftest() -> None:
    assert "stated_line" in FILL_SYSTEM
    assert "matching the schema in the user message" not in FILL_SYSTEM
    assert FILL_SYSTEM != EXTRACT_SYSTEM
    notes = load_dual_library(include_heldout=True)
    assert len(notes) == 108


def main() -> None:
    parser = argparse.ArgumentParser(description="Goal 2 Dual analog extract fill")
    parser.add_argument("--selftest", action="store_true")
    parser.add_argument("--recover-hf", action="store_true")
    parser.add_argument("--census", action="store_true")
    parser.add_argument("--promote", action="store_true")
    parser.add_argument("--model-id", default="")
    args = parser.parse_args()
    if args.selftest:
        selftest()
        print("selftest ok")
        return
    extracts = None
    model_id = args.model_id
    if args.recover_hf:
        blob = recover_fill(model_id=args.model_id)
        extracts = blob["extracts"]
        model_id = blob["model_id"]
    elif args.census and EXTRACT_PATH.is_file():
        saved = json.loads(EXTRACT_PATH.read_text(encoding="utf-8"))
        extracts = saved.get("extracts")
        model_id = model_id or saved.get("model_id") or ""
    if args.census or args.recover_hf:
        if not extracts:
            raise SystemExit("need --recover-hf")
        panel = build_census(extracts=extracts, model_id=model_id or "unknown")
        print(
            json.dumps(
                {
                    "thin_drop": panel["thin_drop"],
                    "fill_worked": panel["fill_worked"],
                    "counts": panel["counts"],
                    "artifact": str(OUT_PATH),
                },
                indent=2,
            )
        )
        return
    if args.promote:
        panel = build_promote()
        print(
            json.dumps(
                {
                    "counts": panel["counts"],
                    "map_cell": panel["map_cell"],
                    "artifact": str(PROMOTE_PATH),
                },
                indent=2,
            )
        )
        return
    parser.print_help()


if __name__ == "__main__":
    main()
