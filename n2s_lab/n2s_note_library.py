"""Dual-shaped note library for Batch 0x expand (no new synthesis).

Unions existing lab data/*.json. Does not invent notes, open
temporal-family-test for phenotype design, or touch claim_analysis_tools.
"""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any

from .paths import (
    ARTIFACTS_DIR,
    DATA_PATH,
    HELDOUT_BMTCART_PHASE7_PATH,
    HELDOUT_PATH,
    HELDOUT_PHASE4_PATH,
    HELDOUT_PHASE11_PATH,
    HELDOUT_PHASE12B_PATH,
    HELDOUT_PHASE14_PATH,
    HELDOUT_UA_PARAPHRASE_PATH,
    TEMPORAL_FAMILY_DEV_PATH,
    TEMPORAL_FAMILY_EXPAND_PATH,
    TEMPORAL_FAMILY_TEST_PATH,
)

MANIFEST_PATH = ARTIFACTS_DIR / "n2s-batch0x-library.json"
GOLD_VOCAB = frozenset({"SATISFIED", "NOT_SATISFIED", "CONTRADICTION", "UNCERTAIN"})
CLASS_TO_CAT = {"temporal": "temporal", "contra": "conflicting"}

# Design-safe Dual notes. Frozen TFT excluded from all slices.
DESIGN_SOURCES: tuple[tuple[str, Path], ...] = (
    ("cases", DATA_PATH),
    ("temporal-family-dev", TEMPORAL_FAMILY_DEV_PATH),
    ("temporal-family-dev-expand", TEMPORAL_FAMILY_EXPAND_PATH),
)
HELDOUT_SOURCES: tuple[tuple[str, Path], ...] = (
    ("heldout-phase3", HELDOUT_PATH),
    ("heldout-phase4", HELDOUT_PHASE4_PATH),
    ("heldout-bmtcart-phase7", HELDOUT_BMTCART_PHASE7_PATH),
    ("heldout-phase11", HELDOUT_PHASE11_PATH),
    ("heldout-phase12b", HELDOUT_PHASE12B_PATH),
    ("heldout-phase14", HELDOUT_PHASE14_PATH),
    ("heldout-ua-paraphrase", HELDOUT_UA_PARAPHRASE_PATH),
)


def _rows(path: Path) -> list[dict[str, Any]]:
    blob = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(blob.get("cases"), list):
        return list(blob["cases"])
    if isinstance(blob.get("notes"), list):
        return list(blob["notes"])
    raise ValueError(f"no cases/notes in {path}")


def _evidence_hash(text: str) -> str:
    return hashlib.md5((text or "").encode("utf-8")).hexdigest()


def normalize_note(
    raw: dict[str, Any],
    *,
    source: str,
    slice_name: str,
    phenotype_design: bool,
) -> dict[str, Any] | None:
    nid = str(raw.get("id") or "").strip()
    evidence = str(raw.get("evidence") or "").strip()
    gold = str(raw.get("expected") or raw.get("gold") or "").strip()
    if not nid or not evidence or gold not in GOLD_VOCAB:
        return None
    cat = str(raw.get("category") or "").strip()
    if not cat:
        cat = CLASS_TO_CAT.get(str(raw.get("class") or "").strip(), "")
    return {
        "id": nid,
        "evidence": evidence,
        "expected": gold,
        "category": cat or "unspecified",
        "why": str(raw.get("why") or raw.get("role") or ""),
        "subtype": raw.get("subtype"),
        "role": raw.get("role"),
        "source": source,
        "slice": slice_name,
        "phenotype_design": phenotype_design,
        "evidence_md5": _evidence_hash(evidence),
        "analog_of": raw.get("analog_of"),
    }


def load_dual_library(*, include_heldout: bool = True) -> list[dict[str, Any]]:
    """Unique Dual notes. First-seen wins (design sources first). TFT excluded."""
    seen_ids: set[str] = set()
    seen_text: set[str] = set()
    out: list[dict[str, Any]] = []

    def _add(source: str, path: Path, slice_name: str, design: bool) -> None:
        for raw in _rows(path):
            note = normalize_note(
                raw, source=source, slice_name=slice_name, phenotype_design=design
            )
            if note is None:
                continue
            if note["id"] in seen_ids or note["evidence_md5"] in seen_text:
                continue
            seen_ids.add(note["id"])
            seen_text.add(note["evidence_md5"])
            out.append(note)

    for source, path in DESIGN_SOURCES:
        _add(source, path, "design", True)
    if include_heldout:
        for source, path in HELDOUT_SOURCES:
            _add(source, path, "heldout_score_only", False)
    return out


def tft_excluded_count() -> int:
    return len(_rows(TEMPORAL_FAMILY_TEST_PATH))


def build_manifest(*, include_heldout: bool = True) -> dict[str, Any]:
    notes = load_dual_library(include_heldout=include_heldout)
    gold = Counter(n["expected"] for n in notes)
    cats = Counter(n["category"] for n in notes)
    slices = Counter(n["slice"] for n in notes)
    sources = Counter(n["source"] for n in notes)
    panel = {
        "ok": True,
        "kind": "n2s_batch0x_library_v1",
        "protocol": "docs/TRACKB-FAILURE-INTERVENTION-MAP.md",
        "not": [
            "new note synthesis",
            "temporal-family-test phenotype design",
            "oncology jsonl relabel",
            "claim_analysis_tools",
            "intervention",
            "ground() rewrite",
        ],
        "n": len(notes),
        "n_design": slices.get("design", 0),
        "n_heldout_score_only": slices.get("heldout_score_only", 0),
        "n_tft_excluded": tft_excluded_count(),
        "honest_cap": (
            "Existing Dual-shaped lab library is ~120 unique / ~64 design-safe. "
            "Not 100–300. This expand uses all Dual notes except frozen TFT."
        ),
        "counts": {
            "gold": dict(gold),
            "category": dict(cats),
            "slice": dict(slices),
            "source": dict(sources),
        },
        "ids": [n["id"] for n in notes],
        "notes": [
            {k: n[k] for k in n if k != "evidence"}
            for n in notes
        ],
    }
    MANIFEST_PATH.write_text(json.dumps(panel, indent=2) + "\n", encoding="utf-8")
    return panel
