#!/usr/bin/env python3
"""Replay the frozen Dual Translate ladder on one note. CPU only.

Does not call a language model. Does not rewrite ground(). Does not add cells.

Usage (from cxr-evidence-grounding-lab):

  ../cxrlabs/faiss_gpu1/bin/python curriculum-mapping/snippets/replay_ladder.py I2
  ../cxrlabs/faiss_gpu1/bin/python curriculum-mapping/snippets/replay_ladder.py C4 T1 T2
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

LAB = Path(__file__).resolve().parents[2]
if str(LAB) not in sys.path:
    sys.path.insert(0, str(LAB))

from n2s_lab.n2s_g2_analog_fill import EXTRACT_PATH  # noqa: E402
from n2s_lab.n2s_note_library import load_dual_library  # noqa: E402
from n2s_lab.n2s_phase2_translate import atoms_payload  # noqa: E402
from n2s_lab.n2s_v1_selection_contrast import (  # noqa: E402
    CELLS,
    _apply,
    _match,
    oracle_cell,
    select_cell,
)
from n2s_lab.types import Extraction  # noqa: E402


def show(note_id: str) -> None:
    notes = {n["id"]: n for n in load_dual_library(include_heldout=True)}
    extracts = json.loads(EXTRACT_PATH.read_text(encoding="utf-8"))["extracts"]
    if note_id not in notes or note_id not in extracts:
        raise SystemExit(f"unknown id {note_id}")
    note = notes[note_id]
    rec = extracts[note_id]
    ex = Extraction.from_dict(rec["extract"])
    gold = note["expected"]
    print(f"\n=== {note_id}  gold={gold}  slice={note['slice']}")
    print((note["evidence"] or "")[:280].replace("\n", " "))
    print(f"selector (signature depth) = {select_cell(ex)}")
    print(f"oracle (shallowest match)  = {oracle_cell(ex, gold)}")
    prev = None
    for name, _fn in CELLS:
        after, _meta = _apply(ex, name)
        payload = atoms_payload(after)
        verdict = payload["verdict"]
        atoms = payload["atoms"]
        match = _match(gold, after)
        admin = after.administration_status
        line = (
            f"  {name:4}  {verdict:16}  match={str(match):5}  "
            f"A={atoms['A_first_line_identified']} "
            f"B={atoms['B_first_line_administered']} "
            f"C={atoms['C_failure_event']} "
            f"D={atoms['D_failure_of_first_line']} "
            f"X={atoms['X_contradiction']}  admin={admin}"
        )
        print(line)
        prev = verdict
        _ = prev


def main() -> None:
    ids = sys.argv[1:] or ["I2", "C4", "T1", "T2"]
    if not EXTRACT_PATH.is_file():
        raise SystemExit(f"missing frozen extracts: {EXTRACT_PATH}")
    for nid in ids:
        show(nid)


if __name__ == "__main__":
    main()
