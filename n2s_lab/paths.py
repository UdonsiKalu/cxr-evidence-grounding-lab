from __future__ import annotations

from pathlib import Path

LAB_ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = LAB_ROOT / "data" / "cases.json"
HELDOUT_PATH = LAB_ROOT / "data" / "heldout-phase3.json"
HELDOUT_PHASE4_PATH = LAB_ROOT / "data" / "heldout-phase4.json"
STATIC_DIR = LAB_ROOT / "static"
ARTIFACTS_DIR = LAB_ROOT / "artifacts"
