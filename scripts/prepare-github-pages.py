#!/usr/bin/env python3
"""Copy curated frozen artifacts into docs/artifacts/ for GitHub Pages (replay-only)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "artifacts"
DST = ROOT / "docs" / "artifacts"

CURATED = [
    "phase1-transition-c-u.json",
    "phase2-model-panel.json",
    "phase2-mistral_instruct.json",
    "phase2-qwen2.5-coder_32b.json",
    "phase3-heldout-panel.json",
    "phase3-llama3_8b-instruct-q4_0.json",
    "phase3-mistral_instruct.json",
    "phase3-qwen2.5-coder_32b.json",
    "phase4-heldout-panel.json",
    "phase4-llama3_8b-instruct-q4_0.json",
    "phase4-mistral_instruct.json",
    "phase4-qwen2.5-coder_32b.json",
]

MANIFEST = {
    "mode": "artifact_replay_only",
    "title": "N2S Evidence Grounding — frozen findings demo",
    "note": (
        "No live inference. Load Phase 1–4 JSON only. "
        "Clone the repo to run Ollama locally."
    ),
    "default_artifact": "phase4-heldout-panel.json",
    "artifacts": [
        {
            "name": "phase4-heldout-panel.json",
            "label": "Phase-4 held-out (C9–U12)",
            "kind": "panel",
        },
        {
            "name": "phase3-heldout-panel.json",
            "label": "Phase-3 held-out (C5–U8)",
            "kind": "panel",
        },
        {
            "name": "phase2-model-panel.json",
            "label": "Phase-2 model panel (C1–U4)",
            "kind": "panel",
        },
        {
            "name": "phase1-transition-c-u.json",
            "label": "Phase-1 control (Llama C/U)",
            "kind": "report",
        },
    ],
}


def main() -> int:
    if not SRC.is_dir():
        print(f"missing artifacts dir: {SRC}", file=sys.stderr)
        return 1
    DST.mkdir(parents=True, exist_ok=True)
    for name in CURATED:
        path = SRC / name
        if not path.is_file():
            print(f"skip missing: {name}")
            continue
        data = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(data.get("models"), list):
            for entry in data["models"]:
                art = entry.get("artifact")
                if isinstance(art, str) and art:
                    entry["artifact"] = Path(art).name
        out = DST / name
        out.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
        print(f"wrote {out.relative_to(ROOT)}")

    manifest_path = ROOT / "docs" / "manifest.json"
    manifest_path.write_text(json.dumps(MANIFEST, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {manifest_path.relative_to(ROOT)}")
    print("OK — serve docs/ with a static server (see docs/README.md)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
