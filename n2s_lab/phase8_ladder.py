"""Phase-8: model ladder — 14B + non-coder 32B on frozen Phase-4 / BMT-CART slices.

Does not overwrite Phase-1–7 artifacts. See docs/PHASE8-PROTOCOL.md.
"""

from __future__ import annotations

from typing import Any

from .phase7_crossdomain import run_phase7_panel

# Mid + non-coder large (same family as frozen qwen2.5-coder:32b for fairer size read).
PHASE8_LADDER_MODELS = (
    "qwen2.5:14b",
    "qwen2.5:32b",
)


def run_phase8_ladder(
    *,
    case_set: str,
    models: list[str] | None = None,
) -> dict[str, Any]:
    import json

    from .paths import ARTIFACTS_DIR

    models = models or list(PHASE8_LADDER_MODELS)
    if case_set == "phase4":
        out_name = "phase8-ladder-phase4-panel.json"
    elif case_set == "bmtcart":
        out_name = "phase8-ladder-bmtcart-panel.json"
    else:
        raise ValueError(case_set)
    panel = run_phase7_panel(
        models=models,
        case_set=case_set,
        out_name=out_name,
        artifact_prefix="phase8",
        experiment="phase8_model_ladder",
    )
    panel["experiment"] = "phase8_model_ladder"
    panel["phase"] = 8
    panel["ladder_models"] = list(models)
    panel["note"] = (
        "Phase-8 model ladder (14B + non-coder 32B). Same Phase-7 stack. "
        "Does not overwrite Phase-1–7 citation panels. Soft claim only."
    )
    path = ARTIFACTS_DIR / out_name
    to_write = {k: v for k, v in panel.items() if not k.startswith("_")}
    path.write_text(json.dumps(to_write, indent=2) + "\n", encoding="utf-8")
    panel["_artifact"] = str(path)
    return panel
