"""Live α intervention on a pasted note (Track B expand L20 direction).

Uses the frozen expand editor direction (Class A − Class B @ L20) from
``artifacts/n2s-forensics-directions-qwen7b.json`` — same vector family as
``docs/TRACKB-ALPHA8-FREEZE.md`` (limited claim @ α=8).

Does **not** use the Phase-10 BC_E2−BC_E1 workbench vector (:8256 default).
Does not open sealed temporal-test evidence. Does not change Track A Dual.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any

import torch

from .hf_client import unload_model
from .n2s_forensics import DIRECTIONS_PATH, MIN_FREE_VRAM_GB, MODEL_7B, cuda_free_gb
from .phase10_activation import FAIL_MODEL
from .trackb_expand_endpoint import GAUSS_SEED
from .trackb_falsex_cluster import _repair_run
from .trackb_family_vector import _unit

STEER_LAYER = 20
DEFAULT_ALPHA = 8.0
CLAIM_NOTE = (
    "Frozen expand L20 editor (unit(mean_A−mean_B)); limited α=8 claim — "
    "partial editor, not family-wide fix. See docs/TRACKB-ALPHA8-FREEZE.md."
)


def load_expand_direction() -> tuple[torch.Tensor, dict[str, Any]]:
    meta = json.loads(DIRECTIONS_PATH.read_text(encoding="utf-8"))
    vec = torch.tensor(meta["direction_a_minus_b"], dtype=torch.float32)
    vec = _unit(vec)
    return vec, meta


def _summarize_arm(row: dict[str, Any], *, label: str) -> dict[str, Any]:
    return {
        "label": label,
        "intervention": row.get("intervention"),
        "alpha": row.get("alpha"),
        "repair_final_x": row.get("repair_final_x"),
        "verdict": row.get("verdict"),
        "margin": row.get("logit_margin_true_minus_false"),
        "commit_step": row.get("commit_step"),
        "commit_value": row.get("commit_value"),
        "parse_ok": row.get("parse_ok"),
        "raw_x": row.get("raw_x"),
        "verify_ok": row.get("verify_ok"),
    }


def run_intervene_live(
    note: str,
    *,
    case_id: str = "LIVE",
    gold: str = "",
    alpha: float = DEFAULT_ALPHA,
    include_controls: bool = True,
    model_id: str = FAIL_MODEL,
    unload_after: bool = True,
) -> dict[str, Any]:
    """Baseline + steered (± controls) HF repair on one note.

    Arms:
      - baseline (intervention=none)
      - full_vector @ α (expand A−B)
      - gaussian_unit @ α (specificity)
      - reverse_vector @ α (−full)
    """
    free = cuda_free_gb()
    if free is not None and free < MIN_FREE_VRAM_GB:
        return {
            "ok": False,
            "status": "deferred_gpu_busy",
            "free_vram_gb": round(free, 2),
            "need_vram_gb": MIN_FREE_VRAM_GB,
            "message": (
                f"GPU has only {free:.1f} GiB free; α intervene needs ~{MIN_FREE_VRAM_GB:.0f} GiB. "
                "Unload Ollama 32B / finish other HF jobs, then Intervene again."
            ),
        }

    try:
        vec, dir_meta = load_expand_direction()
    except Exception as exc:  # noqa: BLE001
        return {"ok": False, "status": "error", "message": f"direction load failed: {exc}"}

    case = {
        "id": case_id,
        "evidence": note,
        "expected": gold if gold and gold != "UNKNOWN" else "SATISFIED",
    }
    arms: list[dict[str, Any]] = []

    try:
        base = _repair_run(
            case,
            model_id=model_id,
            intervention="none",
            layer_indices=(STEER_LAYER,),
        )
        arms.append(_summarize_arm(base, label="baseline"))

        full = _repair_run(
            case,
            model_id=model_id,
            intervention="activation_steer",
            vectors_by_layer={STEER_LAYER: vec},
            alpha=float(alpha),
            layer_indices=(STEER_LAYER,),
        )
        arms.append(_summarize_arm(full, label="full_vector"))

        if include_controls:
            g = torch.Generator(device="cpu")
            g.manual_seed(GAUSS_SEED + int(alpha))
            gauss = _unit(torch.randn(vec.shape, generator=g, dtype=torch.float32))
            g_row = _repair_run(
                case,
                model_id=model_id,
                intervention="activation_steer",
                vectors_by_layer={STEER_LAYER: gauss},
                alpha=float(alpha),
                layer_indices=(STEER_LAYER,),
            )
            arms.append(_summarize_arm(g_row, label="gaussian_unit"))

            rev = _repair_run(
                case,
                model_id=model_id,
                intervention="activation_steer",
                vectors_by_layer={STEER_LAYER: -vec},
                alpha=float(alpha),
                layer_indices=(STEER_LAYER,),
            )
            arms.append(_summarize_arm(rev, label="reverse_vector"))
    except Exception as exc:  # noqa: BLE001
        return {
            "ok": False,
            "status": "error",
            "message": str(exc),
            "arms_partial": arms,
        }
    finally:
        if unload_after:
            try:
                unload_model()
            except Exception:  # noqa: BLE001
                pass

    by_label = {a["label"]: a for a in arms}
    base_x = by_label.get("baseline", {}).get("repair_final_x")
    full_x = by_label.get("full_vector", {}).get("repair_final_x")
    flipped = base_x is True and full_x is False

    return {
        "ok": True,
        "status": "ok",
        "kind": "n2s_intervene_v1",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "case_id": case_id,
        "model_id": model_id or MODEL_7B,
        "steer_layer": STEER_LAYER,
        "alpha": float(alpha),
        "include_controls": include_controls,
        "claim_note": CLAIM_NOTE,
        "direction": {
            "path": str(DIRECTIONS_PATH),
            "fit_layer": dir_meta.get("fit_layer"),
            "class_a": dir_meta.get("class_a"),
            "class_b": dir_meta.get("class_b"),
            "formula": "activation_steer: h += α · unit(mean_A − mean_B) @ L20 (commit token)",
        },
        "arms": arms,
        "summary": {
            "baseline_x": base_x,
            "full_vector_x": full_x,
            "baseline_verdict": by_label.get("baseline", {}).get("verdict"),
            "full_vector_verdict": by_label.get("full_vector", {}).get("verdict"),
            "x_flipped_true_to_false": flipped,
            "gaussian_x": by_label.get("gaussian_unit", {}).get("repair_final_x"),
            "reverse_x": by_label.get("reverse_vector", {}).get("repair_final_x"),
        },
    }


def format_intervene_text(report: dict[str, Any]) -> str:
    if not report.get("ok"):
        return f"N2S INTERVENE — deferred/error\n{report.get('message') or report}"
    lines = [
        f"N2S INTERVENE — {report.get('case_id') or '?'}",
        f"model: {report.get('model_id')}",
        f"layer: L{report.get('steer_layer')} · α={report.get('alpha')}",
        "",
        str(report.get("claim_note") or ""),
        "",
        f"{'Arm':<18}{'X':>6}{'verdict':>16}{'margin':>10}",
        "─" * 52,
    ]
    for a in report.get("arms") or []:
        m = a.get("margin")
        m_s = f"{m:.2f}" if isinstance(m, (int, float)) else "—"
        lines.append(
            f"{str(a.get('label')):<18}{str(a.get('repair_final_x')):>6}"
            f"{str(a.get('verdict') or '—'):>16}{m_s:>10}"
        )
    s = report.get("summary") or {}
    lines += [
        "",
        f"baseline X → full_vector X flip (true→false): {s.get('x_flipped_true_to_false')}",
        "Controls: gaussian should not mimic full; reverse is sign check.",
    ]
    return "\n".join(lines)
