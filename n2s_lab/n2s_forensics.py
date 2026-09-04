"""N2S forensics — layer-wise temporal-change vs contradiction evidence.

Directions fitted at L20 from Track B expand Class A / Class B commit hiddens:
  Class A = gold SATISFIED temporal-change with repair X=false
  Class B = gold CONTRADICTION with repair X=true

At each scored layer, report softmax over cosines to (mean_A, mean_B).
L20-fitted centroids are reused at other layers as a **transfer readout**
(same hidden size; not a per-layer refit). Document this in every payload.

Does not modify G3. Does not open temporal-test evidence for fitting.
"""

from __future__ import annotations

import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch

from .hf_client import unload_model
from .paths import ARTIFACTS_DIR
from .phase10_activation import FAIL_MODEL
from .trackb_falsex_cluster import _repair_run
from .trackb_family_vector import _unit

MODEL_7B = FAIL_MODEL
FORENSICS_LAYERS = (8, 12, 16, 20, 24)
PRIMARY_LAYER = 20
DIRECTIONS_PATH = ARTIFACTS_DIR / "n2s-forensics-directions-qwen7b.json"
HIDDENS_PATH = (
    ARTIFACTS_DIR / "trackb-expand-hiddens-Qwen_Qwen2.5-7B-Instruct.json"
)
FIT_PATH = ARTIFACTS_DIR / "trackb-expand-fit-Qwen_Qwen2.5-7B-Instruct-L20.json"

# Soft gate: need roughly this much free VRAM for 7B bf16 + headroom
MIN_FREE_VRAM_GB = 14.0


def _softmax2(a: float, b: float) -> tuple[float, float]:
    m = max(a, b)
    ea = math.exp(a - m)
    eb = math.exp(b - m)
    z = ea + eb
    return ea / z, eb / z


def _cos(h: list[float] | torch.Tensor, ref: torch.Tensor) -> float:
    if isinstance(h, list):
        t = torch.tensor(h, dtype=torch.float32)
    else:
        t = h.detach().float().cpu()
    r = ref.detach().float().cpu()
    denom = float(t.norm().item() * r.norm().item())
    if denom < 1e-12:
        return 0.0
    return float(torch.dot(t, r).item() / denom)


def cuda_free_gb() -> float | None:
    if not torch.cuda.is_available():
        return None
    try:
        free, _total = torch.cuda.mem_get_info(0)
        return free / (1024**3)
    except Exception:  # noqa: BLE001
        return None


def build_directions_from_expand(
    *,
    write: bool = True,
) -> dict[str, Any]:
    """CPU-only: mean Class A / B commit hiddens @ L20 from expand artifacts."""
    fit = json.loads(FIT_PATH.read_text(encoding="utf-8"))
    hiddens = json.loads(HIDDENS_PATH.read_text(encoding="utf-8"))
    class_a = list(fit["class_a"])
    class_b = list(fit["class_b"])

    def _stack(ids: list[str]) -> torch.Tensor:
        vecs = []
        for cid in ids:
            row = hiddens.get(cid) or {}
            h = row.get("L20")
            if not h:
                raise KeyError(f"missing L20 hidden for {cid}")
            vecs.append(torch.tensor(h, dtype=torch.float32))
        return torch.stack(vecs, dim=0)

    a = _stack(class_a)
    b = _stack(class_b)
    mean_a = a.mean(dim=0)
    mean_b = b.mean(dim=0)
    direction = _unit(mean_a - mean_b)

    payload = {
        "kind": "n2s_forensics_directions_v1",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "model_id": MODEL_7B,
        "fit_layer": PRIMARY_LAYER,
        "class_a": class_a,
        "class_b": class_b,
        "formula": "centroids = mean(commit h @ L20); scores = softmax(cos(h, mean_A), cos(h, mean_B))",
        "transfer_note": (
            "Centroids fitted at L20 only. Other layers use the same L20 centroids "
            "as a transfer readout (not per-layer refit)."
        ),
        "mean_a": mean_a.tolist(),
        "mean_b": mean_b.tolist(),
        "direction_a_minus_b": direction.tolist(),
        "dim": int(mean_a.numel()),
        "source_fit": str(FIT_PATH.name),
        "source_hiddens": str(HIDDENS_PATH.name),
    }
    if write:
        DIRECTIONS_PATH.write_text(json.dumps(payload) + "\n", encoding="utf-8")
    return payload


def load_directions() -> dict[str, Any]:
    if not DIRECTIONS_PATH.is_file():
        return build_directions_from_expand(write=True)
    return json.loads(DIRECTIONS_PATH.read_text(encoding="utf-8"))


def score_layer_hiddens(
    commit_layer_hidden: dict[str, list[float]],
    *,
    directions: dict[str, Any] | None = None,
    final_commitment: str | None = None,
    case_id: str = "",
    temperature: float = 5.0,
) -> dict[str, Any]:
    """Build the forensics table from {L20: [...], ...} or {20: [...]} maps.

    Primary scores use projection onto unit(mean_A − mean_B) with a softmax
    temperature so the UI is readable (raw cosines to centroids are often close).
    """
    directions = directions or load_directions()
    mean_a = torch.tensor(directions["mean_a"], dtype=torch.float32)
    mean_b = torch.tensor(directions["mean_b"], dtype=torch.float32)
    direction = torch.tensor(directions["direction_a_minus_b"], dtype=torch.float32)

    # normalize keys to int layer idx
    by_idx: dict[int, list[float]] = {}
    for k, v in (commit_layer_hidden or {}).items():
        if not v:
            continue
        if isinstance(k, int):
            by_idx[k] = v
        elif isinstance(k, str) and k.startswith("L") and k[1:].isdigit():
            by_idx[int(k[1:])] = v
        elif isinstance(k, str) and k.isdigit():
            by_idx[int(k)] = v

    rows: list[dict[str, Any]] = []
    for layer in FORENSICS_LAYERS:
        h = by_idx.get(layer)
        if h is None:
            rows.append(
                {
                    "layer": layer,
                    "temporal_change_score": None,
                    "contradiction_score": None,
                    "cos_to_A": None,
                    "cos_to_B": None,
                    "proj_A_minus_B": None,
                    "missing": True,
                    "primary": layer == PRIMARY_LAYER,
                }
            )
            continue
        ca = _cos(h, mean_a)
        cb = _cos(h, mean_b)
        proj = _cos(h, direction)
        # positive proj → toward temporal-change (A); negative → contradiction (B)
        st, sc = _softmax2(proj * temperature, -proj * temperature)
        rows.append(
            {
                "layer": layer,
                "temporal_change_score": round(st, 4),
                "contradiction_score": round(sc, 4),
                "cos_to_A": round(ca, 4),
                "cos_to_B": round(cb, 4),
                "proj_A_minus_B": round(proj, 4),
                "missing": False,
                "primary": layer == PRIMARY_LAYER,
            }
        )

    primary = next((r for r in rows if r["layer"] == PRIMARY_LAYER and not r["missing"]), None)
    if primary is None:
        upstream = "UNKNOWN"
        mismatch = None
    else:
        if primary["temporal_change_score"] >= primary["contradiction_score"]:
            upstream = "TEMPORAL CHANGE"
        else:
            upstream = "CONTRADICTION"
        commit = (final_commitment or "").upper()
        # Probe label space is temporal-change vs contradiction. Compare only when
        # final commitment is CONTRADICTION; SATISFIED / NOT_SATISFIED / UNCERTAIN /
        # REVIEW sit outside that space → comparison not evaluated.
        if commit == "CONTRADICTION":
            mismatch = upstream == "TEMPORAL CHANGE"
        else:
            mismatch = None

    return {
        "kind": "n2s_forensics_v1",
        "case_id": case_id,
        "model_id": directions.get("model_id", MODEL_7B),
        "method": (
            "scores = softmax(T·cos(h, u), −T·cos(h, u)) with "
            f"u=unit(mean_A−mean_B) @ L{PRIMARY_LAYER}, T={temperature}"
        ),
        "transfer_note": directions.get("transfer_note"),
        "fit_layer": PRIMARY_LAYER,
        "temperature": temperature,
        "layers": rows,
        "upstream_signal": upstream,
        "final_commitment": final_commitment,
        "representation_commitment_mismatch": mismatch,
        "class_a": directions.get("class_a"),
        "class_b": directions.get("class_b"),
    }


def forensics_from_frozen_expand_case(
    case_id: str,
    *,
    final_commitment: str = "CONTRADICTION",
) -> dict[str, Any]:
    """CPU demo: score a saved expand L20 hidden (other layers missing)."""
    hiddens = json.loads(HIDDENS_PATH.read_text(encoding="utf-8"))
    row = hiddens.get(case_id)
    if not row:
        raise KeyError(case_id)
    payload = score_layer_hiddens(
        row,
        final_commitment=final_commitment,
        case_id=case_id,
    )
    payload["mode"] = "frozen_expand_L20_only"
    payload["note"] = (
        "Demo/offline readout from expand commit hiddens (L20 only). "
        "Live notes need HF capture at FORENSICS_LAYERS."
    )
    return payload


def run_forensics_live(
    note: str,
    *,
    case_id: str = "LIVE",
    gold: str = "",
    final_commitment: str | None = None,
    model_id: str = MODEL_7B,
    unload_after: bool = True,
) -> dict[str, Any]:
    """HF repair-path commit capture + layer scores. Needs free GPU (~15GB)."""
    free = cuda_free_gb()
    if free is not None and free < MIN_FREE_VRAM_GB:
        return {
            "ok": False,
            "status": "deferred_gpu_busy",
            "free_vram_gb": round(free, 2),
            "need_vram_gb": MIN_FREE_VRAM_GB,
            "message": (
                f"GPU has only {free:.1f} GiB free; HF forensics needs ~{MIN_FREE_VRAM_GB:.0f} GiB. "
                "Wait for Ollama Phase-7 / unload the 32B runner, then Deep dive again."
            ),
        }

    directions = load_directions()
    case = {
        "id": case_id,
        "evidence": note,
        "expected": gold if gold and gold != "UNKNOWN" else "SATISFIED",
    }
    try:
        row = _repair_run(
            case,
            model_id=model_id,
            intervention="none",
            store_commit_hidden=True,
            layer_indices=FORENSICS_LAYERS,
        )
    except Exception as exc:  # noqa: BLE001
        return {
            "ok": False,
            "status": "error",
            "message": str(exc),
        }
    finally:
        if unload_after:
            try:
                unload_model()
            except Exception:  # noqa: BLE001
                pass

    hidden = row.get("commit_layer_hidden") or {}
    # _repair_run may key by fraction or Lidx — normalize
    normalized: dict[str, list[float]] = {}
    for k, v in hidden.items():
        if isinstance(v, list):
            normalized[str(k)] = v

    commitment = final_commitment
    if not commitment:
        # prefer repair_final_x → CONTRADICTION-ish
        if row.get("repair_final_x") is True:
            commitment = "CONTRADICTION"
        else:
            commitment = row.get("verdict") or "UNKNOWN"

    report = score_layer_hiddens(
        normalized,
        directions=directions,
        final_commitment=commitment,
        case_id=case_id,
    )
    report["ok"] = True
    report["status"] = "ok"
    report["mode"] = "live_hf"
    report["repair_final_x"] = row.get("repair_final_x")
    report["margin"] = row.get("logit_margin_true_minus_false")
    report["hf_verdict"] = row.get("verdict")
    report["commit_step"] = row.get("commit_step")
    return report


def format_forensics_text(report: dict[str, Any]) -> str:
    """ASCII panel matching the research mockup."""
    if report.get("ok") is False and report.get("status") != "ok":
        return f"N2S FORENSICS — deferred/error\n{report.get('message') or report}"

    cid = report.get("case_id") or "?"
    lines = [
        f"N2S FORENSICS — {cid}",
        "",
        "Final symbolic commitment",
        str(report.get("final_commitment") or "—"),
        "",
        "INTERNAL REPRESENTATION EVIDENCE",
        "─" * 44,
        f"{'Layer':<8}{'Temporal-change':>16}{'Contradiction':>16}",
    ]
    for r in report.get("layers") or []:
        layer = f"L{r['layer']}"
        if r.get("missing"):
            lines.append(f"{layer:<8}{'—':>16}{'—':>16}")
            continue
        mark = "  ←" if r.get("primary") else ""
        lines.append(
            f"{layer:<8}{r['temporal_change_score']:>16.2f}"
            f"{r['contradiction_score']:>16.2f}{mark}"
        )
    lines += [
        "",
        "REPRESENTATION → COMMITMENT",
        "─" * 44,
        "L20 representation readout:",
        str(report.get("upstream_signal") or "—"),
        "",
        "Final commitment:",
        str(report.get("final_commitment") or "—"),
        "",
    ]
    mm = report.get("representation_commitment_mismatch")
    commit = (report.get("final_commitment") or "").upper()
    if mm is True:
        lines.append("⚠ REPRESENTATION–COMMITMENT MISMATCH")
    elif mm is False and commit == "CONTRADICTION":
        lines.append("✓ representation and commitment agree (under this probe)")
    else:
        lines.append(
            "Representation–commitment comparison: NOT EVALUATED — final commitment "
            "is outside the temporal-change/contradiction probe label space."
        )
    if report.get("transfer_note"):
        lines += ["", "Note: " + str(report["transfer_note"])]
    return "\n".join(lines)
