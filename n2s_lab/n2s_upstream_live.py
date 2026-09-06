"""Live Upstream scoring — paste a note, score vs frozen U-A formation direction d.

Not ablation / SAE / α·v. Correlational readout only:
  residual_token · unit(μ_T − μ_C) at prefill layers, plus secondary cos to freeze-v.

Directions are cached in artifacts/n2s-upstream-ua-directions.pt (built once from
the frozen EX_TEMPORAL_FOLFOX / EX_CONTRA pair unless rebuilt).

See docs/TRACKB-UPSTREAM-PROGRAM.md.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch

from .hf_client import unload_model
from .n2s_forensics import MIN_FREE_VRAM_GB, cuda_free_gb
from .n2s_upstream_prefill import (
    DEFAULT_NOTES,
    PREFILL_LAYERS,
    _cos,
    _load_v,
    _proj,
)
from .n2s_upstream_ua_map import (
    CONTRA_ID,
    MAP_PATH,
    TEMPORAL_ID,
    TOP_K,
    _capture_note,
)
from .paths import ARTIFACTS_DIR
from .phase10_activation import FAIL_MODEL
from .trackb_family_vector import _unit

DIRECTIONS_PATH = ARTIFACTS_DIR / "n2s-upstream-ua-directions.pt"
LIVE_RUNS_DIR = ARTIFACTS_DIR / "n2s-upstream-live-runs"


def _gpu_gate() -> dict[str, Any] | None:
    free = cuda_free_gb()
    if free is not None and free < MIN_FREE_VRAM_GB:
        return {
            "ok": False,
            "status": "deferred_gpu_busy",
            "free_vram_gb": round(free, 2),
            "need_vram_gb": MIN_FREE_VRAM_GB,
            "message": f"Need ~{MIN_FREE_VRAM_GB:.0f} GiB free; have {free:.1f}.",
        }
    return None


def ensure_ua_directions(
    *,
    model_id: str = FAIL_MODEL,
    layer_indices: tuple[int, ...] = PREFILL_LAYERS,
    force: bool = False,
    unload_after: bool = True,
) -> dict[str, Any]:
    """Load or rebuild per-layer unit(μ_T−μ_C) from the frozen lab pair."""
    if DIRECTIONS_PATH.is_file() and not force:
        try:
            blob = torch.load(DIRECTIONS_PATH, map_location="cpu", weights_only=False)
        except TypeError:
            blob = torch.load(DIRECTIONS_PATH, map_location="cpu")
        return {
            "ok": True,
            "cached": True,
            "path": str(DIRECTIONS_PATH),
            "layers": list(blob.get("layers") or []),
            "model_id": blob.get("model_id"),
            "pair": blob.get("pair"),
            "timestamp": blob.get("timestamp"),
            "_blob": blob,
        }

    gate = _gpu_gate()
    if gate:
        return gate

    v, v_meta = _load_v()
    notes_raw: dict[str, dict[str, Any]] = {}
    try:
        for nid in (TEMPORAL_ID, CONTRA_ID):
            print(f"[ua-live] baking directions · capture {nid} …", flush=True)
            notes_raw[nid] = _capture_note(
                DEFAULT_NOTES[nid], model_id=model_id, layer_indices=layer_indices
            )
    finally:
        if unload_after:
            try:
                unload_model()
            except Exception:  # noqa: BLE001
                pass

    by_layer: dict[str, torch.Tensor] = {}
    meta_layers: dict[str, Any] = {}
    for L in layer_indices:
        key = f"L{L}"
        ht = notes_raw[TEMPORAL_ID]["residuals"].get(key)
        hc = notes_raw[CONTRA_ID]["residuals"].get(key)
        if ht is None or hc is None:
            continue
        d = _unit((ht.mean(dim=0) - hc.mean(dim=0)).float()).cpu()
        by_layer[key] = d
        meta_layers[key] = {
            "layer": L,
            "delta_mean_norm": round(float((ht.mean(0) - hc.mean(0)).norm().item()), 6),
            "cos_d_vs_freeze_v": round(_cos(d, v), 6),
        }

    blob = {
        "kind": "n2s_upstream_ua_directions_v1",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "model_id": model_id,
        "layers": list(layer_indices),
        "pair": {"temporal": TEMPORAL_ID, "contra": CONTRA_ID},
        "freeze_v_source": str(v_meta.get("source") or v_meta.get("path") or "directions"),
        "by_layer_meta": meta_layers,
        "by_layer": by_layer,
    }
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    torch.save(blob, DIRECTIONS_PATH)
    return {
        "ok": True,
        "cached": False,
        "path": str(DIRECTIONS_PATH),
        "layers": list(layer_indices),
        "model_id": model_id,
        "pair": blob["pair"],
        "timestamp": blob["timestamp"],
        "_blob": blob,
    }


def score_note_live(
    evidence: str,
    *,
    case_id: str = "LIVE_NOTE",
    gold: str = "UNKNOWN",
    model_id: str = FAIL_MODEL,
    layer_indices: tuple[int, ...] = PREFILL_LAYERS,
    top_k: int = TOP_K,
    unload_after: bool = True,
    rebuild_directions: bool = False,
) -> dict[str, Any]:
    """Capture one pasted note and score note-body tokens against frozen U-A d."""
    evidence = (evidence or "").strip()
    if not evidence:
        return {"ok": False, "error": "evidence required"}

    gate = _gpu_gate()
    if gate:
        return gate

    dirs = ensure_ua_directions(
        model_id=model_id,
        layer_indices=layer_indices,
        force=rebuild_directions,
        unload_after=True,  # free VRAM before scoring capture
    )
    if not dirs.get("ok"):
        return dirs
    blob = dirs["_blob"]
    by_layer: dict[str, torch.Tensor] = blob["by_layer"]

    v, v_meta = _load_v()
    note = {"id": case_id, "gold": gold, "evidence": evidence}
    try:
        print(f"[ua-live] scoring {case_id} …", flush=True)
        captured = _capture_note(note, model_id=model_id, layer_indices=layer_indices)
    finally:
        if unload_after:
            try:
                unload_model()
            except Exception:  # noqa: BLE001
                pass

    layers_out: dict[str, Any] = {}
    for key, d in by_layer.items():
        h = captured["residuals"].get(key)
        if h is None:
            continue
        d = d.to(dtype=torch.float32)
        scores = (h.float() @ d.reshape(-1)).tolist()
        norms = h.float().norm(dim=-1).tolist()
        cos_vs = [_cos(h[i], v) for i in range(h.shape[0])]
        order = sorted(range(len(scores)), key=lambda i: abs(scores[i]), reverse=True)
        top = []
        for i in order[:top_k]:
            top.append(
                {
                    "local_pos": i,
                    "global_pos": int(captured["note_token_start"] + i),
                    "token": captured["tokens"][i],
                    "score_d": round(float(scores[i]), 6),
                    "abs_score_d": round(abs(float(scores[i])), 6),
                    "norm": round(float(norms[i]), 6),
                    "cos_v": round(float(cos_vs[i]), 6),
                    "proj_v": round(_proj(h[i], v), 6),
                }
            )
        L = int(key[1:]) if key.startswith("L") else key
        layers_out[key] = {
            "layer": L,
            "mean_score_d": round(float(sum(scores) / max(len(scores), 1)), 6),
            "mean_abs_score_d": round(
                float(sum(abs(s) for s in scores) / max(len(scores), 1)), 6
            ),
            "cos_d_vs_freeze_v": (blob.get("by_layer_meta") or {})
            .get(key, {})
            .get("cos_d_vs_freeze_v"),
            "top_sites": top,
        }

    # Prefer L24 (or best sep from frozen map) for headline
    best_key = "L24" if "L24" in layers_out else (list(layers_out)[-1] if layers_out else None)
    if MAP_PATH.is_file():
        try:
            ua = json.loads(MAP_PATH.read_text(encoding="utf-8"))
            best_sep, pick = -1.0, best_key
            for k, block in (ua.get("by_layer") or {}).items():
                sep = float(block.get("delta_mean_norm") or 0.0)
                if sep > best_sep and k in layers_out:
                    best_sep, pick = sep, k
            best_key = pick or best_key
        except Exception:  # noqa: BLE001
            pass

    payload = {
        "ok": True,
        "kind": "n2s_upstream_live_score_v1",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "quest": (
            "Live: where do this note's prefill residuals project onto the frozen "
            "U-A formation direction d = unit(μ_T−μ_C)?"
        ),
        "case_id": case_id,
        "gold": gold,
        "model_id": model_id,
        "directions_path": str(DIRECTIONS_PATH),
        "directions_cached": bool(dirs.get("cached")),
        "pair_frozen": blob.get("pair"),
        "freeze_v_source": str(v_meta.get("source") or v_meta.get("path") or "directions"),
        "n_note_tokens": captured["n_note_tokens"],
        "best_layer": best_key,
        "by_layer": layers_out,
        "claim_hygiene": {
            "say": "live score vs frozen U-A d; correlational formation readout",
            "do_not_say": "ablation; circuit found; upstream editor; freeze-v is temporality",
        },
    }
    LIVE_RUNS_DIR.mkdir(parents=True, exist_ok=True)
    safe = "".join(c if c.isalnum() or c in "-_" else "_" for c in case_id)[:48]
    out = LIVE_RUNS_DIR / f"live-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}-{safe}.json"
    slim = {k: v for k, v in payload.items()}
    out.write_text(json.dumps(slim, indent=2) + "\n", encoding="utf-8")
    payload["artifact"] = str(out)
    return payload


def run_custom_pair_map(
    temporal_evidence: str,
    contra_evidence: str,
    *,
    temporal_id: str = "LIVE_TEMPORAL",
    contra_id: str = "LIVE_CONTRA",
    model_id: str = FAIL_MODEL,
    layer_indices: tuple[int, ...] = PREFILL_LAYERS,
    top_k: int = TOP_K,
    unload_after: bool = True,
) -> dict[str, Any]:
    """Full U-A-style map on a user-supplied temporal/contra pair (not freeze bake)."""
    temporal_evidence = (temporal_evidence or "").strip()
    contra_evidence = (contra_evidence or "").strip()
    if not temporal_evidence or not contra_evidence:
        return {"ok": False, "error": "both temporal_evidence and contra_evidence required"}

    gate = _gpu_gate()
    if gate:
        return gate

    v, v_meta = _load_v()
    notes = {
        temporal_id: {"id": temporal_id, "gold": "SATISFIED", "evidence": temporal_evidence},
        contra_id: {"id": contra_id, "gold": "CONTRADICTION", "evidence": contra_evidence},
    }
    notes_raw: dict[str, dict[str, Any]] = {}
    try:
        for nid, note in notes.items():
            print(f"[ua-live] custom pair · capture {nid} …", flush=True)
            notes_raw[nid] = _capture_note(note, model_id=model_id, layer_indices=layer_indices)
    finally:
        if unload_after:
            try:
                unload_model()
            except Exception:  # noqa: BLE001
                pass

    layers_out: dict[str, Any] = {}
    best_key, best_sep = None, -1.0
    for L in layer_indices:
        key = f"L{L}"
        ht = notes_raw[temporal_id]["residuals"].get(key)
        hc = notes_raw[contra_id]["residuals"].get(key)
        if ht is None or hc is None:
            continue
        d = _unit((ht.mean(dim=0) - hc.mean(dim=0)).float())
        cos_dv = _cos(d, v)
        sep = float((ht.mean(0) - hc.mean(0)).norm().item())
        if sep > best_sep:
            best_sep, best_key = sep, key
        per_note: dict[str, Any] = {}
        for nid, block in notes_raw.items():
            h = block["residuals"][key]
            scores = (h.float() @ d.reshape(-1)).tolist()
            norms = h.float().norm(dim=-1).tolist()
            cos_vs = [_cos(h[i], v) for i in range(h.shape[0])]
            order = sorted(range(len(scores)), key=lambda i: abs(scores[i]), reverse=True)
            top = [
                {
                    "local_pos": i,
                    "global_pos": int(block["note_token_start"] + i),
                    "token": block["tokens"][i],
                    "score_d": round(float(scores[i]), 6),
                    "abs_score_d": round(abs(float(scores[i])), 6),
                    "norm": round(float(norms[i]), 6),
                    "cos_v": round(float(cos_vs[i]), 6),
                    "proj_v": round(_proj(h[i], v), 6),
                }
                for i in order[:top_k]
            ]
            per_note[nid] = {
                "n_tokens": len(scores),
                "mean_score_d": round(float(sum(scores) / max(len(scores), 1)), 6),
                "top_sites": top,
            }
        layers_out[key] = {
            "layer": L,
            "delta_mean_norm": round(sep, 6),
            "cos_d_vs_freeze_v": round(float(cos_dv), 6),
            "notes": per_note,
        }

    payload = {
        "ok": True,
        "kind": "n2s_upstream_live_pair_v1",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "quest": "Live custom pair: U-A-style class-mean Δ map on pasted temporal vs contra.",
        "model_id": model_id,
        "pair": {"temporal": temporal_id, "contra": contra_id},
        "best_layer": best_key,
        "best_delta_mean_norm": round(best_sep, 6) if best_key else None,
        "freeze_v_source": str(v_meta.get("source") or v_meta.get("path") or "directions"),
        "by_layer": layers_out,
        "claim_hygiene": {
            "say": "live custom-pair formation map; correlational",
            "do_not_say": "ablation; circuit; replaces frozen U-A claim without controls",
        },
    }
    LIVE_RUNS_DIR.mkdir(parents=True, exist_ok=True)
    out = LIVE_RUNS_DIR / (
        f"pair-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}.json"
    )
    out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    payload["artifact"] = str(out)
    return payload
