"""U-A held-out paraphrase / lexical generalization panel.

Scores notes in data/heldout-ua-paraphrase.json against the *frozen* U-A
direction d = unit(μ_T−μ_C). Does not rebuild d from held-out text.
Does not ablate / SAE / circuit.

See docs/TRACKB-UPSTREAM-UA-GENERALIZE.md.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch

from .hf_client import unload_model
from .n2s_forensics import MIN_FREE_VRAM_GB, cuda_free_gb
from .n2s_upstream_live import DIRECTIONS_PATH, ensure_ua_directions
from .n2s_upstream_prefill import PREFILL_LAYERS, _cos, _load_v, _proj
from .n2s_upstream_ua_map import MAP_PATH, TOP_K, _capture_note
from .paths import ARTIFACTS_DIR, HELDOUT_UA_PARAPHRASE_PATH
from .phase10_activation import FAIL_MODEL

PANEL_PATH = ARTIFACTS_DIR / "n2s-upstream-ua-gen-panel.json"
READOUT_PATH = ARTIFACTS_DIR / "n2s-upstream-ua-gen-readout.json"


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


def _load_cases() -> list[dict[str, Any]]:
    data = json.loads(HELDOUT_UA_PARAPHRASE_PATH.read_text(encoding="utf-8"))
    return list(data.get("notes") or [])


def _pick_best_layer(layers_out: dict[str, Any]) -> str | None:
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
    return best_key


def _score_captured(
    captured: dict[str, Any],
    *,
    by_layer: dict[str, torch.Tensor],
    v: torch.Tensor,
    blob_meta: dict[str, Any],
    top_k: int,
) -> dict[str, Any]:
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
            "cos_d_vs_freeze_v": (blob_meta.get("by_layer_meta") or {})
            .get(key, {})
            .get("cos_d_vs_freeze_v"),
            "top_sites": top,
        }
    best_key = _pick_best_layer(layers_out)
    headline = (layers_out.get(best_key) or {}).get("mean_score_d") if best_key else None
    return {
        "n_note_tokens": captured["n_note_tokens"],
        "best_layer": best_key,
        "headline_mean_score_d": headline,
        "by_layer": layers_out,
    }


def _soft_gate(rows: list[dict[str, Any]], *, best_layer: str) -> dict[str, Any]:
    held_t = [
        r
        for r in rows
        if r.get("class") == "temporal" and r.get("role") != "anchor" and r.get("ok")
    ]
    held_c = [
        r
        for r in rows
        if r.get("class") == "contra" and r.get("role") != "anchor" and r.get("ok")
    ]

    def _means(group: list[dict[str, Any]]) -> list[float]:
        out: list[float] = []
        for r in group:
            block = (r.get("by_layer") or {}).get(best_layer) or {}
            if "mean_score_d" in block:
                out.append(float(block["mean_score_d"]))
        return out

    mt, mc = _means(held_t), _means(held_c)
    mean_t = sum(mt) / len(mt) if mt else None
    mean_c = sum(mc) / len(mc) if mc else None
    sep = (mean_t - mean_c) if mean_t is not None and mean_c is not None else None
    strong = (
        bool(mt)
        and bool(mc)
        and min(mt) > max(mc)
    )
    soft = mean_t is not None and mean_c is not None and mean_t > mean_c
    return {
        "best_layer": best_layer,
        "heldout_temporal_n": len(mt),
        "heldout_contra_n": len(mc),
        "heldout_temporal_mean_score_d": None if mean_t is None else round(mean_t, 6),
        "heldout_contra_mean_score_d": None if mean_c is None else round(mean_c, 6),
        "heldout_mean_sep": None if sep is None else round(sep, 6),
        "heldout_min_temporal": None if not mt else round(min(mt), 6),
        "heldout_max_contra": None if not mc else round(max(mc), 6),
        "soft_pass_mean_T_gt_mean_C": soft,
        "strong_pass_min_T_gt_max_C": strong,
        "note": (
            "YES soft — held-out class means separate in expected direction"
            if soft and strong
            else (
                "YES soft / NO strong — means separate but classes overlap"
                if soft
                else "NO — held-out means do not separate (or missing scores)"
            )
        ),
    }


def run_ua_generalize_panel(
    *,
    model_id: str = FAIL_MODEL,
    layer_indices: tuple[int, ...] = PREFILL_LAYERS,
    top_k: int = TOP_K,
    unload_after: bool = True,
    rebuild_directions: bool = False,
) -> dict[str, Any]:
    """Score held-out paraphrases vs frozen U-A d (one GPU session)."""
    gate = _gpu_gate()
    if gate:
        return gate

    dirs = ensure_ua_directions(
        model_id=model_id,
        layer_indices=layer_indices,
        force=rebuild_directions,
        unload_after=True,
    )
    if not dirs.get("ok"):
        return dirs
    blob = dirs["_blob"]
    by_layer: dict[str, torch.Tensor] = blob["by_layer"]
    v, v_meta = _load_v()
    cases = _load_cases()
    rows: list[dict[str, Any]] = []

    try:
        for case in cases:
            cid = case["id"]
            print(f"[ua-gen] capture {cid} ({case.get('role')}/{case.get('class')}) …", flush=True)
            note = {
                "id": cid,
                "gold": case.get("gold") or "UNKNOWN",
                "evidence": case["evidence"],
            }
            try:
                captured = _capture_note(
                    note, model_id=model_id, layer_indices=layer_indices
                )
                scored = _score_captured(
                    captured,
                    by_layer=by_layer,
                    v=v,
                    blob_meta=blob,
                    top_k=top_k,
                )
                rows.append(
                    {
                        "ok": True,
                        "case_id": cid,
                        "role": case.get("role"),
                        "class": case.get("class"),
                        "gold": case.get("gold"),
                        **scored,
                    }
                )
            except Exception as exc:  # noqa: BLE001
                rows.append(
                    {
                        "ok": False,
                        "case_id": cid,
                        "role": case.get("role"),
                        "class": case.get("class"),
                        "error": str(exc),
                    }
                )
    finally:
        if unload_after:
            try:
                unload_model()
            except Exception:  # noqa: BLE001
                pass

    # Headline layer from first successful row (U-A pick)
    best_layer = "L24"
    for r in rows:
        if r.get("ok") and r.get("best_layer"):
            best_layer = str(r["best_layer"])
            break

    soft = _soft_gate(rows, best_layer=best_layer)
    panel: dict[str, Any] = {
        "ok": True,
        "kind": "n2s_upstream_ua_gen_panel_v1",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "protocol": "docs/TRACKB-UPSTREAM-UA-GENERALIZE.md",
        "quest": (
            "Does frozen U-A d separate temporal vs contradiction on "
            "held-out paraphrases / lexical forms?"
        ),
        "model_id": model_id,
        "directions_path": str(DIRECTIONS_PATH),
        "directions_cached": bool(dirs.get("cached")),
        "pair_frozen": blob.get("pair"),
        "heldout": str(HELDOUT_UA_PARAPHRASE_PATH.name),
        "freeze_v_source": str(v_meta.get("source") or v_meta.get("path") or "directions"),
        "soft_gate": soft,
        "rows": rows,
        "claim_hygiene": {
            "say": "correlational lexical generalization of frozen U-A d",
            "do_not_say": "causal editor; temporality neuron; reopen U-B ablation from this",
        },
    }
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    PANEL_PATH.write_text(json.dumps(panel, indent=2) + "\n", encoding="utf-8")
    panel["artifact"] = str(PANEL_PATH)
    write_ua_gen_readout(PANEL_PATH)
    return panel


def write_ua_gen_readout(panel_path: Path | None = None) -> dict[str, Any]:
    path = panel_path or PANEL_PATH
    if not path.is_file():
        return {"ok": False, "error": f"missing {path}"}
    panel = json.loads(path.read_text(encoding="utf-8"))
    soft = panel.get("soft_gate") or {}
    lines = [
        f"# U-A paraphrase generalization readout ({panel.get('timestamp')})",
        "",
        f"Quest: {panel.get('quest')}",
        f"Directions: {panel.get('directions_path')} (cached={panel.get('directions_cached')})",
        f"Held-out: {panel.get('heldout')}",
        "",
        f"Soft gate @ {soft.get('best_layer')}: {soft.get('note')}",
        f"  mean_T={soft.get('heldout_temporal_mean_score_d')}  "
        f"mean_C={soft.get('heldout_contra_mean_score_d')}  "
        f"sep={soft.get('heldout_mean_sep')}",
        f"  min_T={soft.get('heldout_min_temporal')}  max_C={soft.get('heldout_max_contra')}",
        f"  soft_pass={soft.get('soft_pass_mean_T_gt_mean_C')}  "
        f"strong_pass={soft.get('strong_pass_min_T_gt_max_C')}",
        "",
        "## Rows (headline mean_score_d)",
    ]
    for r in panel.get("rows") or []:
        if not r.get("ok"):
            lines.append(f"- {r.get('case_id')}: ERROR {r.get('error')}")
            continue
        lines.append(
            f"- {r.get('case_id')} [{r.get('role')}/{r.get('class')}] "
            f"{r.get('best_layer')} mean_score_d={r.get('headline_mean_score_d')}"
        )
    lines.extend(
        [
            "",
            "## Claim hygiene",
            f"- Say: {(panel.get('claim_hygiene') or {}).get('say')}",
            f"- Do not say: {(panel.get('claim_hygiene') or {}).get('do_not_say')}",
            "",
        ]
    )
    text = "\n".join(lines)
    READOUT_PATH.write_text(
        json.dumps(
            {
                "ok": True,
                "kind": "n2s_upstream_ua_gen_readout_v1",
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "panel": str(path),
                "soft_gate": soft,
                "text": text,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(text, flush=True)
    return {"ok": True, "path": str(READOUT_PATH), "text": text}


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="U-A held-out paraphrase generalization")
    p.add_argument("cmd", choices=("panel", "readout"))
    p.add_argument("--rebuild-directions", action="store_true")
    args = p.parse_args(argv)
    if args.cmd == "panel":
        out = run_ua_generalize_panel(rebuild_directions=args.rebuild_directions)
        print(json.dumps({k: v for k, v in out.items() if k != "rows"}, indent=2))
        if not out.get("ok"):
            return 1
        return 0
    out = write_ua_gen_readout()
    return 0 if out.get("ok") else 1


if __name__ == "__main__":
    raise SystemExit(main())
