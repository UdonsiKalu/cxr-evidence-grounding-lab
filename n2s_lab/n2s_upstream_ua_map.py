"""U-A — multi-token prefill residual contrast map (formation sites).

Data-driven sites from class-mean Δ(temporal − contra), not hand-picked cues.
Freeze-v scored only as a secondary alignment check.

See docs/TRACKB-UPSTREAM-PROGRAM.md.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch

from .hf_client import load_model, unload_model
from .n2s_forensics import MIN_FREE_VRAM_GB, cuda_free_gb
from .n2s_upstream_prefill import (
    DEFAULT_NOTES,
    PREFILL_LAYERS,
    _build_prompt,
    _capture_prefill_residuals,
    _cos,
    _load_v,
    _proj,
)
from .paths import ARTIFACTS_DIR
from .phase10_activation import FAIL_MODEL
from .trackb_family_vector import _unit

MAP_PATH = ARTIFACTS_DIR / "n2s-upstream-ua-map.json"
READOUT_PATH = ARTIFACTS_DIR / "n2s-upstream-ua-readout.json"
TOP_K = 12
TEMPORAL_ID = "EX_TEMPORAL_FOLFOX"
CONTRA_ID = "EX_CONTRA"


def _note_body_token_range(
    tokenizer: Any, prompt: str, note_span: tuple[int, int]
) -> tuple[int, int]:
    """Inclusive-exclusive token indices covering the note body char span."""
    enc = tokenizer(
        prompt,
        return_offsets_mapping=True,
        add_special_tokens=False,
        return_tensors="pt",
    )
    offsets = enc["offset_mapping"][0].tolist()
    lo, hi = note_span
    idxs = [
        i
        for i, (a, b) in enumerate(offsets)
        if not (b <= lo or a >= hi) and not (a == b == 0)
    ]
    if not idxs:
        raise RuntimeError("empty note-body token range")
    return idxs[0], idxs[-1] + 1


def _decode_tokens(tokenizer: Any, input_ids: torch.Tensor, start: int, end: int) -> list[str]:
    ids = input_ids[0, start:end].tolist()
    return [tokenizer.decode([tid], skip_special_tokens=False) for tid in ids]


def _capture_note(
    note: dict[str, str],
    *,
    model_id: str,
    layer_indices: tuple[int, ...],
) -> dict[str, Any]:
    _model, tokenizer = load_model(model_id)
    prompt, input_ids, note_span = _build_prompt(tokenizer, note["evidence"])
    t0, t1 = _note_body_token_range(tokenizer, prompt, note_span)
    captured = _capture_prefill_residuals(
        input_ids=input_ids, model_id=model_id, layer_indices=layer_indices
    )
    tokens = _decode_tokens(tokenizer, input_ids, t0, t1)
    body: dict[str, Any] = {}
    for L in layer_indices:
        if L not in captured:
            continue
        # (n_body, d)
        body[f"L{L}"] = captured[L][t0:t1].clone()
    return {
        "id": note["id"],
        "gold": note.get("gold"),
        "prompt_token_count": int(input_ids.shape[-1]),
        "note_token_start": t0,
        "note_token_end": t1,
        "n_note_tokens": t1 - t0,
        "tokens": tokens,
        "residuals": body,
    }


def run_ua_map(
    *,
    model_id: str = FAIL_MODEL,
    layer_indices: tuple[int, ...] = PREFILL_LAYERS,
    top_k: int = TOP_K,
    unload_after: bool = True,
) -> dict[str, Any]:
    free = cuda_free_gb()
    if free is not None and free < MIN_FREE_VRAM_GB:
        return {
            "ok": False,
            "status": "deferred_gpu_busy",
            "free_vram_gb": round(free, 2),
            "need_vram_gb": MIN_FREE_VRAM_GB,
            "message": f"Need ~{MIN_FREE_VRAM_GB:.0f} GiB free; have {free:.1f}.",
        }

    v, v_meta = _load_v()
    notes_raw: dict[str, dict[str, Any]] = {}
    try:
        for nid in (TEMPORAL_ID, CONTRA_ID):
            print(f"[ua-map] capturing {nid} …", flush=True)
            notes_raw[nid] = _capture_note(
                DEFAULT_NOTES[nid], model_id=model_id, layer_indices=layer_indices
            )
    finally:
        if unload_after:
            try:
                unload_model()
            except Exception:  # noqa: BLE001
                pass

    layers_out: dict[str, Any] = {}
    for L in layer_indices:
        key = f"L{L}"
        ht = notes_raw[TEMPORAL_ID]["residuals"].get(key)
        hc = notes_raw[CONTRA_ID]["residuals"].get(key)
        if ht is None or hc is None:
            continue
        mu_t = ht.mean(dim=0)
        mu_c = hc.mean(dim=0)
        d = _unit((mu_t - mu_c).float())
        cos_dv = _cos(d, v)
        proj_dv = _proj(d, v)  # same as ||d|| * cos if unit — keep for hygiene

        per_note: dict[str, Any] = {}
        for nid, block in notes_raw.items():
            h = block["residuals"][key]
            scores = (h.float() @ d.reshape(-1)).tolist()
            norms = h.float().norm(dim=-1).tolist()
            cos_vs = [_cos(h[i], v) for i in range(h.shape[0])]
            order = sorted(range(len(scores)), key=lambda i: abs(scores[i]), reverse=True)
            top = []
            for i in order[:top_k]:
                top.append(
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
                )
            per_note[nid] = {
                "n_tokens": len(scores),
                "mean_score_d": round(float(sum(scores) / max(len(scores), 1)), 6),
                "top_sites": top,
            }

        layers_out[key] = {
            "layer": L,
            "cos_d_vs_freeze_v": round(float(cos_dv), 6),
            "proj_d_vs_freeze_v": round(float(proj_dv), 6),
            "mean_norm_temporal": round(float(mu_t.norm().item()), 6),
            "mean_norm_contra": round(float(mu_c.norm().item()), 6),
            "delta_mean_norm": round(float((mu_t - mu_c).norm().item()), 6),
            "notes": per_note,
        }
        print(
            f"[ua-map] {key} cos(d,v)={cos_dv:.4f} "
            f"topT={per_note[TEMPORAL_ID]['top_sites'][0]['token']!r} "
            f"topC={per_note[CONTRA_ID]['top_sites'][0]['token']!r}",
            flush=True,
        )

    # Drop bulky residuals from artifact
    notes_meta = {
        nid: {
            "id": b["id"],
            "gold": b["gold"],
            "prompt_token_count": b["prompt_token_count"],
            "note_token_start": b["note_token_start"],
            "note_token_end": b["note_token_end"],
            "n_note_tokens": b["n_note_tokens"],
            "tokens": b["tokens"],
        }
        for nid, b in notes_raw.items()
    }

    payload = {
        "ok": True,
        "kind": "n2s_upstream_ua_map_v1",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "quest": (
            "U-A: where do note-body residuals separate temporal-change vs contradiction "
            "during prefill (class-mean Δ direction)?"
        ),
        "model_id": model_id,
        "layers": list(layer_indices),
        "top_k": top_k,
        "pair": {"temporal": TEMPORAL_ID, "contra": CONTRA_ID},
        "freeze_v_source": str(v_meta.get("source") or v_meta.get("path") or "directions"),
        "notes": notes_meta,
        "by_layer": layers_out,
        "claim_hygiene": {
            "say": "multi-token formation map; class Δ direction; candidate patch sites",
            "do_not_say": "upstream circuit; freeze-v is the temporality feature; U1 null closed formation",
        },
    }
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    MAP_PATH.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    payload["artifact"] = str(MAP_PATH)
    return payload


def write_ua_readout(map_path: Path | None = None) -> dict[str, Any]:
    path = map_path or MAP_PATH
    if not path.exists():
        raise SystemExit(f"Missing U-A map: {path}")
    panel = json.loads(path.read_text(encoding="utf-8"))
    findings: list[str] = []
    best_layer = None
    best_sep = -1.0
    for key, block in (panel.get("by_layer") or {}).items():
        sep = float(block.get("delta_mean_norm") or 0.0)
        if sep > best_sep:
            best_sep = sep
            best_layer = key
        cos_dv = block.get("cos_d_vs_freeze_v")
        findings.append(
            f"{key}: ||μ_T−μ_C||={sep:.3f}; cos(d, freeze-v)={cos_dv}"
        )

    site_lines: list[str] = []
    if best_layer and best_layer in (panel.get("by_layer") or {}):
        block = panel["by_layer"][best_layer]
        for nid in (TEMPORAL_ID, CONTRA_ID):
            tops = (block.get("notes") or {}).get(nid, {}).get("top_sites") or []
            toks = " | ".join(
                f"{t['token'].strip()!r}({t['score_d']:+.2f})" for t in tops[:5]
            )
            site_lines.append(f"{best_layer} {nid} top: {toks}")

    cos_at_best = (
        (panel.get("by_layer") or {}).get(best_layer or "", {}).get("cos_d_vs_freeze_v")
    )
    hypothesis = (
        f"Strongest mean separation at {best_layer} "
        f"(||μ_T−μ_C||≈{best_sep:.2f}); freeze-v alignment cos≈{cos_at_best}. "
        "Next U-B: path-patch top sites at that layer (attn vs MLP) — not α·v."
    )
    findings.insert(
        0,
        "U-A uses class-mean Δ over all note-body tokens (not hand-picked cues).",
    )
    findings.append(
        "Freeze-v is secondary: weak |cos(d,v)| means formation ≠ commit editor direction."
        if cos_at_best is not None and abs(float(cos_at_best)) < 0.15
        else "Freeze-v shows non-trivial alignment with class Δ — still secondary to site map."
    )

    out = {
        "ok": True,
        "kind": "n2s_upstream_ua_readout_v1",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "source_map": str(path),
        "best_layer": best_layer,
        "findings": findings,
        "top_sites_summary": site_lines,
        "formation_hypothesis": hypothesis,
        "exit_criterion": "U-A exit met if best_layer + hypothesis written; then wait go for U-B.",
        "next": "U-B component patch at top sites — wait go; no browser yet; no α-chase.",
    }
    READOUT_PATH.write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
    out["artifact"] = str(READOUT_PATH)
    return out


def _print_map(panel: dict[str, Any]) -> None:
    if not panel.get("ok"):
        print(panel.get("message") or panel)
        return
    print(f"\n=== U-A MAP · {panel.get('artifact')} ===")
    for key, block in (panel.get("by_layer") or {}).items():
        print(
            f"{key}: ||Δμ||={block.get('delta_mean_norm')}  "
            f"cos(d,v)={block.get('cos_d_vs_freeze_v')}"
        )
        for nid, nb in (block.get("notes") or {}).items():
            tops = nb.get("top_sites") or []
            head = ", ".join(
                f"{t['token'].strip()!r}:{t['score_d']:+.2f}" for t in tops[:4]
            )
            print(f"  {nid}: {head}")


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description="U-A multi-token residual contrast map")
    sub = p.add_subparsers(dest="cmd", required=True)

    sp = sub.add_parser("map", help="Capture + class Δ map (GPU)")
    sp.add_argument("--model", default=FAIL_MODEL)
    sp.add_argument("--top-k", type=int, default=TOP_K)
    sp.add_argument("--keep-model", action="store_true")

    sp = sub.add_parser("readout", help="Interpret U-A map (CPU)")
    sp.add_argument("--map", default=str(MAP_PATH))

    args = p.parse_args(argv)
    if args.cmd == "map":
        panel = run_ua_map(
            model_id=args.model,
            top_k=args.top_k,
            unload_after=not args.keep_model,
        )
        _print_map(panel)
        if not panel.get("ok"):
            raise SystemExit(2)
        readout = write_ua_readout()
        print("\n--- readout ---")
        for line in readout.get("findings") or []:
            print(f"- {line}")
        print(f"hypothesis: {readout.get('formation_hypothesis')}")
        print(f"artifact: {readout.get('artifact')}")
    elif args.cmd == "readout":
        readout = write_ua_readout(Path(args.map))
        for line in readout.get("findings") or []:
            print(f"- {line}")
        for line in readout.get("top_sites_summary") or []:
            print(f"* {line}")
        print(f"hypothesis: {readout.get('formation_hypothesis')}")
        print(f"artifact: {readout.get('artifact')}")
    else:
        raise SystemExit(f"unknown cmd {args.cmd}")


if __name__ == "__main__":
    main()
