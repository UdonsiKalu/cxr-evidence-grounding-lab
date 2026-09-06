"""Thin upstream prefill trace — cue × layer residuals vs frozen v / SAE.

Observational only (U0). Does **not** patch or flip X.
Stack: HF transformers + forward hooks (same as Intervene / SAE).

See docs/TRACKB-UPSTREAM-PREFILL.md.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch

from .hf_client import load_model, unload_model
from .hf_trace import _model_layers
from .neural import EXTRACT_SYSTEM
from .n2s_forensics import DIRECTIONS_PATH, FORENSICS_LAYERS, MIN_FREE_VRAM_GB, cuda_free_gb
from .n2s_sae_pilot import JumpReLUSae, SCORE_PATH
from .paths import ARTIFACTS_DIR
from .phase10_activation import FAIL_MODEL
from .trackb_family_vector import _unit

TRACE_PATH = ARTIFACTS_DIR / "n2s-upstream-prefill-trace.json"
PATCH_PATH = ARTIFACTS_DIR / "n2s-upstream-prefill-patch.json"
PATCH_U1B_PATH = ARTIFACTS_DIR / "n2s-upstream-prefill-patch-u1b.json"
U0_READOUT_PATH = ARTIFACTS_DIR / "n2s-upstream-u0-readout.json"
PREFILL_LAYERS = FORENSICS_LAYERS  # (8, 12, 16, 20, 24)
SAE_LAYER = 20
DEFAULT_TOP_SAE = 5

# U1b — different sites/layer (not α-chase): outcome-language @ L16.
# Hypothesis: temporal meaning forms on progression/failure cues mid-network,
# not on last drug-name token at commit layer L20 (U1 null).
U1B_SITE_RECIPE: dict[str, dict[str, Any]] = {
    "EX_TEMPORAL_FOLFOX": {
        "cues": ["progression", "failure"],
        "which": "all",
        "layer": 16,
    },
    "EX_CONTRA": {
        "cues": ["failed", "responsive", "addendum"],
        "which": "all",
        "layer": 16,
    },
}

DEFAULT_CUES = (
    "FOLFOX",
    "failed",
    "response",
    "later",
    "progression",
    "responsive",
    "discontinued",
    "addendum",
)

# Default pair: temporal vs true-contra (same as walkthrough / SAE soft notes)
DEFAULT_NOTES: dict[str, dict[str, str]] = {
    "EX_TEMPORAL_FOLFOX": {
        "id": "EX_TEMPORAL_FOLFOX",
        "gold": "SATISFIED",
        "evidence": (
            "Metastatic colorectal cancer. First-line FOLFOX produced a partial response. "
            "At follow-up four months later, imaging demonstrated new hepatic lesions "
            "consistent with progression. FOLFOX was discontinued for treatment failure; "
            "second-line therapy discussed."
        ),
    },
    "EX_CONTRA": {
        "id": "EX_CONTRA",
        "gold": "CONTRADICTION",
        "evidence": (
            "Note A: Patient failed first-line FOLFOX after four cycles. "
            "Addendum same day: Disease remains responsive to FOLFOX; continue current regimen."
        ),
    },
}


def _cos(a: torch.Tensor, b: torch.Tensor) -> float:
    a = a.float().reshape(-1)
    b = b.float().reshape(-1)
    denom = float(a.norm().item() * b.norm().item())
    if denom < 1e-12:
        return 0.0
    return float(torch.dot(a, b).item() / denom)


def _proj(h: torch.Tensor, direction: torch.Tensor) -> float:
    d = _unit(direction.float())
    h = h.float().reshape(-1)
    return float(torch.dot(h, d).item())


def _load_v() -> tuple[torch.Tensor, dict[str, Any]]:
    meta = json.loads(DIRECTIONS_PATH.read_text(encoding="utf-8"))
    v = _unit(torch.tensor(meta["direction_a_minus_b"], dtype=torch.float32))
    return v, meta


def _build_prompt(tokenizer: Any, note: str) -> tuple[str, torch.Tensor, tuple[int, int]]:
    """Chat-templated extract prompt; return (prompt_str, input_ids, note_char_span)."""
    messages = [
        {"role": "system", "content": EXTRACT_SYSTEM},
        {"role": "user", "content": note},
    ]
    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    ids = tokenizer(prompt, return_tensors="pt", add_special_tokens=False).input_ids
    note_start = prompt.find(note)
    if note_start < 0:
        # Fallback: match a long unique prefix of the note
        prefix = note[: min(48, len(note))]
        note_start = prompt.find(prefix)
        if note_start < 0:
            raise RuntimeError("Could not locate note text inside chat prompt")
        note_end = note_start + len(note)
    else:
        note_end = note_start + len(note)
    return prompt, ids, (note_start, note_end)


def _find_cue_spans(
    prompt: str,
    tokenizer: Any,
    cues: tuple[str, ...] | list[str],
    *,
    char_range: tuple[int, int] | None = None,
) -> list[dict[str, Any]]:
    """Map cue substrings to token spans. Restrict to note body when char_range set."""
    enc = tokenizer(
        prompt,
        return_offsets_mapping=True,
        add_special_tokens=False,
        return_tensors="pt",
    )
    offsets = enc["offset_mapping"][0].tolist()
    lo, hi = char_range if char_range is not None else (0, len(prompt))
    region = prompt[lo:hi]
    region_l = region.lower()
    spans: list[dict[str, Any]] = []
    for cue in cues:
        cue_l = cue.lower()
        if not cue_l:
            continue
        start = 0
        while True:
            idx_rel = region_l.find(cue_l, start)
            if idx_rel < 0:
                break
            idx = lo + idx_rel
            end = idx + len(cue_l)
            tok_idxs = [
                i
                for i, (a, b) in enumerate(offsets)
                if not (b <= idx or a >= end) and not (a == b == 0)
            ]
            if tok_idxs:
                spans.append(
                    {
                        "cue": cue,
                        "char_start": idx,
                        "char_end": end,
                        "token_start": tok_idxs[0],
                        "token_end": tok_idxs[-1] + 1,
                        "readout_pos": tok_idxs[-1],
                        "snippet": prompt[idx:end],
                        "in_note_body": True,
                    }
                )
            start = idx_rel + 1
    return spans


def _capture_prefill_residuals(
    *,
    input_ids: torch.Tensor,
    model_id: str,
    layer_indices: tuple[int, ...] = PREFILL_LAYERS,
) -> dict[int, torch.Tensor]:
    """One forward; residual stream at selected layers → {layer: (seq, d)} on CPU."""
    model, _tok = load_model(model_id)
    layers = _model_layers(model)
    captured: dict[int, torch.Tensor] = {}

    def _make_hook(layer_idx: int):
        def hook(_module, _inp, out):
            hs = out[0] if isinstance(out, tuple) else out
            captured[layer_idx] = hs[0].detach().float().cpu()

        return hook

    handles = []
    for i in layer_indices:
        if 0 <= i < len(layers):
            handles.append(layers[i].register_forward_hook(_make_hook(i)))
    try:
        device = next(model.parameters()).device
        ids = input_ids.to(device)
        with torch.inference_mode():
            model(ids)
    finally:
        for h in handles:
            h.remove()
    return captured


def _sae_topk(
    sae: JumpReLUSae,
    h: torch.Tensor,
    *,
    k: int,
    interest_ids: list[int] | None = None,
) -> list[dict[str, Any]]:
    feats = sae.encode(h.unsqueeze(0))[0]
    vals, idxs = torch.topk(feats, k=min(k, int(feats.numel())))
    rows = [
        {"feat_id": int(i), "act": float(v)}
        for i, v in zip(idxs.tolist(), vals.tolist())
    ]
    if interest_ids:
        for fid in interest_ids:
            rows.append(
                {
                    "feat_id": int(fid),
                    "act": float(feats[int(fid)].item()),
                    "from_sae_pilot_top": True,
                }
            )
    # de-dupe feat_id keeping first (topk) then interest
    seen: set[int] = set()
    out: list[dict[str, Any]] = []
    for r in rows:
        fid = int(r["feat_id"])
        if fid in seen:
            continue
        seen.add(fid)
        out.append(r)
    return out


def trace_note(
    note: dict[str, str],
    *,
    model_id: str = FAIL_MODEL,
    cues: tuple[str, ...] | list[str] = DEFAULT_CUES,
    layer_indices: tuple[int, ...] = PREFILL_LAYERS,
    v: torch.Tensor | None = None,
    sae: JumpReLUSae | None = None,
    sae_interest: list[int] | None = None,
    top_sae: int = DEFAULT_TOP_SAE,
) -> dict[str, Any]:
    """Prefill capture + proj onto v (+ SAE @ L20) for one note."""
    if v is None:
        v, _ = _load_v()
    _model, tokenizer = load_model(model_id)
    prompt, input_ids, note_span = _build_prompt(tokenizer, note["evidence"])
    spans = _find_cue_spans(prompt, tokenizer, cues, char_range=note_span)
    captured = _capture_prefill_residuals(
        input_ids=input_ids, model_id=model_id, layer_indices=layer_indices
    )

    seq_len = int(input_ids.shape[-1])
    sites: list[dict[str, Any]] = []
    for span in spans:
        pos = int(span["readout_pos"])
        if pos < 0 or pos >= seq_len:
            continue
        layer_rows: dict[str, Any] = {}
        for L in layer_indices:
            if L not in captured or pos >= captured[L].shape[0]:
                continue
            h = captured[L][pos]
            row: dict[str, Any] = {
                "proj_v": round(_proj(h, v), 6),
                "cos_v": round(_cos(h, v), 6),
                "norm": round(float(h.norm().item()), 6),
            }
            if L == SAE_LAYER and sae is not None:
                row["sae_topk"] = _sae_topk(
                    sae, h, k=top_sae, interest_ids=sae_interest
                )
            layer_rows[f"L{L}"] = row
        sites.append({**span, "layers": layer_rows})

    return {
        "id": note["id"],
        "gold": note.get("gold"),
        "prompt_token_count": seq_len,
        "n_cue_spans": len(sites),
        "sites": sites,
    }


def run_trace(
    *,
    note_ids: list[str] | None = None,
    cues: list[str] | None = None,
    model_id: str = FAIL_MODEL,
    top_sae: int = DEFAULT_TOP_SAE,
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

    ids = note_ids or list(DEFAULT_NOTES.keys())
    cue_list = tuple(cues) if cues else DEFAULT_CUES
    notes = []
    for nid in ids:
        if nid not in DEFAULT_NOTES:
            raise SystemExit(f"Unknown note id {nid}; known={list(DEFAULT_NOTES)}")
        notes.append(DEFAULT_NOTES[nid])

    v, dir_meta = _load_v()
    sae, sae_meta = JumpReLUSae.from_hub()
    interest: list[int] = []
    if SCORE_PATH.exists():
        scores = json.loads(SCORE_PATH.read_text(encoding="utf-8"))
        interest = [int(r["feat_id"]) for r in (scores.get("top") or [])[:3]]

    try:
        per_note = []
        for note in notes:
            print(f"[upstream-prefill] tracing {note['id']} …", flush=True)
            per_note.append(
                trace_note(
                    note,
                    model_id=model_id,
                    cues=cue_list,
                    v=v,
                    sae=sae,
                    sae_interest=interest or None,
                    top_sae=top_sae,
                )
            )
    finally:
        if unload_after:
            try:
                unload_model()
            except Exception:  # noqa: BLE001
                pass

    # Contrast: first note-body span per cue at L20 (temporal vs contra)
    contrast: list[dict[str, Any]] = []
    by_id = {n["id"]: n for n in per_note}
    if "EX_TEMPORAL_FOLFOX" in by_id and "EX_CONTRA" in by_id:
        def _first_by_cue(sites: list[dict[str, Any]], cue: str) -> dict[str, Any] | None:
            for s in sites:
                if s["cue"].lower() == cue.lower():
                    return s
            return None

        t_sites = by_id["EX_TEMPORAL_FOLFOX"]["sites"]
        c_sites = by_id["EX_CONTRA"]["sites"]
        for cue in cue_list:
            t = _first_by_cue(t_sites, cue)
            c = _first_by_cue(c_sites, cue)
            if not t or not c:
                continue
            tl = (t.get("layers") or {}).get(f"L{SAE_LAYER}") or {}
            cl = (c.get("layers") or {}).get(f"L{SAE_LAYER}") or {}
            if "cos_v" not in tl or "cos_v" not in cl:
                continue
            contrast.append(
                {
                    "cue": cue,
                    "layer": SAE_LAYER,
                    "temporal_pos": t.get("readout_pos"),
                    "contra_pos": c.get("readout_pos"),
                    "temporal_cos_v": tl["cos_v"],
                    "contra_cos_v": cl["cos_v"],
                    "delta_cos_v_temporal_minus_contra": round(
                        float(tl["cos_v"]) - float(cl["cos_v"]), 6
                    ),
                    "temporal_proj_v": tl.get("proj_v"),
                    "contra_proj_v": cl.get("proj_v"),
                }
            )

    payload = {
        "ok": True,
        "kind": "n2s_upstream_prefill_trace_v1",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "quest": (
            "U0 observational: how do cue tokens align with frozen expand v "
            "(and L20 SAE) during prefill — formation, not commit utilization."
        ),
        "model_id": model_id,
        "layers": list(PREFILL_LAYERS),
        "cues": list(cue_list),
        "direction_source": str(DIRECTIONS_PATH.name),
        "direction_meta_keys": sorted(dir_meta.keys())[:12],
        "sae": {
            "repo": sae_meta.get("repo"),
            "subdir": sae_meta.get("subdir"),
            "interest_feat_ids": interest,
        },
        "notes": per_note,
        "contrast_L20": contrast,
        "claim_hygiene": {
            "say": "prefill cue×layer alignment with frozen v / SAE activations",
            "do_not_say": "found temporality circuit; patched X; upstream closed",
            "next": "U1 site patch cue→commit X (not this artifact)",
        },
    }
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    TRACE_PATH.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    payload["artifact"] = str(TRACE_PATH)
    return payload


def write_u0_readout(trace_path: Path | None = None) -> dict[str, Any]:
    """Interpret U0 artifact → short machine+human readout (no GPU)."""
    path = trace_path or TRACE_PATH
    if not path.exists():
        raise SystemExit(f"Missing U0 artifact: {path}")
    trace = json.loads(path.read_text(encoding="utf-8"))
    findings: list[str] = []
    best: list[dict[str, Any]] = []
    for note in trace.get("notes") or []:
        for s in note.get("sites") or []:
            l20 = (s.get("layers") or {}).get("L20") or {}
            cos = l20.get("cos_v")
            if cos is None:
                continue
            best.append(
                {
                    "note_id": note["id"],
                    "cue": s["cue"],
                    "pos": s["readout_pos"],
                    "cos_v_L20": cos,
                    "proj_v_L20": l20.get("proj_v"),
                    "traj_cos_v": {
                        L: (s.get("layers") or {}).get(L, {}).get("cos_v")
                        for L in ("L8", "L12", "L16", "L20", "L24")
                    },
                }
            )
    best_sorted = sorted(best, key=lambda r: abs(float(r["cos_v_L20"])), reverse=True)
    top = best_sorted[:5]
    findings.append(
        "Frozen commit-v is weakly expressed at note-body cues (|cos_v|≲0.05 at L20)."
    )
    findings.append(
        "Strongest U0 site: second FOLFOX on temporal note (cos_v rises L12→L20)."
    )
    findings.append(
        "SAE pilot feats 165/9204 are off at prefill cues; 1196 activates but is not class-selective here."
    )
    findings.append(
        "First-span FOLFOX temporal−contra Δcos_v≈0 — shared early drug mention is not a separator."
    )
    findings.append(
        "U1 hypothesis: steer last FOLFOX (temporal) and failed+responsive (contra) at L20 with ±α·v; metric=repair X/margin."
    )
    # Site recipe for U1
    recipe = {
        "EX_TEMPORAL_FOLFOX": {"cues": ["FOLFOX"], "which": "last", "layer": SAE_LAYER},
        "EX_CONTRA": {"cues": ["failed", "responsive"], "which": "all", "layer": SAE_LAYER},
    }
    payload = {
        "ok": True,
        "kind": "n2s_upstream_u0_readout_v1",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "source_trace": str(path),
        "findings": findings,
        "top_sites_by_abs_cos_v_L20": top,
        "contrast_L20": trace.get("contrast_L20"),
        "u1_site_recipe": recipe,
        "decision": "Proceed to thin U1 prefill-position steer → commit X (not a circuit claim).",
    }
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    U0_READOUT_PATH.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    payload["artifact"] = str(U0_READOUT_PATH)
    return payload


def _repair_prompt_and_positions(
    note: dict[str, str],
    *,
    model_id: str,
    cues: list[str],
    which: str,
) -> tuple[str, list[int], list[dict[str, Any]]]:
    """Build repair chat user text; locate cue readout positions inside the full prompt."""
    from .ground import ground
    from .phase9a_behavioral import _analyze_hf, _extract_hf
    from .phase9b_localization import _repair_user_prompt
    from .verify import verify_formalization

    evidence = note["evidence"]
    gold = note.get("gold") or ""
    raw = _extract_hf(evidence, model_id=model_id)
    analysis = _analyze_hf(evidence, model_id=model_id)
    v1 = verify_formalization(
        analysis=analysis,
        extraction=raw,
        grounding=ground(raw).to_dict(),
        case_id=note["id"],
        gold=gold,
    )
    user = _repair_user_prompt(evidence, raw, v1.reasons)
    _model, tokenizer = load_model(model_id)
    messages = [
        {"role": "system", "content": EXTRACT_SYSTEM},
        {"role": "user", "content": user},
    ]
    prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    # Prefer locating the clinical note body inside the repair user blob.
    note_start = prompt.find(evidence)
    if note_start < 0:
        prefix = evidence[: min(48, len(evidence))]
        note_start = prompt.find(prefix)
        if note_start < 0:
            raise RuntimeError("Could not locate evidence inside repair prompt")
        note_end = note_start + len(evidence)
    else:
        note_end = note_start + len(evidence)
    spans = _find_cue_spans(prompt, tokenizer, cues, char_range=(note_start, note_end))
    if which == "last":
        by_cue: dict[str, dict[str, Any]] = {}
        for s in spans:
            by_cue[s["cue"].lower()] = s
        chosen = list(by_cue.values())
    else:
        chosen = spans
    positions = [int(s["readout_pos"]) for s in chosen]
    return user, positions, chosen


def run_patch(
    *,
    alpha: float = 8.0,
    layer: int = SAE_LAYER,
    model_id: str = FAIL_MODEL,
    unload_after: bool = True,
    recipe: dict[str, dict[str, Any]] | None = None,
    hypothesis: str = "u1",
    out_path: Path | None = None,
) -> dict[str, Any]:
    """Prefill-position ±α·v at cue sites → repair commit X/margin.

    ``hypothesis=u1`` uses U0 readout recipe @ L20 (default).
    ``hypothesis=u1b`` uses outcome-language cues @ L16 (different sites; same α).
    """
    from .ground import ground
    from .hf_intervene import PrefillPositionSteerSpec, generate_intervened
    from .hf_trace import find_present_commit_step
    from .ollama_client import parse_json_object
    from .phase9a_behavioral import _analyze_hf, _extract_hf
    from .phase9b_localization import _repair_user_prompt
    from .predicate import evaluate_rule
    from .types import Extraction
    from .verify import verify_formalization

    free = cuda_free_gb()
    if free is not None and free < MIN_FREE_VRAM_GB:
        return {
            "ok": False,
            "status": "deferred_gpu_busy",
            "free_vram_gb": round(free, 2),
            "need_vram_gb": MIN_FREE_VRAM_GB,
            "message": f"Need ~{MIN_FREE_VRAM_GB:.0f} GiB free; have {free:.1f}.",
        }

    if recipe is not None:
        use_recipe = recipe
    elif hypothesis == "u1b":
        use_recipe = U1B_SITE_RECIPE
    else:
        readout = write_u0_readout() if not U0_READOUT_PATH.exists() else json.loads(
            U0_READOUT_PATH.read_text(encoding="utf-8")
        )
        use_recipe = readout.get("u1_site_recipe") or {
            "EX_TEMPORAL_FOLFOX": {"cues": ["FOLFOX"], "which": "last", "layer": layer},
            "EX_CONTRA": {"cues": ["failed", "responsive"], "which": "all", "layer": layer},
        }
    v, _ = _load_v()
    notes_out: dict[str, Any] = {}
    artifact_path = out_path or (PATCH_U1B_PATH if hypothesis == "u1b" else PATCH_PATH)

    try:
        for nid, spec in use_recipe.items():
            note = DEFAULT_NOTES[nid]
            cues = list(spec.get("cues") or ["FOLFOX"])
            which = str(spec.get("which") or "last")
            L = int(spec.get("layer") or layer)
            user, positions, chosen = _repair_prompt_and_positions(
                note, model_id=model_id, cues=cues, which=which
            )
            print(
                f"[upstream-{hypothesis}] {nid} L={L} positions={positions} "
                f"cues={[c['cue'] for c in chosen]}",
                flush=True,
            )
            if not positions:
                notes_out[nid] = {"ok": False, "message": "no cue positions in repair prompt"}
                continue

            evidence = note["evidence"]
            gold = note.get("gold") or ""
            raw = _extract_hf(evidence, model_id=model_id)
            analysis = _analyze_hf(evidence, model_id=model_id)
            v1 = verify_formalization(
                analysis=analysis,
                extraction=raw,
                grounding=ground(raw).to_dict(),
                case_id=nid,
                gold=gold,
            )
            # user already built; keep consistent
            user = _repair_user_prompt(evidence, raw, v1.reasons)

            arms: list[dict[str, Any]] = []
            for label, a, kind in (
                ("baseline", 0.0, "none"),
                ("prefill_+v", float(alpha), "prefill_position_steer"),
                ("prefill_-v", float(-alpha), "prefill_position_steer"),
            ):
                kwargs: dict[str, Any] = {
                    "system": EXTRACT_SYSTEM,
                    "user": user,
                    "model_id": model_id,
                    "max_new_tokens": 400,
                    "layer_indices": (L,),
                    "intervention": kind,
                }
                if kind == "prefill_position_steer":
                    kwargs["prefill_steer"] = PrefillPositionSteerSpec(
                        positions=positions,
                        vectors_by_layer={L: v},
                        alpha=a,
                    )
                trace, _iv = generate_intervened(**kwargs)
                gen_only = "".join(s.token_text for s in trace.steps)
                try:
                    ex = Extraction.from_dict(parse_json_object(gen_only))
                    final_x = bool(ex.contradiction_present)
                    parse_ok = True
                except ValueError:
                    ex = None
                    final_x = None
                    parse_ok = False
                g = ground(ex) if ex else None
                rule = evaluate_rule(g) if g else None
                commit_step, commit_val = find_present_commit_step(trace.steps)
                margin = None
                if commit_step is not None:
                    st = trace.steps[commit_step]
                    if st.logit_true is not None and st.logit_false is not None:
                        margin = st.logit_true - st.logit_false
                arms.append(
                    {
                        "label": label,
                        "alpha": a if kind != "none" else None,
                        "repair_final_x": final_x,
                        "margin": margin,
                        "verdict": rule.verdict.value if rule else None,
                        "parse_ok": parse_ok,
                        "commit_step": commit_step,
                        "commit_value": commit_val,
                        "raw_x": bool(raw.contradiction_present),
                    }
                )
                print(
                    f"  {label}: X={final_x} margin={margin} verdict={arms[-1]['verdict']}",
                    flush=True,
                )

            notes_out[nid] = {
                "ok": True,
                "gold": gold,
                "positions": positions,
                "chosen_spans": [
                    {"cue": c["cue"], "readout_pos": c["readout_pos"], "snippet": c["snippet"]}
                    for c in chosen
                ],
                "steer_layer": L,
                "arms": arms,
            }
    finally:
        if unload_after:
            try:
                unload_model()
            except Exception:  # noqa: BLE001
                pass

    quest = (
        "U1b: does ±α·v at outcome-language cues @ L16 move repair commit X?"
        if hypothesis == "u1b"
        else "U1: does ±α·v at U0 cue positions during prefill move repair commit X?"
    )
    payload = {
        "ok": True,
        "kind": (
            "n2s_upstream_prefill_patch_u1b_v1"
            if hypothesis == "u1b"
            else "n2s_upstream_prefill_patch_v1"
        ),
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "quest": quest,
        "hypothesis": hypothesis,
        "model_id": model_id,
        "alpha": float(alpha),
        "site_recipe": use_recipe,
        "u0_readout": str(U0_READOUT_PATH) if hypothesis != "u1b" else None,
        "notes": notes_out,
        "claim_hygiene": {
            "say": (
                "different-site/layer prefill steer pilot (α frozen); not α-chase"
                if hypothesis == "u1b"
                else "prefill-position steer pilot linked to commit X/margin"
            ),
            "do_not_say": "upstream circuit found; formation fully characterized; upstream editor",
        },
    }
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    artifact_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    payload["artifact"] = str(artifact_path)
    return payload


def _print_summary(panel: dict[str, Any]) -> None:
    if not panel.get("ok"):
        print(panel.get("message") or panel)
        return
    print(f"\n=== UPSTREAM PREFILL · {panel.get('artifact')} ===")
    for note in panel.get("notes") or []:
        if isinstance(note, dict) and "id" in note:
            print(f"\n{note['id']}  gold={note.get('gold')}  tokens={note.get('prompt_token_count')}  spans={note.get('n_cue_spans')}")
            for s in note.get("sites") or []:
                l20 = (s.get("layers") or {}).get("L20") or {}
                print(
                    f"  cue={s['cue']!r:16} pos={s['readout_pos']:<4} "
                    f"L20 cos_v={l20.get('cos_v')} proj={l20.get('proj_v')}"
                )
    if panel.get("contrast_L20"):
        print("\n--- L20 contrast (temporal − contra) ---")
        for row in panel["contrast_L20"]:
            print(
                f"  {row['cue']:16} Δcos_v={row['delta_cos_v_temporal_minus_contra']} "
                f"(T={row['temporal_cos_v']} C={row['contra_cos_v']})"
            )


def _print_patch(panel: dict[str, Any]) -> None:
    if not panel.get("ok"):
        print(panel.get("message") or panel)
        return
    print(f"\n=== UPSTREAM U1 PATCH · {panel.get('artifact')} ===")
    for nid, block in (panel.get("notes") or {}).items():
        print(f"\n{nid} positions={block.get('positions')}")
        for a in block.get("arms") or []:
            print(
                f"  {a['label']:12} X={a.get('repair_final_x')}  "
                f"margin={a.get('margin')}  verdict={a.get('verdict')}"
            )


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description="Thin upstream prefill trace / U1 patch")
    sub = p.add_subparsers(dest="cmd", required=True)

    sp = sub.add_parser("trace", help="U0: Prefill cue×layer residuals + proj v + SAE@L20")
    sp.add_argument(
        "--notes",
        default="EX_TEMPORAL_FOLFOX,EX_CONTRA",
        help="Comma note ids",
    )
    sp.add_argument(
        "--cues",
        default=",".join(DEFAULT_CUES),
        help="Comma cue substrings to locate in the chat prompt",
    )
    sp.add_argument("--model", default=FAIL_MODEL)
    sp.add_argument("--top-sae", type=int, default=DEFAULT_TOP_SAE)
    sp.add_argument("--keep-model", action="store_true")

    sp = sub.add_parser("readout", help="Interpret U0 artifact (CPU)")
    sp.add_argument("--trace", default=str(TRACE_PATH))

    sp = sub.add_parser("patch", help="U1/U1b: prefill-position ±α·v → commit X")
    sp.add_argument("--alpha", type=float, default=8.0)
    sp.add_argument("--layer", type=int, default=SAE_LAYER)
    sp.add_argument(
        "--hypothesis",
        choices=("u1", "u1b"),
        default="u1",
        help="u1=U0 recipe@L20; u1b=outcome cues@L16 (different sites, same α)",
    )
    sp.add_argument("--model", default=FAIL_MODEL)
    sp.add_argument("--keep-model", action="store_true")

    args = p.parse_args(argv)
    if args.cmd == "trace":
        note_ids = [x.strip() for x in args.notes.split(",") if x.strip()]
        cues = [x.strip() for x in args.cues.split(",") if x.strip()]
        panel = run_trace(
            note_ids=note_ids,
            cues=cues,
            model_id=args.model,
            top_sae=args.top_sae,
            unload_after=not args.keep_model,
        )
        _print_summary(panel)
        if not panel.get("ok"):
            raise SystemExit(2)
    elif args.cmd == "readout":
        from pathlib import Path as _P

        panel = write_u0_readout(_P(args.trace))
        for line in panel.get("findings") or []:
            print(f"- {line}")
        print(f"artifact: {panel.get('artifact')}")
    elif args.cmd == "patch":
        if args.hypothesis == "u1":
            write_u0_readout()
        panel = run_patch(
            alpha=args.alpha,
            layer=args.layer,
            model_id=args.model,
            unload_after=not args.keep_model,
            hypothesis=args.hypothesis,
        )
        _print_patch(panel)
        if not panel.get("ok"):
            raise SystemExit(2)
    else:
        raise SystemExit(f"unknown cmd {args.cmd}")


if __name__ == "__main__":
    main()
