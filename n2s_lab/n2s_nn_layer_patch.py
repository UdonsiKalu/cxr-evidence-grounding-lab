"""First-lost-layer n2n patch along frozen U-A d (Dual-shaped control).

Detect (CPU) → prefill h += sign·α·d at earlier layer → re-read layers + extract.
If Dual analog would AUTO a wrong answer → fail patch, fallback REVIEW.

See docs/TRACKB-NN-LAYER-PATCH.md. Not α=8 / U-C / U-D / L24-only causal.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from typing import Any

import torch

from .auto_contract import classify_case
from .ground import ground
from .hf_client import load_model, unload_model
from .hf_intervene import PrefillPositionSteerSpec, generate_intervened
from .hf_trace import _model_layers, find_present_commit_step
from .neural import EXTRACT_SYSTEM
from .n2s_forensics import MIN_FREE_VRAM_GB, cuda_free_gb
from .n2s_nn_layer_loss import (
    LAYERS,
    analyze_note,
    apply_n2n_lost_review,
    detect_patch_plan,
    first_lost_step,
    frozen_temporal_scores,
)
from .n2s_upstream_live import ensure_ua_directions
from .n2s_upstream_prefill import DEFAULT_NOTES, _build_prompt, _unit
from .n2s_upstream_ua_map import CONTRA_ID, MAP_PATH, TEMPORAL_ID, _note_body_token_range
from .ollama_client import parse_json_object
from .paths import ARTIFACTS_DIR
from .phase10_activation import FAIL_MODEL
from .predicate import evaluate_rule
from .trackb_family_vector import _unit as _unit_vec
from .types import Extraction

PANEL_PATH = ARTIFACTS_DIR / "n2s-nn-layer-patch-panel.json"
READOUT_PATH = ARTIFACTS_DIR / "n2s-nn-layer-patch-readout.json"
STACK_PANEL_PATH = ARTIFACTS_DIR / "n2s-nn-layer-patch-l12l16-panel.json"
STACK_READOUT_PATH = ARTIFACTS_DIR / "n2s-nn-layer-patch-l12l16-readout.json"
RESIDUAL_PANEL_PATH = ARTIFACTS_DIR / "n2s-nn-layer-patch-l12l16l20-panel.json"
RESIDUAL_READOUT_PATH = ARTIFACTS_DIR / "n2s-nn-layer-patch-l12l16l20-readout.json"
DEFAULT_ALPHAS = (1.0, 2.0)


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


def _note_body_positions(nid: str) -> list[int]:
    ua = json.loads(MAP_PATH.read_text(encoding="utf-8"))
    meta = (ua.get("notes") or {}).get(nid) or {}
    t0 = int(meta.get("note_token_start") or 0)
    t1 = int(meta.get("note_token_end") or 0)
    if t1 <= t0:
        raise RuntimeError(f"bad note-body range for {nid}: [{t0},{t1})")
    return list(range(t0, t1))


def _side_for_note(nid: str) -> str:
    gold = (DEFAULT_NOTES[nid].get("gold") or "").upper()
    return "contra" if gold == "CONTRADICTION" else "temporal"


def _signed_vec(d: torch.Tensor, side: str) -> torch.Tensor:
    """Temporal +d (keep score up); contra −d (keep score down/negative)."""
    v = _unit_vec(d.float().cpu())
    return -v if side == "contra" else v


def _capture_prefill_patched(
    *,
    evidence: str,
    model_id: str,
    layer_indices: tuple[int, ...],
    patch_spec: dict[int, torch.Tensor],
    alpha: float,
    positions: list[int],
) -> tuple[dict[int, torch.Tensor], int, int]:
    """One forward; optional prefill push at each patch layer on note-body positions."""
    model, tokenizer = load_model(model_id)
    prompt, input_ids, note_span = _build_prompt(tokenizer, evidence)
    t0, t1 = _note_body_token_range(tokenizer, prompt, note_span)
    layers = _model_layers(model)
    captured: dict[int, torch.Tensor] = {}
    pos_set = sorted({int(p) for p in positions if p >= 0})
    need_len = (max(pos_set) + 1) if pos_set else 0

    def _make_capture(layer_idx: int):
        def hook(_module, _inp, out):
            hs = out[0] if isinstance(out, tuple) else out
            captured[layer_idx] = hs[0].detach().float().cpu()

        return hook

    def _make_patch(vec_cpu: torch.Tensor, a: float):
        def hook(_module, _inp, out):
            hs = out[0] if isinstance(out, tuple) else out
            if hs.shape[1] < need_len:
                return
            modified = hs.clone()
            v = vec_cpu.to(device=modified.device, dtype=modified.dtype)
            for pos in pos_set:
                if pos < modified.shape[1]:
                    modified[0, pos, :] = modified[0, pos, :] + a * v
            if isinstance(out, tuple):
                return (modified,) + out[1:]
            return modified

        return hook

    handles = []
    if alpha and pos_set:
        for L, vec in patch_spec.items():
            if 0 <= int(L) < len(layers) and vec is not None:
                handles.append(
                    layers[int(L)].register_forward_hook(_make_patch(vec, alpha))
                )
    for i in layer_indices:
        if 0 <= i < len(layers):
            handles.append(layers[i].register_forward_hook(_make_capture(i)))
    try:
        device = next(model.parameters()).device
        ids = input_ids.to(device)
        with torch.inference_mode():
            model(ids)
    finally:
        for h in handles:
            h.remove()
    body: dict[int, torch.Tensor] = {}
    for L, h in captured.items():
        body[L] = h[t0:t1].clone()
    return body, t0, t1


def _mean_score_d(h: torch.Tensor, d: torch.Tensor) -> float:
    scores = h.float() @ _unit(d.float().reshape(-1))
    return float(scores.mean().item())


def _scores_from_body(
    body: dict[int, torch.Tensor],
    d_by_layer: dict[str, torch.Tensor],
) -> dict[int, float]:
    out: dict[int, float] = {}
    for L in LAYERS:
        h = body.get(L)
        d = d_by_layer.get(f"L{L}")
        if h is None or d is None:
            continue
        out[L] = _mean_score_d(h, d)
    return out


def _run_extract(
    *,
    evidence: str,
    model_id: str,
    patch_spec: dict[int, torch.Tensor] | None,
    alpha: float,
    positions: list[int],
    max_new_tokens: int = 400,
) -> dict[str, Any]:
    layer_keys = tuple(sorted(patch_spec.keys())) if patch_spec else (12,)
    kwargs: dict[str, Any] = {
        "system": EXTRACT_SYSTEM,
        "user": evidence,
        "model_id": model_id,
        "max_new_tokens": max_new_tokens,
        "layer_indices": layer_keys,
    }
    if not patch_spec or abs(float(alpha)) < 1e-12:
        kwargs["intervention"] = "none"
    else:
        kwargs["intervention"] = "prefill_position_steer"
        kwargs["prefill_steer"] = PrefillPositionSteerSpec(
            positions=list(positions),
            vectors_by_layer={int(k): v for k, v in patch_spec.items()},
            alpha=float(alpha),
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
            margin = float(st.logit_true - st.logit_false)
    verdict = rule.verdict.value if rule else None
    gold = None
    return {
        "extract_x": final_x,
        "margin": None if margin is None else round(margin, 6),
        "commit_step": commit_step,
        "commit_value": commit_val,
        "parse_ok": parse_ok,
        "verdict": verdict,
        "grounded_X": None if g is None else bool(g.contradiction),
        "gold_unused": gold,
    }


def _dual_control(
    *,
    gold: str,
    extract_x: bool | None,
    verdict: str | None,
    baseline_extract_x: bool | None = None,
    baseline_bucket: str | None = None,
) -> dict[str, Any]:
    """Dual-shaped AUTO contract.

    Patch fail = introduced AUTO-wrong vs baseline, or contra flipped to all-good.
    Pre-existing 7B Dual analog errors (e.g. FOLFOX NOT_SATISFIED) are not patch fails.
    """
    classified = classify_case(gold=gold, verdict=verdict)
    all_good = extract_x is False or classified["verdict"] == "SATISFIED"
    wreck_contra = gold == "CONTRADICTION" and all_good
    if baseline_bucket is None:
        fail = wreck_contra
        introduced = False
    else:
        introduced = bool(classified["wrong_AUTO"] and baseline_bucket != "wrong_AUTO")
        newly_flipped = bool(
            gold == "CONTRADICTION"
            and extract_x is False
            and baseline_extract_x is not False
        )
        fail = bool(introduced or newly_flipped or wreck_contra)
    return {
        **classified,
        "contra_flipped_all_good": wreck_contra,
        "introduced_wrong_AUTO": introduced,
        "control_fail": fail,
        "fallback": "REVIEW" if fail else None,
    }


def _layer_rescue(
    *,
    baseline_steps: list[dict[str, Any]],
    patched_steps: list[dict[str, Any]],
) -> dict[str, Any]:
    first = first_lost_step(baseline_steps)
    if first is None:
        still = first_lost_step(patched_steps)
        remaining = [s for s in patched_steps if s.get("lost")]
        return {
            "had_loss": False,
            "rescued": False,
            "no_new_loss": still is None,
            "first_lost_still": still,
            "n_lost_after": len(remaining),
            "cleared_all": len(remaining) == 0,
            "residual_rescued": False,
        }
    match = next(
        (
            s
            for s in patched_steps
            if s["from_layer"] == first["from_layer"] and s["to_layer"] == first["to_layer"]
        ),
        None,
    )
    rescued = bool(match) and not match.get("lost")
    remaining = [s for s in patched_steps if s.get("lost")]
    baseline_lost = [s for s in baseline_steps if s.get("lost")]
    residual_before = baseline_lost[1] if len(baseline_lost) > 1 else None
    residual_after = None
    residual_rescued = False
    if residual_before is not None:
        residual_after = next(
            (
                s
                for s in patched_steps
                if s["from_layer"] == residual_before["from_layer"]
                and s["to_layer"] == residual_before["to_layer"]
            ),
            None,
        )
        residual_rescued = bool(residual_after) and not residual_after.get("lost")
    return {
        "had_loss": True,
        "rescued": rescued,
        "first_lost_before": first,
        "first_lost_after": match,
        "n_lost_after": len(remaining),
        "cleared_all": len(remaining) == 0,
        "residual_before": residual_before,
        "residual_after": residual_after,
        "residual_rescued": residual_rescued,
    }


def run_layer_patch_panel(
    *,
    model_id: str = FAIL_MODEL,
    alphas: tuple[float, ...] = DEFAULT_ALPHAS,
    unload_after: bool = True,
    stack: bool = False,
    residual: bool = False,
) -> dict[str, Any]:
    plan = detect_patch_plan()
    patch_layer = plan.get("patch_layer")
    if patch_layer is None:
        return {"ok": False, "error": "no first lost step on frozen CONTRA"}
    if residual:
        patch_layers = list(
            plan.get("residual_layers") or plan.get("patch_layers") or [patch_layer]
        )
    elif stack:
        patch_layers = list(plan.get("patch_layers") or [patch_layer])
    else:
        patch_layers = [int(patch_layer)]

    gate = _gpu_gate()
    if gate:
        return {**gate, "detect": plan}

    dirs = ensure_ua_directions(model_id=model_id, unload_after=False)
    if not dirs.get("ok"):
        return dirs
    blob = dirs["_blob"]
    d_by_layer: dict[str, torch.Tensor] = blob["by_layer"]
    d_fallback = d_by_layer.get("L24")
    t_scores = frozen_temporal_scores()
    notes = (CONTRA_ID, TEMPORAL_ID)
    rows: list[dict[str, Any]] = []
    if residual:
        out_path = RESIDUAL_PANEL_PATH
        kind = "n2s_nn_layer_patch_residual_v1"
    elif stack:
        out_path = STACK_PANEL_PATH
        kind = "n2s_nn_layer_patch_stack_v1"
    else:
        out_path = PANEL_PATH
        kind = "n2s_nn_layer_patch_panel_v1"

    def _vecs_for_side(side: str) -> dict[int, torch.Tensor]:
        out: dict[int, torch.Tensor] = {}
        for L in patch_layers:
            d = d_by_layer.get(f"L{L}")
            if d is None:
                d = d_fallback
            if d is None:
                continue
            out[int(L)] = _signed_vec(d, side)
        return out

    try:
        for nid in notes:
            note = DEFAULT_NOTES[nid]
            evidence = note["evidence"]
            gold = str(note.get("gold") or "")
            side = _side_for_note(nid)
            positions = _note_body_positions(nid)
            patch_spec = _vecs_for_side(side)
            case = {
                "id": nid,
                "expected": gold,
                "category": "ua_pair",
                "evidence": evidence,
            }

            print(f"[nn-patch] {nid} baseline capture + extract …", flush=True)
            body0, _, _ = _capture_prefill_patched(
                evidence=evidence,
                model_id=model_id,
                layer_indices=LAYERS,
                patch_spec=patch_spec,
                alpha=0.0,
                positions=positions,
            )
            scores0 = _scores_from_body(body0, d_by_layer)
            rec0 = analyze_note(case, scores=scores0, temporal_scores=t_scores)
            ex0 = _run_extract(
                evidence=evidence,
                model_id=model_id,
                patch_spec=None,
                alpha=0.0,
                positions=positions,
            )
            dual0 = _dual_control(
                gold=gold, extract_x=ex0.get("extract_x"), verdict=ex0.get("verdict")
            )
            rows.append(
                {
                    "ok": True,
                    "case_id": nid,
                    "gold": gold,
                    "side": side,
                    "label": "baseline",
                    "alpha": 0.0,
                    "patch_layer": patch_layer,
                    "patch_layers": patch_layers,
                    "sign": -1.0 if side == "contra" else 1.0,
                    "mean_score_d": rec0["mean_score_d"],
                    "steps": rec0["steps"],
                    "n_lost_steps": rec0["n_lost_steps"],
                    "layer_control_ok": rec0["control_ok"],
                    "n2n_review": rec0.get("n2n_review"),
                    **ex0,
                    "dual": dual0,
                    "rescue": None,
                }
            )

            for a in alphas:
                print(
                    f"[nn-patch] {nid} patch L{'+'.join(str(x) for x in patch_layers)} α={a} …",
                    flush=True,
                )
                body, _, _ = _capture_prefill_patched(
                    evidence=evidence,
                    model_id=model_id,
                    layer_indices=LAYERS,
                    patch_spec=patch_spec,
                    alpha=float(a),
                    positions=positions,
                )
                scores = _scores_from_body(body, d_by_layer)
                rec = analyze_note(case, scores=scores, temporal_scores=t_scores)
                ex = _run_extract(
                    evidence=evidence,
                    model_id=model_id,
                    patch_spec=patch_spec,
                    alpha=float(a),
                    positions=positions,
                )
                dual = _dual_control(
                    gold=gold,
                    extract_x=ex.get("extract_x"),
                    verdict=ex.get("verdict"),
                    baseline_extract_x=ex0.get("extract_x"),
                    baseline_bucket=dual0.get("bucket"),
                )
                rescue = _layer_rescue(
                    baseline_steps=rec0["steps"], patched_steps=rec["steps"]
                )
                rows.append(
                    {
                        "ok": True,
                        "case_id": nid,
                        "gold": gold,
                        "side": side,
                        "label": "patch",
                        "alpha": float(a),
                        "patch_layer": patch_layer,
                        "patch_layers": patch_layers,
                        "sign": -1.0 if side == "contra" else 1.0,
                        "mean_score_d": rec["mean_score_d"],
                        "steps": rec["steps"],
                        "n_lost_steps": rec["n_lost_steps"],
                        "layer_control_ok": rec["control_ok"],
                        "n2n_review": rec.get("n2n_review"),
                        **ex,
                        "dual": dual,
                        "rescue": rescue,
                    }
                )
    finally:
        if unload_after:
            try:
                unload_model()
            except Exception:  # noqa: BLE001
                pass

    gate_out = _panel_gate(rows, patch_layer=int(patch_layer))
    panel = {
        "ok": True,
        "kind": kind,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "protocol": "docs/TRACKB-NN-LAYER-PATCH.md",
        "quest": plan["quest"],
        "model_id": model_id,
        "detect": plan,
        "patch_layer": patch_layer,
        "patch_layers": patch_layers,
        "stack": bool(stack) and not residual,
        "residual": bool(residual),
        "alphas": list(alphas),
        "site": "prefill_note_body",
        "vector": (
            f"frozen U-A d @ L{'+L'.join(str(x) for x in patch_layers)} "
            "(sign + temporal / − contra)"
        ),
        "rows": rows,
        "gate": gate_out,
        "claim_hygiene": {
            "say": (
                "residual L12+L16+L20 frozen-d patch did/did not clear remaining "
                "L16→L20 lost under Dual AUTO-wrong control"
                if residual
                else (
                    "first-lost-layer frozen-d patch did/did not rescue the step "
                    "under Dual AUTO-wrong control"
                )
            ),
            "do_not_say": (
                "inside-net REVIEW; temporality circuit; production editor; Dual_full 32B"
            ),
        },
    }
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(panel, indent=2) + "\n", encoding="utf-8")
    return panel


def _panel_gate(rows: list[dict[str, Any]], *, patch_layer: int) -> dict[str, Any]:
    def _arm(nid: str, label: str, alpha: float | None = None) -> dict[str, Any] | None:
        for r in rows:
            if r.get("case_id") != nid or r.get("label") != label:
                continue
            if alpha is not None and abs(float(r.get("alpha") or 0) - alpha) > 1e-9:
                continue
            return r
        return None

    by_alpha: list[dict[str, Any]] = []
    any_control_fail = False
    any_rescue = False
    any_full = False
    for a in DEFAULT_ALPHAS:
        c = _arm(CONTRA_ID, "patch", float(a))
        t = _arm(TEMPORAL_ID, "patch", float(a))
        c_fail = bool(c and (c.get("dual") or {}).get("control_fail"))
        t_fail = bool(t and (t.get("dual") or {}).get("control_fail"))
        rescue = (c or {}).get("rescue") or {}
        rescued = bool(c and rescue.get("rescued"))
        residual_rescued = bool(c and rescue.get("residual_rescued"))
        n_lost = None if not c else c.get("n_lost_steps")
        cleared_all = n_lost == 0
        control_fail = c_fail or t_fail
        any_control_fail = any_control_fail or control_fail
        any_rescue = any_rescue or (rescued and not control_fail)
        any_full = any_full or (cleared_all and not control_fail)
        fallback = "REVIEW" if control_fail else None
        if control_fail:
            status = "FAIL_CONTROL"
        elif cleared_all:
            status = "FULL"
        elif rescued:
            status = "RESCUE"
        else:
            status = "NULL"
        by_alpha.append(
            {
                "alpha": a,
                "patch_layer": patch_layer,
                "contra_rescued": rescued,
                "contra_residual_rescued": residual_rescued,
                "contra_cleared_all": cleared_all,
                "contra_n_lost": n_lost,
                "contra_extract_x": None if not c else c.get("extract_x"),
                "contra_verdict": None if not c else c.get("verdict"),
                "temporal_extract_x": None if not t else t.get("extract_x"),
                "temporal_verdict": None if not t else t.get("verdict"),
                "control_fail": control_fail,
                "fallback": fallback,
                "status": status,
            }
        )

    if any_control_fail:
        note = (
            "FAIL — Dual analog would AUTO a wrong answer "
            "(or contra flipped all-good). Fallback REVIEW."
        )
        overall = "FAIL_CONTROL"
    elif any_full:
        note = "PASS — CONTRA n_lost=0 at some α; Dual control held."
        overall = "FULL"
    elif any_rescue:
        note = (
            "PASS/PARTIAL — first lost step rescued at some α; "
            "Dual control held. Residual L16→L20 may remain."
        )
        overall = "RESCUE"
    else:
        note = (
            "NULL — patch did not clear CONTRA’s first lost step; Dual control held."
        )
        overall = "NULL"

    return {
        "overall": overall,
        "note": note,
        "fallback_if_fail": "REVIEW",
        "by_alpha": by_alpha,
    }


def recompute_gate_from_artifact() -> dict[str, Any]:
    """Rewrite dual/gate on a saved panel without GPU (after control-delta fix)."""
    panel = json.loads(PANEL_PATH.read_text(encoding="utf-8"))
    rows = panel.get("rows") or []
    by_base: dict[str, dict[str, Any]] = {}
    for r in rows:
        if r.get("label") == "baseline":
            by_base[r["case_id"]] = r
    for r in rows:
        gold = str(r.get("gold") or "")
        if r.get("label") == "baseline":
            r["dual"] = _dual_control(
                gold=gold,
                extract_x=r.get("extract_x"),
                verdict=r.get("verdict"),
            )
            continue
        base = by_base.get(r["case_id"]) or {}
        r["dual"] = _dual_control(
            gold=gold,
            extract_x=r.get("extract_x"),
            verdict=r.get("verdict"),
            baseline_extract_x=base.get("extract_x"),
            baseline_bucket=(base.get("dual") or {}).get("bucket"),
        )
        if r.get("rescue") and not (r["rescue"].get("had_loss")):
            r["rescue"]["rescued"] = False
            r["rescue"].setdefault("no_new_loss", True)
    panel["gate"] = _panel_gate(rows, patch_layer=int(panel.get("patch_layer") or 0))
    panel["rows"] = rows
    PANEL_PATH.write_text(json.dumps(panel, indent=2) + "\n", encoding="utf-8")
    write_readout(panel)
    return panel


def write_readout(
    panel: dict[str, Any] | None = None,
    *,
    stack: bool = False,
    residual: bool = False,
) -> dict[str, Any]:
    if residual:
        src, dest_default, artifact = (
            RESIDUAL_PANEL_PATH,
            RESIDUAL_READOUT_PATH,
            RESIDUAL_PANEL_PATH.name,
        )
    elif stack:
        src, dest_default, artifact = (
            STACK_PANEL_PATH,
            STACK_READOUT_PATH,
            STACK_PANEL_PATH.name,
        )
    else:
        src, dest_default, artifact = PANEL_PATH, READOUT_PATH, PANEL_PATH.name
    if panel is None:
        panel = json.loads(src.read_text(encoding="utf-8"))
    residual = bool(panel.get("residual") or residual)
    stack = bool(panel.get("stack") or stack) and not residual
    if residual:
        dest, artifact = RESIDUAL_READOUT_PATH, RESIDUAL_PANEL_PATH.name
    elif stack:
        dest, artifact = STACK_READOUT_PATH, STACK_PANEL_PATH.name
    else:
        dest, artifact = dest_default, artifact
    gate = panel.get("gate") or {}
    detect = panel.get("detect") or {}
    readout = {
        "ok": panel.get("ok"),
        "kind": "n2s_nn_layer_patch_readout_v1",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "patch_layer": panel.get("patch_layer"),
        "patch_layers": panel.get("patch_layers"),
        "stack": stack,
        "residual": residual,
        "first_lost": detect.get("first_lost"),
        "overall": gate.get("overall"),
        "note": gate.get("note"),
        "by_alpha": gate.get("by_alpha"),
        "artifact": artifact,
    }
    dest.write_text(json.dumps(readout, indent=2) + "\n", encoding="utf-8")
    return readout


def main() -> None:
    parser = argparse.ArgumentParser(description="N2S first-lost-layer d patch")
    parser.add_argument(
        "cmd",
        choices=("detect", "panel", "panel-stack", "panel-residual", "readout", "recompute"),
    )
    args = parser.parse_args()
    if args.cmd == "detect":
        plan = detect_patch_plan()
        print(json.dumps(plan, indent=2))
        return
    if args.cmd == "readout":
        print(json.dumps(write_readout(), indent=2))
        return
    if args.cmd == "recompute":
        panel = recompute_gate_from_artifact()
        print(json.dumps(panel.get("gate"), indent=2))
        return
    residual = args.cmd == "panel-residual"
    stack = args.cmd == "panel-stack"
    panel = run_layer_patch_panel(stack=stack, residual=residual)
    write_readout(panel, stack=stack, residual=residual)
    print(json.dumps(panel.get("gate") or panel, indent=2))
    if not panel.get("ok"):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
