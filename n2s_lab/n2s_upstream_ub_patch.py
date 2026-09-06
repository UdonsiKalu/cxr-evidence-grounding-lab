"""U-B / U-B2 — component ablation at U-A top sites (attn vs MLP).

U-B: zero-ablate @ L24. U-B2: mean-ablate @ L20 (rec after U-B null).
Measures extract commit X / margin. Not α·v. See docs/TRACKB-UPSTREAM-PROGRAM.md.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

from .ground import ground
from .hf_client import unload_model
from .hf_intervene import PrefillComponentAblateSpec, PrefillComponentZeroSpec, generate_intervened
from .hf_trace import find_present_commit_step
from .neural import EXTRACT_SYSTEM
from .n2s_forensics import MIN_FREE_VRAM_GB, cuda_free_gb
from .n2s_upstream_prefill import DEFAULT_NOTES
from .n2s_upstream_ua_map import MAP_PATH, READOUT_PATH, TEMPORAL_ID, CONTRA_ID
from .ollama_client import parse_json_object
from .paths import ARTIFACTS_DIR
from .phase10_activation import FAIL_MODEL
from .predicate import evaluate_rule
from .types import Extraction

PATCH_PATH = ARTIFACTS_DIR / "n2s-upstream-ub-patch.json"
PATCH_UB2_PATH = ARTIFACTS_DIR / "n2s-upstream-ub2-patch.json"
UB_READOUT_PATH = ARTIFACTS_DIR / "n2s-upstream-ub-readout.json"
UB2_READOUT_PATH = ARTIFACTS_DIR / "n2s-upstream-ub2-readout.json"
DEFAULT_TOP_SITES = 5
DEFAULT_LAYER = 24
Component = Literal["attn", "mlp", "resid"]
AblateMode = Literal["zero", "mean"]


def _sites_from_ua(
    map_path: Path,
    *,
    layer: int,
    top_k: int,
) -> tuple[dict[str, list[dict[str, Any]]], dict[str, tuple[int, int]]]:
    panel = json.loads(map_path.read_text(encoding="utf-8"))
    key = f"L{layer}"
    block = (panel.get("by_layer") or {}).get(key)
    if not block:
        raise SystemExit(f"U-A map missing {key}")
    out: dict[str, list[dict[str, Any]]] = {}
    ranges: dict[str, tuple[int, int]] = {}
    for nid in (TEMPORAL_ID, CONTRA_ID):
        tops = (block.get("notes") or {}).get(nid, {}).get("top_sites") or []
        out[nid] = tops[:top_k]
        meta = (panel.get("notes") or {}).get(nid) or {}
        ranges[nid] = (
            int(meta.get("note_token_start") or 0),
            int(meta.get("note_token_end") or 0),
        )
    return out, ranges


def _run_arm(
    *,
    evidence: str,
    model_id: str,
    layer: int,
    positions: list[int],
    component: Component | None,
    mode: AblateMode = "zero",
    note_token_start: int | None = None,
    note_token_end: int | None = None,
    max_new_tokens: int = 400,
) -> dict[str, Any]:
    kwargs: dict[str, Any] = {
        "system": EXTRACT_SYSTEM,
        "user": evidence,
        "model_id": model_id,
        "max_new_tokens": max_new_tokens,
        "layer_indices": (layer,),
    }
    if component is None:
        kwargs["intervention"] = "none"
    elif mode == "zero":
        kwargs["intervention"] = "prefill_component_zero"
        kwargs["prefill_component_zero"] = PrefillComponentZeroSpec(
            layer=layer,
            positions=positions,
            component=component,
        )
    else:
        kwargs["intervention"] = "prefill_component_ablate"
        kwargs["prefill_component_ablate"] = PrefillComponentAblateSpec(
            layer=layer,
            positions=positions,
            component=component,
            mode="mean",
            note_token_start=note_token_start,
            note_token_end=note_token_end,
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
    return {
        "component": component or "baseline",
        "mode": mode if component else None,
        "extract_x": final_x,
        "margin": margin,
        "verdict": rule.verdict.value if rule else None,
        "parse_ok": parse_ok,
        "commit_step": commit_step,
        "commit_value": commit_val,
        "n_gen_tokens": len(trace.steps),
    }


def run_ub_patch(
    *,
    map_path: Path | None = None,
    model_id: str = FAIL_MODEL,
    layer: int = DEFAULT_LAYER,
    top_k: int = DEFAULT_TOP_SITES,
    mode: AblateMode = "zero",
    variant: str = "ub",
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

    path = map_path or MAP_PATH
    if not path.exists():
        raise SystemExit(f"Missing U-A map (run ua map first): {path}")
    sites, ranges = _sites_from_ua(path, layer=layer, top_k=top_k)
    notes_out: dict[str, Any] = {}
    out_path = PATCH_UB2_PATH if variant == "ub2" else PATCH_PATH

    try:
        for nid in (TEMPORAL_ID, CONTRA_ID):
            note = DEFAULT_NOTES[nid]
            tops = sites[nid]
            positions = [int(t["global_pos"]) for t in tops]
            tokens = [t.get("token") for t in tops]
            t0, t1 = ranges[nid]
            print(
                f"[ub] {nid} L{layer} mode={mode} positions={positions} tokens={tokens}",
                flush=True,
            )
            arms: list[dict[str, Any]] = []
            for comp in (None, "attn", "mlp", "resid"):
                label = comp or "baseline"
                print(f"  arm={label} …", flush=True)
                arm = _run_arm(
                    evidence=note["evidence"],
                    model_id=model_id,
                    layer=layer,
                    positions=positions,
                    component=comp,  # type: ignore[arg-type]
                    mode=mode,
                    note_token_start=t0,
                    note_token_end=t1,
                )
                arms.append(arm)
                print(
                    f"    X={arm['extract_x']} margin={arm['margin']} "
                    f"verdict={arm['verdict']}",
                    flush=True,
                )
            baseline = next(a for a in arms if a["component"] == "baseline")
            deltas = []
            for a in arms:
                if a["component"] == "baseline":
                    continue
                dm = None
                if a["margin"] is not None and baseline["margin"] is not None:
                    dm = a["margin"] - baseline["margin"]
                deltas.append(
                    {
                        "component": a["component"],
                        "x_flipped": a["extract_x"] != baseline["extract_x"]
                        and a["extract_x"] is not None
                        and baseline["extract_x"] is not None,
                        "delta_margin": dm,
                        "extract_x": a["extract_x"],
                        "baseline_x": baseline["extract_x"],
                    }
                )
            notes_out[nid] = {
                "ok": True,
                "gold": note.get("gold"),
                "layer": layer,
                "mode": mode,
                "note_token_range": [t0, t1],
                "positions": positions,
                "tokens": tokens,
                "arms": arms,
                "deltas_vs_baseline": deltas,
            }
    finally:
        if unload_after:
            try:
                unload_model()
            except Exception:  # noqa: BLE001
                pass

    payload = {
        "ok": True,
        "kind": (
            "n2s_upstream_ub2_patch_v1" if variant == "ub2" else "n2s_upstream_ub_patch_v1"
        ),
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "quest": (
            f"U-B2: does {mode}-ablating attn vs mlp vs resid at U-A L{layer} top sites "
            "during prefill move extract commit X/margin?"
            if variant == "ub2"
            else (
                "U-B: does zeroing attn vs mlp vs resid at U-A top sites during prefill "
                "move extract commit X/margin?"
            )
        ),
        "variant": variant,
        "model_id": model_id,
        "layer": layer,
        "mode": mode,
        "top_k": top_k,
        "ua_map": str(path),
        "ua_readout": str(READOUT_PATH) if READOUT_PATH.exists() else None,
        "notes": notes_out,
        "claim_hygiene": {
            "say": "component ablation at formation sites; attn vs MLP shortlist",
            "do_not_say": "full circuit; freeze-v editor; α-chase",
        },
    }
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    payload["artifact"] = str(out_path)
    return payload


def write_ub_readout(
    patch_path: Path | None = None,
    *,
    variant: str = "ub",
) -> dict[str, Any]:
    path = patch_path or (PATCH_UB2_PATH if variant == "ub2" else PATCH_PATH)
    if not path.exists():
        raise SystemExit(f"Missing U-B patch: {path}")
    panel = json.loads(path.read_text(encoding="utf-8"))
    findings: list[str] = []
    shortlist: list[dict[str, Any]] = []
    flips = 0
    for nid, block in (panel.get("notes") or {}).items():
        for d in block.get("deltas_vs_baseline") or []:
            dm = d.get("delta_margin")
            findings.append(
                f"{nid} {panel.get('mode', 'zero')}-{d['component']}@L{block.get('layer')}: "
                f"X {d.get('baseline_x')}→{d.get('extract_x')} "
                f"Δmargin={dm} flip={d.get('x_flipped')}"
            )
            if d.get("x_flipped"):
                flips += 1
                shortlist.append(
                    {
                        "note_id": nid,
                        "layer": block.get("layer"),
                        "component": d["component"],
                        "positions": block.get("positions"),
                        "tokens": block.get("tokens"),
                        "delta_margin": dm,
                    }
                )
            elif dm is not None and abs(float(dm)) >= 1.0:
                shortlist.append(
                    {
                        "note_id": nid,
                        "layer": block.get("layer"),
                        "component": d["component"],
                        "positions": block.get("positions"),
                        "tokens": block.get("tokens"),
                        "delta_margin": dm,
                        "note": "margin shift ≥1 without X flip",
                    }
                )

    tag = "U-B2" if variant == "ub2" else "U-B"
    if flips == 0 and not shortlist:
        hypothesis = (
            f"{tag} null at L{panel.get('layer')} ({panel.get('mode')}-ablate): "
            "no X flip and |Δmargin|<1. Formation sites still not causally sufficient "
            "under this ablation — pause before SAE/circuits, or one more different question."
        )
    elif flips:
        hypothesis = (
            f"{tag}: {flips} X flip(s) under component ablation — shortlist "
            "those (layer, component, tokens) for U-C SAE / U-D pathway."
        )
    else:
        hypothesis = (
            f"{tag}: no X flip but margin moved ≥1 on some components — weak causal "
            "signal; shortlist for cautious U-C or pause."
        )

    out_path = UB2_READOUT_PATH if variant == "ub2" else UB_READOUT_PATH
    out = {
        "ok": True,
        "kind": (
            "n2s_upstream_ub2_readout_v1"
            if variant == "ub2"
            else "n2s_upstream_ub_readout_v1"
        ),
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "source_patch": str(path),
        "variant": variant,
        "findings": findings,
        "shortlist": shortlist,
        "formation_hypothesis": hypothesis,
        "exit_criterion_met": True,
        "next": (
            "U-C upstream SAE at shortlist sites — wait go"
            if shortlist and flips
            else "Pause upstream before circuits/SAE — or new fidelity question — wait go"
        ),
    }
    out_path.write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
    out["artifact"] = str(out_path)
    return out


def _print_patch(panel: dict[str, Any]) -> None:
    if not panel.get("ok"):
        print(panel.get("message") or panel)
        return
    print(f"\n=== {panel.get('kind')} · {panel.get('artifact')} ===")
    for nid, block in (panel.get("notes") or {}).items():
        print(
            f"\n{nid} L{block.get('layer')} mode={block.get('mode')} "
            f"pos={block.get('positions')}"
        )
        for a in block.get("arms") or []:
            print(
                f"  {a['component']:10} X={a.get('extract_x')}  "
                f"margin={a.get('margin')}  verdict={a.get('verdict')}"
            )


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description="U-B / U-B2 component ablation at U-A sites")
    sub = p.add_subparsers(dest="cmd", required=True)

    sp = sub.add_parser("patch", help="Ablate attn/mlp/resid → extract X (GPU)")
    sp.add_argument("--map", default=str(MAP_PATH))
    sp.add_argument("--layer", type=int, default=DEFAULT_LAYER)
    sp.add_argument("--top-k", type=int, default=DEFAULT_TOP_SITES)
    sp.add_argument("--mode", choices=("zero", "mean"), default="zero")
    sp.add_argument(
        "--variant",
        choices=("ub", "ub2"),
        default="ub",
        help="ub2 writes separate artifacts (U-B2 refine)",
    )
    sp.add_argument("--model", default=FAIL_MODEL)
    sp.add_argument("--keep-model", action="store_true")

    sp = sub.add_parser("readout", help="Interpret U-B/U-B2 patch (CPU)")
    sp.add_argument("--patch", default="")
    sp.add_argument("--variant", choices=("ub", "ub2"), default="ub")

    args = p.parse_args(argv)
    if args.cmd == "patch":
        panel = run_ub_patch(
            map_path=Path(args.map),
            model_id=args.model,
            layer=args.layer,
            top_k=args.top_k,
            mode=args.mode,
            variant=args.variant,
            unload_after=not args.keep_model,
        )
        _print_patch(panel)
        if not panel.get("ok"):
            raise SystemExit(2)
        readout = write_ub_readout(variant=args.variant)
        print("\n--- readout ---")
        for line in readout.get("findings") or []:
            print(f"- {line}")
        print(f"hypothesis: {readout.get('formation_hypothesis')}")
        print(f"shortlist_n={len(readout.get('shortlist') or [])}")
        print(f"artifact: {readout.get('artifact')}")
    elif args.cmd == "readout":
        patch = Path(args.patch) if args.patch else None
        readout = write_ub_readout(patch, variant=args.variant)
        for line in readout.get("findings") or []:
            print(f"- {line}")
        print(f"hypothesis: {readout.get('formation_hypothesis')}")
        print(f"artifact: {readout.get('artifact')}")
    else:
        raise SystemExit(f"unknown cmd {args.cmd}")


if __name__ == "__main__":
    main()
