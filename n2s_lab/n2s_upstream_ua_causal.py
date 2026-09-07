"""U-A causal — intervene along frozen formation direction d at L24.

Primary: commit-token activation_steer (h += α·d).
Optional: prefill note-body prefill_position_steer.

Not top-site ablation. See docs/TRACKB-UPSTREAM-UA-CAUSAL.md.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

import torch

from .ground import ground
from .hf_client import unload_model
from .hf_intervene import ActivationSteerSpec, PrefillPositionSteerSpec, generate_intervened
from .hf_trace import find_present_commit_step
from .neural import EXTRACT_SYSTEM
from .n2s_forensics import MIN_FREE_VRAM_GB, cuda_free_gb
from .n2s_upstream_live import ensure_ua_directions
from .n2s_upstream_prefill import DEFAULT_NOTES
from .n2s_upstream_ua_map import CONTRA_ID, MAP_PATH, TEMPORAL_ID
from .ollama_client import parse_json_object
from .paths import ARTIFACTS_DIR
from .phase10_activation import FAIL_MODEL
from .predicate import evaluate_rule
from .trackb_expand_endpoint import GAUSS_SEED
from .trackb_family_vector import _unit
from .types import Extraction

PANEL_PATH = ARTIFACTS_DIR / "n2s-upstream-ua-causal-panel.json"
PANEL_COMMIT_PATH = ARTIFACTS_DIR / "n2s-upstream-ua-causal-commit-panel.json"
PANEL_PREFILL_PATH = ARTIFACTS_DIR / "n2s-upstream-ua-causal-prefill-panel.json"
READOUT_PATH = ARTIFACTS_DIR / "n2s-upstream-ua-causal-readout.json"
READOUT_COMMIT_PATH = ARTIFACTS_DIR / "n2s-upstream-ua-causal-commit-readout.json"
READOUT_PREFILL_PATH = ARTIFACTS_DIR / "n2s-upstream-ua-causal-prefill-readout.json"
DEFAULT_LAYER = 24
DEFAULT_ALPHAS = (1.0, 2.0, 4.0, 8.0)
Site = Literal["commit", "prefill"]


def _panel_path(site: Site) -> Path:
    return PANEL_PREFILL_PATH if site == "prefill" else PANEL_COMMIT_PATH


def _readout_path(site: Site) -> Path:
    return READOUT_PREFILL_PATH if site == "prefill" else READOUT_COMMIT_PATH


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
    if not MAP_PATH.is_file():
        raise FileNotFoundError(f"missing U-A map {MAP_PATH}")
    ua = json.loads(MAP_PATH.read_text(encoding="utf-8"))
    meta = (ua.get("notes") or {}).get(nid) or {}
    t0 = int(meta.get("note_token_start") or 0)
    t1 = int(meta.get("note_token_end") or 0)
    if t1 <= t0:
        raise RuntimeError(f"bad note-body range for {nid}: [{t0},{t1})")
    return list(range(t0, t1))


def _run_extract_arm(
    *,
    evidence: str,
    model_id: str,
    layer: int,
    site: Site,
    label: str,
    alpha: float,
    vec: torch.Tensor | None,
    positions: list[int] | None,
    max_new_tokens: int = 400,
) -> dict[str, Any]:
    kwargs: dict[str, Any] = {
        "system": EXTRACT_SYSTEM,
        "user": evidence,
        "model_id": model_id,
        "max_new_tokens": max_new_tokens,
        "layer_indices": (layer,),
    }
    if vec is None or label == "baseline":
        kwargs["intervention"] = "none"
    elif site == "commit":
        kwargs["intervention"] = "activation_steer"
        kwargs["activation_steer"] = ActivationSteerSpec(
            vectors_by_layer={layer: vec},
            alpha=float(alpha),
        )
    else:
        kwargs["intervention"] = "prefill_position_steer"
        kwargs["prefill_steer"] = PrefillPositionSteerSpec(
            positions=list(positions or []),
            vectors_by_layer={layer: vec},
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
    return {
        "label": label,
        "site": site,
        "alpha": float(alpha) if label != "baseline" else 0.0,
        "extract_x": final_x,
        "margin": None if margin is None else round(margin, 6),
        "commit_step": commit_step,
        "commit_value": commit_val,
        "parse_ok": parse_ok,
        "verdict": rule.verdict.value if rule else None,
        "grounded_X": None if g is None else bool(g.contradiction),
    }


def _soft_gate(rows: list[dict[str, Any]], *, site: Site) -> dict[str, Any]:
    by_note: dict[str, list[dict[str, Any]]] = {}
    for r in rows:
        by_note.setdefault(r["case_id"], []).append(r)

    temporal = by_note.get(TEMPORAL_ID) or []
    contra = by_note.get(CONTRA_ID) or []

    def _arm(note_rows: list[dict[str, Any]], label: str, alpha: float | None = None) -> dict[str, Any] | None:
        for r in note_rows:
            if r.get("label") != label:
                continue
            if alpha is not None and abs(float(r.get("alpha") or 0) - alpha) > 1e-9:
                continue
            return r
        return None

    base_t = _arm(temporal, "baseline")
    base_c = _arm(contra, "baseline")

    # Dose: +α·d margins on temporal vs α
    plus_margins: list[tuple[float, float | None, bool | None]] = []
    for a in DEFAULT_ALPHAS:
        row = _arm(temporal, "+d", float(a))
        if row:
            plus_margins.append((float(a), row.get("margin"), row.get("extract_x")))

    # Sign / specificity at α=8
    a8 = 8.0
    plus8 = _arm(temporal, "+d", a8)
    minus8 = _arm(temporal, "-d", a8)
    gauss8 = _arm(temporal, "gaussian", a8)

    contra_plus_ok = True
    contra_wrecks: list[dict[str, Any]] = []
    for a in DEFAULT_ALPHAS:
        row = _arm(contra, "+d", float(a))
        if row and row.get("extract_x") is False:
            contra_wrecks.append({"alpha": a, **{k: row.get(k) for k in ("margin", "extract_x")}})
            contra_plus_ok = False

    # Monotone-ish: margin decreases with α for +d when baseline X true (push toward false)
    # or X flips true→false at some α and stays
    dose_partial = False
    dose_yes = False
    if base_t and plus_margins:
        b_x = base_t.get("extract_x")
        b_m = base_t.get("margin")
        xs = [x for _, _, x in plus_margins]
        ms = [m for _, m, _ in plus_margins if m is not None]
        if b_x is True and any(x is False for x in xs):
            dose_yes = True
        if b_m is not None and len(ms) >= 2:
            # prefer lower margin (less true) as α grows when pushing toward temporal
            if ms[-1] < ms[0] - 0.5 or (ms[-1] < float(b_m) - 0.5):
                dose_partial = True
                if ms == sorted(ms, reverse=True) or ms[-1] == min(ms):
                    dose_yes = dose_yes or True

    sign_ok = False
    if plus8 and minus8 and plus8.get("margin") is not None and minus8.get("margin") is not None:
        # Require a material gap, not float noise
        if abs(float(plus8["margin"]) - float(minus8["margin"])) >= 1.0:
            sign_ok = True
        if plus8.get("extract_x") != minus8.get("extract_x"):
            sign_ok = True

    gauss_ok = False
    if plus8 and gauss8:
        if plus8.get("extract_x") != gauss8.get("extract_x"):
            gauss_ok = True
        elif (
            plus8.get("margin") is not None
            and gauss8.get("margin") is not None
            and abs(float(plus8["margin"]) - float(gauss8["margin"])) >= 1.0
        ):
            gauss_ok = True

    # Meaningful dose: |Δmargin| vs baseline ≥ 1.0 at some α, or X flip
    if base_t and plus_margins:
        b_m = base_t.get("margin")
        for _a, m, x in plus_margins:
            if base_t.get("extract_x") is True and x is False:
                dose_yes = True
            if b_m is not None and m is not None and abs(float(m) - float(b_m)) >= 1.0:
                dose_partial = True

    note = (
        "YES — dose/sign/specificity signals with contra control held"
        if dose_yes and sign_ok and gauss_ok and contra_plus_ok
        else (
            "PARTIAL — some movement; see rows"
            if dose_yes or (dose_partial and sign_ok)
            else "NULL / weak — no clear dose-dependent causal effect of d on this panel"
        )
    )
    return {
        "site": site,
        "baseline_temporal_x": None if not base_t else base_t.get("extract_x"),
        "baseline_temporal_margin": None if not base_t else base_t.get("margin"),
        "baseline_contra_x": None if not base_c else base_c.get("extract_x"),
        "plus_d_temporal_by_alpha": [
            {"alpha": a, "margin": m, "extract_x": x} for a, m, x in plus_margins
        ],
        "dose_yes": dose_yes,
        "dose_partial": dose_partial,
        "sign_ok": sign_ok,
        "gaussian_distinct": gauss_ok,
        "contra_control_ok": contra_plus_ok,
        "contra_wrecks": contra_wrecks,
        "note": note,
    }


def run_ua_causal_panel(
    *,
    model_id: str = FAIL_MODEL,
    layer: int = DEFAULT_LAYER,
    site: Site = "commit",
    alphas: tuple[float, ...] = DEFAULT_ALPHAS,
    unload_after: bool = True,
) -> dict[str, Any]:
    gate = _gpu_gate()
    if gate:
        return gate

    dirs = ensure_ua_directions(model_id=model_id, unload_after=True)
    if not dirs.get("ok"):
        return dirs
    blob = dirs["_blob"]
    key = f"L{layer}"
    d = blob["by_layer"].get(key)
    if d is None:
        return {"ok": False, "error": f"directions missing {key}"}
    d = _unit(d.float().cpu())

    g = torch.Generator()
    g.manual_seed(GAUSS_SEED)
    gauss = _unit(torch.randn(d.shape, generator=g))

    rows: list[dict[str, Any]] = []
    positions_by_note: dict[str, list[int]] = {}
    if site == "prefill":
        for nid in (TEMPORAL_ID, CONTRA_ID):
            positions_by_note[nid] = _note_body_positions(nid)

    try:
        for nid in (TEMPORAL_ID, CONTRA_ID):
            note = DEFAULT_NOTES[nid]
            evidence = note["evidence"]
            positions = positions_by_note.get(nid)
            print(f"[ua-causal] {nid} baseline ({site}) …", flush=True)
            base = _run_extract_arm(
                evidence=evidence,
                model_id=model_id,
                layer=layer,
                site=site,
                label="baseline",
                alpha=0.0,
                vec=None,
                positions=positions,
            )
            rows.append({"ok": True, "case_id": nid, "gold": note.get("gold"), **base})

            for a in alphas:
                for label, vec in (
                    ("+d", d),
                    ("-d", -d),
                    ("gaussian", gauss),
                ):
                    print(f"[ua-causal] {nid} {label} α={a} ({site}) …", flush=True)
                    arm = _run_extract_arm(
                        evidence=evidence,
                        model_id=model_id,
                        layer=layer,
                        site=site,
                        label=label,
                        alpha=float(a),
                        vec=vec,
                        positions=positions,
                    )
                    rows.append({"ok": True, "case_id": nid, "gold": note.get("gold"), **arm})
    finally:
        if unload_after:
            try:
                unload_model()
            except Exception:  # noqa: BLE001
                pass

    soft = _soft_gate(rows, site=site)
    panel: dict[str, Any] = {
        "ok": True,
        "kind": "n2s_upstream_ua_causal_panel_v1",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "protocol": "docs/TRACKB-UPSTREAM-UA-CAUSAL.md",
        "quest": (
            "Does intervening along frozen U-A d change extract X/margin dose-dependently?"
        ),
        "model_id": model_id,
        "layer": layer,
        "site": site,
        "alphas": list(alphas),
        "directions_path": str(ARTIFACTS_DIR / "n2s-upstream-ua-directions.pt"),
        "directions_cached": bool(dirs.get("cached")),
        "pair_frozen": blob.get("pair"),
        "gauss_seed": GAUSS_SEED,
        "soft_gate": soft,
        "rows": rows,
        "claim_hygiene": {
            "say": "causal test of frozen U-A d via steer (not top-site ablation)",
            "do_not_say": "U-B reopen; temporality neuron; expand-v α=8 freeze",
        },
    }
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = _panel_path(site)
    out_path.write_text(json.dumps(panel, indent=2) + "\n", encoding="utf-8")
    # Convenience alias for latest run
    PANEL_PATH.write_text(json.dumps(panel, indent=2) + "\n", encoding="utf-8")
    panel["artifact"] = str(out_path)
    write_ua_causal_readout(out_path, site=site)
    return panel


def write_ua_causal_readout(
    panel_path: Path | None = None,
    *,
    site: Site | None = None,
) -> dict[str, Any]:
    path = panel_path or PANEL_PATH
    if not path.is_file():
        return {"ok": False, "error": f"missing {path}"}
    panel = json.loads(path.read_text(encoding="utf-8"))
    use_site: Site = site or panel.get("site") or "commit"  # type: ignore[assignment]
    soft = _soft_gate(list(panel.get("rows") or []), site=use_site)
    panel["soft_gate"] = soft
    path.write_text(json.dumps(panel, indent=2) + "\n", encoding="utf-8")
    lines = [
        f"# U-A causal-d readout ({panel.get('timestamp')})",
        "",
        f"Quest: {panel.get('quest')}",
        f"Site={panel.get('site')} layer=L{panel.get('layer')} alphas={panel.get('alphas')}",
        f"Soft gate: {soft.get('note')}",
        f"  dose_yes={soft.get('dose_yes')} dose_partial={soft.get('dose_partial')} "
        f"sign_ok={soft.get('sign_ok')} gauss_distinct={soft.get('gaussian_distinct')} "
        f"contra_ok={soft.get('contra_control_ok')}",
        f"  baseline temporal X={soft.get('baseline_temporal_x')} "
        f"margin={soft.get('baseline_temporal_margin')}",
        "",
        "## Rows",
    ]
    for r in panel.get("rows") or []:
        lines.append(
            f"- {r.get('case_id')} {r.get('label')} α={r.get('alpha')} "
            f"X={r.get('extract_x')} margin={r.get('margin')} verdict={r.get('verdict')}"
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
    rpath = _readout_path(use_site)
    payload = {
        "ok": True,
        "kind": "n2s_upstream_ua_causal_readout_v1",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "panel": str(path),
        "soft_gate": soft,
        "text": text,
    }
    rpath.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    READOUT_PATH.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(text, flush=True)
    return {"ok": True, "path": str(rpath), "text": text}


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="U-A causal intervene along frozen d")
    p.add_argument("cmd", choices=("panel", "readout"))
    p.add_argument("--site", choices=("commit", "prefill"), default="commit")
    p.add_argument("--layer", type=int, default=DEFAULT_LAYER)
    args = p.parse_args(argv)
    if args.cmd == "panel":
        out = run_ua_causal_panel(site=args.site, layer=args.layer)
        slim = {k: v for k, v in out.items() if k != "rows"}
        print(json.dumps(slim, indent=2))
        return 0 if out.get("ok") else 1
    out = write_ua_causal_readout()
    return 0 if out.get("ok") else 1


if __name__ == "__main__":
    raise SystemExit(main())
