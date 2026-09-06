"""Thin circuit pilot — C0 write audit + C1 L20 MLP causal patch.

HF hooks only (no TransformerLens). C0: project attn/mlp writes onto frozen v /
SAE dirs at commit. C1: zero/replace L20 MLP write at contradiction.present;
metric = repair X / margin; attn + other-layer controls.

See docs/TRACKB-CIRCUIT-PILOT.md.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch

from .ground import ground
from .hf_client import load_model, unload_model
from .hf_intervene import CommitComponentSpec, _at_contradiction_present_commit
from .hf_trace import _model_layers
from .n2s_forensics import DIRECTIONS_PATH, MIN_FREE_VRAM_GB, cuda_free_gb
from .n2s_sae_pilot import SCORE_PATH, JumpReLUSae
from .paths import ARTIFACTS_DIR, TEMPORAL_FAMILY_EXPAND_PATH
from .phase10_activation import FAIL_MODEL
from .phase9a_behavioral import _analyze_hf, _extract_hf
from .phase9b_localization import _repair_user_prompt
from .trackb_falsex_cluster import _repair_run
from .trackb_family_vector import _unit
from .verify import verify_formalization

FIT_PATH = ARTIFACTS_DIR / "trackb-expand-fit-Qwen_Qwen2.5-7B-Instruct-L20.json"
AUDIT_PATH = ARTIFACTS_DIR / "n2s-circuit-write-audit.json"
C1_PATH = ARTIFACTS_DIR / "n2s-circuit-c1-l20-mlp-patch.json"
C2_PATH = ARTIFACTS_DIR / "n2s-circuit-c2-path-restrict.json"
AUDIT_LAYERS = (16, 20, 24)
C1_LAYER = 20
C2_FEEDER = 16
C2_POST = 24


def _load_expand_case(case_id: str) -> dict[str, Any]:
    data = json.loads(TEMPORAL_FAMILY_EXPAND_PATH.read_text(encoding="utf-8"))
    for c in data.get("cases") or []:
        if c.get("id") == case_id:
            return c
    raise KeyError(case_id)


def _proj(h: torch.Tensor, direction: torch.Tensor) -> float:
    d = _unit(direction.float())
    h = h.float().reshape(-1)
    return float(torch.dot(h, d).item())


def _load_directions() -> tuple[torch.Tensor, list[dict[str, Any]], dict[str, Any]]:
    meta = json.loads(DIRECTIONS_PATH.read_text(encoding="utf-8"))
    v = _unit(torch.tensor(meta["direction_a_minus_b"], dtype=torch.float32))
    dirs: list[dict[str, Any]] = [{"name": "frozen_v", "feat_id": None, "vec": v}]
    if SCORE_PATH.exists():
        scores = json.loads(SCORE_PATH.read_text(encoding="utf-8"))
        top = list(scores.get("top") or [])[:3]
        if top:
            sae, _ = JumpReLUSae.from_hub()
            for row in top:
                fid = int(row["feat_id"])
                dirs.append(
                    {
                        "name": f"sae_{fid}",
                        "feat_id": fid,
                        "vec": sae.decoder_unit(fid),
                        "delta_A_minus_B": row.get("delta_A_minus_B"),
                        "cos_to_v": row.get("cos_to_v"),
                    }
                )
    return v, dirs, meta


def _capture_writes_at_commit(
    *,
    case: dict[str, Any],
    model_id: str,
    layer_indices: tuple[int, ...] = AUDIT_LAYERS,
    include_repair: bool = True,
) -> dict[str, Any]:
    """One repair generation; at commit, record attn/mlp last-token writes."""
    evidence = case["evidence"]
    gold = case["expected"]
    case_id = case["id"]

    raw = _extract_hf(evidence, model_id=model_id)
    analysis = _analyze_hf(evidence, model_id=model_id)
    v1 = verify_formalization(
        analysis=analysis,
        extraction=raw,
        grounding=ground(raw).to_dict(),
        case_id=case_id,
        gold=gold,
    )
    user = _repair_user_prompt(evidence, raw, v1.reasons)

    model, tokenizer = load_model(model_id)
    layers = _model_layers(model)
    messages = [
        {"role": "system", "content": "You extract structured JSON for clinical predicates."},
        {"role": "user", "content": user},
    ]
    prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )

    writes: dict[str, dict[str, torch.Tensor]] = {
        f"L{i}": {} for i in layer_indices
    }
    commit_hit = {"ok": False}

    def _out_vec(out: Any) -> torch.Tensor:
        hs = out[0] if isinstance(out, tuple) else out
        return hs[0, -1, :].detach().float().cpu()

    handles: list[Any] = []
    for i in layer_indices:
        if i < 0 or i >= len(layers):
            continue
        block = layers[i]
        attn = getattr(block, "self_attn", None)
        mlp = getattr(block, "mlp", None)

        def _make(kind: str, idx: int):
            def hook(_m, _inp, out):
                if not commit_hit["ok"]:
                    return
                writes[f"L{idx}"][kind] = _out_vec(out)

            return hook

        if attn is not None:
            handles.append(attn.register_forward_hook(_make("attn", i)))
        if mlp is not None:
            handles.append(mlp.register_forward_hook(_make("mlp", i)))

    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(model.device)
    generated = input_ids.clone()
    gen_prefix = ""
    past = None
    max_new = 600

    try:
        with torch.inference_mode():
            for _ in range(max_new):
                at_commit = _at_contradiction_present_commit(gen_prefix)
                commit_hit["ok"] = at_commit
                out = model(
                    generated if past is None else generated[:, -1:],
                    past_key_values=past,
                    use_cache=True,
                )
                logits = out.logits[0, -1]
                past = out.past_key_values
                next_id = int(torch.argmax(logits).item())
                tok = tokenizer.decode([next_id], skip_special_tokens=False)
                gen_prefix += tok
                generated = torch.cat(
                    [generated, torch.tensor([[next_id]], device=generated.device)],
                    dim=1,
                )
                if at_commit:
                    # one more forward already captured writes at this step
                    break
                if next_id == tokenizer.eos_token_id:
                    break
    finally:
        for h in handles:
            h.remove()

    out: dict[str, Any] = {
        "case_id": case_id,
        "gold": gold,
        "commit_captured": bool(writes.get("L20") or writes.get(f"L{layer_indices[0]}")),
        "writes": {
            layer: {k: t.tolist() for k, t in kinds.items()}
            for layer, kinds in writes.items()
        },
    }
    if include_repair:
        row = _repair_run(
            case,
            model_id=model_id,
            intervention="none",
            layer_indices=(20,),
        )
        out["repair_final_x"] = row.get("repair_final_x")
        out["margin"] = row.get("logit_margin_true_minus_false")
        out["verdict"] = row.get("verdict")
    return out


def _mean_component_write(
    rows: list[dict[str, Any]],
    *,
    layer: int,
    component: str,
) -> torch.Tensor:
    key = f"L{layer}"
    vecs: list[torch.Tensor] = []
    for r in rows:
        w = ((r.get("writes") or {}).get(key) or {}).get(component)
        if not w:
            raise RuntimeError(f"missing {key}/{component} write for {r.get('case_id')}")
        vecs.append(torch.tensor(w, dtype=torch.float32))
    return torch.stack(vecs, dim=0).mean(dim=0)


def _arm_row_summary(row: dict[str, Any], *, arm: str, cls: str) -> dict[str, Any]:
    return {
        "arm": arm,
        "class": cls,
        "case_id": row.get("case_id"),
        "gold": row.get("gold"),
        "repair_final_x": row.get("repair_final_x"),
        "margin": row.get("logit_margin_true_minus_false"),
        "verdict": row.get("verdict"),
        "parse_ok": row.get("parse_ok"),
        "commit_step": row.get("commit_step"),
        "intervened_steps": row.get("intervened_steps"),
    }


def _c1_soft_gate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Soft gate: Class B patch←A mean should lower margin / flip X vs baseline.

    Controls (attn zero, L16 mlp zero) should move less than primary replace.
    """
    by: dict[tuple[str, str], dict[str, Any]] = {}
    for r in rows:
        by[(str(r["case_id"]), str(r["arm"]))] = r

    b_cases = sorted({r["case_id"] for r in rows if r.get("class") == "B"})
    a_cases = sorted({r["case_id"] for r in rows if r.get("class") == "A"})

    def delta_margin(cid: str, arm: str) -> float | None:
        base = by.get((cid, "baseline"))
        arm_row = by.get((cid, arm))
        if not base or not arm_row:
            return None
        mb, ma = base.get("margin"), arm_row.get("margin")
        if mb is None or ma is None:
            return None
        return float(ma) - float(mb)

    def flipped_x(cid: str, arm: str, *, want: bool) -> bool | None:
        base = by.get((cid, "baseline"))
        arm_row = by.get((cid, arm))
        if not base or not arm_row:
            return None
        if base.get("repair_final_x") is None or arm_row.get("repair_final_x") is None:
            return None
        return bool(arm_row["repair_final_x"]) is want and bool(base["repair_final_x"]) is not want

    primary = "l20_mlp_replace_A_mean"
    controls = ("l20_attn_zero", "l16_mlp_zero")

    b_primary_dm = [delta_margin(c, primary) for c in b_cases]
    b_primary_dm = [d for d in b_primary_dm if d is not None]
    b_zero_dm = [delta_margin(c, "l20_mlp_zero") for c in b_cases]
    b_zero_dm = [d for d in b_zero_dm if d is not None]

    # Predicted: B←A replace lowers true−false margin (more A-like / X-false)
    mean_primary = (
        float(sum(b_primary_dm) / len(b_primary_dm)) if b_primary_dm else None
    )
    mean_zero = float(sum(b_zero_dm) / len(b_zero_dm)) if b_zero_dm else None
    ctrl_means: dict[str, float | None] = {}
    for ctrl in controls:
        vals = [delta_margin(c, ctrl) for c in b_cases]
        vals = [d for d in vals if d is not None]
        ctrl_means[ctrl] = float(sum(vals) / len(vals)) if vals else None

    n_flip_false = sum(
        1 for c in b_cases if flipped_x(c, primary, want=False) is True
    )
    primary_beats_controls = False
    if mean_primary is not None:
        ctrl_ok = [
            abs(mean_primary) > abs(v) + 0.5
            for v in ctrl_means.values()
            if v is not None
        ]
        # stronger downward move than controls (more negative)
        primary_beats_controls = bool(ctrl_ok) and all(
            mean_primary < (v if v is not None else 0) - 0.5
            for v in ctrl_means.values()
            if v is not None
        )

    passed = bool(
        mean_primary is not None
        and mean_primary < -1.0
        and (n_flip_false > 0 or primary_beats_controls)
    )
    # A←B secondary: margin should rise if MLP write is causal both ways
    a_replace = "l20_mlp_replace_B_mean"
    a_dm = [delta_margin(c, a_replace) for c in a_cases]
    a_dm = [d for d in a_dm if d is not None]
    mean_a_replace = float(sum(a_dm) / len(a_dm)) if a_dm else None

    return {
        "passed": passed,
        "verdict": "YES" if passed else "NULL/weak",
        "class_b_mean_delta_margin_primary": mean_primary,
        "class_b_mean_delta_margin_l20_mlp_zero": mean_zero,
        "class_b_control_mean_delta_margin": ctrl_means,
        "class_b_n_flip_to_x_false_primary": n_flip_false,
        "class_b_n": len(b_cases),
        "primary_beats_controls": primary_beats_controls,
        "class_a_mean_delta_margin_replace_B": mean_a_replace,
        "note": (
            "Primary = Class B repair with L20 MLP replaced by Class A mean write. "
            "Pass if mean Δmargin < −1 and (X flips false on ≥1 B case or primary "
            "beats controls)."
        ),
    }


def run_write_audit(
    *,
    max_per_class: int = 2,
    model_id: str = FAIL_MODEL,
    unload_after: bool = True,
) -> dict[str, Any]:
    free = cuda_free_gb()
    if free is not None and free < MIN_FREE_VRAM_GB:
        return {
            "ok": False,
            "status": "deferred_gpu_busy",
            "free_vram_gb": round(free, 2),
            "message": f"Need ~{MIN_FREE_VRAM_GB:.0f} GiB free; have {free:.1f}.",
        }

    fit = json.loads(FIT_PATH.read_text(encoding="utf-8"))
    class_a = list(fit["class_a"])[:max_per_class]
    class_b = list(fit["class_b"])[:max_per_class]
    _v, dir_list, dir_meta = _load_directions()

    print(f"[circuit] write audit A={class_a} B={class_b} dirs={[d['name'] for d in dir_list]}")
    rows_a: list[dict[str, Any]] = []
    rows_b: list[dict[str, Any]] = []
    try:
        for cid in class_a:
            case = _load_expand_case(cid)
            print(f"[circuit] A · {cid}")
            rows_a.append(
                _capture_writes_at_commit(case=case, model_id=model_id)
            )
        for cid in class_b:
            case = _load_expand_case(cid)
            print(f"[circuit] B · {cid}")
            rows_b.append(
                _capture_writes_at_commit(case=case, model_id=model_id)
            )
    finally:
        if unload_after:
            try:
                unload_model()
            except Exception:  # noqa: BLE001
                pass

    # Aggregate mean projection Δ(A−B) per layer × kind × direction
    summary: list[dict[str, Any]] = []
    for layer in [f"L{i}" for i in AUDIT_LAYERS]:
        for kind in ("attn", "mlp"):
            for d in dir_list:
                name = d["name"]
                vec = d["vec"]

                def mean_proj(rows: list[dict[str, Any]]) -> float | None:
                    vals = []
                    for r in rows:
                        w = ((r.get("writes") or {}).get(layer) or {}).get(kind)
                        if not w:
                            continue
                        vals.append(_proj(torch.tensor(w, dtype=torch.float32), vec))
                    if not vals:
                        return None
                    return float(sum(vals) / len(vals))

                ma = mean_proj(rows_a)
                mb = mean_proj(rows_b)
                if ma is None or mb is None:
                    continue
                summary.append(
                    {
                        "layer": layer,
                        "component": kind,
                        "direction": name,
                        "feat_id": d.get("feat_id"),
                        "mean_proj_A": ma,
                        "mean_proj_B": mb,
                        "delta_A_minus_B": ma - mb,
                        "abs_delta": abs(ma - mb),
                    }
                )

    summary.sort(key=lambda r: r["abs_delta"], reverse=True)
    payload = {
        "ok": True,
        "kind": "n2s_circuit_write_audit_v1",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "quest": (
            "Which layer×attn/mlp writes along frozen v / SAE dirs at commit "
            "differ for Class A vs B?"
        ),
        "model_id": model_id,
        "layers": list(AUDIT_LAYERS),
        "class_a": class_a,
        "class_b": class_b,
        "direction_source": str(DIRECTIONS_PATH.name),
        "sae_scores_used": SCORE_PATH.exists(),
        "fit_layer": dir_meta.get("fit_layer"),
        "cases_A": [{k: r[k] for k in ("case_id", "gold", "repair_final_x", "margin", "verdict", "commit_captured")} for r in rows_a],
        "cases_B": [{k: r[k] for k in ("case_id", "gold", "repair_final_x", "margin", "verdict", "commit_captured")} for r in rows_b],
        "summary_ranked": summary,
        "next": (
            "C1: activation-patch top abs_delta sites between A/B; "
            "metric = repair X / margin; random-site control."
        ),
    }
    # Drop bulky write vectors from disk payload (keep summary)
    AUDIT_PATH.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"[circuit] wrote {AUDIT_PATH}")
    print("[circuit] top write sites by |Δ(A−B)|:")
    for row in summary[:12]:
        print(
            f"  {row['layer']}/{row['component']}/{row['direction']}: "
            f"Δ={row['delta_A_minus_B']:+.4f} (A={row['mean_proj_A']:+.4f} B={row['mean_proj_B']:+.4f})"
        )
    return payload


def run_c1_l20_mlp_patch(
    *,
    max_per_class: int = 2,
    model_id: str = FAIL_MODEL,
    unload_after: bool = True,
) -> dict[str, Any]:
    """C1: causal patch of L20 MLP at contradiction.present commit.

    Arms: baseline · l20_mlp_zero · l20_mlp_replace_{A,B}_mean ·
    l20_attn_zero · l16_mlp_zero. Metric = repair X / margin.
    """
    free = cuda_free_gb()
    if free is not None and free < MIN_FREE_VRAM_GB:
        return {
            "ok": False,
            "status": "deferred_gpu_busy",
            "free_vram_gb": round(free, 2),
            "message": f"Need ~{MIN_FREE_VRAM_GB:.0f} GiB free; have {free:.1f}.",
        }

    fit = json.loads(FIT_PATH.read_text(encoding="utf-8"))
    class_a = list(fit["class_a"])[:max_per_class]
    class_b = list(fit["class_b"])[:max_per_class]

    print(f"[circuit-c1] capture donor MLP writes A={class_a} B={class_b}")
    rows_a: list[dict[str, Any]] = []
    rows_b: list[dict[str, Any]] = []
    try:
        for cid in class_a:
            case = _load_expand_case(cid)
            print(f"[circuit-c1] capture A · {cid}")
            rows_a.append(
                _capture_writes_at_commit(
                    case=case,
                    model_id=model_id,
                    layer_indices=(16, C1_LAYER),
                    include_repair=False,
                )
            )
        for cid in class_b:
            case = _load_expand_case(cid)
            print(f"[circuit-c1] capture B · {cid}")
            rows_b.append(
                _capture_writes_at_commit(
                    case=case,
                    model_id=model_id,
                    layer_indices=(16, C1_LAYER),
                    include_repair=False,
                )
            )

        mean_a = _mean_component_write(rows_a, layer=C1_LAYER, component="mlp")
        mean_b = _mean_component_write(rows_b, layer=C1_LAYER, component="mlp")

        arms: list[tuple[str, str, CommitComponentSpec | None]] = [
            ("baseline", "none", None),
            (
                "l20_mlp_zero",
                "commit_component",
                CommitComponentSpec(layer=C1_LAYER, component="mlp", mode="zero"),
            ),
            (
                "l20_mlp_replace_A_mean",
                "commit_component",
                CommitComponentSpec(
                    layer=C1_LAYER, component="mlp", mode="replace", fill=mean_a
                ),
            ),
            (
                "l20_mlp_replace_B_mean",
                "commit_component",
                CommitComponentSpec(
                    layer=C1_LAYER, component="mlp", mode="replace", fill=mean_b
                ),
            ),
            (
                "l20_attn_zero",
                "commit_component",
                CommitComponentSpec(layer=C1_LAYER, component="attn", mode="zero"),
            ),
            (
                "l16_mlp_zero",
                "commit_component",
                CommitComponentSpec(layer=16, component="mlp", mode="zero"),
            ),
        ]

        panel_rows: list[dict[str, Any]] = []
        cases = [(cid, "A") for cid in class_a] + [(cid, "B") for cid in class_b]
        for cid, cls in cases:
            case = _load_expand_case(cid)
            for arm_name, intervention, spec in arms:
                print(f"[circuit-c1] {cls}/{cid} · {arm_name}")
                row = _repair_run(
                    case,
                    model_id=model_id,
                    intervention=intervention,
                    commit_component=spec,
                    layer_indices=(int(spec.layer),) if spec is not None else (C1_LAYER,),
                )
                panel_rows.append(_arm_row_summary(row, arm=arm_name, cls=cls))
                print(
                    f"  X={row.get('repair_final_x')} margin={row.get('logit_margin_true_minus_false')}"
                )
    finally:
        if unload_after:
            try:
                unload_model()
            except Exception:  # noqa: BLE001
                pass

    soft = _c1_soft_gate(panel_rows)
    payload = {
        "ok": True,
        "kind": "n2s_circuit_c1_l20_mlp_patch_v1",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "quest": (
            "Does causal patch of L20 MLP at contradiction.present move repair "
            "X / margin in the direction predicted by the C0 write map?"
        ),
        "model_id": model_id,
        "c0_top_site": "L20/mlp/frozen_v",
        "layer": C1_LAYER,
        "class_a": class_a,
        "class_b": class_b,
        "arms": [a[0] for a in arms],
        "donor": {
            "mean_A_l20_mlp_norm": float(mean_a.norm().item()),
            "mean_B_l20_mlp_norm": float(mean_b.norm().item()),
            "cos_meanA_meanB": float(
                torch.nn.functional.cosine_similarity(
                    mean_a.unsqueeze(0), mean_b.unsqueeze(0)
                ).item()
            ),
        },
        "rows": panel_rows,
        "soft_gate": soft,
        "claim_hygiene": (
            "Say: C1 soft_gate YES — L20 MLP commit write is a causal site for "
            "X/margin. Do not say: full temporality circuit found."
        ),
        "next": (
            "Optional C2 path restrict wait-go. "
            "Do not reopen Upstream U-C/U-D from this Downstream site hit."
        ),
    }
    C1_PATH.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"[circuit-c1] wrote {C1_PATH}")
    print(
        f"[circuit-c1] soft_gate={soft['verdict']} "
        f"B_primary_Δmargin={soft.get('class_b_mean_delta_margin_primary')}"
    )
    return payload


def _c2_soft_gate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """C2: is L20 MLP locally sufficient, or does L16 MLP feed the effect?

    Local-sufficient YES if:
      - C1 primary (l20 replace A) still Δmargin_B < −1
      - |Δ(l16_replace_A)| < |Δ(l20)| / 2 on B
      - |Δ(path_stack) − Δ(l20)| < 1.0 on B
    Feeder-matters if L16 alone is strong or path_stack collapses the L20 effect.
    """
    by: dict[tuple[str, str], dict[str, Any]] = {}
    for r in rows:
        by[(str(r["case_id"]), str(r["arm"]))] = r
    b_cases = sorted({r["case_id"] for r in rows if r.get("class") == "B"})
    a_cases = sorted({r["case_id"] for r in rows if r.get("class") == "A"})

    def mean_dm(cids: list[str], arm: str) -> float | None:
        vals = []
        for cid in cids:
            base = by.get((cid, "baseline"))
            arm_row = by.get((cid, arm))
            if not base or not arm_row:
                continue
            mb, ma = base.get("margin"), arm_row.get("margin")
            if mb is None or ma is None:
                continue
            vals.append(float(ma) - float(mb))
        if not vals:
            return None
        return float(sum(vals) / len(vals))

    dm_l20 = mean_dm(b_cases, "l20_mlp_replace_A_mean")
    dm_l16 = mean_dm(b_cases, "l16_mlp_replace_A_mean")
    dm_stack = mean_dm(b_cases, "path_l16_zero_plus_l20_replace_A")
    dm_post = mean_dm(b_cases, "l24_mlp_zero")
    # A←B flip rate under L20 vs L16 alone
    def n_flip_true(cids: list[str], arm: str) -> int:
        n = 0
        for cid in cids:
            base = by.get((cid, "baseline"))
            arm_row = by.get((cid, arm))
            if not base or not arm_row:
                continue
            if base.get("repair_final_x") is False and arm_row.get("repair_final_x") is True:
                n += 1
        return n

    a_flip_l20 = n_flip_true(a_cases, "l20_mlp_replace_B_mean")
    a_flip_l16 = n_flip_true(a_cases, "l16_mlp_replace_B_mean")

    local = False
    feeder = False
    if dm_l20 is not None and dm_l20 < -1.0:
        l16_weak = dm_l16 is None or abs(dm_l16) < abs(dm_l20) / 2
        stack_close = (
            dm_stack is not None and abs(dm_stack - dm_l20) < 1.0
        )
        local = bool(l16_weak and stack_close)
        feeder = bool(
            (dm_l16 is not None and abs(dm_l16) >= abs(dm_l20) / 2)
            or (dm_stack is not None and abs(dm_stack - dm_l20) >= 1.0)
        )

    if local and not feeder:
        verdict = "LOCAL_SUFFICIENT"
        passed = True
    elif feeder:
        verdict = "FEEDER_MATTERS"
        passed = True
    else:
        verdict = "NULL/weak"
        passed = False

    return {
        "passed": passed,
        "verdict": verdict,
        "class_b_mean_delta_margin_l20_replace_A": dm_l20,
        "class_b_mean_delta_margin_l16_replace_A": dm_l16,
        "class_b_mean_delta_margin_path_stack": dm_stack,
        "class_b_mean_delta_margin_l24_mlp_zero": dm_post,
        "class_a_n_flip_x_true_l20_replace_B": a_flip_l20,
        "class_a_n_flip_x_true_l16_replace_B": a_flip_l16,
        "local_sufficient": local,
        "feeder_matters": feeder,
        "note": (
            "LOCAL_SUFFICIENT = L20 MLP replace retains C1 effect; L16 alone weak; "
            "path stack ≈ L20-only. FEEDER_MATTERS = L16 alone strong or stack shifts effect."
        ),
    }


def run_c2_path_restrict(
    *,
    max_per_class: int = 2,
    model_id: str = FAIL_MODEL,
    unload_after: bool = True,
) -> dict[str, Any]:
    """C2: path restrict — L16 feeder vs L20 MLP local sufficiency at commit."""
    free = cuda_free_gb()
    if free is not None and free < MIN_FREE_VRAM_GB:
        return {
            "ok": False,
            "status": "deferred_gpu_busy",
            "free_vram_gb": round(free, 2),
            "message": f"Need ~{MIN_FREE_VRAM_GB:.0f} GiB free; have {free:.1f}.",
        }

    fit = json.loads(FIT_PATH.read_text(encoding="utf-8"))
    class_a = list(fit["class_a"])[:max_per_class]
    class_b = list(fit["class_b"])[:max_per_class]

    print(f"[circuit-c2] capture L16/L20 MLP writes A={class_a} B={class_b}")
    rows_a: list[dict[str, Any]] = []
    rows_b: list[dict[str, Any]] = []
    try:
        for cid in class_a:
            case = _load_expand_case(cid)
            print(f"[circuit-c2] capture A · {cid}")
            rows_a.append(
                _capture_writes_at_commit(
                    case=case,
                    model_id=model_id,
                    layer_indices=(C2_FEEDER, C1_LAYER, C2_POST),
                    include_repair=False,
                )
            )
        for cid in class_b:
            case = _load_expand_case(cid)
            print(f"[circuit-c2] capture B · {cid}")
            rows_b.append(
                _capture_writes_at_commit(
                    case=case,
                    model_id=model_id,
                    layer_indices=(C2_FEEDER, C1_LAYER, C2_POST),
                    include_repair=False,
                )
            )

        mean_a_l20 = _mean_component_write(rows_a, layer=C1_LAYER, component="mlp")
        mean_b_l20 = _mean_component_write(rows_b, layer=C1_LAYER, component="mlp")
        mean_a_l16 = _mean_component_write(rows_a, layer=C2_FEEDER, component="mlp")
        mean_b_l16 = _mean_component_write(rows_b, layer=C2_FEEDER, component="mlp")

        arms: list[tuple[str, list[CommitComponentSpec] | None]] = [
            ("baseline", None),
            (
                "l20_mlp_replace_A_mean",
                [
                    CommitComponentSpec(
                        layer=C1_LAYER, component="mlp", mode="replace", fill=mean_a_l20
                    )
                ],
            ),
            (
                "l16_mlp_replace_A_mean",
                [
                    CommitComponentSpec(
                        layer=C2_FEEDER, component="mlp", mode="replace", fill=mean_a_l16
                    )
                ],
            ),
            (
                "path_l16_zero_plus_l20_replace_A",
                [
                    CommitComponentSpec(
                        layer=C2_FEEDER, component="mlp", mode="zero"
                    ),
                    CommitComponentSpec(
                        layer=C1_LAYER, component="mlp", mode="replace", fill=mean_a_l20
                    ),
                ],
            ),
            (
                "l20_mlp_replace_B_mean",
                [
                    CommitComponentSpec(
                        layer=C1_LAYER, component="mlp", mode="replace", fill=mean_b_l20
                    )
                ],
            ),
            (
                "l16_mlp_replace_B_mean",
                [
                    CommitComponentSpec(
                        layer=C2_FEEDER, component="mlp", mode="replace", fill=mean_b_l16
                    )
                ],
            ),
            (
                "l24_mlp_zero",
                [CommitComponentSpec(layer=C2_POST, component="mlp", mode="zero")],
            ),
        ]

        panel_rows: list[dict[str, Any]] = []
        cases = [(cid, "A") for cid in class_a] + [(cid, "B") for cid in class_b]
        for cid, cls in cases:
            case = _load_expand_case(cid)
            for arm_name, specs in arms:
                print(f"[circuit-c2] {cls}/{cid} · {arm_name}")
                layers = (
                    tuple(sorted({int(s.layer) for s in specs}))
                    if specs
                    else (C1_LAYER,)
                )
                row = _repair_run(
                    case,
                    model_id=model_id,
                    intervention="none" if not specs else "commit_component",
                    commit_components=specs,
                    layer_indices=layers,
                )
                panel_rows.append(_arm_row_summary(row, arm=arm_name, cls=cls))
                print(
                    f"  X={row.get('repair_final_x')} margin={row.get('logit_margin_true_minus_false')}"
                )
    finally:
        if unload_after:
            try:
                unload_model()
            except Exception:  # noqa: BLE001
                pass

    soft = _c2_soft_gate(panel_rows)
    payload = {
        "ok": True,
        "kind": "n2s_circuit_c2_path_restrict_v1",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "quest": (
            "Is the C1 L20 MLP causal effect locally sufficient at commit, "
            "or does an L16 MLP feeder path matter?"
        ),
        "model_id": model_id,
        "site": C1_LAYER,
        "feeder": C2_FEEDER,
        "post_control": C2_POST,
        "class_a": class_a,
        "class_b": class_b,
        "arms": [a[0] for a in arms],
        "rows": panel_rows,
        "soft_gate": soft,
        "claim_hygiene": (
            "Say: C2 path-restrict readout (local vs feeder). "
            "Do not say: full circuit diagram / named temporality path."
        ),
        "next": "Freeze C0–C2; leave Upstream U-C/U-D locked; stop ladder unless new Q.",
    }
    C2_PATH.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"[circuit-c2] wrote {C2_PATH}")
    print(f"[circuit-c2] soft_gate={soft['verdict']}")
    return payload


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(
        description="Thin N2S circuit pilot (C0 audit + C1 patch + C2 path)"
    )
    sub = p.add_subparsers(dest="cmd", required=True)
    sp = sub.add_parser("audit", help="C0 write audit Class A vs B")
    sp.add_argument("--max-per-class", type=int, default=2)
    sp.add_argument("--model", default=FAIL_MODEL)
    sp2 = sub.add_parser("patch", help="C1 causal patch L20 MLP at commit")
    sp2.add_argument("--max-per-class", type=int, default=2)
    sp2.add_argument("--model", default=FAIL_MODEL)
    sp3 = sub.add_parser("path", help="C2 path restrict L16 feeder vs L20 MLP")
    sp3.add_argument("--max-per-class", type=int, default=2)
    sp3.add_argument("--model", default=FAIL_MODEL)
    args = p.parse_args(argv)
    if args.cmd == "audit":
        run_write_audit(max_per_class=args.max_per_class, model_id=args.model)
    elif args.cmd == "patch":
        run_c1_l20_mlp_patch(max_per_class=args.max_per_class, model_id=args.model)
    elif args.cmd == "path":
        run_c2_path_restrict(max_per_class=args.max_per_class, model_id=args.model)


if __name__ == "__main__":
    main()
