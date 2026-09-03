"""Track B — frozen expand panel collect / optional family fit (DEV expand only).

Wording in data/temporal-family-dev-expand.json was frozen before any HF run.
Does not read temporal-family-test.json evidence. Does not modify G3.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any

import torch

from .hf_client import load_model, unload_model
from .hf_trace import _layer_count
from .paths import ARTIFACTS_DIR, TEMPORAL_FAMILY_EXPAND_PATH
from .phase10_activation import FAIL_MODEL
from .trackb_falsex_cluster import _repair_run
from .trackb_family_vector import SWEEP_ALPHAS, _unit

MODEL_7B = FAIL_MODEL
MODEL_14B = "Qwen/Qwen2.5-14B-Instruct"
ENOUGH_A = 3
ENOUGH_B = 2
NEARLY_EMPTY_A = 2  # n_A <= this → do not fit on that model


def load_expand() -> dict[str, Any]:
    return json.loads(TEMPORAL_FAMILY_EXPAND_PATH.read_text(encoding="utf-8"))


def _case(payload: dict[str, Any], case_id: str) -> dict[str, Any]:
    for c in payload["cases"]:
        if c["id"] == case_id:
            return c
    raise KeyError(case_id)


def _is_temporal_satisfied(c: dict[str, Any]) -> bool:
    return c["expected"] == "SATISFIED" and c.get("role") == "expand_temporal"


def _is_contradiction(c: dict[str, Any]) -> bool:
    return c["expected"] == "CONTRADICTION"


def _is_nofail(c: dict[str, Any]) -> bool:
    return c["expected"] == "NOT_SATISFIED" and "nofail" in c.get("role", "")


def _collect_one(
    case: dict[str, Any],
    *,
    model_id: str,
    layer_indices: tuple[int, ...],
) -> dict[str, Any]:
    cid = case["id"]
    print(f"\n=== expand · {model_id.split('/')[-1]} · collect · {cid} ===")
    row = _repair_run(
        case,
        model_id=model_id,
        intervention="none",
        store_commit_hidden=True,
        layer_indices=layer_indices,
    )
    hidden = row.get("commit_layer_hidden") or {}
    return {
        "case_id": cid,
        "gold": case["expected"],
        "role": case.get("role"),
        "subtype": case.get("subtype"),
        "repair_final_x": row["repair_final_x"],
        "margin": row["logit_margin_true_minus_false"],
        "verdict": row["verdict"],
        "usable": bool(hidden),
        "commit_layer_hidden": hidden,
    }


def _classify(rows: dict[str, Any], payload: dict[str, Any]) -> dict[str, list[str]]:
    by_id = {c["id"]: c for c in payload["cases"]}
    class_a, class_b, fail_temporal, nofail = [], [], [], []
    for cid, r in rows.items():
        c = by_id[cid]
        x = r["repair_final_x"]
        if _is_temporal_satisfied(c) and x is False and r["usable"]:
            class_a.append(cid)
        if _is_contradiction(c) and x is True and r["usable"]:
            class_b.append(cid)
        if _is_temporal_satisfied(c) and x is True:
            fail_temporal.append(cid)
        if _is_nofail(c):
            nofail.append(cid)
    return {
        "class_a": class_a,
        "class_b": class_b,
        "fail_temporal": fail_temporal,
        "nofail": nofail,
    }


def _strip_hidden(rows: dict[str, Any]) -> dict[str, Any]:
    out = {}
    for cid, r in rows.items():
        d = {k: v for k, v in r.items() if k != "commit_layer_hidden"}
        out[cid] = d
    return out


def _fit_and_sweep(
    *,
    model_id: str,
    payload: dict[str, Any],
    rows: dict[str, Any],
    classes: dict[str, list[str]],
    layer_idx: int,
) -> dict[str, Any]:
    mean_a = torch.stack(
        [
            torch.tensor(rows[c]["commit_layer_hidden"][f"L{layer_idx}"], dtype=torch.float32)
            for c in classes["class_a"]
        ]
    ).mean(dim=0)
    mean_b = torch.stack(
        [
            torch.tensor(rows[c]["commit_layer_hidden"][f"L{layer_idx}"], dtype=torch.float32)
            for c in classes["class_b"]
        ]
    ).mean(dim=0)
    vec = _unit(mean_a - mean_b)
    vectors = {layer_idx: vec}

    eval_ids = list(classes["fail_temporal"])[:8]  # cap GPU
    control_ids = list(classes["class_b"]) + list(classes["nofail"])
    by_alpha: dict[str, Any] = {}
    for alpha in SWEEP_ALPHAS:
        if alpha == 0.0:
            continue
        print(f"\n=== expand · sweep α={alpha} @ L{layer_idx} ===")
        results = {}
        for cid in eval_ids:
            print(f"  · target {cid}")
            row = _repair_run(
                _case(payload, cid),
                model_id=model_id,
                intervention="activation_steer",
                vectors_by_layer=vectors,
                alpha=alpha,
                layer_indices=(layer_idx,),
            )
            base = rows[cid]
            results[cid] = {
                "baseline_x": base["repair_final_x"],
                "steered_x": row["repair_final_x"],
                "baseline_margin": base["margin"],
                "steered_margin": row["logit_margin_true_minus_false"],
                "x_flipped_true_to_false": base["repair_final_x"] is True
                and row["repair_final_x"] is False,
            }
        controls = {}
        for cid in control_ids:
            print(f"  · control {cid}")
            row = _repair_run(
                _case(payload, cid),
                model_id=model_id,
                intervention="activation_steer",
                vectors_by_layer=vectors,
                alpha=alpha,
                layer_indices=(layer_idx,),
            )
            controls[cid] = {
                "baseline_x": rows[cid]["repair_final_x"],
                "steered_x": row["repair_final_x"],
                "baseline_margin": rows[cid]["margin"],
                "steered_margin": row["logit_margin_true_minus_false"],
            }
        n_flip = sum(1 for r in results.values() if r["x_flipped_true_to_false"])
        contra_ok = all(
            controls[c]["steered_x"] is True
            for c in classes["class_b"]
            if c in controls
        )
        nofail_ok = all(
            controls[c]["steered_x"] is not True
            for c in classes["nofail"]
            if c in controls
        )
        by_alpha[str(alpha)] = {
            "alpha": alpha,
            "n_target_flips": n_flip,
            "n_eval": len(results),
            "contra_stay_true": contra_ok,
            "nofail_stay_false": nofail_ok,
            "safe": n_flip >= 1 and contra_ok and nofail_ok,
            "results": results,
            "controls": controls,
        }
    unload_model()
    return {
        "layer_idx": layer_idx,
        "formula": "unit(mean(A)-mean(B))",
        "vector_norm_raw": float((mean_a - mean_b).norm().item()),
        "by_alpha": by_alpha,
    }


def _sweep_layers_for(n_layers: int) -> tuple[int, ...]:
    """Coarse absolute indices; never the final block."""
    last = n_layers - 2
    fracs = (0.15, 0.30, 0.45, 0.60, 0.75, 0.90)
    out = []
    for f in fracs:
        i = min(last, max(0, int(round(f * (n_layers - 1)))))
        if i not in out and i != n_layers - 1:
            out.append(i)
    return tuple(out)


def run_expand_collect(
    *,
    model_id: str,
    layer_indices: tuple[int, ...] | None = None,
) -> dict[str, Any]:
    payload = load_expand()
    if layer_indices is None:
        model, _ = load_model(model_id)
        n = _layer_count(model)
        del model
        unload_model()
        # 7B: L20; other: mid-late site as default collect (patch-depth may override)
        if n == 28:
            layer_indices = (20,)
        else:
            layer_indices = _sweep_layers_for(n)
            # also keep a 0.75 analogue
            idx75 = min(n - 2, max(0, int(round(0.75 * (n - 1)))))
            if idx75 not in layer_indices:
                layer_indices = tuple(sorted(layer_indices + (idx75,)))
    else:
        n = None

    rows: dict[str, Any] = {}
    for case in payload["cases"]:
        rows[case["id"]] = _collect_one(
            case, model_id=model_id, layer_indices=layer_indices
        )
    unload_model()
    classes = _classify(rows, payload)
    n_a, n_b = len(classes["class_a"]), len(classes["class_b"])
    enough = n_a >= ENOUGH_A and n_b >= ENOUGH_B
    thin = n_a <= NEARLY_EMPTY_A

    panel = {
        "kind": "trackb_expand_collect",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "model_id": model_id,
        "case_source": TEMPORAL_FAMILY_EXPAND_PATH.name,
        "test_set_not_used": "temporal-family-test.json",
        "n_layers_hint": n,
        "layer_indices": list(layer_indices),
        "predeclared": {
            "enough": f"n_A>={ENOUGH_A} and n_B>={ENOUGH_B}",
            "nearly_empty": f"n_A<={NEARLY_EMPTY_A}",
        },
        "classes": classes,
        "n_class_a": n_a,
        "n_class_b": n_b,
        "enough_to_fit": enough,
        "nearly_empty_class_a": thin,
        "baselines": _strip_hidden(rows),
        "note": "Wording frozen before this collect. G3 untouched.",
    }
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    slug = model_id.replace("/", "_")
    out = ARTIFACTS_DIR / f"trackb-expand-collect-{slug}.json"
    sidecar = ARTIFACTS_DIR / f"trackb-expand-hiddens-{slug}.json"
    sidecar.write_text(
        json.dumps(
            {cid: r.get("commit_layer_hidden") for cid, r in rows.items()},
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    out.write_text(json.dumps(panel, indent=2) + "\n", encoding="utf-8")
    panel["_rows"] = rows
    panel["_artifact"] = str(out)
    panel["_hiddens"] = str(sidecar)
    print(f"\nwrote {out}")
    print(f"class A n={n_a} {classes['class_a']}")
    print(f"class B n={n_b} {classes['class_b']}")
    print(f"fail_temporal n={len(classes['fail_temporal'])} enough={enough} thin={thin}")
    return panel


def run_expand_fit_from_collect(
    collect_panel: dict[str, Any],
    *,
    layer_idx: int,
) -> dict[str, Any]:
    payload = load_expand()
    rows = collect_panel["_rows"]
    classes = collect_panel["classes"]
    if collect_panel["n_class_a"] < ENOUGH_A or collect_panel["n_class_b"] < ENOUGH_B:
        raise RuntimeError("fit blocked: class A/B below predeclared threshold")
    sweep = _fit_and_sweep(
        model_id=collect_panel["model_id"],
        payload=payload,
        rows=rows,
        classes=classes,
        layer_idx=layer_idx,
    )
    panel = {
        "kind": "trackb_expand_family_fit",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "model_id": collect_panel["model_id"],
        "class_a": classes["class_a"],
        "class_b": classes["class_b"],
        "sweep": sweep,
        "test_set_not_used": "temporal-family-test.json",
    }
    slug = collect_panel["model_id"].replace("/", "_")
    out = ARTIFACTS_DIR / f"trackb-expand-fit-{slug}-L{layer_idx}.json"
    out.write_text(json.dumps(panel, indent=2) + "\n", encoding="utf-8")
    panel["_artifact"] = str(out)
    print(f"\nwrote {out}")
    for a, b in sweep["by_alpha"].items():
        print(
            f"  α={a}: flips={b['n_target_flips']}/{b['n_eval']} "
            f"contra={b['contra_stay_true']} nofail={b['nofail_stay_false']} safe={b['safe']}"
        )
    return panel


def _patch_expand(
    payload: dict[str, Any],
    recipient_id: str,
    *,
    model_id: str,
    layer_idx: int,
    donor_vec: torch.Tensor,
    donor_id: str,
    baseline: dict[str, Any],
) -> dict[str, Any]:
    print(f"\n=== expand-patch · L{layer_idx} · {recipient_id} ← {donor_id} ===")
    row = _repair_run(
        _case(payload, recipient_id),
        model_id=model_id,
        intervention="activation_patch",
        patch_by_layer={layer_idx: donor_vec},
        layer_indices=(layer_idx,),
    )
    base_x = baseline["repair_final_x"]
    return {
        "recipient": recipient_id,
        "donor": donor_id,
        "layer_idx": layer_idx,
        "baseline_x": base_x,
        "patched_x": row["repair_final_x"],
        "baseline_margin": baseline["margin"],
        "patched_margin": row["logit_margin_true_minus_false"],
        "x_flipped_true_to_false": base_x is True and row["repair_final_x"] is False,
    }


def run_expand_patch_depth(
    collect_panel: dict[str, Any],
    *,
    sweep_layers: tuple[int, ...] | None = None,
) -> dict[str, Any]:
    """Patch-depth on expand failures using a nofail donor that is X=false."""
    payload = load_expand()
    model_id = collect_panel["model_id"]
    rows = collect_panel["_rows"]
    classes = collect_panel["classes"]
    nofail_ok = [
        c for c in classes["nofail"] if rows[c]["repair_final_x"] is False
    ]
    fails = classes["fail_temporal"][:3]
    cons = classes["class_b"][:2]
    if not nofail_ok or len(fails) < 2:
        raise RuntimeError("patch-depth needs X=false nofail donor and ≥2 fail temporals")
    donor_id = nofail_ok[0]

    model, _ = load_model(model_id)
    n = _layer_count(model)
    del model
    unload_model()
    layers = sweep_layers or _sweep_layers_for(n)

    need_ids = list(dict.fromkeys([donor_id, *fails, *cons]))
    collected = {}
    for cid in need_ids:
        collected[cid] = _collect_one(
            _case(payload, cid), model_id=model_id, layer_indices=layers
        )

    by_layer = {}
    earliest = None
    for L in layers:
        donor_vec = torch.tensor(
            collected[donor_id]["commit_layer_hidden"][f"L{L}"], dtype=torch.float32
        )
        layer_results = [
            _patch_expand(
                payload,
                tid,
                model_id=model_id,
                layer_idx=L,
                donor_vec=donor_vec,
                donor_id=donor_id,
                baseline=collected[tid],
            )
            for tid in fails
        ]
        n_flip = sum(1 for r in layer_results if r["x_flipped_true_to_false"])
        by_layer[f"L{L}"] = {
            "layer_idx": L,
            "n_flips": n_flip,
            "results": layer_results,
        }
        if earliest is None and n_flip >= 2:
            earliest = L

    unload_model()
    panel = {
        "kind": "trackb_expand_patch_depth",
        "model_id": model_id,
        "n_layers": n,
        "sweep_layers": list(layers),
        "donor": donor_id,
        "targets": fails,
        "earliest_sufficient_layer": earliest,
        "by_layer": by_layer,
        "note": "expand-panel patch-depth; test sealed; G3 untouched",
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    slug = model_id.replace("/", "_")
    out = ARTIFACTS_DIR / f"trackb-expand-patch-depth-{slug}.json"
    out.write_text(json.dumps(panel, indent=2) + "\n", encoding="utf-8")
    panel["_artifact"] = str(out)
    print(f"\nwrote {out}")
    print(f"earliest sufficient={earliest}")
    for k, b in by_layer.items():
        print(f"  {k}: flips={b['n_flips']}/{len(fails)}")
    return panel


def run_expand_sequence() -> dict[str, Any]:
    """7B collect → fit if enough A/B; else 14B collect → patch-depth → fit if split exists."""
    print("\n======== EXPAND SEQUENCE · HF 7B collect ========")
    p7 = run_expand_collect(model_id=MODEL_7B)
    if p7["enough_to_fit"]:
        print("7B has enough Class A/B — fitting L20 family vector")
        fit = run_expand_fit_from_collect(p7, layer_idx=20)
        summary = {"stage": "7b_fit", "collect_7b": _public(p7), "fit": _public(fit)}
        _write_sequence_summary(summary)
        return summary

    print(
        f"7B Class A n={p7['n_class_a']} (thin/empty) — do not fit on 7B; moving to 14B"
    )
    print("\n======== EXPAND SEQUENCE · HF 14B collect ========")
    p14 = run_expand_collect(model_id=MODEL_14B)
    depth = None
    try:
        print("\n======== EXPAND SEQUENCE · 14B patch-depth ========")
        depth = run_expand_patch_depth(p14)
    except RuntimeError as exc:
        print(f"14B patch-depth skipped: {exc}")

    layer = (depth or {}).get("earliest_sufficient_layer")
    if layer is None:
        model, _ = load_model(MODEL_14B)
        n = _layer_count(model)
        del model
        unload_model()
        layer = min(n - 2, max(0, int(round(0.75 * (n - 1)))))
        print(f"no earliest patch layer; using 0.75 analogue L{layer}")

    both_correct_and_fail = bool(p14["classes"]["class_a"]) and bool(
        p14["classes"]["fail_temporal"]
    )
    if p14["enough_to_fit"] and both_correct_and_fail:
        print(f"14B enough Class A/B with fail split — fitting @ L{layer}")
        fit = run_expand_fit_from_collect(p14, layer_idx=layer)
        summary = {
            "stage": "14b_fit",
            "collect_7b": _public(p7),
            "collect_14b": _public(p14),
            "depth": _public(depth) if depth else None,
            "fit": _public(fit),
        }
        _write_sequence_summary(summary)
        return summary

    summary = {
        "stage": "14b_no_fit",
        "collect_7b": _public(p7),
        "collect_14b": _public(p14),
        "depth": _public(depth) if depth else None,
        "reason": "14B Class A still below threshold or no fail/correct split",
    }
    _write_sequence_summary(summary)
    return summary


def _write_sequence_summary(summary: dict[str, Any]) -> None:
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    out = ARTIFACTS_DIR / "trackb-expand-sequence-summary.json"
    out.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(f"\nwrote {out}")


def _public(panel: dict[str, Any] | None) -> dict[str, Any] | None:
    if panel is None:
        return None
    return {k: v for k, v in panel.items() if not str(k).startswith("_")}
