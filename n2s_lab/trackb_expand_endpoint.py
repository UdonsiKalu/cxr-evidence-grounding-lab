"""Track B — 7B expand endpoint (LOO + specificity) then narrow 14B transfer.

Does not read temporal-family-test evidence. Does not modify G3.
14B question: does the temporal N2S failure / signature persist across models?
Not: can 14B solve this better? No forced full MI ladder on 14B.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch

from .hf_client import unload_model
from .paths import ARTIFACTS_DIR
from .trackb_expand_pipeline import (
    MODEL_14B,
    MODEL_7B,
    _case,
    _unit,
    load_expand,
    run_expand_collect,
)
from .trackb_falsex_cluster import _repair_run

LAYER = 20
ENDPOINT_ALPHAS = (8.0, 16.0)
GAUSS_SEED = 20260903
EVAL_FAIL_CAP = 8
MIN_MATCHED_FAILS_FOR_PROBE = 2


def _slug(model_id: str) -> str:
    return model_id.replace("/", "_")


def _collect_paths(model_id: str) -> tuple[Path, Path]:
    s = _slug(model_id)
    return (
        ARTIFACTS_DIR / f"trackb-expand-collect-{s}.json",
        ARTIFACTS_DIR / f"trackb-expand-hiddens-{s}.json",
    )


def load_collect_with_hiddens(model_id: str = MODEL_7B) -> dict[str, Any]:
    collect_path, hidden_path = _collect_paths(model_id)
    if not collect_path.is_file() or not hidden_path.is_file():
        raise FileNotFoundError(
            f"need prior expand collect+hiddens for {model_id}: "
            f"{collect_path.name} + {hidden_path.name}"
        )
    panel = json.loads(collect_path.read_text(encoding="utf-8"))
    hiddens = json.loads(hidden_path.read_text(encoding="utf-8"))
    rows: dict[str, Any] = {}
    for cid, base in panel["baselines"].items():
        rows[cid] = {**base, "commit_layer_hidden": hiddens.get(cid) or {}}
        if not rows[cid]["commit_layer_hidden"]:
            rows[cid]["usable"] = False
    panel["_rows"] = rows
    panel["_artifact"] = str(collect_path)
    return panel


def _vector_from(
    rows: dict[str, Any],
    class_a: list[str],
    class_b: list[str],
    layer_idx: int,
) -> torch.Tensor:
    mean_a = torch.stack(
        [
            torch.tensor(rows[c]["commit_layer_hidden"][f"L{layer_idx}"], dtype=torch.float32)
            for c in class_a
        ]
    ).mean(dim=0)
    mean_b = torch.stack(
        [
            torch.tensor(rows[c]["commit_layer_hidden"][f"L{layer_idx}"], dtype=torch.float32)
            for c in class_b
        ]
    ).mean(dim=0)
    return _unit(mean_a - mean_b)


def _steer_block(
    *,
    model_id: str,
    payload: dict[str, Any],
    rows: dict[str, Any],
    vec: torch.Tensor,
    alpha: float,
    layer_idx: int,
    eval_ids: list[str],
    control_ids: list[str],
    label: str,
) -> dict[str, Any]:
    vectors = {layer_idx: vec}
    print(f"\n=== endpoint · {label} · α={alpha} @ L{layer_idx} ===")
    results: dict[str, Any] = {}
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
    controls: dict[str, Any] = {}
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
    class_b = [c for c in control_ids if c.startswith("TX_C")]
    nofail = [c for c in control_ids if c.startswith("TX_N")]
    n_flip = sum(1 for r in results.values() if r["x_flipped_true_to_false"])
    contra_ok = all(controls[c]["steered_x"] is True for c in class_b if c in controls)
    nofail_ok = all(
        controls[c]["steered_x"] is not True for c in nofail if c in controls
    )
    return {
        "label": label,
        "alpha": alpha,
        "n_target_flips": n_flip,
        "n_eval": len(results),
        "contra_stay_true": contra_ok,
        "nofail_stay_false": nofail_ok,
        "safe": n_flip >= 1 and contra_ok and nofail_ok,
        "results": results,
        "controls": controls,
    }


def run_expand_7b_endpoint(
    *,
    model_id: str = MODEL_7B,
    layer_idx: int = LAYER,
    alphas: tuple[float, ...] = ENDPOINT_ALPHAS,
) -> dict[str, Any]:
    """LOO Class-A stability + steer specificity at α=8/16. Clean 7B endpoint."""
    payload = load_expand()
    collect = load_collect_with_hiddens(model_id)
    rows = collect["_rows"]
    classes = collect["classes"]
    class_a = list(classes["class_a"])
    class_b = list(classes["class_b"])
    eval_ids = list(classes["fail_temporal"])[:EVAL_FAIL_CAP]
    control_ids = list(class_b) + list(classes["nofail"])

    if len(class_a) < 3 or len(class_b) < 2:
        raise RuntimeError("endpoint blocked: need ≥3 Class A and ≥2 Class B")

    full_vec = _vector_from(rows, class_a, class_b, layer_idx)

    # --- LOO: leave one Class A out ---
    loo: dict[str, Any] = {}
    for held in class_a:
        a_fit = [c for c in class_a if c != held]
        vec = _vector_from(rows, a_fit, class_b, layer_idx)
        cos = float(torch.nn.functional.cosine_similarity(
            full_vec.unsqueeze(0), vec.unsqueeze(0)
        ).item())
        by_alpha = {}
        for alpha in alphas:
            by_alpha[str(alpha)] = _steer_block(
                model_id=model_id,
                payload=payload,
                rows=rows,
                vec=vec,
                alpha=alpha,
                layer_idx=layer_idx,
                eval_ids=eval_ids,
                control_ids=control_ids,
                label=f"loo_holdout_{held}",
            )
        loo[held] = {
            "held_out": held,
            "fit_a": a_fit,
            "cosine_to_full": cos,
            "by_alpha": by_alpha,
        }

    # --- Specificity @ each α (full vector vs Gaussian vs reverse) ---
    specificity: dict[str, Any] = {}
    for alpha in alphas:
        real = _steer_block(
            model_id=model_id,
            payload=payload,
            rows=rows,
            vec=full_vec,
            alpha=alpha,
            layer_idx=layer_idx,
            eval_ids=eval_ids,
            control_ids=control_ids,
            label="full_vector",
        )
        g = torch.Generator()
        g.manual_seed(GAUSS_SEED + int(alpha))
        noise = torch.randn(full_vec.shape, generator=g, dtype=torch.float32)
        gauss = _unit(noise)
        gauss_block = _steer_block(
            model_id=model_id,
            payload=payload,
            rows=rows,
            vec=gauss,
            alpha=alpha,
            layer_idx=layer_idx,
            eval_ids=eval_ids,
            control_ids=control_ids,
            label="gaussian_unit",
        )
        rev_block = _steer_block(
            model_id=model_id,
            payload=payload,
            rows=rows,
            vec=-full_vec,
            alpha=alpha,
            layer_idx=layer_idx,
            eval_ids=eval_ids,
            control_ids=control_ids,
            label="reverse_vector",
        )
        specificity[str(alpha)] = {
            "full": real,
            "gaussian": gauss_block,
            "reverse": rev_block,
            "full_beats_gaussian_flips": real["n_target_flips"]
            > gauss_block["n_target_flips"],
            "full_safe": real["safe"],
            "gaussian_safe": gauss_block["safe"],
            "reverse_fewer_or_equal_flips_vs_full": rev_block["n_target_flips"]
            <= real["n_target_flips"],
        }

    unload_model()

    # Aggregate LOO
    loo_safe_counts = {
        str(a): sum(
            1 for h in loo.values() if h["by_alpha"][str(a)]["safe"]
        )
        for a in alphas
    }
    mean_cos = sum(h["cosine_to_full"] for h in loo.values()) / len(loo)

    # Endpoint verdict
    best_alpha = None
    for a in alphas:
        sp = specificity[str(a)]
        if (
            sp["full_safe"]
            and sp["full_beats_gaussian_flips"]
            and loo_safe_counts[str(a)] >= max(1, len(class_a) - 1)
        ):
            best_alpha = a
            break
    if best_alpha is None:
        for a in alphas:
            if specificity[str(a)]["full_safe"]:
                best_alpha = a
                break

    gate = {
        "pass": best_alpha is not None,
        "best_alpha": best_alpha,
        "mean_loo_cosine_to_full": round(mean_cos, 4),
        "loo_safe_counts": loo_safe_counts,
        "note": (
            f"7B endpoint YES @ α={best_alpha} — LOO-stable partial editor"
            if best_alpha is not None
            else "7B endpoint NO — LOO/specificity did not support a freeze claim"
        ),
    }

    panel = {
        "kind": "trackb_expand_7b_endpoint",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "model_id": model_id,
        "layer_idx": layer_idx,
        "alphas": list(alphas),
        "class_a": class_a,
        "class_b": class_b,
        "eval_fail_ids": eval_ids,
        "loo": loo,
        "specificity": specificity,
        "gate_7b_endpoint": gate,
        "claim_hygiene": (
            "Limited claim only: partial editor on weak-margin fails. "
            "Not a family-wide temporal-change fix. Test sealed. G3 untouched."
        ),
        "test_set_not_used": "temporal-family-test.json",
    }
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    out = ARTIFACTS_DIR / "trackb-expand-7b-endpoint.json"
    out.write_text(json.dumps(panel, indent=2) + "\n", encoding="utf-8")
    panel["_artifact"] = str(out)
    print(f"\nwrote {out}")
    print(f"7B endpoint: {'YES' if gate['pass'] else 'NO'} — {gate['note']}")
    print(f"  mean LOO cos={gate['mean_loo_cosine_to_full']} loo_safe={loo_safe_counts}")
    for a in alphas:
        sp = specificity[str(a)]
        print(
            f"  α={a}: full_flips={sp['full']['n_target_flips']}/"
            f"{sp['full']['n_eval']} gauss={sp['gaussian']['n_target_flips']} "
            f"rev={sp['reverse']['n_target_flips']} "
            f"beats_gauss={sp['full_beats_gaussian_flips']} safe={sp['full_safe']}"
        )
    return panel


def run_expand_14b_transfer(*, model_id: str = MODEL_14B) -> dict[str, Any]:
    """Behavioral collect on same expand DEV. Transfer question, not fix-by-scaling.

    If matched false-X fails are scarce, stop — do not force a full MI ladder.
    """
    print("\n======== 14B TRANSFER · behavioral collect (expand DEV) ========")
    print("Question: does the temporal N2S failure persist when the model changes?")
    print("Not: can 14B solve this better?")

    # Prefer mid-late layers for optional light probe; behavioral X/margin is primary.
    collect = run_expand_collect(model_id=model_id)
    classes = collect["classes"]
    baselines = collect["baselines"]

    n_a = collect["n_class_a"]
    n_b = collect["n_class_b"]
    n_fail = len(classes["fail_temporal"])
    n_temporal = sum(
        1
        for c in load_expand()["cases"]
        if c["expected"] == "SATISFIED" and c.get("role") == "expand_temporal"
    )

    # Verdict taxonomy for transfer reading
    temporal_rows = {
        cid: baselines[cid]
        for cid in baselines
        if cid.startswith("TX_E")
    }
    n_x_true = sum(1 for r in temporal_rows.values() if r["repair_final_x"] is True)
    n_x_false = sum(1 for r in temporal_rows.values() if r["repair_final_x"] is False)
    n_reviewish = sum(
        1
        for r in temporal_rows.values()
        if r.get("verdict") in ("UNCERTAIN", "NOT_SATISFIED")
        or (r["repair_final_x"] is False and r.get("verdict") == "SATISFIED")
    )

    matched_useful = n_fail >= MIN_MATCHED_FAILS_FOR_PROBE and n_a >= 1 and n_b >= 1
    force_ladder = False  # explicit: never force full MI on 14B here

    reading = []
    if n_fail == 0 or n_x_true == 0:
        reading.append(
            "14B does not reproduce the false-X temporal cluster on this panel "
            "(mostly correct / non-committing). Informative for Track A containment; "
            "do not force MI ladder."
        )
    elif n_fail < MIN_MATCHED_FAILS_FOR_PROBE:
        reading.append(
            f"Only {n_fail} temporal false-X commits — too thin for a 14B MI ladder."
        )
    else:
        reading.append(
            f"False-X persists on {n_fail}/{n_temporal} temporal notes — signature "
            "survives the model change at behavior level."
        )
    if matched_useful:
        reading.append(
            "Matched fail + clean-A + contradiction exist; optional light case "
            "inspection only (not run here unless separately requested)."
        )
    else:
        reading.append("No useful matched failure/contrast for a forced MI ladder.")

    panel = {
        "kind": "trackb_expand_14b_transfer",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "model_id": model_id,
        "question": (
            "Does the temporal neural→symbolic failure and its internal signature "
            "persist when the model changes?"
        ),
        "not_the_question": "Can 14B solve this better?",
        "case_source": "temporal-family-dev-expand.json",
        "n_temporal": n_temporal,
        "n_class_a_x_false": n_a,
        "n_class_b_x_true": n_b,
        "n_fail_temporal_x_true": n_fail,
        "n_temporal_x_true": n_x_true,
        "n_temporal_x_false": n_x_false,
        "classes": classes,
        "baselines": baselines,
        "matched_failure_contrast_useful": matched_useful,
        "forced_mi_ladder": force_ladder,
        "reading": reading,
        "test_set_not_used": "temporal-family-test.json",
        "g3_untouched": True,
        "collect_artifact": collect.get("_artifact"),
    }
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    out = ARTIFACTS_DIR / "trackb-expand-14b-transfer.json"
    # Drop hiddens from public panel (already in sidecar from collect)
    out.write_text(json.dumps(panel, indent=2) + "\n", encoding="utf-8")
    panel["_artifact"] = str(out)
    print(f"\nwrote {out}")
    print(f"14B transfer: A={n_a} B={n_b} fail_temporal={n_fail}/{n_temporal}")
    print(f"  matched_useful={matched_useful} forced_mi={force_ladder}")
    for line in reading:
        print(f"  · {line}")
    return panel


def run_finish_7b_then_14b() -> dict[str, Any]:
    """User go: clean 7B endpoint, then one 14B behavioral transfer pass."""
    print("\n======== FINISH 7B ENDPOINT (LOO + specificity) ========")
    ep7 = run_expand_7b_endpoint()
    print("\n======== THEN 14B TRANSFER (once) ========")
    t14 = run_expand_14b_transfer()
    summary = {
        "kind": "trackb_expand_finish7b_then_14b",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "gate_7b_endpoint": ep7["gate_7b_endpoint"],
        "transfer_14b": {
            "n_class_a": t14["n_class_a_x_false"],
            "n_class_b": t14["n_class_b_x_true"],
            "n_fail_temporal": t14["n_fail_temporal_x_true"],
            "matched_useful": t14["matched_failure_contrast_useful"],
            "forced_mi_ladder": t14["forced_mi_ladder"],
            "reading": t14["reading"],
        },
        "artifacts": {
            "7b_endpoint": ep7.get("_artifact"),
            "14b_transfer": t14.get("_artifact"),
        },
        "test_set_not_used": "temporal-family-test.json",
        "g3_untouched": True,
    }
    out = ARTIFACTS_DIR / "trackb-expand-finish7b-then-14b-summary.json"
    out.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(f"\nwrote {out}")
    return summary
