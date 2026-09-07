"""Thin SAE pilot: decompose frozen L20 expand direction v.

Uses Chanin Qwen2.5-7B-Instruct L20 JumpReLU SAE
(``chanind/qwen2.5-7B-it-layer-20-saes``, pile/matryoshka/k-100,
hook ``blocks.20.hook_resid_post``).

No framework. No English feature labels.

Phases
------
1. **score** (CPU): encode frozen Class A/B (+ expand) L20 commit hiddens;
   rank by Δ(A−B), |cos| to v, and X/margin tracking.
2. **causal** (GPU): top-k feature steers on EX_TEMPORAL_FOLFOX + EX_CONTRA
   with reverse + random-feature controls.

CLI::

    python -m n2s_lab.n2s_sae_pilot score
    python -m n2s_lab.n2s_sae_pilot causal --top-k 5 --alpha 8
    python -m n2s_lab.n2s_sae_pilot all --top-k 5 --alpha 8
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch
from huggingface_hub import hf_hub_download
from safetensors import safe_open

from .hf_client import unload_model
from .n2s_forensics import DIRECTIONS_PATH, HIDDENS_PATH, MIN_FREE_VRAM_GB, cuda_free_gb
from .n2s_intervene import STEER_LAYER
from .paths import ARTIFACTS_DIR
from .phase10_activation import FAIL_MODEL
from .trackb_falsex_cluster import _repair_run
from .trackb_family_vector import _unit

SAE_REPO = "chanind/qwen2.5-7B-it-layer-20-saes"
SAE_SUBDIR = "pile/matryoshka/k-100"
FIT_PATH = ARTIFACTS_DIR / "trackb-expand-fit-Qwen_Qwen2.5-7B-Instruct-L20.json"
COLLECT_PATH = ARTIFACTS_DIR / "trackb-expand-collect-Qwen_Qwen2.5-7B-Instruct.json"
SCORE_PATH = ARTIFACTS_DIR / "n2s-sae-pilot-scores.json"
CAUSAL_PATH = ARTIFACTS_DIR / "n2s-sae-pilot-causal.json"

# Live causal notes — prefer expand false-X + class-B contra when available.
# Soft walkthrough FOLFOX often already X=false on 7B (poor flip demo).
DEFAULT_CAUSAL_IDS = ("TX_E01", "TX_C01")  # expand false-X + true contra


def _load_expand_case(case_id: str) -> dict[str, str]:
    from .paths import TEMPORAL_FAMILY_EXPAND_PATH

    data = json.loads(TEMPORAL_FAMILY_EXPAND_PATH.read_text(encoding="utf-8"))
    for c in data.get("cases") or []:
        if c.get("id") == case_id:
            return {
                "id": case_id,
                "gold": str(c.get("expected") or "UNKNOWN"),
                "evidence": c["evidence"],
            }
    raise KeyError(case_id)


def causal_notes(ids: tuple[str, ...] | list[str] | None = None) -> dict[str, dict[str, str]]:
    """Build note map: expand IDs preferred; fall back to walkthrough FOLFOX pair."""
    chosen = tuple(ids) if ids else DEFAULT_CAUSAL_IDS
    out: dict[str, dict[str, str]] = {}
    for cid in chosen:
        if cid in CAUSAL_NOTES:
            out[cid] = CAUSAL_NOTES[cid]
            continue
        try:
            out[cid] = _load_expand_case(cid)
        except Exception as exc:  # noqa: BLE001
            raise SystemExit(f"Cannot load causal note {cid}: {exc}") from exc
    return out


# Walkthrough FOLFOX pair (kept for optional --notes EX_TEMPORAL_FOLFOX,EX_CONTRA)
CAUSAL_NOTES: dict[str, dict[str, str]] = {
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

RANDOM_FEATURE_SEED = 20260905


class JumpReLUSae:
    """Minimal JumpReLU SAE (SAELens-compatible weight names)."""

    def __init__(
        self,
        *,
        W_enc: torch.Tensor,
        b_enc: torch.Tensor,
        W_dec: torch.Tensor,
        b_dec: torch.Tensor,
        threshold: torch.Tensor,
        apply_b_dec_to_input: bool = True,
    ) -> None:
        self.W_enc = W_enc
        self.b_enc = b_enc
        self.W_dec = W_dec  # (d_sae, d_in)
        self.b_dec = b_dec
        self.threshold = threshold
        self.apply_b_dec_to_input = apply_b_dec_to_input
        self.d_sae = int(W_dec.shape[0])
        self.d_in = int(W_dec.shape[1])

    @classmethod
    def from_hub(cls, repo: str = SAE_REPO, subdir: str = SAE_SUBDIR) -> tuple["JumpReLUSae", dict[str, Any]]:
        cfg_path = hf_hub_download(repo, f"{subdir}/cfg.json")
        w_path = hf_hub_download(repo, f"{subdir}/sae_weights.safetensors")
        cfg = json.loads(Path(cfg_path).read_text(encoding="utf-8"))
        with safe_open(w_path, framework="pt", device="cpu") as f:
            tensors = {k: f.get_tensor(k) for k in f.keys()}
        sae = cls(
            W_enc=tensors["W_enc"].float(),
            b_enc=tensors["b_enc"].float(),
            W_dec=tensors["W_dec"].float(),
            b_dec=tensors["b_dec"].float(),
            threshold=tensors["threshold"].float(),
            apply_b_dec_to_input=bool(cfg.get("apply_b_dec_to_input", True)),
        )
        meta = {
            "repo": repo,
            "subdir": subdir,
            "cfg_path": cfg_path,
            "weights_path": w_path,
            "d_sae": sae.d_sae,
            "d_in": sae.d_in,
            "hook_name": (cfg.get("metadata") or {}).get("hook_name"),
            "model_name": (cfg.get("metadata") or {}).get("model_name"),
            "architecture": cfg.get("architecture"),
            "apply_b_dec_to_input": sae.apply_b_dec_to_input,
        }
        return sae, meta

    @torch.no_grad()
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """x: (..., d_in) → features (..., d_sae)."""
        h = x.float()
        if self.apply_b_dec_to_input:
            h = h - self.b_dec
        pre = h @ self.W_enc + self.b_enc
        return pre * (pre > self.threshold)

    def decoder_unit(self, feat_id: int) -> torch.Tensor:
        return _unit(self.W_dec[feat_id].float())


def _cos(a: torch.Tensor, b: torch.Tensor) -> float:
    a = a.float().reshape(-1)
    b = b.float().reshape(-1)
    denom = float(a.norm().item() * b.norm().item())
    if denom < 1e-12:
        return 0.0
    return float(torch.dot(a, b).item() / denom)


def _pearson(xs: list[float], ys: list[float]) -> float | None:
    if len(xs) < 3 or len(xs) != len(ys):
        return None
    x = torch.tensor(xs, dtype=torch.float64)
    y = torch.tensor(ys, dtype=torch.float64)
    x = x - x.mean()
    y = y - y.mean()
    denom = float(x.norm().item() * y.norm().item())
    if denom < 1e-12:
        return None
    return float((x * y).sum().item() / denom)


def hook_sanity(sae: JumpReLUSae, hiddens: torch.Tensor, v: torch.Tensor) -> dict[str, Any]:
    """Recon MSE on A/B commit hiddens + decoder alignment mass with v."""
    feats = sae.encode(hiddens)
    recon = feats @ sae.W_dec + sae.b_dec
    mse = float(((recon - hiddens) ** 2).mean().item())
    # fraction of variance explained (rough)
    var = float(((hiddens - hiddens.mean(dim=0)) ** 2).mean().item())
    fve = 1.0 - (mse / var) if var > 1e-12 else None
    # top |cos| of decoder cols vs v
    # W_dec is (d_sae, d_in); compute in chunks
    v_u = _unit(v)
    abs_cos = (sae.W_dec @ v_u) / (sae.W_dec.norm(dim=1).clamp_min(1e-12))
    abs_cos = abs_cos.abs()
    top_vals, top_idx = torch.topk(abs_cos, k=10)
    return {
        "n_hiddens": int(hiddens.shape[0]),
        "recon_mse": mse,
        "frac_var_explained_approx": fve,
        "max_abs_cos_decoder_to_v": float(abs_cos.max().item()),
        "mean_abs_cos_decoder_to_v": float(abs_cos.mean().item()),
        "top10_abs_cos": [
            {"feat_id": int(i), "abs_cos": float(c)}
            for i, c in zip(top_idx.tolist(), top_vals.tolist())
        ],
        "pass_heuristic": bool(
            mse < 2.0  # residual units; soft
            and float(abs_cos.max().item()) > 0.05
        ),
        "note": (
            "HF L20 block-output hiddens vs SAELens resid_post — soft match. "
            "If pass_heuristic is false, stop and consider training a local SAE."
        ),
    }


def run_score(*, top_report: int = 30) -> dict[str, Any]:
    fit = json.loads(FIT_PATH.read_text(encoding="utf-8"))
    directions = json.loads(DIRECTIONS_PATH.read_text(encoding="utf-8"))
    hiddens_raw = json.loads(HIDDENS_PATH.read_text(encoding="utf-8"))
    collect = json.loads(COLLECT_PATH.read_text(encoding="utf-8"))
    raw_base = collect.get("baselines") or {}
    if isinstance(raw_base, dict):
        baselines = raw_base
    else:
        baselines = {r["case_id"]: r for r in raw_base}

    class_a = list(fit["class_a"])
    class_b = list(fit["class_b"])
    v = torch.tensor(directions["direction_a_minus_b"], dtype=torch.float32)
    v = _unit(v)

    def stack_ids(ids: list[str]) -> torch.Tensor:
        rows = []
        for cid in ids:
            h = (hiddens_raw.get(cid) or {}).get("L20")
            if not h:
                raise KeyError(f"missing L20 for {cid}")
            rows.append(torch.tensor(h, dtype=torch.float32))
        return torch.stack(rows, dim=0)

    print("[sae-pilot] loading Chanin L20 SAE (CPU)…")
    sae, sae_meta = JumpReLUSae.from_hub()
    assert sae.d_in == int(v.numel()), f"d_in mismatch {sae.d_in} vs v {v.numel()}"

    a_h = stack_ids(class_a)
    b_h = stack_ids(class_b)
    sanity = hook_sanity(sae, torch.cat([a_h, b_h], dim=0), v)
    print(
        f"[sae-pilot] sanity recon_mse={sanity['recon_mse']:.4f} "
        f"max|cos|={sanity['max_abs_cos_decoder_to_v']:.4f} "
        f"pass={sanity['pass_heuristic']}"
    )

    print("[sae-pilot] encoding Class A/B…")
    feat_a = sae.encode(a_h)  # (n_a, d_sae)
    feat_b = sae.encode(b_h)
    mean_a = feat_a.mean(dim=0)
    mean_b = feat_b.mean(dim=0)
    delta = mean_a - mean_b

    # decoder cos to v (signed)
    dec_norms = sae.W_dec.norm(dim=1).clamp_min(1e-12)
    cos_to_v = (sae.W_dec @ v) / dec_norms

    # Expand-wide X / margin tracking (all cases with L20 + baseline)
    expand_ids = [cid for cid in hiddens_raw if "L20" in (hiddens_raw.get(cid) or {})]
    expand_h = stack_ids(expand_ids)
    print(f"[sae-pilot] encoding expand panel n={len(expand_ids)}…")
    feat_all = sae.encode(expand_h)  # (n, d_sae)
    x_list: list[float] = []
    margin_list: list[float] = []
    keep_idx: list[int] = []
    for i, cid in enumerate(expand_ids):
        b = baselines.get(cid) or {}
        if b.get("repair_final_x") is None or b.get("margin") is None:
            continue
        keep_idx.append(i)
        x_list.append(1.0 if b["repair_final_x"] is True else 0.0)
        margin_list.append(float(b["margin"]))
    feat_track = feat_all[keep_idx] if keep_idx else feat_all[:0]

    # Rank: prefer features that move A vs B AND align with v AND track X/margin
    # Score = |Δ| * (0.5 + |cos|) * (0.5 + |r_x| or |r_margin|)
    candidates: list[dict[str, Any]] = []
    # Only score features with any mass on A or B (sparse)
    mass = (mean_a.abs() + mean_b.abs())
    active = (mass > 1e-6).nonzero(as_tuple=False).view(-1)
    print(f"[sae-pilot] active features on A∪B: {int(active.numel())} / {sae.d_sae}")

    for j in active.tolist():
        dlt = float(delta[j].item())
        ctv = float(cos_to_v[j].item())
        acts = feat_track[:, j].tolist() if feat_track.numel() else []
        r_x = _pearson(acts, x_list) if acts else None
        r_m = _pearson(acts, margin_list) if acts else None
        track = max(abs(r_x or 0.0), abs(r_m or 0.0))
        rank_score = abs(dlt) * (0.5 + abs(ctv)) * (0.5 + track)
        candidates.append(
            {
                "feat_id": int(j),
                "mean_act_A": float(mean_a[j].item()),
                "mean_act_B": float(mean_b[j].item()),
                "delta_A_minus_B": dlt,
                "cos_to_v": ctv,
                "abs_cos_to_v": abs(ctv),
                "corr_with_X": r_x,
                "corr_with_margin": r_m,
                "rank_score": rank_score,
            }
        )

    candidates.sort(key=lambda r: r["rank_score"], reverse=True)
    top = candidates[:top_report]

    # Also keep a low-|cos| active feature pool for random control
    low_align = sorted(
        [c for c in candidates if c["abs_cos_to_v"] < 0.02],
        key=lambda r: abs(r["delta_A_minus_B"]),
    )
    random_pool = [c["feat_id"] for c in low_align[:200]]

    payload = {
        "kind": "n2s_sae_pilot_scores_v1",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "quest": (
            "Decompose frozen L20 v=unit(mean_A−mean_B); find sparse features that "
            "distinguish A/B, align with v, track X/margin — then causal-test top-k. "
            "No English labels."
        ),
        "model_id": FAIL_MODEL,
        "sae": sae_meta,
        "class_a": class_a,
        "class_b": class_b,
        "hook_sanity": sanity,
        "n_candidates_scored": len(candidates),
        "top": top,
        "random_control_pool": random_pool,
        "rank_formula": "|Δ(A−B)| * (0.5+|cos_to_v|) * (0.5+max(|r_X|,|r_margin|))",
        "success_criteria": (
            "Feature or small group: distinguishes A/B, aligns with v, tracks X/margin, "
            "causally moves X on temporal FOLFOX without destroying true-contra control."
        ),
    }
    SCORE_PATH.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"[sae-pilot] wrote {SCORE_PATH}")
    print("[sae-pilot] top 10:")
    for row in top[:10]:
        print(
            f"  feat={row['feat_id']:<6} Δ={row['delta_A_minus_B']:+.4f} "
            f"cos_v={row['cos_to_v']:+.4f} rX={row['corr_with_X']} "
            f"rM={row['corr_with_margin']} score={row['rank_score']:.5f}"
        )
    return payload


def _summarize_arm(row: dict[str, Any], label: str) -> dict[str, Any]:
    return {
        "label": label,
        "repair_final_x": row.get("repair_final_x"),
        "margin": row.get("logit_margin_true_minus_false"),
        "verdict": row.get("verdict"),
        "commit_value": row.get("commit_value"),
        "parse_ok": row.get("parse_ok"),
    }


def _pick_random_feat(pool: list[int], exclude: set[int]) -> int:
    g = torch.Generator()
    g.manual_seed(RANDOM_FEATURE_SEED)
    cand = [i for i in pool if i not in exclude]
    if not cand:
        # fallback: any mid-range id
        return 12345
    idx = int(torch.randint(0, len(cand), (1,), generator=g).item())
    return int(cand[idx])


def run_causal(
    *,
    top_k: int = 5,
    alpha: float = 8.0,
    group: bool = True,
    model_id: str = FAIL_MODEL,
    note_ids: tuple[str, ...] | list[str] | None = None,
    notes_override: dict[str, dict[str, str]] | None = None,
) -> dict[str, Any]:
    if not SCORE_PATH.exists():
        raise FileNotFoundError("Run `score` first (missing n2s-sae-pilot-scores.json).")

    free = cuda_free_gb()
    if free is not None and free < MIN_FREE_VRAM_GB:
        return {
            "ok": False,
            "status": "deferred_gpu_busy",
            "free_vram_gb": round(free, 2),
            "need_vram_gb": MIN_FREE_VRAM_GB,
            "message": f"Need ~{MIN_FREE_VRAM_GB:.0f} GiB free; have {free:.1f}.",
        }

    scores = json.loads(SCORE_PATH.read_text(encoding="utf-8"))
    if not (scores.get("hook_sanity") or {}).get("pass_heuristic"):
        print("[sae-pilot] WARNING: hook_sanity pass_heuristic=false — proceeding anyway for thin pilot.")

    top = list(scores.get("top") or [])[:top_k]
    if not top:
        raise RuntimeError("No scored candidates.")
    pool = list(scores.get("random_control_pool") or [])
    feat_ids = [int(r["feat_id"]) for r in top]
    rand_id = _pick_random_feat(pool, set(feat_ids))
    notes = notes_override if notes_override else causal_notes(note_ids)

    print("[sae-pilot] loading SAE for decoder directions…")
    sae, sae_meta = JumpReLUSae.from_hub()

    # Precompute unit decoder dirs
    dirs = {fid: sae.decoder_unit(fid) for fid in feat_ids + [rand_id]}
    # Group direction: unit mean of top-k decoder units (signed by cos_to_v so + points with v)
    group_vec = None
    if group and len(feat_ids) > 1:
        acc = torch.zeros(sae.d_in, dtype=torch.float32)
        for row in top:
            fid = int(row["feat_id"])
            sign = 1.0 if float(row.get("cos_to_v") or 0.0) >= 0 else -1.0
            acc = acc + sign * dirs[fid]
        group_vec = _unit(acc)

    results: dict[str, Any] = {
        "ok": True,
        "kind": "n2s_sae_pilot_causal_v1",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "model_id": model_id,
        "steer_layer": STEER_LAYER,
        "alpha": float(alpha),
        "top_k": top_k,
        "candidates": top,
        "random_feat_id": rand_id,
        "sae": {"repo": sae_meta["repo"], "subdir": sae_meta["subdir"]},
        "note_ids": list(notes.keys()),
        "notes": {},
    }

    try:
        for note_key, note in notes.items():
            case = {
                "id": note["id"],
                "evidence": note["evidence"],
                "expected": note["gold"],
            }
            print(f"\n=== SAE causal · {note_key} · baseline ===")
            base = _repair_run(
                case,
                model_id=model_id,
                intervention="none",
                layer_indices=(STEER_LAYER,),
            )
            note_arms: list[dict[str, Any]] = [_summarize_arm(base, "baseline")]

            for row in top:
                fid = int(row["feat_id"])
                vec = dirs[fid]
                # Positive steer along decoder (feature "on" direction)
                print(f"=== SAE causal · {note_key} · feat {fid} +α ===")
                pos = _repair_run(
                    case,
                    model_id=model_id,
                    intervention="activation_steer",
                    vectors_by_layer={STEER_LAYER: vec},
                    alpha=float(alpha),
                    layer_indices=(STEER_LAYER,),
                )
                note_arms.append(_summarize_arm(pos, f"feat_{fid}_pos"))

                print(f"=== SAE causal · {note_key} · feat {fid} −α (sign) ===")
                neg = _repair_run(
                    case,
                    model_id=model_id,
                    intervention="activation_steer",
                    vectors_by_layer={STEER_LAYER: -vec},
                    alpha=float(alpha),
                    layer_indices=(STEER_LAYER,),
                )
                note_arms.append(_summarize_arm(neg, f"feat_{fid}_neg"))

            print(f"=== SAE causal · {note_key} · random feat {rand_id} +α ===")
            rnd = _repair_run(
                case,
                model_id=model_id,
                intervention="activation_steer",
                vectors_by_layer={STEER_LAYER: dirs[rand_id]},
                alpha=float(alpha),
                layer_indices=(STEER_LAYER,),
            )
            note_arms.append(_summarize_arm(rnd, f"random_feat_{rand_id}_pos"))

            if group_vec is not None:
                print(f"=== SAE causal · {note_key} · top-{top_k} group +α ===")
                gpos = _repair_run(
                    case,
                    model_id=model_id,
                    intervention="activation_steer",
                    vectors_by_layer={STEER_LAYER: group_vec},
                    alpha=float(alpha),
                    layer_indices=(STEER_LAYER,),
                )
                note_arms.append(_summarize_arm(gpos, f"top{top_k}_group_pos"))
                print(f"=== SAE causal · {note_key} · top-{top_k} group −α ===")
                gneg = _repair_run(
                    case,
                    model_id=model_id,
                    intervention="activation_steer",
                    vectors_by_layer={STEER_LAYER: -group_vec},
                    alpha=float(alpha),
                    layer_indices=(STEER_LAYER,),
                )
                note_arms.append(_summarize_arm(gneg, f"top{top_k}_group_neg"))

            by = {a["label"]: a for a in note_arms}
            results["notes"][note_key] = {
                "gold": note["gold"],
                "arms": note_arms,
                "baseline_x": by["baseline"].get("repair_final_x"),
                "baseline_margin": by["baseline"].get("margin"),
            }
    finally:
        try:
            unload_model()
        except Exception:  # noqa: BLE001
            pass

    # Compact success readout (first note = temporal/false-X target; second = contra)
    note_keys = list(results["notes"].keys())
    temporal = results["notes"].get(note_keys[0]) if note_keys else {}
    contra = results["notes"].get(note_keys[1]) if len(note_keys) > 1 else {}
    results["readout"] = {
        "temporal_note": note_keys[0] if note_keys else None,
        "contra_note": note_keys[1] if len(note_keys) > 1 else None,
        "temporal_baseline_x": (temporal or {}).get("baseline_x"),
        "contra_baseline_x": (contra or {}).get("baseline_x"),
        "note": (
            "Success if some feat/group moves temporal X true→false (or margin down) "
            "while contra stays X=true; random control should not mimic the effect."
        ),
    }

    CAUSAL_PATH.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    print(f"\n[sae-pilot] wrote {CAUSAL_PATH}")
    _print_causal_table(results)
    return results


def run_workbench_sae(
    *,
    note: str,
    case_id: str = "LIVE",
    gold: str = "SATISFIED",
    top_k: int = 3,
    alpha: float = 8.0,
    include_contra_control: bool = True,
    refresh_scores: bool = False,
    model_id: str = FAIL_MODEL,
) -> dict[str, Any]:
    """Workbench entry: ensure ranked scores, then causal on pasted note (+ EX_CONTRA)."""
    if refresh_scores or not SCORE_PATH.exists():
        print("[sae-pilot] workbench: running score phase…")
        run_score(top_report=max(30, top_k))

    notes: dict[str, dict[str, str]] = {
        case_id: {
            "id": case_id,
            "gold": gold if gold and gold != "UNKNOWN" else "SATISFIED",
            "evidence": note,
        }
    }
    if include_contra_control:
        notes["EX_CONTRA"] = CAUSAL_NOTES["EX_CONTRA"]

    return run_causal(
        top_k=top_k,
        alpha=alpha,
        group=True,
        model_id=model_id,
        notes_override=notes,
    )


def _print_causal_table(results: dict[str, Any]) -> None:
    print("\n=== CAUSAL SUMMARY ===")
    for note_key, block in (results.get("notes") or {}).items():
        print(f"\n{note_key}  gold={block.get('gold')}  baseline_X={block.get('baseline_x')}")
        print(f"{'arm':<28} {'X':<8} {'margin'}")
        print("-" * 50)
        for a in block.get("arms") or []:
            print(
                f"{a.get('label'):<28} {str(a.get('repair_final_x')):<8} {a.get('margin')}"
            )


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description="Thin N2S SAE pilot (Chanin L20)")
    sub = p.add_subparsers(dest="cmd", required=True)

    sp = sub.add_parser("score", help="Encode A/B, rank features (CPU)")
    sp.add_argument("--top-report", type=int, default=30)

    sp = sub.add_parser("causal", help="Intervene top-k on FOLFOX pair (GPU)")
    sp.add_argument("--top-k", type=int, default=5)
    sp.add_argument("--alpha", type=float, default=8.0)
    sp.add_argument("--no-group", action="store_true")
    sp.add_argument("--model", default=FAIL_MODEL)
    sp.add_argument(
        "--notes",
        default="TX_E01,TX_C01",
        help="Comma ids: expand false-X + contra (default TX_E01,TX_C01)",
    )

    sp = sub.add_parser("all", help="score then causal")
    sp.add_argument("--top-k", type=int, default=5)
    sp.add_argument("--top-report", type=int, default=30)
    sp.add_argument("--alpha", type=float, default=8.0)
    sp.add_argument("--no-group", action="store_true")
    sp.add_argument("--model", default=FAIL_MODEL)
    sp.add_argument("--notes", default="TX_E01,TX_C01")

    args = p.parse_args(argv)

    def _note_ids() -> list[str]:
        return [x.strip() for x in str(getattr(args, "notes", "") or "").split(",") if x.strip()]

    if args.cmd == "score":
        run_score(top_report=args.top_report)
    elif args.cmd == "causal":
        run_causal(
            top_k=args.top_k,
            alpha=args.alpha,
            group=not args.no_group,
            model_id=args.model,
            note_ids=_note_ids(),
        )
    elif args.cmd == "all":
        run_score(top_report=args.top_report)
        run_causal(
            top_k=args.top_k,
            alpha=args.alpha,
            group=not args.no_group,
            model_id=args.model,
            note_ids=_note_ids(),
        )


if __name__ == "__main__":
    main()
