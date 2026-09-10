# Neural-to-neural first-lost-layer patch (pilot)

**Status:** **CLOSED** 2026-09-10 (user: furthest). L12-only **NULL** · L12+L16 α=2 **PARTIAL** · L12+L16+L20 α=2 **same PARTIAL** (gap≈9 vs α=2). Auto = Dual REVIEW. Not α-chase.  
**Detect:** [TRACKB-NN-LAYER-LOSS.md](./TRACKB-NN-LAYER-LOSS.md) (`:8259`)  
**Not this:** α=8 expand-`v` Intervene · U-A causal @ L24-only · U-C SAE · U-D · sealed-test redesign

Lab-scale only. Mirror Dual’s shape: **detect → act at the fail site → check a control.**

---

## Question

> If class-correct `mean_score_d` is first lost on a step L→L′, does a **small**
> prefill push along frozen U-A `d` at **L** stop the fall on later layers
> **without** Dual AUTO-ing a wrong answer?

Null / control-fail / REVIEW-fallback are allowed honest outcomes.

---

## Design

| Piece | Choice |
|-------|--------|
| **Detect** | Frozen U-A traces; first lost step (τ≥5 or sign flip). Example: CONTRA **L12→L16**. |
| **Act** | Prefill note-body `h ← h + sign·α·d` at the **earlier** layer (L12). |
| **Vector** | Frozen per-layer `d` from `n2s-upstream-ua-directions.pt` (do not rebuild). |
| **Sign** | Temporal: **+α·d** (keep score up). Contradiction: **−α·d** (keep score down/negative). |
| **α** | **1 and 2 only** (not 8). Stacks: L12 · L12+L16 · residual **L12+L16+L20**. |
| **Re-read** | Later listed layers’ `mean_score_d` + 7B extract yes/no (`contradiction.present` + rule). |
| **Notes** | `EX_TEMPORAL_FOLFOX` + `EX_CONTRA` (true-contra control). |
| **Model** | HF `Qwen/Qwen2.5-7B-Instruct` |

### Dual analog (this panel)

Patched 7B extract → `ground` → `evaluate_rule`.  
**If Dual would AUTO a wrong answer** that the baseline did not (disposition AUTO and verdict ≠ gold, *introduced* by the patch) → **fail the patch**, **fallback REVIEW**.  
True-contradiction must not flip to “all good” (`extract_x=false` or AUTO SATISFIED).  
7B extract+rule is Dual-**shaped**, not Dual_full 32B. A pre-existing 7B Dual analog error (e.g. FOLFOX `NOT_SATISFIED` at α=0) is **not** a patch fail.

This is Dual-**shaped** (detect / act / control + AUTO-wrong → REVIEW). It is **not** Ollama Dual_full 32B Path C+D.

### Soft gates

| Gate | Pass if |
|------|---------|
| **Rescue** | CONTRA’s first lost step is no longer lost after the patch |
| **Control** | CONTRA does not AUTO-wrong / does not lose `extract_x=true` |
| **Temporal** | FOLFOX is not Dual AUTO-wrong after the same-site +α·d push |

Control fail beats rescue: wrecked contra → **FAIL**, fallback REVIEW even if scores look better.

---

## Run

```bash
cd cxr-evidence-grounding-lab
./scripts/run_nn_layer_patch.sh detect       # CPU, frozen traces
./scripts/run_nn_layer_patch.sh panel        # GPU L12-only (first live NULL)
./scripts/run_nn_layer_patch.sh panel-stack  # GPU L12+L16
./scripts/run_nn_layer_patch.sh panel-residual  # GPU L12+L16+L20 (remaining L16→L20)
```

Artifact: `artifacts/n2s-nn-layer-patch-panel.json` (do not overwrite with later stacks)  
Residual: `artifacts/n2s-nn-layer-patch-l12l16l20-panel.json`  
GUI: http://127.0.0.1:8259/ (detect is CPU; panel GPU)

---

## Claim hygiene

| Say | Do not say |
|-----|------------|
| First-lost-layer `d` patch did / did not rescue the step under Dual control | Inside-net REVIEW; temporality circuit; production editor |
| Distinct from L24 U-A causal (null) and from α=8 L20 `v` | Reopen U-C / U-D / α-chase |
| Pilot on two notes + α∈{1,2} | Family-wide n2n correction |
