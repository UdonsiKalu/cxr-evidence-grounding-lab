# n2n-lost → Dual REVIEW (Track A contain)

**Status:** wired 2026-09-10 · optional gate · **not** a G3 change  
**Sibling:** [TRACKB-NN-LAYER-LOSS.md](./TRACKB-NN-LAYER-LOSS.md) · [TRACKB-NN-LAYER-PATCH.md](./TRACKB-NN-LAYER-PATCH.md) · [AUTO-CONTRACT.md](./AUTO-CONTRACT.md)

When a **layer trace exists** and class-correct `mean_score_d` has **≥1 lost step**, Dual **REVIEW**.  
No trace → gate does not fire. Does **not** rescore frozen Phase-7 Dual_full.

```text
if has_trace and n_lost_steps ≥ 1:
    Dual → REVIEW  (reason: n2n_layer_loss)
else:
    leave Dual as Dual (G1–G3 unchanged)
```

Frozen pair: CONTRA (2 lost) → REVIEW. FOLFOX (0 lost) → gate pass.

GUI: http://127.0.0.1:8259/  
Artifact: `artifacts/n2s-nn-lost-review-panel.json`

**Say:** lost n2n steps contain as REVIEW when we measured them.  
**Do not say:** G3 changed; Dual_full 32B rescore; inside-net editor.
