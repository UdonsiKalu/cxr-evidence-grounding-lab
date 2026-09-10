# Neural-to-neural layer-loss — ground rules (v1)

**Status:** setup 2026-09-10 · correlational replay · **not** a causal editor  
**GUI:** http://127.0.0.1:8259/  
**Locator (first-break):** http://127.0.0.1:8260/ · [TRACKB-LOCATOR-V1.md](./TRACKB-LOCATOR-V1.md)  
**Sibling:** neural→symbolic lab **:8253** · encode Score **:8258** · Dual/Intervene **:8257**  
**Next (new, not proven):** first-lost-layer `d` patch — [TRACKB-NN-LAYER-PATCH.md](./TRACKB-NN-LAYER-PATCH.md)  
**Not this:** U-C SAE · U-D circuits · α-chase · sealed-test redesign · Shannon bit-loss

Lab-scale only. Frozen U-A direction `d = unit(μ_T − μ_C)` from the FOLFOX vs contradiction pair.

---

## Four rules (locked before the GUI)

| # | Rule | v1 choice |
|---|------|-----------|
| 1 | **What “the information” is** | Mean projection of **note-body** residuals onto frozen `d` (`mean_score_d`). **+** = temporal-change side. **−** = contradiction side. |
| 2 | **Where we read it** | Layers **{8, 12, 16, 20, 24}** only (same as U-A). Not every layer 1…N. |
| 3 | **What “lost” means** | On a step L→L′: **drop** if class-correct score falls by **τ ≥ 5.0**, or **flip** if the sign of `mean_score_d` changes. Temporal gold: class-correct = `mean_score_d`. Contradiction gold: class-correct = `−mean_score_d`. |
| 4 | **Control** | A contradiction note must stay **below** the temporal note’s `mean_score_d` at every listed layer (does not “look temporal throughout”). |

τ=5.0 is a lab threshold on this model’s score scale (temporal L24 ≈ 71). It is not a universal constant.

---

## What the GUI does

- Loads **coding-set** notes used on :8253 (`T1`, `T3`, `C1`, `C4`) plus **BC_E1** and the U-A pair **EX_TEMPORAL_FOLFOX** / **EX_CONTRA**.
- **Analyze** on notes that already have a frozen U-A / live Score trace (today: the U-A pair).
- Other coding notes show text + gold + “no layer trace yet” — same list as the N2S GUI, ready for a later live HF pass.

This is **not** “loss from layer 1 to 30.” It is a first contract + viewer for the compute surface.

---

## Claim hygiene

**Say:** on the frozen pair, we can plot `mean_score_d` across five layers and flag τ-drops / sign flips.  
**Do not say:** we measured information theory loss; we can auto-fix a middle layer; we found a temporality neuron.
