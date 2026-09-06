# Upstream U-A generalize — held-out paraphrase / lexical forms

**Status:** **FROZEN 2026-09-06** — [TRACKB-UPSTREAM-UA-GEN-FREEZE.md](./TRACKB-UPSTREAM-UA-GEN-FREEZE.md) · first live panel soft+strong YES @ L24 · correlational only (not U-B/U-C/U-D reopen)  
**Portfolio story:** representation discovery → held-out generalization → (causal later, only if asked)  
**Prior close:** [TRACKB-UPSTREAM-PORTFOLIO.md](./TRACKB-UPSTREAM-PORTFOLIO.md) · [TRACKB-UPSTREAM-PROGRAM.md](./TRACKB-UPSTREAM-PROGRAM.md)

**First live (2026-09-06):** held-out mean_T≈58.4 > mean_C≈23.5 @ L24; min_T≈45.5 > max_C≈29.0; anchors 71.4 vs 9.4. Artifacts: `n2s-upstream-ua-gen-panel.json` · `n2s-upstream-ua-gen-readout.json`. **Freeze doc locked.**

---

## Question

> Can the frozen U-A formation direction `d = unit(μ_T − μ_C)`, built on the discovery pair EX_TEMPORAL_FOLFOX / EX_CONTRA, still **separate** temporal-change from same-time contradiction on **unseen paraphrases and different lexical forms**?

This is correlational generalization of the **map**, not a causal editor test.

---

## Design

| Piece | Rule |
|-------|------|
| **Discovery (frozen)** | Only EX_TEMPORAL_FOLFOX + EX_CONTRA bake `d` — **do not** add held-out text to `DEFAULT_NOTES` |
| **Held-out** | `data/heldout-ua-paraphrase.json` — paraphrases + alternate regimens/lexemes; same class labels |
| **Score** | Prefill note-body residuals · `d` at layers 8/12/16/20/**24** (reuse live scorer) |
| **Headline metric** | At U-A best layer (usually **L24**): `mean_score_d` per note |
| **Soft gate** | Held-out temporal mean(`mean_score_d`) **>** held-out contra mean(`mean_score_d`); report also min_T vs max_C (stronger) |
| **Anchors** | Re-score discovery pair as sanity (expect clear sep) |
| **Not this** | Ablation, SAE, circuits, α·v, sealed-test redesign, rebuilding `d` from held-out |

### Roles in the JSON

| Role | Meaning |
|------|---------|
| `anchor` | Discovery notes (optional re-score) |
| `near_paraphrase` | Same clinical story, heavily reworded |
| `far_paraphrase` | Same distinction, different disease / regimen / surface forms |

---

## Run

```bash
cd cxr-evidence-grounding-lab
./scripts/run_upstream_ua_gen.sh panel
./scripts/run_upstream_ua_gen.sh readout
```

Needs ~14 GiB free GPU; prefer `../cxrlabs/faiss_gpu1/bin/python`.

Artifacts:

| File | Kind |
|------|------|
| `artifacts/n2s-upstream-ua-gen-panel.json` | per-note layer scores + soft gate |
| `artifacts/n2s-upstream-ua-gen-readout.json` | short human readout |

---

## Claim hygiene

| Say | Do not say |
|-----|------------|
| Frozen `d` did / did not separate held-out paraphrases at L24 | “Temporality feature proven” |
| Correlational lexical generalization of the formation map | Causal control / editor |
| Soft pilot on small n | Production clinical guarantee |
| Next causal step only if separation holds | Jump to U-C SAE from a null or weak gen |

---

## Exit

Write the panel + one sentence:

- **YES (soft):** held-out class means separate in the expected direction at best layer.  
- **NO / weak:** means overlap or reverse — map may be discovery-lexicon brittle; do **not** claim a general formation direction.

Then stop or open a **new** causal Q only with explicit go — still **not** more L20/L24 top-site ablation thrash from the old U-B null.
