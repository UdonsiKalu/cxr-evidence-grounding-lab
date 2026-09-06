# Upstream portfolio package — formation ≠ control

**Status:** frozen write-up 2026-09-06 · science branch **left alone**  
**Audience:** portfolio / faculty / visitor one-pager  
**Detail:** [TRACKB-UPSTREAM-PROGRAM.md](./TRACKB-UPSTREAM-PROGRAM.md) · [N2S-SYSTEM-CLAIM.md](./N2S-SYSTEM-CLAIM.md) · [TRACKB-UPSTREAM-PREFILL.md](./TRACKB-UPSTREAM-PREFILL.md)  
**Live demos:** workbench `:8258` (score vs frozen `d` · optional custom pair) — correlational only

---

## 1. Summary (one page)

| Piece | What we did | Outcome |
|-------|-------------|---------|
| **U-A** | Multi-token class-mean Δ map on temporal vs contradiction notes (prefill residuals × layers) | Formation map; **strongest separation at L24**; cos(d, freeze-v)≈**0.03** |
| **U-B** | Zero-ablate attn / mlp / resid at L24 top sites → commit X / margin | **Null** — no X flip |
| **U-B2** | Mean-ablate at L20 U-A top sites | **Null again** — no X flip |
| **`:8258` demos** | Score FOLFOX temporal + EX_CONTRA vs frozen U-A `d`; Map custom pair | Correlational only (L24 `|μ·d|` strong on temporal; weaker on contra / custom) |

**Conclusion:** we **localized** where a treatment-course vs contradiction distinction becomes visible in residual space. Ablating those strongest sites did **not** control the model’s contradiction commit. **Localized representation ≠ causal control.**

### Claim hygiene

| Say | Do not say |
|-----|------------|
| U-A formation map; L24 strongest correlational sep | “Found the temporality circuit / neuron” |
| U-B / U-B2 ablation **null** at tested sites | Upstream editor / causal prefill lever |
| `:8258` scores are correlational demos | Live ablate / SAE / circuit from these nulls |
| Localization alone is insufficient for causality | Jump to U-C SAE / U-D from null |

---

## 2. Diagram

```text
  Clinical note (temporal-change vs same-time contradiction)
           │
           ▼
  Prefill — representation forms across layers
           │
           ▼
  U-A map: class-mean Δ direction d = unit(μ_T − μ_C)
           │
           ├─► strongest correlation @ L24
           │     (top sites ≈ imaging / lesions vs Disease / failed …)
           │
           ▼
  U-B / U-B2: ablate top sites (zero@L24 · mean@L20)
           │
           ▼
  Commit X / margin  ──►  no flip (null)
           │
           ▼
  Claim stays limited:
  correlation at late layers  ≠  causal control of the decision
```

Workbench `:8258` sits **beside** this story: paste a note → score vs frozen `d` (or map a new pair). That is **readout / demo**, not a causal editor.

---

## 3. Plain English (portfolio line)

> I traced where a treatment-failure representation becomes visible inside a transformer, then tested whether the strongest locations actually controlled the model’s decision. They did not. This showed that localization alone was insufficient to establish causality.

Shorter variant:

> Late-layer residuals separate temporal-change from contradiction notes, but ablating those sites did not move the contradiction commit — so the map is correlational, not a causal editor.

---

## 4. Leave this branch alone

**Locked — do not reopen from this null:**

- U-C upstream SAE  
- U-D circuit sketch  
- More L20 / L24 ablation thrash  
- α-chase on freeze-`v` × cue  
- Sealed-test redesign from upstream

**Next technical project (later, new question — not more of this ablation family):**

> Can a representation found on one set of clinical examples generalize to unseen paraphrases and different lexical forms?

**Opened:** [TRACKB-UPSTREAM-UA-GENERALIZE.md](./TRACKB-UPSTREAM-UA-GENERALIZE.md) · `./scripts/run_upstream_ua_gen.sh panel`  
Story shape: **representation discovery → held-out generalization → causal test** (causal only if gen bites; still not old U-B thrash).
---

## Pointers

| What | Where |
|------|--------|
| Program + hard stop | `TRACKB-UPSTREAM-PROGRAM.md` |
| System claim snapshot | `N2S-SYSTEM-CLAIM.md` |
| U0/U1 null prefill | `TRACKB-UPSTREAM-PREFILL.md` |
| Lab runners | `scripts/run_upstream_ua.sh` · `run_upstream_ub.sh` |
| Live UI | `cxr-n2s-eval-workbench` port **8258** |
| Artifacts | `n2s-upstream-ua-*.json` · `n2s-upstream-ub*.json` · `n2s-upstream-live-runs/` |
