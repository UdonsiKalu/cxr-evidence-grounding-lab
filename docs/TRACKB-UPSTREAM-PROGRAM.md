# Upstream program — after U0/U1 null (intent freeze)

**Status:** program locked 2026-09-06 · **U-A done · U-B null · U-B2 null → pause** · `:8258` demos OK · **portfolio package frozen** — **leave this branch alone**  
**Portfolio write-up:** [TRACKB-UPSTREAM-PORTFOLIO.md](./TRACKB-UPSTREAM-PORTFOLIO.md)  
**Does not reopen:** α-chase on freeze-`v`, U1/U1b cue×layer squeeze, sealed-test redesign, **U-C SAE / U-D circuits / live ablate thrash** from null U-B/U-B2  
**Prior null:** [TRACKB-UPSTREAM-PREFILL.md](./TRACKB-UPSTREAM-PREFILL.md)

---

## Two meanings of “complete”

| Question | Status |
|----------|--------|
| Does freeze commit-`v` sit at obvious cue sites and causally drive X? | **Done. Null. Stop.** (U0/U1/U1b) |
| How does the model form temporal-change vs contradiction reps in prefill? | **Not done** — this program |

Null kills the *v × cue × layer* editor story. It does **not** characterize upstream computation.

---

## North-star question

> Where and how does prefill build a representation that distinguishes *sequenced temporal-change* from *same-time contradiction* — and which components are causally necessary for that distinction (and for commit X)?

---

## Phases (one at a time; exit before next)

| Phase | Name | What | Exit criterion |
|-------|------|------|----------------|
| **U-A** | Multi-token map | Note-body residuals × layers; class mean Δ direction; top sites by proj — **not** hand-picked cues first | Site map + one formation hypothesis sentence |
| **U-B** | Component patch | Activation/path patch at U-A sites; attn vs MLP | Shortlist of (layer, component, token) that move margin/X |
| **U-C** | Upstream SAE | Encode at U-A sites; class-separating features; clamp check | Few features with causal evidence **or** clear sparse-null |
| **U-D** | Circuit sketch | Only if U-B/C bite; token→head→token | Minimal pathway figure + controls |

**Do not** run U-B/C/D until the prior exit is written.  
**No live upstream ablate/SAE UI** — JSON + CLI first; thin **read-only** viewer on `:8257` (`GET /api/upstream`) loads frozen U-A/U-B/U-B2 artifacts only.

---

## Hard rules

- Stack: HF `transformers` + forward hooks (+ existing Chanin SAE). No TL dependency required.  
- Track A Dual remains the fidelity judge.  
- Freeze-`v` may be scored as a **secondary** alignment check — never the primary search target.  
- No α=16/32 reopen. No sealed-test redesign from upstream.

---

**U-A first live (2026-09-06):** strongest mean separation **L24**; freeze-v alignment **cos≈0.03** (formation ≠ commit editor). Top temporal sites ≈ imaging/lesions/metastatic fragments; contra ≈ Disease/failed/remains. Artifacts: `n2s-upstream-ua-map.json` · `n2s-upstream-ua-readout.json`. **Exit met.**

**U-B first live (2026-09-06):** zero-ablate attn / mlp / resid at L24 top-5 sites on extract prefill → commit X/margin. **Null** — no X flip; |Δmargin|≤0.5. Artifacts: `n2s-upstream-ub-patch.json` · `n2s-upstream-ub-readout.json`.

**U-B2 refine (2026-09-06):** mean-ablate attn / mlp / resid at **L20** U-A top-5. **Null again** — no X flip; |Δmargin|<1. Artifacts: `n2s-upstream-ub2-patch.json` · `n2s-upstream-ub2-readout.json`.

**Decision:** **Pause** this ablation family before SAE/circuits. U-A map stands as correlational formation evidence; causal lever at these top sites is not established under zero@L24 or mean@L20.

**Viewer / live Upstream (2026-09-06):** dedicated workbench **`:8258`** only (`upstream_server.py`) — frozen U-A/U-B replay + paste-note score vs frozen U-A `d` + optional custom pair. Module: `n2s_lab/n2s_upstream_live.py`. **Not** on `:8257` (Track A / Intervene / SAE only). Not ablation / SAE / α·v.

**Operator demos (2026-09-06):** Score EX_TEMPORAL_FOLFOX vs frozen `d` (best L24 `|μ·d|≈71`); Score EX_CONTRA vs frozen `d` (best L24 `|μ·d|≈18.5`; top `Disease`/`remains`); Map custom pair short progression vs no-new-lesions (new `d`, L24 `||Δμ||≈25.6`). Correlational only.

**Hard stop (locked):** do **not** open U-C SAE, U-D circuits, or live ablate thrash from U-B/U-B2 null. **Portfolio package shipped** (`TRACKB-UPSTREAM-PORTFOLIO.md`) — leave this ablation-family branch alone. **New Q (open):** held-out paraphrase generalization — [TRACKB-UPSTREAM-UA-GENERALIZE.md](./TRACKB-UPSTREAM-UA-GENERALIZE.md).

## U-A run

```bash
cd cxr-evidence-grounding-lab
./scripts/run_upstream_ua.sh map
./scripts/run_upstream_ua.sh readout
```

## U-B / U-B2 run

```bash
./scripts/run_upstream_ub.sh patch
./scripts/run_upstream_ub.sh patch --variant ub2 --layer 20 --mode mean
```

Artifacts: `n2s-upstream-ua-*.json` · `n2s-upstream-ub-*.json` · `n2s-upstream-ub2-*.json`

## Claim hygiene

**Say:** multi-token formation map; class-separating residual direction at layer L; component ablation null at those sites (zero@L24, mean@L20).  
**Do not say:** upstream circuit found; freeze-`v` is the temporality feature; U0/U1 null means upstream is fully characterized.
