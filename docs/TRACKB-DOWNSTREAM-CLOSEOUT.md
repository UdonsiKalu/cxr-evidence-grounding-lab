# Downstream closeout — utilization + neural→symbolic boundary

**Status:** **ROUNDED UP 2026-09-06**  
**Scope:** Experimental **downstream** (representation utilization / generation) + **external** neural→symbolic boundary.  
**Not closed:** **Upstream** (representation formation on prefill / cue tokens) — next build.  
**Sealed:** `temporal-family-test.json` — still not for fitting.  
**G3:** untouched.

**Canonical system map:** [RESEARCH-APPROACH.md](./RESEARCH-APPROACH.md) § System characterization.  
**α=8 claim:** [TRACKB-ALPHA8-FREEZE.md](./TRACKB-ALPHA8-FREEZE.md) (unchanged).  
**SAE pilot:** [TRACKB-SAE-PILOT.md](./TRACKB-SAE-PILOT.md).

---

## Three pieces (locked)

```text
Doctor's note
      ↓  UPSTREAM — representation formation / evolution (NEXT)
Tokenization → layer-wise compute → internal reps
      ↓  DOWNSTREAM — utilization / readout (THIS CLOSEOUT)
Generation → structured LLM output (e.g. contradiction.present)
════════════════════════════
   NEURAL → SYMBOLIC BOUNDARY  (Track A)
════════════════════════════
CXR atoms / Dual / gates → AUTO | REVIEW
```

Upstream vs downstream is an **experimental partition** (formation → utilization), not a hard layer cut inside the transformer. The neural→symbolic boundary **is** a hard cut: LLM structured output leaves the neural model and enters CXR symbolic machinery.

---

## What downstream shipped

| Piece | Role | Where |
|-------|------|--------|
| Evaluate | Track A extract → ground → Dual → mismatch → gates | `:8257` / `eval_api.evaluate_fast` |
| Deep dive | Observe L20 temporal vs contradiction readout | `run_forensics_live` |
| Intervene α | Steer `h ← h + α·v` @ L20 commit; baseline + controls | `run_intervene_live` |
| SAE features | Chanin L20 decompose **v**; top-k steers + EX_CONTRA control | `run_workbench_sae` |
| Docs / demos | Newcomer tour PDF; CLI walkthrough with **implement sketches** | `cxr-n2s-eval-workbench/docs/` |

Stack: HF `transformers` + forward hooks (not TransformerLens rewrite). Prefer `faiss_gpu1` for GPU phases.

---

## Claims (allowed)

**Track A (boundary):** uncertain transforms → **REVIEW**; Dual / N2S mismatch (neural X vs grounded X) are containment, not product failure. Temporal change ≠ contradiction in grounding.

**Track B (downstream utilization):**

- Frozen expand **v** @ L20, **α=8**: **partial** editor — selective flip on some temporal false-X; **no flip** on true contradiction controls; beats gaussian / reverse in the freeze panel.
- SAE: ranked sparse features vs **v** + causal top-k arms — **pilot**, not English-labeled “temporality neuron.”
- Dull demos when 7B baseline already X=false are **valid** (nothing dramatic to flip).

## Claims (forbidden)

- Steering fixed healthcare / all temporal false-X.  
- α=16/32 is the freeze.  
- 14B MI editor transfer (behavioral only; cluster mostly non-persist).  
- SAE ranking = discovered named clinical feature.  
- Upstream formation already characterized (it is **not**).  
- Sealed test used for fitting.

---

## Freeze decision

| Workstream | Status |
|------------|--------|
| Downstream MI buttons + α=8 + SAE pilot | **Stop expanding** unless a new fidelity question requires it |
| Track A Dual / grounding / gates | Keep as fidelity judge; no G3 redesign from one case |
| Upstream prefill (cue × layer) | **Next** — thin `prefill_trace`; reuse **v**/SAE/hooks; measure against commit X + Dual |
| Circuit C0 scaffold | Optional later; not blocking upstream start |

---

## Next (upstream)

| Step | Status |
|------|--------|
| U0 prefill trace | **Done** |
| U0 readout | **Done** |
| U1 patch cue → commit X | **Done (null on FOLFOX pair @ α=8)** — no flip |
| Further upstream squeeze | **Stop** unless new site hypothesis |

Do **not** adopt external thought-tracing / ACDC / MLSAE stacks as the product; borrow ideas only.
