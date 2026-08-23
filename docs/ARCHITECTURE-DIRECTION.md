# Architecture direction — toward safe coverage (not universal labels)

**Status:** approved direction (2026-08-23). Pilot lab evidence only; not a product claim.  
**Lab:** `cxr-evidence-grounding-lab/` · **Next build:** [PHASE5-PROTOCOL.md](./PHASE5-PROTOCOL.md)

This note exists so a future session can resume without re-deriving the thread.

---

## North star

> The model **proposes**; a **finite semantic contract** + **verify/abstain** decides;  
> anything the system cannot safely represent becomes **REVIEW**, not a confident wrong answer.

We aim for **safe handling of most cases** (correct auto-decision **or** justified REVIEW), not 100% automatic labels on every clinical note.

---

## What the pilot showed (soft claim)

Do **not** say the lab “proved” that the contract beats the LLM in general.

Defensible statement:

> Our pilot results show that explicitly preserving contradiction and uncertainty can prevent semantic states observed in analysis from being lost before symbolic evaluation.

Study size is small (synthetic notes, one predicate, local models). Stronger claims wait for more evidence.

---

## Layers

| Layer | Role | Status |
|-------|------|--------|
| **L1** | Extract → first-class states (`contradiction`, `uncertainty`) → atoms → **fixed rule** | Done (M2/M3); frozen A–D Phases 1–4 |
| **L2** | **Verify** formalization against analysis/source; regenerate **once** if clash | Phase-5 |
| **L3** | If still unverified → **REVIEW** (not UNCERTAIN) | Phase-5 |
| **L4** | Grow ontology / predicates / temporal structure carefully | Later |
| **L5** | CXR: Archetypes + Qdrant (B2) + fusion; same propose/verify/REVIEW pattern | Later — **not** until L2/L3 helps in lab |

---

## UNCERTAIN vs REVIEW (do not conflate)

| Label | Meaning |
|-------|---------|
| **UNCERTAIN** | Clinical / evidence uncertainty (incomplete, hedged, pending) — rule may fire this from unknown atoms / uncertainty field |
| **REVIEW** | Formalization **cannot be verified** against what analysis/source said — trust failure, not “the evidence is uncertain” |

Example:

- “Possible progression; scan pending.” → often **UNCERTAIN** (real uncertainty).  
- Analysis says contradiction; JSON has `contradiction.present=false` after repair → **REVIEW**.

Forcing UNCERTAIN after verify-fail invents clinical meaning. **Forbidden.**

---

## Decision flow (V1 research architecture)

```text
                  NOTE
                    ↓
             LLM INTERPRETATION
                    ↓
           SEMANTIC CONTRACT
                    ↓
              FORMALIZATION
                    ↓
                VERIFY
              ↙        ↘
         VERIFIED     FAILED
             ↓           ↓
       SYMBOLIC RULE   regenerate once
                         ↓
                       VERIFY
                      ↙      ↘
                   PASS      FAIL
                    ↓         ↓
                 RULE       REVIEW
```

Later (not Phase-5):

- **Archetypes** — what kind of reasoning problem  
- **Qdrant** — which evidence is relevant (boundary B2)  
- **Contract** — how evidence is represented  
- **Verification** — whether representation still matches  
- **Symbolic rules** — what follows from verified evidence  
- **REVIEW** — what the system cannot safely determine  

---

## Metrics (two dimensions — do not collapse)

Do **not** optimize a single “useful rate” that rewards REVIEW-everything.

1. **Safety** — among cases with an automatic decision (not REVIEW), how often is the verdict correct vs gold?  
2. **Coverage** — what fraction of cases receive an automatic decision (not REVIEW)?  
3. **Abstention quality** (secondary) — when REVIEW, was review warranted?

C4-class (temporal/identity): V1 may **REVIEW** without “solving” it. Later structure can reclaim that class.

---

## Hard scope freeze until Phase-5 reports

**In scope:** L2 verification + L3 REVIEW in the isolated lab; frozen A–D prompts; existing held-out D-loss slices.

**Out of scope for now:** Qdrant, Archetypes, CXR production, C4 field hacks, new predicates, B′ iteration, minting a first-class artefact per failure mode.

If L2/L3 does not help on the controlled experiment, **stop** and learn that before building CXR around it.

---

## Related artifacts

| Doc | Role |
|-----|------|
| [PHASE5-PROTOCOL.md](./PHASE5-PROTOCOL.md) | Experiment freeze before live run |
| [PHASE4-PROTOCOL.md](./PHASE4-PROTOCOL.md) | Prior held-out preservation study |
| `notes/progress-notes.pdf` | Full frozen M1–M4 record |
| `notes/progress-notes-simple.pdf` | Newcomer guide + demo |
| Public demo | https://udonsikalu.github.io/cxr-evidence-grounding-lab/ |
