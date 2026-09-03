# Architecture direction — toward universal *safe* handling (not universal labels)

**Status:** approved direction (2026-08-23); protocol-universality framing frozen same day.  
**Pilot lab evidence only; not a product claim.**  
**Lab:** `cxr-evidence-grounding-lab/` · **Phase-5 (done):** [PHASE5-PROTOCOL.md](./PHASE5-PROTOCOL.md) · **Phase-6 (frozen; live pending):** [PHASE6-PROTOCOL.md](./PHASE6-PROTOCOL.md)

**Two-track research spine (2026-09-03):** [RESEARCH-APPROACH.md](./RESEARCH-APPROACH.md) — Track A (wrong AUTO / REVIEW gates) vs Track B (MI/RepEng causal ladder). Gates **contain** fidelity failures; they do not **explain** them.

This note exists so a future session can resume without re-deriving the thread.

---

## North star

> The model **proposes**; a **finite semantic contract** + **verify/abstain** decides;  
> anything the system cannot safely represent becomes **REVIEW**, not a confident wrong answer.

**Universal** means: *universally safe in how arbitrary notes are handled* — either a demonstrably faithful formalization (then decide) or **REVIEW**.  
It does **not** mean: correctly understanding every possible clinical note, or 100% automatic labels.

Universality comes from the **protocol**, not from enumerating every semantic state (`contradiction`, `uncertainty`, temporal logic, …).

---

## Technical ambition (careful wording)

Do **not** ask: “How do we make CXR universal?”

Ask:

> Can we define a **domain-general verification protocol** for neural-to-symbolic grounding such that arbitrary natural-language evidence is either translated into a **demonstrably faithful** symbolic representation or **explicitly rejected** as unverifiable?

If that works, it is larger than CXR. Healthcare sits above it:

```text
          GENERAL N→S ASSURANCE
                  │
       evidence → formalization
       preservation / verification
       abstention (REVIEW)
                  │
                  ↓
        HEALTHCARE ONTOLOGY
                  │
       clinical entities, events/time
       policies / evidence
                  │
                  ↓
                 CXR
                  │
       claims / Archetypes
       CMS / payer rules
       denial prevention
```

We are **nowhere near proving** domain-general N→S assurance. Phase-5 is a lab pilot of propose → verify → REVIEW on one predicate.

---

## Protocol universality (any note)

Imagine a situation we have never modeled. The system should still follow:

```text
Any clinical note
       ↓
Interpret evidence
       ↓
Construct formal representation
       ↓
Can we demonstrate that the representation
faithfully preserves the source evidence?
       ↓
     YES                    NO
      ↓                      ↓
Can applicable rules      REVIEW
be established?
      ↓
   YES    NO
    ↓      ↓
 DECIDE   REVIEW
```

The hard research sentence is: **“Can we demonstrate that the representation faithfully preserves the source evidence?”**

Phase-5 / Phase-6 checks (analysis↔structure, repair, round-trip, dual-path) are **imperfect experiments** toward that question. None alone *proves* faithfulness.

---

## Evidence-preservation object (stronger abstraction)

Prefer not to feed the symbolic engine a naked fact:

```text
first_line_failed = TRUE
```

Prefer a **claim accompanied by evidence and justification** (conceptual shape — not yet implemented):

```text
PROPOSITION:     first_line_failed
STATUS:          supported
SOURCE:          oncology_note_27
EVIDENCE:        "After four months of therapy, imaging demonstrated progression."
RELATIONSHIP:    progression_after_therapy → evidence_of_failure
QUALIFIERS:      certainty = definite; temporal_order = after; therapy_identity = matched
VERIFICATION:    source supports proposition = PASS
```

If a bizarre note appears tomorrow with a phenomenon we never categorized, the system need not invent a new first-class field first. It asks:

> Can I produce the required evidence / proof object for the proposition I am about to give the symbolic engine?

If not → **REVIEW**.

That is the more universal abstraction than minting a field per failure mode.

---

## Safety vs coverage (do not confuse)

| Lever | Improves | Mechanism |
|-------|----------|-----------|
| **Verification + abstention** (L2/L3) | **Safety** | Do not decide when the handoff cannot be demonstrated |
| **Ontology growth** (L4) + CXR stack (L5) | **Coverage** | More notes can establish verified facts → fewer REVIEW |

Approach “universality” by **shrinking REVIEW**, not by eliminating it.

```text
Early:     DECIDE ~55%   REVIEW ~45%
Later:     DECIDE ~80%   REVIEW ~20%
Ambition:  DECIDE ~90%+  REVIEW residual
```

Some cases are genuinely undecidable (“Possible progression; awaiting pathology”). The correct universal behavior is **not to decide** — often **UNCERTAIN** (clinical) or **REVIEW** (unverifiable formalization), never a manufactured pathology result.

---

## What the pilot showed (soft claim)

Do **not** say the lab “proved” that the contract beats the LLM in general.

Defensible statement:

> Our pilot results show that explicitly preserving contradiction and uncertainty can prevent semantic states observed in analysis from being lost before symbolic evaluation.

Phase-5 addendum (same soft register): on the Phase-4 held-out slice, verify → regenerate once → REVIEW cleared measured D `representation_loss` (fix or abstain) without forcing UNCERTAIN on verify-fail. Soft claim; \(n=7\); one predicate.

---

## Layers

| Layer | Role | Status |
|-------|------|--------|
| **L1** | Extract → first-class states (`contradiction`, `uncertainty`) → atoms → **fixed rule** | Done (M2/M3); frozen A–D Phases 1–4 |
| **L2** | **Verify** formalization against analysis/source; regenerate **once** if clash | Phase-5 (frozen) |
| **L3** | If still unverified → **REVIEW** (not UNCERTAIN) | Phase-5 (frozen) |
| **L2b / L2c** | Broader faithfulness checks (round-trip, dual-path) — experiments, not final proof | Phase-6 frozen; live pending |
| **L4** | Grow ontology / predicates / temporal structure carefully → **coverage** | Later |
| **L5** | CXR: Archetypes + Qdrant (B2) + fusion; same propose/verify/REVIEW pattern | Later — **not** until lab L2/L3 direction holds |

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

C4-class (temporal/identity): V1 may **REVIEW** without “solving” it. Later structure (coverage) can reclaim that class.

---

## Scope freezes

### After Phase-5 live (frozen 2026-08-23)

Phase-5 L2+L3 on Phase-4 held-out: D-rep-loss cleared (fix or REVIEW) on all three models. Soft support for safe-coverage architecture. PDF: `notes/progress-notes.pdf`.

**Still out of scope without a new protocol:** Qdrant, Archetypes, CXR production, C4 field hacks, new predicates, B′ iteration, minting a first-class artefact per failure mode, claiming domain-general N→S assurance.

### Phase-6 (protocol frozen 2026-08-23)

L2b round-trip + L2c dual-path on Phase-4 held-out; same three models. Implementation: `run_phase6.py`, `n2s_lab/roundtrip.py`, `n2s_lab/phase6_faithfulness.py`. Selftest OK. **Live panel wait user `python3 run_phase6.py`.**

Still out of scope: full evidence-preservation object, Qdrant, Archetypes, CXR production, C4 field hacks, claiming domain-general N→S assurance.

---

## Related artifacts

| Doc | Role |
|-----|------|
| [PHASE5-PROTOCOL.md](./PHASE5-PROTOCOL.md) | Verify → regen → REVIEW; live record frozen |
| [PHASE6-PROTOCOL.md](./PHASE6-PROTOCOL.md) | Research stub: faithfulness experiments |
| [PHASE4-PROTOCOL.md](./PHASE4-PROTOCOL.md) | Prior held-out preservation study |
| `notes/progress-notes.pdf` | Full frozen M1–M5 record |
| `notes/progress-notes-simple.pdf` | Newcomer guide + demo |
| Public demo | https://udonsikalu.github.io/cxr-evidence-grounding-lab/ |
