# MAP-01 — What we set out to achieve

**Prerequisite:** MAP-00 · **Next:** [MAP-02](./02-backend-pipeline.md)

## Theory — the original aim

The ambitious sentence was:

> Can we automatically change and correct every failure mode inside a large language model?

That is a **later thesis**, not a closeable v1. It is open-ended: there is no finite list of “every failure.”

This pass asked a **smaller, empirical** question (design doc `docs/TRACKB-FAILURE-INTERVENTION-MAP.md`):

> Can recurring Dual-shaped clinical-note failure signatures be mapped to correction methods that repair those failures while preserving unaffected notes — and could that map later support *choosing* an intervention?

Two halves:

1. **Mapping** — group similar Dual mistakes; attach a patch; check collateral (did we break notes that were already right?).
2. **Selection** — given a note, pick patch A vs B vs abstain, not “always apply the deepest stack.”

We completed a **partial mapping** on the Translate/extract side. We did **not** complete selection or auto-correct.

## Background — why Dual exists

Doctors write messy notes. Rules need four crisp facts:

| Atom | Plain meaning |
|------|----------------|
| **A** | A first-line (1L) therapy is identified |
| **B** | That therapy was actually given, not only planned |
| **C** | Something that counts as failure happened |
| **D** | That failure is *of* first-line, not some other line |

Plus **X**: two claims in the note cannot both be true about those facts.

The model is allowed to fill the extract. The rule is not allowed to “use vibes.” If C is unknown, the rule says UNCERTAIN. If X is true, CONTRADICTION. If any of A–D is false, NOT_SATISFIED. Only A and B and C and D true, and not X, is SATISFIED.

That split is the **neural to symbolic boundary**. Fidelity means: did the meaning in the note survive into those atoms?

## What “several inputs” means here

Each row is a **synthetic note** (not a real patient), with a gold label. Different notes do different clinical jobs:

- sequenced response then later progression (T1)
- failure implied by “platinum-refractory” (I2)
- two incompatible 1L stories on the same chart (C4)
- treatment just started, no failure yet (T2)

A one-size patch that helps I2 can hurt a P-neg note if you are careless. That is why we measured **P-neg**, **P-xspan**, and **P-bind** as arms, not only “miss count went down.”

## Results you will need later

| Original part | This-pass answer |
|---------------|------------------|
| Map Dual failures to corrections | Partial yes — six nested Translate cells, n=108 |
| Repair without breaking others | Yes on measured arms (P-neg leak 0 after G3; P-xspan 10/10 after G6) |
| Choose among independent editors | **No** — only one class (nested Translate) |
| Locator picks Encode vs Compute vs Translate | **No** |
| Auto-correct every LLM failure | **No** |

## Backend — find the locked question

```bash
cd ~/staging/cxr-evidence-grounding-lab
sed -n '1,12p' docs/TRACKB-FAILURE-INTERVENTION-MAP.md
```

## Week 1 — send-back

Write the smaller Dual question in one sentence. Write the ambitious auto-correct sentence. Mark which one this curriculum canonizes.
