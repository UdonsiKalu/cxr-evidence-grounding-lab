# Phase 9 — simple progress notes (rep-eng pilot)

**Purpose:** Learn mechanistic interpretability and representation engineering on one real lab miss.  
**Status:** **Frozen completed pilot** (2026-08-24). Artifacts only — do not overwrite Phase 1–8.  
**Full protocol:** [PHASE9-PROTOCOL.md](./PHASE9-PROTOCOL.md)

---

## The question we cared about

On case **BC_E1**, the note describes a normal treatment **course** (response, then progression). Gold is **SATISFIED** (first-line therapy failed). Some Qwen models still set **`contradiction.present = true`** and the pipeline can AUTO-label **CONTRADICTION** — wrong.

We wanted to know: *where* in the model/pipeline does that false “contradiction” come from?

---

## What we already knew (from Phase 7–8, Ollama)

- Raw extract often has **X = false** (no contradiction).
- Verify can fail because analysis text triggers a regex (“mentions contradiction”) even when the note denies conflict.
- **Repair** then sets **X = true** with spans like “initial response” vs “progression” — treating **course as conflict**.
- Dual and evidence can amplify; that’s a separate story.

Phase 9 does **not** re-run Phase 7–8. Those JSON files stay frozen citations.

---

## Phase 9 in three steps (Observe → Localize → Intervene)

### 9A — Observe (HF, same note + schema)

We loaded **HuggingFace** Qwen (not Ollama) with the **same frozen extract prompt** on BC_E1.

| Model | Raw X | After repair | Outcome |
|-------|-------|--------------|---------|
| **7B-Instruct** | false | **true** | Wrong (like the fail class) |
| **14B-Instruct** | false | false | Safer (REVIEW, not AUTO wrong) |

**Gate: YES** — we reproduced an X divergence on the same probe.

**Surprise:** HF 14B did *not* replay Ollama 14B’s AUTO miss; HF **7B** was the fail proxy for activation work.

---

### 9B — Localize (where does X get decided?)

We traced **repair JSON generation** token-by-token and hooked **hidden states** at four depths (25%, 50%, 75%, 100% of layers).

At the moment the model writes **`contradiction.present`**:

| Model | Commits | Logit margin (true − false) |
|-------|---------|----------------------------|
| **7B** | **true** | **+1.63** (leans true) |
| **14B** | **false** | **−3.97** (leans false) |

**Finding:** The split is at the **repair commit token**, not in raw extract. Same verify-fail upstream; different boolean choice downstream.

**Gate: YES** — logit margins and layer activity diverge at commit.

Artifacts: `artifacts/phase9b-bc-e1-panel.json`, `*-trace.json`.

---

### 9C — Intervene (causal test at the commit)

We added **+4 logit bias toward `false`** only at the **contradiction.present** commit step (7B, repair path).

| | Baseline | After intervention |
|---|----------|-------------------|
| **repair X** | true | **false** |
| **Rule verdict** | CONTRADICTION | **SATISFIED** |
| **Gold** | SATISFIED | SATISFIED |

**Controls (unchanged):**

- **BC_C1** (real contradiction): stayed X = true.
- **BC_E2** (no failure): stayed X = false.

**Gate: YES** for a **behavioral / output-level** causal chain:

```text
repair → wrong boolean commit (true) → X=true → CONTRADICTION
         ↑ fix commit (bias toward false)
         → X=false → SATISFIED
```

---

## What Phase 9 proved (say this)

1. The BC_E1 false contradiction is **localized to the repair commit** on `contradiction.present`.
2. **7B vs 14B** differ in logit preference at that commit (not necessarily in raw extract).
3. **Output-level** intervention at that commit **causally fixes** symbolic X and the rule verdict on BC_E1 without breaking two controls.

---

## What Phase 9 did *not* prove (important)

9C used **logit bias on the output token** (`true` / `false`). That is **not** the same as:

> We found an internal neural feature and steered **hidden activations** so the model *naturally* prefers false.

For **mechanistic interpretability / representation engineering**, that gap is exactly why **Phase 10** exists.

---

## Artifacts (citation record)

| File | What |
|------|------|
| `artifacts/phase9a-bc-e1-panel.json` | 9A behavioral reproduce |
| `artifacts/phase9b-bc-e1-panel.json` | 9B repair traces + commit margins |
| `artifacts/phase9c-bc-e1-panel.json` | 9C logit intervention + controls |
| `docs/PHASE9-PROTOCOL.md` | Frozen protocol |
| `run_phase9a.py`, `run_phase9b.py`, `run_phase9c.py` | Runners (HF; `.venv-phase9`) |

---

## What comes next: Phase 10

**Stricter question:**

> Can we change the 7B model’s contradiction decision by intervening on **hidden activations** rather than directly biasing the output token?

See [PHASE10-PROTOCOL.md](./PHASE10-PROTOCOL.md). Wait user **go** before implementation.

---

## One-line north star (rep-eng track)

Discover whether **internal representations** at the repair commit carry the false “course = contradiction” signal — and whether steering them changes the decision **without** forcing the `false` token.
