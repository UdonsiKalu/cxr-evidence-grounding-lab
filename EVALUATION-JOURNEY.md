# Evaluation journey — Neural-to-Symbolic Evidence Grounding Lab

**Purpose of this note:** Source material for writing up how the lab was built and what was evaluated. Not a paper. Not a claim of clinical validity.

**Lab path:** `staging/cxr-evidence-grounding-lab/`  
**PDF notes:** `notes/progress-notes.pdf` (source: `notes/progress-notes.tex`)  
**UI:** http://127.0.0.1:8253/  
**Dates:** 2026-08-17 (Milestones 1, 2, 2.1, 3 — sequence frozen; optional Phase-1 diagnostic added)  
**Model:** local Ollama `llama3:8b-instruct-q4_0`  
**Constraint:** does not modify CXR production (`claim_analysis_tools`, Claim Studio, rehearsal UI).

---

## 1. The question we are evaluating

> Can a system reliably determine whether evidence expressed in natural language satisfies a formally defined condition?

Milestone 1 was deliberately tiny: **one** formal condition, **~20** hand-authored notes, and a **direct-LLM baseline** so the structured pipeline could be compared rather than merely demonstrated.

We did **not** optimize accuracy. Version 1 succeeds if the neural-to-symbolic boundary is **observable** (same requirement, different wording, extraction, atoms, rule) and we can mark succeed / fail / uncertain / contradiction.

---

## 2. What we built (the system under test)

Two paths, **same notes**, **same predicate**, gold labels **not** fed to the model.

```
                    ┌─────────────────────────────┐
                    │  Direct LLM baseline        │
 note ──────────────┤  → SATISFIED / NOT_SATISFIED│
                    │    / UNCERTAIN / CONTRADICTION
                    └─────────────────────────────┘

                    ┌─────────────────────────────┐
                    │  Pipeline                   │
 note ── extract ──►│  descriptive JSON           │
                    │  (no verdict)               │
                    │         ↓                   │
                    │  deterministic grounding    │
                    │  → atoms A B C D + X        │
                    │         ↓                   │
                    │  fixed symbolic rule        │
                    └─────────────────────────────┘
```

**Predicate:** `FIRST_LINE_THERAPY_FAILED`

| Atom | Meaning |
|------|---------|
| **A** | first-line therapy identified (explicit or conventional implication) |
| **B** | that therapy was given/attempted, not merely planned |
| **C** | failure event (progression, refractory language, stop for intolerance) |
| **D** | that failure is of the first-line regimen |
| **X** | mutually incompatible claims about A–D |

**Rule (unchanged across milestones):** CONTRADICTION if X; else NOT_SATISFIED if any of A–D is false; else UNCERTAIN if any is unknown; else SATISFIED if A∧B∧C∧D.

**UI marks:** `P+` / `P−` = pipeline matches / misses **gold**. `B+` / `B−` = direct LLM matches / misses gold. Not “predicate is true.”

**How to run**

```bash
cd /home/udonsi-kalu/staging/cxr-evidence-grounding-lab
python3 run_experiment.py --mode live    # Ollama extract + baseline
python3 server.py                        # http://127.0.0.1:8253/
```

`--mode mock` is a keyword extractor for inspecting the symbolic path. It is **not** a research result.

---

## 3. The evaluation set (what each case is for)

Twenty synthetic oncology-note snippets. Four wording categories. Gold is a four-way label, not yes/no.

### Explicit (E) — the note almost says it

| ID | Gold | Why we included it |
|----|------|--------------------|
| E1 | SATISFIED | Named first-line + explicit progression |
| E2 | SATISFIED | Stopped for intolerance (counts as line failure) |
| E3 | NOT_SATISFIED | Ongoing first-line with response |
| E4 | NOT_SATISFIED | First-line only planned; never given |

### Implicit (I) — you have to infer it

| ID | Gold | Why |
|----|------|-----|
| I1 | SATISFIED | New lesions → switch to second-line (never says “failed”) |
| I2 | SATISFIED | “Platinum-refractory” as conventional 1L failure language |
| I3 | NOT_SATISFIED | Future plan only |
| I4 | NOT_SATISFIED | Completed adjuvant + NED ≠ metastatic first-line failure |

### Temporal (T) — order in time matters

| ID | Gold | Why |
|----|------|-----|
| T1 | SATISFIED | Initial response, then later progression on 1L |
| T2 | NOT_SATISFIED | Cycle 1 day 2; too early |
| T3 | SATISFIED | Response then progression; now on second-line |
| T4 | NOT_SATISFIED | Hold for neutropenia with plan to resume |

### Uncertain (U) — incomplete or mixed

| ID | Gold | Why |
|----|------|-----|
| U1 | UNCERTAIN | Possible progression; awaiting confirmatory scan |
| U2 | UNCERTAIN | May have received platinum; outcome unknown |
| U3 | UNCERTAIN | Mixed response |
| U4 | UNCERTAIN | Unclear whether metastatic 1L or adjuvant; no clear progression |

### Conflicting (C) — two parts of the note disagree

| ID | Gold | Why |
|----|------|-----|
| C1 | CONTRADICTION | Same-day assessment “failed” vs addendum “continue 1L with SD” |
| C2 | CONTRADICTION | “Never received systemic therapy” vs six cycles then progression |
| C3 | CONTRADICTION | Referring “failed 1L” vs chart review “never started” |
| C4 | CONTRADICTION | Same regimen: “progressed” vs later “excellent ongoing response” |

Full text lives in `data/cases.json`. Gold rationale is the `why` field (shown in the UI, **not** sent to the model).

---

## 4. Journey of the evaluation (what we actually did)

### Milestone 1 — make the boundary visible + don’t skip the baseline

**Intervention:** Isolated lab, one predicate, 20 cases, two paths, UI.

**Live 20-case run** (`artifacts/live-m1-before-contradiction.json`):

| | Match gold |
|---|---|
| Pipeline | **13/20** |
| Direct LLM | **12/20** |
| Disagree with each other | I2, T2, U1, U3, U4 |

By category (pipeline / baseline correct): explicit 4/4 and 4/4; implicit 3/4 and 4/4; temporal 4/4 and 3/4; uncertain 2/4 and 1/4; **conflicting 0/4 and 0/4**.

**What we learned**

- Explicit cases are easy for both. Structure is not “doing extra work” there.
- **I2:** baseline SATISFIED (correct); pipeline missed at **extraction** (implicit platinum-refractory → B false). Fail is not the rule.
- **T2, U1, U3:** pipeline closer to gold; baseline over-called SATISFIED or NOT_SATISFIED.
- **C1–C4:** both missed all four. On C1 the baseline *described* the addendum conflict, then still emitted NOT_SATISFIED.

**Decision:** Do not chase I2 accuracy. The interesting failure was contradiction information **recognized in language and dropped before the rule**.

### Milestone 2 — first-class contradiction at the neural→symbolic boundary

**Intervention:** Extractor must always fill `contradiction.present` = true (two incompatible spans) or false (explicit none). Grounding sets **X** from that field. **Rule unchanged.**

**Finding (the one worth writing):** The LLM can notice contradiction in prose and still not emit CONTRADICTION as a verdict. Putting two spans on the interface lets the symbolic rule fire.

Live pattern on C (from `artifacts/live-c1-c4.json` and the M2 full run):

| Case | Pipeline | Direct LLM |
|------|----------|------------|
| C1 | CONTRADICTION | NOT_SATISFIED (rationale still describes the conflict) |
| C2 | CONTRADICTION | SATISFIED |
| C3 | CONTRADICTION | NOT_SATISFIED |
| C4 | NOT_SATISFIED | NOT_SATISFIED (`present=false`) |

Headline score stayed about **13/20 vs 12/20**. The *kind* of error changed, not the leaderboard.

**Cost of M2:** **U4** (uncertain line of therapy) was called CONTRADICTION by the pipeline — X over-fired on “metastatic vs adjuvant?”

**C4 left failing on purpose.** It is not the same class as C1–C3 (surface incompatible records / addendum). It needs identity of regimen plus time: progressed-at-T1 vs excellent-response-at-T2 might be conflict **or** clinical course. Stretching span extraction to “fix” C4 would hide that class.

### Milestone 2.1 — X is not uncertainty (U4 hygiene)

**Intervention:** Prompt + grounding: unresolved which-line / mixed / pending → uncertainty, **not** X. Do **not** try to make C4 pass.

**Live slice** (`artifacts/live-m2.1-c-u4.json`):

| Case | Pipeline | Direct LLM | X |
|------|----------|------------|---|
| C1 | CONTRADICTION | NOT_SATISFIED | true |
| C2 | CONTRADICTION | SATISFIED | true |
| C3 | CONTRADICTION | NOT_SATISFIED | true |
| C4 | NOT_SATISFIED | NOT_SATISFIED | false (still the hard miss) |
| U4 | NOT_SATISFIED | UNCERTAIN | **false** (no longer CONTRADICTION) |

U4 hygiene succeeded (X off). U4 is **not** gold-correct this run (gold UNCERTAIN; pipeline NOT_SATISFIED from atoms). That is leftover uncertainty calibration, not a reason to reopen C4.

---

## 5. How to read a case in the UI

1. Open http://127.0.0.1:8253/ → **Load last artifact** (or **Run selected**).
2. Left list: category + `P+ B+` vs gold.
3. Right pane, top to bottom: evidence + gold → baseline vs pipeline verdicts → neural JSON → **Contradiction field (X)** → grounded atoms A–D + X → the **same** predicate text on every case.
4. For the write-up, the instructive trio is **C1** (structure preserves conflict), **U4** (X must not eat uncertainty), **C4** (harder temporal/identity class).

---

## 6. What we are evaluating (and what we are not)

**Evaluating**

- Whether a direct LLM verdict and a structured extract→ground→rule path **differ** on the same 20 statements.
- Whether an explicit field at the neural→symbolic boundary **preserves** a semantic state (contradiction) that the baseline narrates then discards.
- Whether that field **over-generalizes** (U4) and can be constrained without “fixing” a harder class (C4).
- Whether the same move for **uncertainty** (`uncertainty.present` → unknown atoms) stops collapse to NOT_SATISFIED while the baseline still over-commits.

**Not evaluating (yet)**

- Production CXR / Archetypes / Qdrant / kernel fusion.
- Accuracy as a goal, or a larger gold set.
- Additional predicates.
- Whether C4 *should* be CONTRADICTION vs course-over-time (gold is a design choice; the miss is the scientific object).

---

## 7. Milestone 3 — first-class uncertainty at the neural→symbolic boundary

**Intervention:** Extractor must always fill `uncertainty.present` = true (one cue + why) or false (explicit none). Grounding keeps implicated A–D atoms **unknown**. **Rule unchanged** (UNCERTAIN still comes from unknown atoms, not a new U atom). Do not rewrite A–D or X.

**Live slice** (`artifacts/live-m3-u-c.json`): U1–U4 + C1–C4.

| ID | Pipeline | Baseline | U.present | X |
|----|----------|----------|-----------|---|
| U1 | UNCERTAIN | SATISFIED | true (pending) | false |
| U2 | UNCERTAIN | NOT_SATISFIED | true (unknown_outcome) | false |
| U3 | UNCERTAIN | NOT_SATISFIED | true | false |
| U4 | UNCERTAIN | UNCERTAIN | true (unresolved_line) | **false** |
| C1 | CONTRADICTION | NOT_SATISFIED | false | true |
| C2 | CONTRADICTION | SATISFIED | false | true |
| C3 | CONTRADICTION | NOT_SATISFIED | false | true |
| C4 | NOT_SATISFIED | NOT_SATISFIED | false | false (still miss) |

**Finding:** Same pattern as M2, but **do not treat U1–U4 as one M3 win.** U1/U3 were often already UNCERTAIN from unknown atoms. The distinctive M3 save is **U2/U4** no longer collapsing to NOT_SATISFIED. Baseline still over-commits on U1–U3; this run it matched gold on U4. C1–C3 still CONTRADICTION; C4 left failing; U4 X stayed false.

**Hypothesis (lab scale):** reliable neuro-symbolic *verdicts here* depend not only on extracting facts, but on preserving semantic states (uncertainty, contradiction) across the neural-to-symbolic boundary. One predicate, 20 notes, one local 8B model. The score is not the result.

### Frozen sequence (M1→M3)

Documented before adding fields or growing the set.

- **C1–C3:** explicit contradiction prevents conflict from flattening into an ordinary verdict.
- **U2/U4:** explicit uncertainty prevents ambiguous evidence from flattening into SATISFIED/NOT_SATISFIED.
- **U1/U3:** supporting cases, not the M3 increment.
- **C4:** harder identity/time class; span-pair X is insufficient. Gold=CONTRADICTION is a design choice. Does **not** yet prove a temporal reasoner is required.
- **Next experiment after this write-up:** whether the same pattern holds on a larger controlled set — not more extract fields.

### Optional Phase-1 (removable, 2026-08-17)

Transition diagnostic on **C1–C4 + U1–U4** only: Conditions A–D, failure-location counts. See `docs/PHASE1-DIAGNOSTIC.md`. Does not change M1–M3 APIs. Undo by deleting Phase-1 files + `artifacts/phase1-*.json`.

**Live control (llama3:8b-instruct-q4_0):** match gold A 1/8, B 3/8, C 7/8, D 4/8. Frozen reading: path comparison, not “C wins”; C’s strength is partly test-set design risk; D’s `representation_loss` (analysis has distinction → structure drops it) is the interesting locus. Prompts frozen; C4 left unresolved.

### Phase-2 — model panel (frozen prompts, 2026-08-17)

Same A–D, same 8 cases, **no** per-model prompt retune. Local panel (no API): control Llama 8B Q4 (reused Phase-1), `mistral:instruct`, `qwen2.5-coder:32b`.

| Model | A | B | C | D | D `representation_loss` |
|-------|---|---|---|---|-------------------------|
| llama3:8b-instruct-q4_0 | 1/8 | 3/8 | 7/8 | 4/8 | 3 |
| mistral:instruct | 7/8 | 3/8 | 7/8 | 5/8 | 2 |
| qwen2.5-coder:32b | 4/8 | 7/8 | 8/8 | 8/8 | 0 |

D drop-at-formalization **repeats on small instruct models**, not only Llama. The 32B local model **wipes D-rep-loss and C4 on this discovery slice**. B analysis→answer mismatch still occurs on all three. Direct A is model-specific; Qwen still misses all four U as NOT_SATISFIED. **n=8 cannot decide** research-problem vs capability — next is unseen cases, prompts still frozen. Artifact: `artifacts/phase2-model-panel.json`.

### B′ ablation (2026-08-17)

Same 8 Phase-2 analyses; frozen B prompt unchanged; only verdict-mapping prompt rerun. Do not overwrite Phase-1/2 JSON.

| Model | B | B′ | B mismatch | B′ mismatch |
|-------|---|----|------------|-------------|
| llama3:8b-instruct-q4_0 | 3/8 | 3/8 | 3 | 3 |
| mistral:instruct | 3/8 | 4/8 | 4 | 3 |
| qwen2.5-coder:32b | 7/8 | 8/8 | 1 | 0 |

Llama unchanged. Mistral saved U1 only. Qwen C2 UNCERTAIN→CONTRADICTION. **Discovery:** B mismatch on small models is not a forgotten-scratchpad / one-line mapping bug; contradiction collapse survived the mapping prompt; uncertainty was the easier remap. Do not iterate B′ on this slice. Do not adopt B′ as Phase-3 default. Artifact: `artifacts/bprime-ablation.json`.

### Phase-3 (frozen, 2026-08-17)

Eight new notes C5–C8 / U5–U8 in `data/heldout-phase3.json` (not merged into locked `cases.json`). Frozen A–D; B′ not used; Phase-1/2 not overwritten. **Phase-3 is frozen.** Next protocol deferred.

| Slice | Model | A | B | C | D | D `representation_loss` |
|-------|-------|---|---|---|---|-------------------------|
| Discovery C1–C4/U1–U4 | llama3:8b-instruct-q4_0 | 1/8 | 3/8 | 7/8 | 4/8 | 3 |
| Discovery | mistral:instruct | 7/8 | 3/8 | 7/8 | 5/8 | 2 |
| Discovery | qwen2.5-coder:32b | 4/8 | 7/8 | 8/8 | 8/8 | 0 |
| Held-out C5–C8/U5–U8 | llama3:8b-instruct-q4_0 | 1/8 | 4/8 | 7/8 | 6/8 | 1 |
| Held-out | mistral:instruct | 7/8 | 4/8 | 7/8 | 6/8 | 2 |
| Held-out | qwen2.5-coder:32b | 3/8 | 6/8 | 8/8 | 6/8 | 1 |

**Exactly:** D-loss on held-out = Llama C8; Mistral C6+C8; Qwen C6 (plus Qwen C7 D symbolic_failure). B mismatch on all three (Llama C7/C8/U7; Mistral C5–C8; Qwen C6/C8). Small-model C miss = C8 only. Qwen A = all four U as NOT_SATISFIED. 32B perfect D on discovery did not survive unseen. Gold unchanged. Artifact: `artifacts/phase3-heldout-panel.json`. Case table in `notes/progress-notes.pdf`.

### Phase-4 (frozen 2026-08-17)

Second held-out C9–C11 / U9–U12 in `data/heldout-phase4.json` (not merged into locked 20). Frozen A–D; B′ off; no C4/C8 analog. Phase-1/2/3 artifacts not overwritten. **Phase-4 is frozen.** Case table in `notes/progress-notes.pdf`.

| Model | A | B | C | D | D `representation_loss` |
|-------|---|---|---|---|-------------------------|
| llama3:8b-instruct-q4_0 | 2/7 | 4/7 | 6/7 | 5/7 | 1 (C11) |
| mistral:instruct | 7/7 | 4/7 | 7/7 | 4/7 | 1 (C11) |
| qwen2.5-coder:32b | 2/7 | 5/7 | 6/7 | 6/7 | 1 (C10) |

**Stop rule (declared before run):** D-loss recurred on **all three** models → formalization-preservation hypothesis **gains support**. B mismatch also on all three; do not collapse B and D.

**Exactly:** D-rep-loss = Llama C11; Mistral C11; Qwen C10. B mismatch Llama C9/C10/C11; Mistral C9/C10/C11; Qwen C9/C10. C miss = Llama C10, Qwen C10 (Mistral C 7/7). All four U matched B/C/D on all three; Qwen A = all four U as NOT_SATISFIED (same A pattern as Phase-3). Regex screen (not headline): D-rep 1/6, 1/5, 1/7. Human confirmation of analyses still required for protocol \(A_{\text{human}}\) rates. Gold unchanged. Artifact: `artifacts/phase4-heldout-panel.json`. No further N2S protocol queued unless designed separately.

---

## 8. Artifact index

| File | What it is |
|------|------------|
| `data/cases.json` | Locked 20 cases + gold + why |
| `data/heldout-phase3.json` | Frozen held-out C5–C8 / U5–U8 (not merged into locked 20) |
| `data/heldout-phase4.json` | Frozen held-out C9–C11 / U9–U12 (not merged into locked 20) |
| `artifacts/phase4-heldout-panel.json` | Phase-4 held-out cross-model summary |
| `artifacts/phase4-*.json` | Per-model Phase-4 A–D rows |
| `artifacts/phase3-heldout-panel.json` | Phase-3 held-out cross-model summary |
| `artifacts/phase3-*.json` | Per-model Phase-3 A–D rows |
| `artifacts/live-m1-before-contradiction.json` | M1 full live 20 |
| `artifacts/live-latest.json` | Last full live 20 (M2-era; may lag M2.1) |
| `artifacts/live-c1-c4.json` | M2 fresh C1–C4 |
| `artifacts/live-m2.1-c-u4.json` | M2.1 C1–C4 + U4 |
| `artifacts/live-m3-u-c.json` | M3 U1–U4 + C1–C4 |
| `artifacts/phase1-transition-c-u.json` | Phase-1 A–D control |
| `artifacts/phase2-model-panel.json` | Phase-2 cross-model summary |
| `artifacts/phase2-*.json` | Per-model Phase-2 A–D rows |
| `artifacts/bprime-ablation.json` | B′ mapping-prompt ablation vs frozen B |
| `artifacts/bprime-*.json` | Per-model B vs B′ rows |
| `artifacts/mock-latest.json` | Heuristic mock; not a result |
| `notes/progress-notes.pdf` | LaTeX progress notes (M1–M3 + Phase-1–4 protocol) |
| `docs/PHASE4-PROTOCOL.md` | Frozen Phase-4 protocol + live record in PDF |

Live JSON is gitignored; this journey note is the durable narrative if artifacts are missing.

---

## 9. One-paragraph abstract (pasteable)

We built a 20-note lab that asks whether natural-language oncology snippets satisfy a single formal predicate, `FIRST_LINE_THERAPY_FAILED`, comparing a direct local LLM verdict to an extract→ground→rule pipeline. Explicit wording is easy for both. The first scientific result is not a higher score (13/20 vs 12/20) but a failure mode: the LLM can describe a contradiction and still not emit CONTRADICTION. Making contradiction a required extract field (two spans or none) let the unchanged symbolic rule fire on C1–C3 while the baseline still would not. C4 remains unresolved because the apparent conflict depends on regimen identity and time, not a surface span pair. A follow-up constraint stopped the new field from labeling unresolved line-of-therapy (U4) as contradiction. The same move for uncertainty (`uncertainty.present` → keep atoms unknown) stopped **U2/U4** collapsing to NOT_SATISFIED while the direct LLM still over-committed on U1–U3. U1/U3 were often already pipeline-UNCERTAIN; they are not the M3 increment. C4 remains the harder identity/time miss — explicit state fields do not cover that class. Not production CXR.
