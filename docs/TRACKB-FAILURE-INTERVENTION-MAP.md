# Failure → Intervention Mapping — design only (2026-09-11)

**Status:** original mapping Q **closed for this pass** — six Dual Translate cells; extract-signature selector matches apply-all miss (6/108); not a selected editor; `locator_chose_repair_class=false`.  
Phase 1–3 / `:8260` / frozen-`d` / Dual_full / `ground()` are **read-only**  
**Not this:** invent one editor per Encode/Compute/Translate · Detect→route→fix · α-chase on frozen `d` · U-C SAE / U-D · GUI expansion · claim automatic correction · invent notes to hit 300 · retune promote from collaterals · CAA

**Central question (later thesis, not this pass):**

> Can recurring Dual-shaped clinical-note failure signatures be empirically mapped to correction methods that repair those failures while preserving unaffected behavior, and could that mapping later support intervention *selection*?

This document is A–E only. Percentages in examples are **not** our n.

Sibling freezes: [TRACKB-PHASE1-3-PORTFOLIO.md](./TRACKB-PHASE1-3-PORTFOLIO.md) · [N2S-SYSTEM-CLAIM.md](./N2S-SYSTEM-CLAIM.md)

---

## Preserved conclusions (do not reopen)

| Surface | What we have | Editor? |
|---------|--------------|---------|
| Encode | L8 class-correct on frozen `d` (FOLFOX/CONTRA only) | **none** |
| Compute | lost-steps; G5 REVIEW contain; frozen-`d` L12+L16+L20 α=2 **PARTIAL then CLOSED** | **none validated** |
| Translate | Dual analog / Dual_full snapshot; FOLFOX Schema quote-promote → SATISFIED | **one** boundary repair |
| Incomplete | T1/T3/C1/C4 and most coding notes | no Encode/Compute measurement |

`locator_chose_repair_class = false`. REVIEW/ABSTAIN = **containment**, not correction.

---

## 1. Audit — measurements we already have

### Encode-associated

| Measurement | Where | n | Claim level |
|-------------|-------|---|-------------|
| L8 `mean_score_d` / class-correct | locator + U-A map | 2 notes (FOLFOX, CONTRA) | correlational readout |
| U-A `d` @ L24 separation | U-A / `:8258` | pair + small paraphrase panel | correlational; gen soft+strong YES @ L24 |
| U-A causal α·`d` @ L24 | `TRACKB-UPSTREAM-UA-CAUSAL.md` | FOLFOX/CONTRA | **NULL/weak** |
| U-B / U-B2 site ablation | U-B freeze | FOLFOX/CONTRA sites | **NULL** (no X flip) |

**Failure we can currently *call* encode:** none demonstrated. Both traced notes **pass** L8. Encode miss would look like: L8 class-correct on the wrong side of `d` *and* later Dual wrong — we have **zero** such rows.

### Compute-associated

| Measurement | Where | n | Claim level |
|-------------|-------|---|-------------|
| Lost-steps (τ≥5 or sign flip) | `:8259` / locator | 2 traces | correlational |
| G5 n2n-lost→REVIEW | `TRACKB-NN-LOST-REVIEW.md` | CONTRA fires; FOLFOX does not | contain, not repair |
| Frozen-`d` first-lost / residual | patch panels | FOLFOX 0 lost Dual still NOT_SATISFIED; CONTRA 2→1 lost Dual still CONTRADICTION | **PARTIAL / CLOSED** |
| α=8 expand-`v` @ L20 | `TRACKB-ALPHA8-FREEZE.md` | small temporal false-X panel | **partial** commit editor; not n2n rewrite |
| C0–C2 L20 MLP along `v` | circuit freeze | site causal, not full circuit | frozen; not an n2n editor |

**Failure we can currently *call* compute:** CONTRA 2 lost with Dual already **correct_AUTO**. Compute loss without Dual miss. Repair of Dual is not the Q on that row.

### Translate-associated

| Measurement | Where | n | Claim level |
|-------------|-------|---|-------------|
| Dual analog 7B extract→`ground`→rule | Phase 2 / patch baselines | FOLFOX, CONTRA | FOLFOX wrong_AUTO; CONTRA correct_AUTO (analog prompt) |
| Dual_full 32B frozen | Phase-7 panels | BC_E1 + temporal family / BMT-CART | **do not rescore** |
| Quote-promote `failure_evidence_from_extract_text` | Phase 2 clean panel | 1 repair (Schema FOLFOX) | boundary; hidden states untouched |
| Schema vs analog prompt split | CONTRA Dual reconcile | 1 note | prompt/extract sample, not two Dual engines |
| Track A AUTO/REVIEW gates G1–G3 (+ optional G5) | `AUTO-CONTRACT.md` | temporal family | safety among AUTO; REVIEW = contain |

### Incomplete / unknown

Locator 7-case matrix: **4/7 incomplete** (no L8, no lost-steps, no Dual snapshot on T1/T3/C1/C4). `cases.json` n=20 and temporal-family DEV/TEST have Dual-shaped gold + note text; almost none have Encode/Compute traces.

**Implication for any batch:** default phenotype is **Translate-only** (or Dual-unknown). Do not force E/C/T on rows without probes.

---

## A. Literature-backed candidate interventions

Only methods with a real paper (or our already-run lab analog). **Do not treat these as unused tools** — several were already tried here and **null/partial**.

| Method | Source | Intervened on | Behavior in paper | Causal vs correlational | Transfer to Dual-shaped clinical notes |
|--------|--------|---------------|-------------------|-------------------------|----------------------------------------|
| **RepEng / LAT + control** | Zou et al., *Representation Engineering*, arXiv:2310.01405, 2023 | reading/control vectors on residual / contrast stimuli; LoRRA is a *tune*, not inference-only | honesty, harmlessness, hallucination, power-seeking (generic safety) | reading correlational; control is causal *on those behaviors* | **Already in-family.** Our U-A `d` is a contrast direction; causal α·`d` @ L24 was **NULL**. Do not rebuild `d` as “new RepEng.” |
| **ActAdd** | Turner et al., *Activation Addition: Steering Language Models Without Optimization*, arXiv:2308.10248, 2023 | residual-stream +α·(act(prompt+)−act(prompt−)) | topic, sentiment, detox; off-target perplexity mostly held | causal on generation style/topic | Prompts are short contrast strings, not 1L-failure atoms. May steer *wording*, not A–D grounding. Collateral risk high on CONTRADICTION controls. |
| **CAA** | Rimsky et al., *Steering Llama 2 via Contrastive Activation Addition*, ACL 2024 (arXiv:2312.06681) | mean residual difference over contrast *pairs*, add after user prompt | sycophancy, hallucination-style MC / open-ended on Llama 2 Chat | causal on targeted behaviors; authors claim small capability drop | Closest published cousin of U-A `d`. We already have **null causal-d** and **PARTIAL frozen-`d`**. New CAA needs a **new contrast set** (e.g. quote-failure vs polarity-only), not bigger α on frozen `d`. |
| **ITI** | Li et al., *Inference-Time Intervention*, NeurIPS 2023 (arXiv:2306.03341) | shift top-K *attention-head* outputs along a truthfulness probe direction; scale by head σ | TruthfulQA / Alpaca truthfulness ↑; helpfulness tradeoff | causal on TruthfulQA-style answers | “Truthfulness” ≠ `FIRST_LINE_THERAPY_FAILED`. Probe would need Dual-gold labels. Risk: flip true CONTRADICTION → SATISFIED. **New Q**, not first batch. |
| **Activation scaling (α)** | Not a separate paper — the **coefficient** in ActAdd/CAA/ITI and our α=8 / α=2 panels | same sites as the parent method | dose–response / helpfulness tradeoff (ITI §) | causal intensity of an existing direction | Our α-chase on frozen `d` is **LOCKED**. Scaling is a *knob*, not a new editor class. |
| **Causal tracing + ROME** | Meng et al., *Locating and Editing Factual Associations in GPT*, NeurIPS 2022 | causal tracing locates mid-MLP; ROME rank-one *weight* edit | factual subject–relation–object (zsRE / CounterFact) | tracing causal; ROME is a **weight** edit | Poor fit: Dual misses are *this-note extract* errors, not a stored world fact (“Paris is the capital of…”). Editing weights to “FOLFOX failed” would poison other notes. **Do not use as per-note corrector.** |
| **MEMIT** | Meng et al., *Mass-Editing Memory in a Transformer*, ICLR 2023 (arXiv:2210.07229) | batched MLP weight updates | thousands of facts | weight edit | Same mismatch as ROME. Out of scope for per-note Dual. |
| **DAS / interchange** | Geiger et al., *Finding Alignments…* (DAS), CLeaR 2024 (arXiv:2303.02536) | distributed interchange of rotated subspaces vs a high-level causal model | align neural vars to interpretable causal vars (synthetic / small tasks) | causal abstraction test | Useful as a *localization* tool if we write a high-level model of A–D/`X`. Heavy; needs paired source/base notes. Not a first-batch repair. |
| **Attention-head / MLP knockout** | e.g. Wang et al., *Interpretability in the Wild* (IOI circuit), 2023; our C1 L20 MLP along `v` | zero or patch specific heads/MLPs | IOI / site causality | causal at *that* site | Our C1 is **frozen** (site causal, not full circuit). Knockout is diagnosis, not Dual repair. U-B knockout was **NULL** on X. |
| **Boundary / structured correction** | *This lab* Phase 2 (not a NeurIPS paper): promote failure already in extract quotes/notes into outcome polarity; Track A `ground()` reconnect | extract JSON / grounding rules — **no hidden state** | FOLFOX analog Dual NOT_SATISFIED→SATISFIED; CONTRA analog no-op | causal on the *symbolic* pipeline | **Only demonstrated repair.** Hypothesis: subtypes where evidence is in the extract but not in the field `ground()` reads. Do **not** rewrite `ground()` to chase CONTRA Schema SATISFIED. |

**Not listed as “published neural editors”:** G5 REVIEW, Dual_full rescore, prompt-only Schema+note (reconcile showed it *changes extract*, not a validated repair class). Prompt change is a **confound** to log, not an editor to ship.

---

## B. Proposed failure taxonomy (data-first, Dual-shaped)

Do **not** assign Encode/Compute/Translate first. Record **clinical/Dual phenotype**, then attach probes if they exist.

Seeded from `cases.json` categories + temporal-family subtypes + Phase 2 FOLFOX/CONTRA — *hypotheses to confirm in Batch 0*, not locked classes.

| Code | Working name | What the note is doing | Gold often | Phase 2 / coding hint |
|------|----------------|------------------------|------------|------------------------|
| **P-seq** | sequenced response→later failure | time-separated response then progression | SATISFIED | T1/T3; FOLFOX; TF `therapy_worked_then_failed` |
| **P-bind** | failure in quotes/notes, not outcome polarity | extractor parked failure in `quotes` | SATISFIED | **FOLFOX Schema miss** (demonstrated) |
| **P-impl** | failure implied (switch / refractory) not stated | second-line / “refractory” | SATISFIED | I1/I2 |
| **P-neg** | no failure / planned 1L / hold-not-fail | ongoing PR, not started, dose hold | NOT_SATISFIED | E3/E4/I3/T2/T4 |
| **P-adj** | adjuvant NED misread as 1L metastatic failure | old adjuvant FOLFOX, NED | NOT_SATISFIED | I4 |
| **P-unc** | pending / mixed / unknown line | confirmatory scan, mixed response | UNCERTAIN | U1–U4 |
| **P-xspan** | same-day or chart contradiction | two incompatible 1L claims | CONTRADICTION | C1–C4; EX_CONTRA analog |
| **P-recon** | extractor `contradiction.present` but sequenced change | reconnect should override X | SATISFIED not X | Track A reconnect; Schema CONTRA trap |
| **P-unk** | no Dual / no gold match yet | incomplete locator | — | T1 locator row |

Optional **probe tags** (nullable): `enc_L8=pass/fail/unknown`, `cmp_lost=0/k/unknown`, `trn_bucket=wrong_AUTO/correct_AUTO/REVIEW/unknown`.

A note can have **two phenotypes** (e.g. P-seq + P-bind). That is allowed. One surface editor is not assumed.

---

## C. Experimental matrix / schema

One row per `(note_id, snapshot, intervention_id)`:

```text
note_id
gold                          # SATISFIED | NOT_SATISFIED | UNCERTAIN | CONTRADICTION
expected_atoms                # optional A–D/X gold if coded
model                         # 7B analog | Dual_full frozen (read-only)
prompt_snapshot               # analog | schema  (log; do not mix blindly)
verdict_before / disposition  # AUTO | REVIEW
phenotype[]                   # P-* codes from Batch 0
probes                        # enc / cmp / trn  (unknown allowed)
route_8260                    # encode|compute|translate|incomplete|none  (read-only label)
intervention_id               # none | quote_promote | …  (no frozen-d / no ground rewrite)
selection                     # none | phenotype_guided | route_guided | blind
verdict_after / disposition
match_gold_before / after
repaired                      # before≠gold and after=gold
broke_control                 # before=gold and after≠gold
signature_cleared             # e.g. C/D now true; or n_lost unchanged (do not require encode/compute)
collateral                    # other notes in the same run whose gold match flipped
hidden_state_patched          # must be false in Batch 0–1
invariants                    # Dual_full not rewritten; ground() unchanged
```

**Eventual map (not this week’s numbers):**

```text
phenotype P-bind  →  {quote_promote, CAA_new, ITI} tested
                    →  quote_promote k/n, CAA …, ITI …
                    →  lowest broke_control
P-xspan           →  quote_promote should be ~no-op (CONTRA analog)
```

**Controls (required before any “works” claim):**

| Control | Pass if |
|---------|---------|
| Correction | gold match after, not merely a different wrong |
| Correct-stay | gold-matched controls do not flip |
| Specificity | intervention that helps P-bind does **not** “repair” P-xspan Dual into SATISFIED |
| Signature | for boundary: C/D or promoted polarity actually change; for neural: say so only if we later allow a **new** vector |
| Held-out | temporal-family-**test** not used to design phenotypes |
| Blind vs guided | only after ≥2 intervention classes exist; compare phenotype-guided vs apply-all vs route_8260-guided vs REVIEW-only |

Locator-guided selection **cannot** be tested honestly until a second real intervention exists. Today route_8260-guided vs quote_promote is circular (FOLFOX already translate-routed).

---

## D. Recommended first batch (still no code until a later go)

| Batch | What | n (honest) | Interventions | Why first |
|-------|------|------------|---------------|-----------|
| **0** | Phenotype census | `cases.json` 20 + optional temporal-family **DEV** 14 | **none** — Dual analog extract only | Let P-* frequencies come from extracts, not from this table |
| **1** | Boundary-only | Dual-wrong AUTO from Batch 0 with gold SATISFIED + P-xspan/P-neg **controls** | `failure_evidence_from_extract_text` only | Replicate FOLFOX-class vs prove it is subtype-specific |
| **2** | Prompt confound | same notes, analog vs Schema snapshot | still no `ground()` change | Log extract deltas; do not pick the prompt that “wins” as an editor |
| **3+** | *Not now* | only if Batch 1 shows ≥1 recurring P-* with k≥3 | **new** CAA/ITI contrast (not frozen `d`); or DAS localization | New Q; locked knobs stay locked |

**Do not** in Batch 0–2: GPU frozen-`d`, α=8 as Compute fix, Dual_full rescore, U-C/U-D, `:8260` new cases, rewrite `ground()`.

Success for Batch 1: quote-promote helps **P-bind** (and maybe P-seq with quotes) and is a **no-op** on P-xspan analog Dual CONTRADICTION and on P-neg. That is a *map cell*, not auto-correct.

---

## E. Existing CXR infrastructure to reuse

| Reuse | For |
|-------|-----|
| `data/cases.json`, temporal-family-dev/test, held-out JSONs | notes + gold + `why` (phenotype seeds) |
| Dual analog path (`extract → ground → evaluate_rule`) | Batch 0–1 verdicts |
| `n2s_phase2_translate.py` quote-promote | Batch 1 intervention (copy, do not overwrite Exp1 artifacts) |
| `AUTO-CONTRACT.md` G1–G3 | disposition; REVIEW ≠ repaired |
| G5 / lost-steps / L8 | nullable probe tags when a trace **already** exists |
| `:8253` / locator **labels** | read-only route column |
| U-A frozen `d` | **readout only** (`:8258`) |
| Phase 2 clean + reconcile JSON | FOLFOX/CONTRA anchors |

| Do not reuse as a new editor | Why |
|------------------------------|-----|
| Frozen-`d` patch runner | CLOSED; α-chase locked |
| Dual_full Phase-7 JSON rewrite | freeze |
| `ground()` reconnect rewrite | CONTRA split was prompt |
| `:8260` GUI | v1 frozen |
| C1 L20 MLP / α=8 `v` | different Q (commit X), already frozen |

---

## Locked (this branch)

- Phase 1–3 artifacts read-only  
- No Encode/Compute editor invented to fill the matrix  
- No Detect→route→fix  
- No U-C / U-D / Dual_full rescore / GUI  
- Auto-correction remains a **later** thesis after Batch 0–1 (and only if a second intervention class exists)

## Batch 0 result (2026-09-11)

Dual analog 7B (`user=evidence`, `intervention=none`) on `data/cases.json` n=20. **Did not** switch to Schema+note.

| Count | n |
|-------|---|
| parse_ok | **20/20** (JSON parse, not a rich extract) |
| thin_extract | **18/20** |
| P-unk | **18** |
| P-bind | **0** |
| P-recon | **0** |
| wrong_AUTO | **12** |
| correct_AUTO | **8** (default empty-extract verdict vs gold NOT_SATISFIED / UNCERTAIN — not a rich match) |

Gold/category tags (proposed, not extract-proven): P-expl 2 · P-impl 2 · P-seq 2 · P-neg 5 · P-adj 1 · P-unc 4 · P-xspan 4.

Non-thin: **C1, C3** — extractor set `contradiction_present` with spans; `ground()` then **overrode X=false** (`no failure polarity / no hard conflict`). Verdict NOT_SATISFIED vs gold CONTRADICTION. **Do not change `ground()`** (same class as CONTRA Dual reconcile).

SATISFIED golds (E1 E2 I1 I2 T1 T3) are all thin + wrong_AUTO. Quote-promote has **no quotes to promote** on analog Batch 0.

Artifacts: `artifacts/n2s-batch0-phenotype-census.json` · `artifacts/n2s-batch0-analog-extracts.json`  
Runner: `n2s_lab/n2s_batch0_phenotype.py` · `scripts/run_batch0_phenotype.sh`

**Implication:** Batch 1 quote-promote on analog Dual-wrong SATISFIED is expected **no-op** (0 P-bind). That is a valid map cell if run. A live P-bind test needs extract text — Batch 2 Schema snapshot — **without** treating Schema as the Dual-matched editor.

---

## Batch 0x result (2026-09-11) — expand then analog+Schema

User asked 100–300 from the existing library. **Honest cap:** Dual-shaped lab notes are **120 unique / 64 design-safe**. Did **not** invent notes, relabel oncology jsonl, or open frozen `temporal-family-test` (n=12) for phenotype design.

**Set used:** 108 unique Dual notes = 64 design (`cases` + temporal-dev + temporal-dev-expand) + 44 held-out **score-only**. Same frozen Batch 0 protocol; both prompts; **no intervention**.

| | analog (all / design) | Schema (all / design) |
|--|--|--|
| n | 108 / 64 | 108 / 64 |
| parse_ok | 108 / 64 | 108 / 64 |
| thin_extract | **91 / 56** | **0 / 0** |
| P-unk | 91 / 56 | 0 / 0 |
| **P-bind** | **0 / 0** | **8 / 7** |
| P-recon | 0 / 0 | 2 / 0 (held-out only) |
| wrong_AUTO | 77 / 46 | 31 / 17 |
| correct_AUTO | 31 / 18 | 77 / 47 |

Schema design P-bind: **E2, I2, T3, TF_E3, TF_E4, TX_E06, TX_E19**. Held-out score-only P-bind: **EX_TEMPORAL_FOLFOX**. P-recon (do not retune tag from held-out): BC14_E2, BC14_E3.

Gold/category tags on design n=64 (same both prompts): P-expl 33 · P-neg 11 · P-xspan 10 · P-unc 5 · P-seq 2 · P-impl 2 · P-adj 2.

Schema is **prompt confound / extract substrate**, not a Dual-matched editor. Analog remains thin. **Do not change `ground()`**.

Artifacts: `n2s-batch0x-library.json` · `n2s-batch0x-analog-extracts.json` · `n2s-batch0x-schema-extracts.json` · `n2s-batch0x-census.json`  
Runner: `n2s_lab/n2s_batch0x_census.py` · `scripts/run_batch0x_census.sh`  
Batch 0 n=20 artifacts **not overwritten**.

## Batch 1 result (2026-09-11) — quote-promote map cell

CPU on saved Batch 0x extracts. Intervention = Phase 2 `failure_evidence_from_extract_text` only. Design n=28 (7 P-bind + 10 P-xspan + 11 P-neg). Same IDs on analog and Schema. FOLFOX/CONTRA anchors score-only. **Did not** retune the intervention after seeing controls.

| Arm | P-bind repaired | P-xspan → SATISFIED | P-neg → SATISFIED | fired |
|-----|-----------------|---------------------|-------------------|-------|
| **Analog** | **0/7** (0 fired) | 0/10 | 0/11 | 0/28 |
| **Schema** | **5/7** (E2 T3 TF_E3 TX_E06 TX_E19) | **1/10 (C4)** | **5/11** | 7/7 bind · 1 xspan · 7 neg |

Schema bind misses: I2 fired (C unknown→true) still NOT_SATISFIED; TF_E4 fired still UNCERTAIN.  
P-neg collateral: TF_T2, TF_N1, BC_E2, TX_N01, TX_N02.  
Anchors: Schema FOLFOX repaired (Phase 2 replicate); analog FOLFOX no-op; both CONTRA no-op on analog CONTRADICTION.

`map_cell.specificity_holds = false`. `locator_chose_repair_class = false`. Schema ≠ Dual editor.

**Map cell (honest):** quote-promote can flip Schema P-bind (k=5/7) and is a **complete analog no-op**. It is **not** subtype-specific — lexical failure markers in P-neg / C4 quotes also promote to SATISFIED. Not auto-correct. Do not pick this as a selected editor.

Artifact: `artifacts/n2s-batch1-quote-promote-panel.json`  
Runner: `n2s_lab/n2s_batch1_quote_promote.py` · `scripts/run_batch1_quote_promote.sh`  
Phase 2 / Batch 0 / 0x extracts **not overwritten**. `ground()` unchanged.

## Held-out check (2026-09-11) — then STOP

Same frozen promote. n=44 held-out Dual notes. Gold labels only (no new P-*). TFT closed. Rule **not** changed.

| Arm | SATISFIED repaired | NOT_SATISFIED → SATISFIED | CONTRADICTION → SATISFIED | fired |
|-----|--------------------|---------------------------|---------------------------|-------|
| Analog | **0/13** | 0/3 | 0/18 | **0/44** |
| Schema | **1/13** (FOLFOX only) | **3/3** (BC11_N1, BC12_N1, BC14_N1) | 0/18 | 10/44 |

`leak_repeats = true`. The P-neg leak was not a 28-note accident.

**Frozen claim (Batch 1 analog):** quote-promote is analog no-op when extracts are empty; Schema lexical patch that can help SATISFIED and wrongly flips NOT_SATISFIED.

Artifact: `artifacts/n2s-batch1h-heldout-promote-panel.json`

## Goal 2 (2026-09-11) — Dual analog extracts fill

Frozen analog system asked for schema in the user message but user was note-only → empty extracts. **Fix (not Schema+note):** put schema in **system**, keep user=evidence.

| | Frozen analog | Fill (schema in system) |
|--|--|--|
| thin | **91/108** | **0/108** |
| P-unk | 91 | 0 |
| P-bind | 0 | 5 (design: E2 I2 T3 TX_E03) |
| wrong_AUTO | 77 | 30 |

`fill_worked=true`. Dual analog now has extract text.

Quote-promote on those Dual extracts (same 28, frozen rule): P-bind **2/7** (E2 T3); P-xspan 0 collateral; P-neg **5/11** still leak. `specificity_holds=false`.

**Toward the full goal:** empty Dual was the blocker; it is cleared. Quote-promote is still not an accurate/specific solution on Dual. Next goal = a **new** predeclared repair (not silent retune, not CAA yet).

Artifacts: `n2s-g2-analog-fill-census.json` · `n2s-g2-analog-fill-extracts.json` · `n2s-g2-analog-fill-promote-panel.json`

## Goal 3 (2026-09-11) — Dual gated quote-promote (negation-aware)

Predeclared gate, **not** fitted to the five P-neg IDs: a span is failure evidence only if a `FAILURE_MARKER` occurs that is **not under local negation** (`no` / `not` / `without` / `never` / `n't` / `denies` / `absent` / `no evidence of` / …, or `-free`). Same Dual-fill extracts. Ungated function **not** mutated. CPU. Held-out scored with the frozen gate (no retune).

| Arm (Dual-fill) | P-bind repaired | P-xspan → SATISFIED | P-neg → SATISFIED |
|-----------------|-----------------|---------------------|-------------------|
| Ungated | **2/7** (E2 T3) | 0/10 | **5/11** |
| **Gated** | **2/7** (E2 T3 kept) | 0/10 | **0/11** |

Design `specificity_holds=true`. Gate cleared the P-neg substring leak (`progression` inside `no/without progression`) and did **not** kill the bind repairs.

Held-out n=44, same frozen gate:

| | SATISFIED repaired | NOT_SATISFIED → SATISFIED | CONTRADICTION → SATISFIED |
|--|--------------------|---------------------------|---------------------------|
| Ungated | 1/13 (FOLFOX) | **3/3** (BC11/12/14_N1) | 1/18 (C8) |
| **Gated** | **1/13 (FOLFOX kept)** | **0/3** | **1/18 (C8)** |

`heldout_leak=true` **only C8**. C8 quotes assert `"progression on first-line …"` **and** `"excellent ongoing response"` to the same regimen; extract `contradiction_present=false`. The gate is working as declared (asserted failure still promotes). C8 is mixed-span / unflagged contradiction, not the P-neg substring bug. **Do not retune the gate from C8.**

`locator_chose_repair_class=false`. `is_selected_editor=false`. Not auto-correct. Not CAA.

**Map cell (honest):** Dual analog now has a **specific** Translate repair for P-bind vs P-neg on the design set. It is **not** a selected editor: bind coverage is still 2/7, and held-out C8 (gold CONTRADICTION, asserted mixed quotes) still flips to SATISFIED.

Artifact: `artifacts/n2s-g3-gated-promote-panel.json`  
Runner: `n2s_lab/n2s_g3_gated_promote.py` · `scripts/run_g3_gated_promote.sh`  
G2 extracts / Batch 1 panels **not overwritten**. `ground()` unchanged.

## Goal 4 (2026-09-11) — Dual mixed-span / simultaneous 1L flag

New intervention on Dual-fill extracts, **not** a Goal 3 retune from C8. Predeclared from P-xspan (two incompatible first-line claims): run Goal 3 gated promote, then if frozen `ground()`'s hard-simultaneous check is true and the extract omitted `contradiction.present`, set `present=true`. **Do not rewrite `ground()`.**

| Arm (Dual-fill) | P-bind repaired | P-neg → SATISFIED | P-xspan match-repaired | held-out SATISFIED leak |
|-----------------|-----------------|-------------------|------------------------|-------------------------|
| Goal 3 gated | 2/7 (E2 T3) | 0/11 | 0/10 | C8 |
| **Goal 4 mixed** | **2/7 kept** | **0/11** | **1/10 (C4)** | **none** |

Held-out n=44: FOLFOX still repaired; NOT_SATISFIED leaks stay 0; C8 and BC14_C2 → CONTRADICTION; `heldout_leak=false`. Design `specificity_holds=true`.

Remaining Dual misses are **other subtypes**, not this cell: bind I2 (planned) + TF_E4 (UNCERTAIN); design xspan C2 / TX_C02 already SATISFIED without ongoing+fail flag, TX_C04 UNCERTAIN. Do not retune from those IDs.

`locator_chose_repair_class=false`. `is_selected_editor=false`. Bind coverage still 2/7. Not auto-correct. Not CAA.

**Map cell (honest):** Dual analog now has (1) negation-aware promote for P-bind vs P-neg and (2) a simultaneous-mix P-xspan subtype repair. Two Translate cells, not a selected editor.

Artifact: `artifacts/n2s-g4-mixed-span-panel.json`  
Runner: `n2s_lab/n2s_g4_mixed_span.py` · `scripts/run_g4_mixed_span.sh`  
G3 panel / G2 extracts **not overwritten**. `ground()` unchanged.

## Goal 5 (2026-09-11) — remaining Dual-miss census (read-only)

Frozen Goal 4 on Dual-fill n=108. **27 still miss gold** (15 design / 12 held-out). No new repair. No G3/G4 retune. `ground()` unchanged.

| Primary shape | n | What the extract is doing |
|---------------|---|---------------------------|
| **S-never-quote** | **9** | “never / no prior” in quotes; Dual still SATISFIED or NOT_SATISFIED. Gold CONTRADICTION. `ground()` conflict blob **omits quotes**, so Goal 4 cannot see never↔given. |
| S-hedge | 6 | possible / pseudo progression; Dual UNCERTAIN vs gold SATISFIED |
| S-g4-stable-seq | 2 | Goal 4 flagged sequential stable→progression (TF_E2, TX_E02) SATISFIED → CONTRADICTION. **G4 not fully specific on n=64.** |
| S-x-override | 2 | `contradiction.present=true` then `ground()` sequenced override → SATISFIED (UA_GEN_C1/C2) |
| S-admin-blocked | 2 | planned / not_given (I2 bind) |
| S-may-have | 2 | outside-facility “may have received” |
| S-g4-notes-promote | 1 | U4 notes promote + flag; gold UNCERTAIN → CONTRADICTION |
| S-too-early / S-response-only / S-other | 1 each | T2; BC12_C1; UA_GEN_C4_FAR (continue-present in quotes; same quote-blind hole) |

`g4_flipped_away = U4, TF_E2, TX_E02`. Largest leftover class = **S-never-quote**.

**Next repair, if named:** predeclare from a shape tag (S-never-quote is the obvious Dual P-xspan leftover). Do **not** retune Goal 3/4 from these IDs. Do not rewrite `ground()`.

Artifact: `artifacts/n2s-g5-remain-census.json`  
Runner: `n2s_lab/n2s_g5_remain_census.py` · `scripts/run_g5_remain_census.sh`

## Goal 6 (2026-09-11) — Dual never↔given quote repair

Predeclared from S-never-quote, **not** a G3/G4 retune. After frozen Goal 4, copy never/no-prior spans from quotes/notes into contradiction cues with a cycles/completed/received counter-span (or `administration_status=given`) and set `present=true`. **Do not rewrite `ground()`.** Planned “no prior, starting next week” does not fire.

| | P-bind | P-neg leak | P-xspan match-repaired | n=108 miss |
|--|--|--|--|--|
| Goal 4 mixed | 2/7 | 0 | 1/10 (C4) | 27 |
| **Goal 6 never** | **2/7 kept** | **0** | **4/10 (C2 C4 TX_C02 TX_C04)** | **19** |

Design P-xspan **10/10 match gold**. Held-out CONTRADICTION match-repaired 7 (includes G4’s C8/BC14_C2 plus never-quote C6/C10/BC_C2/BC11_C1/BC11_C2). FOLFOX kept. `specificity_holds=true`. `heldout_leak=false`. **No new flips vs Goal 4.**

BC14_C1 still misses: “systemic-therapy naive” is in the census never list but not in frozen `ground()`’s never lexicon (`never` / `no prior`). Do not stretch `ground()` to fit it.

G4 collateral U4/TF_E2/TX_E02 remains. Bind still 2/7. `locator_chose_repair_class=false`. `is_selected_editor=false`.

**Map cell (honest):** Dual analog now has three Translate cells (P-neg gate, simultaneous-mix, never↔given quotes). Not a selected editor.

Artifact: `artifacts/n2s-g6-never-quote-panel.json`  
Runner: `n2s_lab/n2s_g6_never_quote.py` · `scripts/run_g6_never_quote.sh`  
G3–G5 artifacts **not overwritten**. `ground()` unchanged.

## Goal 7 (2026-09-11) — Dual restaging confirmation

New wrapper on frozen Goal 6, **not** an in-place G3/G4/G6 retune. Two extract-faithful restaging steps:

1. Confirmed failure span (`clear progressive` / `unequivocal` / `confirmed …`) → clear `uncertainty.present` (possible/pseudo resolved).
2. Goal 4 simultaneous flag from **stable disease then later progression**, without continue-current → revert `contradiction.present` (temporal, not P-xspan).

**Do not rewrite `ground()`.** C4/C8 continue-current still CONTRADICTION.

| | P-bind | P-neg leak | n=108 miss |
|--|--|--|--|
| Goal 6 never | 2/7 | 0 | 19 |
| **Goal 7 restage** | **3/7 (TF_E4 added)** | **0** | **12** |

New SATISFIED repairs: TF_E4, TX_E05, TX_E12, TX_E17, TX_E23. Restored G4 collateral: TF_E2, TX_E02. Design P-xspan still 10/10. `specificity_holds=true`. `heldout_leak=false`. No new flips vs Goal 6.

Leftover 12: I2 planned; T2 too-early; U2/U6 may-have; U4 G4 notes-promote; TX_E03 response-only; C11 not_given; BC12_C1 / BC14_C1; UA_GEN_C* `ground()` X-override. Not a selected editor.

Artifact: `artifacts/n2s-g7-restage-panel.json`  
Runner: `n2s_lab/n2s_g7_restage.py` · `scripts/run_g7_restage.sh`  
G3–G6 artifacts **not overwritten**. `ground()` unchanged.

## Goal 8 (2026-09-11) — Dual same-time mix vs sequenced override

New wrapper on frozen Goal 7, **not** an in-place G3–G7 retune. Three extract-faithful steps:

1. If the extract was already uncertain and Goal 3 only promoted failure from **notes**, revert (meta commentary, not a clinical failure span).
2. Relabel continue-now `response` polarities to `ongoing` so frozen `ground()` hard-simultaneous can fire instead of sequenced override.
3. Copy continue-current quotes into contradiction cues when failure is already extracted but `present` was omitted (quote-blind blob).

**Do not rewrite `ground()`.** Sequential then-progressed stays SATISFIED. C4 continue-current stays CONTRADICTION.

| | P-bind | P-neg leak | n=108 miss |
|--|--|--|--|
| Goal 7 restage | 3/7 | 0 | 12 |
| **Goal 8 simult** | **3/7 kept** | **0** | **8** |

Fixed vs G7: U4 restored UNCERTAIN; UA_GEN_C1_NEAR / C2_FAR / C4_FAR → CONTRADICTION. Design P-xspan still 10/10. `specificity_holds=true`. `heldout_leak=false`. **No new flips vs Goal 7.**

Leftover 8: I2 planned vs refractory; T2 too-early; U2/U6 may-have; TX_E03 held-for-progression; C11 never-dispensed; BC12_C1 missing pole; BC14_C1 naive≠never. Not a selected editor.

Artifact: `artifacts/n2s-g8-simult-mix-panel.json`  
Runner: `n2s_lab/n2s_g8_simult_mix.py` · `scripts/run_g8_simult_mix.sh`  
G3–G7 artifacts **not overwritten**. `ground()` unchanged.

## Goal 9 (2026-09-11) — Dual S-admin-blocked

New wrapper on frozen Goal 8, **not** an in-place G3–G8 retune. Goal 5 leftover shape `S-admin-blocked`:

1. Extractor marked first-line `planned`, but the note is refractory and now considering a next line → `administration_status=given` (planned token is the next line).
2. Fail vs never-dispensed quotes → copy into contradiction cues; clear U if the uncertainty cue is that never-dispensed span.

**Do not rewrite `ground()`.** Truly planned first-line (E4/I3) and neutropenia hold (T4) stay NOT_SATISFIED.

| | P-bind | P-neg leak | n=108 miss |
|--|--|--|--|
| Goal 8 simult | 3/7 | 0 | 8 |
| **Goal 9 admin** | **4/7 (I2 added)** | **0** | **6** |

Fixed vs G8: I2 SATISFIED; C11 CONTRADICTION. Design P-xspan still 10/10. `specificity_holds=true`. `heldout_leak=false`. **No new flips vs Goal 8.**

Leftover 6: T2 too-early; U2/U6 may-have; TX_E03 held-for-progression; BC12_C1 missing pole; BC14_C1 naive≠never. Not a selected editor.

Artifact: `artifacts/n2s-g9-admin-blocked-panel.json`  
Runner: `n2s_lab/n2s_g9_admin_blocked.py` · `scripts/run_g9_admin_blocked.sh`  
G3–G8 artifacts **not overwritten**. `ground()` unchanged.

## Goal 10 (2026-09-11) — leftover-shape prevalence (read-only)

Frozen Goal 9 on Dual-fill **n=108**. Tags the six leftover mechanisms on **every** note, including notes that already match gold. **No new repair.** Does not rerun Goal 5.

| Shape | library | miss | match | Recurrence |
|--|--|--|--|--|
| too-early | 1 | 1 (T2) | 0 | one-off |
| may-have | 3 | 2 (U2, U6) | 1 (BC_U2) | weak pair; 1 already correct |
| held-for-progression | 1 | 1 (TX_E03) | 0 | one-off |
| response-only | 6 | 2 (BC12_C1, BC14_C1) | 4 | library recurs; 4 already match via prior cells |
| naive≠never | 1 | 1 (BC14_C1) | 0 | one-off |

`shapes_that_justify_new_translate_cell=[]`. Design P-bind **7/7 match gold**. n=108 miss still 6. `v1_freeze_recommended=true`.

Artifact: `artifacts/n2s-g10-leftover-census.json`  
Runner: `n2s_lab/n2s_g10_leftover_census.py` · `scripts/run_g10_leftover_census.sh`  
G3–G9 artifacts **not overwritten**. `ground()` unchanged.

## v1 freeze (2026-09-11) — Translate intervention map

**Frozen here.** Six Dual Translate cells, controlled collateral, leftover Dual-wrong notes are one-offs or not a new Translate phenotype.

| Cell | Intervention | Design bind | n=108 miss after |
|--|--|--|--|
| G3 gated promote | asserted failure markers | 2/7 | — |
| G4 mixed-span | simultaneous 1L → X | 2/7 | 27 |
| G6 never-quote | never/no-prior quotes → cues | 2/7 | 19 |
| G7 restage | hedge-clear + stable-seq undo | 3/7 | 12 |
| G8 same-time mix | notes-undo + continue-now | 3/7 | 8 |
| G9 admin-blocked | planned-after-refractory + never-dispensed | **4/7 repaired; 7/7 match** | **6** |

P-neg leak 0 after G3. Design P-xspan 10/10 after G6. Held-out SATISFIED leak false. No new flips on G6–G9.

**Not a selected editor.** `locator_chose_repair_class=false`. `is_selected_editor=false`. Not automatic correction. Do not retune G3–G9 from leftover IDs. Do not rewrite `ground()` to fit naive. Do not jump to CAA / frozen-d / GUI unless a new question.

## Original Q closed (2026-09-11) — mapping + extract-signature selection

The leftover treadmill is stopped. This measurement closes the question this pass set out to answer. **No new repair cell.**

| Original part | This-pass answer |
|--|--|
| Map recurring Dual failures to corrections | **Partial yes** — six Translate cells, Dual analog n=108 |
| Repair without breaking other notes | **Yes on measured arms** — P-neg 0; no new flips G6–G9 |
| Could that map support selection? | **Yes at extract-signature depth** — selector miss **6 = apply-all**; chose G9 on **2/108** (not apply-all in disguise); oracle agreement **79.6%** |
| Locator chooses Encode vs Compute vs Translate | **No** — Encode none; Compute CLOSED |
| Automatic correction | **No** |

Raw Dual miss **30/108** → selector **6/108** = always-G9 **6/108**. Selector lost 0 apply-all repairs. Design P-bind and P-xspan still all match under the selector. Nested wrappers: selecting G6 still runs G3+G4+G6. Not independent editors.

Artifact: `artifacts/n2s-v1-selection-contrast.json`  
Runner: `n2s_lab/n2s_v1_selection_contrast.py`  
G3–G10 artifacts **not overwritten**. `ground()` unchanged. **Stop.**

Walk-back (same day): nested wrappers ≠ independent A vs B vs C vs abstain. V1 demonstrated **mapping**, not class selection. Encode/Compute/locator/auto-correct still open.

## Map viewer :8263 (2026-09-11) — canonize nested approaches

Read-only workbench for the frozen Dual Translate stack. **Not** an expansion of locator `:8260`. **Not** Detect→route→fix.

Open: http://127.0.0.1:8263/  
Run: `cd cxr-evidence-grounding-lab && ./scripts/run_map_viewer.sh`

Each clinical note shows the nested ladder **none → G3 → G4 → G6 → G7 → G8 → G9**. Deeper includes earlier. Highlight = extract-signature depth; dashed = shallowest match; blue bar = that cell actually changed the extract. Filters: raw Dual-wrong / leftover after G9 / design P-bind / selector ≠ none.

| | |
|--|--|
| n | 108 Dual-fill |
| raw Dual-wrong | 30 |
| leftover after G9 | 6 (T2, U2, TX_E03, U6, BC12_C1, BC14_C1) |
| Claim | mapping, not selection |

Artifact: `artifacts/n2s-map-viewer-panel.json`  
Server: `map_viewer_server.py` · page `static/map-viewer.html`  
`ground()` unchanged. G3–G9 not mutated. No new Translate cell.

## Curriculum + PDF freeze (2026-09-11)

Beginner track **MAP-00…09**: aim, Dual backend, nested cells, four walked notes, leftovers, mapping-not-selection, replicate book. Honest ceiling: not auto-correct; not class selection.

| | |
|--|--|
| Track UI | http://127.0.0.1:8264/ |
| PDF | `notes/CXR-Mapping-Curriculum.pdf` (18pp) |
| Modules | `curriculum-mapping/modules/` |
| Replay | `curriculum-mapping/snippets/replay_ladder.py` |

Run: `./scripts/run_mapping_curriculum.sh` · rebuild PDF: `curriculum-mapping/modules/build-mapping-pdf.sh`

This **is** the freeze write-up for this pass. Class B parked. No G11. `:8260` untouched.

PDFs live in `notes/` next to `progress-notes.pdf` (same split as before):

| PDF | Role |
|-----|------|
| `notes/mapping-notes-simple.pdf` | Walkthrough (like `progress-notes-simple.pdf`) |
| `notes/mapping-code-walkthrough.pdf` | Runnable snippets (like `N2S-CLI-Walkthrough.pdf`) |
| `notes/CXR-Mapping-Curriculum.pdf` | MAP-00…09 (like RepEng curriculum PDF) |
| `notes/mapping-notes.pdf` | Full freeze record (like `progress-notes.pdf`) |

Index: `notes/MAPPING-PDFS.md`. Rebuild: `notes/build-mapping-pdfs.sh`.
