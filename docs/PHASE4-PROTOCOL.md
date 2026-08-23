# Phase-4 protocol (frozen before any new notes)

**Status:** wording accepted; live A–D complete; PDF frozen 2026-08-17 (`notes/progress-notes.pdf`).  
**Does not replace** M1–M3 or Phase-1–3 records. B′ is not the default.

Plain copy: `notes/progress-notes.pdf` § Phase-4.

## Research question

When NL analysis already has contradiction or uncertainty, how often does that distinction survive **formalization (D)** and **categorical answer (B)**, on wording the prompts were not tuned on?

Not: whether C is more accurate. Not: a new failure class.

## 1. Object

Semantic **preservation**, conditional on the analysis containing the target distinction.

Two **separate** rates (do not collapse B and D):

Let \(S\) = cases whose gold is CONTRADICTION or UNCERTAIN (all Phase-4 notes).

Let \(A_{\text{human}}\) = subset of \(S\) where a human marks that the free-text analysis states the gold distinction (conflict vs unresolved uncertainty).

- **B loss rate** = fraction of \(A_{\text{human}}\) whose Condition B verdict ≠ gold (`analysis_output_misalignment`).
- **D loss rate** = fraction of \(A_{\text{human}}\) whose Condition D is `representation_loss` (analysis had it; structure dropped it).

Also record the **regex screen** (`analysis_has_distinction` cue flags) but **do not** use it as the headline denominator. Regex ≠ “analysis is correct.”

Match-gold / n remains a secondary table, not the claim.

## 2. Controls

- Frozen Phase-1/2/3 **A–D prompts**. No retune after seeing Phase-4 misses.
- **B′ off** (ablation only; not this run).
- Temperature **0**. Do not resample the same note. Replication is **new wording**, not extra draws.
- Gold **not** fed to models. Do not change gold after seeing output.
- Do not overwrite Phase-1/2/3 artifacts.

## 3. Data (wording accepted; live A–D)

Second independently held-out set in `data/heldout-phase4.json`. **Same concepts** as C1–C3 and U1–U4. **No C4/C8 analog.** Not merged into `data/cases.json`.

| ID (planned) | Analog | Gold | Type |
|--------------|--------|------|------|
| C9 | C1 | CONTRADICTION | same-day reverse |
| C10 | C2 | CONTRADICTION | never vs cycles |
| C11 | C3 | CONTRADICTION | referred fail vs never given |
| U9 | U1 | UNCERTAIN | pending confirmatory scan |
| U10 | U2 | UNCERTAIN | unknown outside therapy/outcome |
| U11 | U3 | UNCERTAIN | mixed / deferred decision |
| U12 | U4 | UNCERTAIN | unresolved which-line (adjuvant vs metastatic) |

**n = 7.** Do not pad with a C4/C8 analog. Identity/time (same regimen progressed vs still responding) waits for a later protocol.

Not in this set: leftover E/I/T, CXR oncology-eval dumps, C8-class notes.

**Order:** protocol frozen → wording accepted → live run (`run_phase4.py`).

## After this freeze

Live runner: `python3 run_phase4.py`. Do not overwrite Phase-1/2/3 artifacts. B′ off.

## 4. Models

Same three, same tags:

- `llama3:8b-instruct-q4_0` (bench)
- `mistral:instruct` (bench)
- `qwen2.5-coder:32b` (capacity check)

No API ceiling in Phase-4. No per-model prompt variants.

## 5. Falsification / stop (declared before seeing cases)

B and D are **not** one score.

- If **D loss rate = 0 on all three models** on this set (among \(A_{\text{human}}\)), evidence for a reproducible **formalization-loss** phenomenon is **weakened**.
- If **D loss recurs on more than one model** despite frozen prompts and new wording, the formalization-preservation hypothesis **gains support**.
- If **B loss = 0 on all three** but D loss remains, that is a D-only finding; do not call it “preservation solved.”
- If **D loss = 0** but B loss remains, that is a B-only finding; do not call formalization-loss falsified.

C4/C8 identity/time is **out of scope**; a miss of that class is not a Phase-4 stop.

## 6. What we will not do in Phase-4

Iterate B′. Fix C8. Add extract fields. Touch CXR production. Grow N for its own sake. Change temperature. Peek at model output while drafting notes.
