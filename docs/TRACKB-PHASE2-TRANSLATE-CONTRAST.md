# Phase 2 — Translate contrast (not auto-correct)

**Status:** Exp1 + Dual reconcile + **clean pairing** + **Compute reverse (read-only)** ran 2026-09-10 · `:8260` v1 stays **FROZEN** · no GUI  
**Goal:** test whether first-break **routing** has **intervention-selection value**.  
**Not this:** automate Detect→route→fix · reopen n2n frozen-`d` · invent Encode/Compute editors · Dual_full Phase-7 rescore · silent AUTO

Locator v1 **contains**. Phase 2 asks whether a **translation-side** intervention can make the symbolic verdict match gold **without** patching hidden states.

```text
PHASE 1 LOCATE     done (:8260 v1)
PHASE 2 CORRECT    done (Translate contrast + Compute reverse)
PHASE 3 THESIS     done (read-only matrix) — [TRACKB-PHASE3-MATRIX.md](./TRACKB-PHASE3-MATRIX.md)
```

REVIEW remains **containment**, not a correction. Success = verdict matches gold, traces unchanged, controls held.

---

## Why Translate / FOLFOX first

| Case | Encode | Compute | Translate analog | Gold |
|------|--------|---------|------------------|------|
| **EX_TEMPORAL_FOLFOX** | pass | pass (0 lost) | AUTO **NOT_SATISFIED** (`wrong_AUTO`) | SATISFIED |
| **EX_CONTRA** | pass | fail (2 lost) | AUTO CONTRADICTION (`correct_AUTO`) | CONTRADICTION |

FOLFOX is encode+compute OK, Dual-shaped 7B extract→ground→rule wrong. `extract_x=false` (not a false contradiction); rule `NOT_SATISFIED` means some atom A–D is false after grounding (`FIRST_LINE_THERAPY_FAILED`).

CONTRA Dual analog is **already correct**. The cross-control is **not** “Dual stays broken.” It is:

- same Translate intervention must **not** clear CONTRA lost-steps
- Dual on CONTRA must stay AUTO CONTRADICTION (control held)
- we do **not** treat Dual/REVIEW as having repaired the compute-routed case

---

## Experiment 1 (FOLFOX)

```text
FOLFOX frozen traces
       ↓
Translation intervention  (boundary only: extract→ground→rule / Dual)
       ↓
Symbolic verdict → gold SATISFIED?   YES / NO
       ↓
Encode L8 + compute lost-steps unchanged?
CONTRA Dual still CONTRADICTION?  lost-steps still lost?
```

**Invariants:** `hidden_state_patched=false` · no Dual_full rewrite of the sealed Phase-7 panel · overlay/protocol only.

## Experiment 1 result (2026-09-10)

Recovered 7B extract (`intervention=none`, not a `d`-patch).

**FOLFOX A–D before:** A true, B true, **C false, D false** → AUTO NOT_SATISFIED.  
Cause: only outcome was `polarity=response` (“partial response”). Failure lived in **quotes** (“FOLFOX was discontinued for treatment failure”), which `ground()` does not read.

**Intervention:** `failure_evidence_from_extract_text` — promote quote/notes failure into an outcome polarity. Hidden states untouched.

| Case | Before | After | Frozen encode/compute |
|------|--------|-------|------------------------|
| FOLFOX | NOT_SATISFIED | **SATISFIED** (repaired) | pass / 0 lost unchanged |
| CONTRA | SATISFIED on *this* recover+ground | unchanged (no-op) | fail / 2 lost unchanged |

**CONTRA caveat (resolved):** Schema+note recover CONTRA is SATISFIED because sequenced-temporal override fires. Frozen Dual analog is CONTRADICTION. Intervention did not cause that. See reconcile below — **do not change `ground()`**.

**Claim:** a boundary-only Translate intervention **can** correct FOLFOX analog wrong_AUTO without rewriting the net. Full locator-selection thesis **not** established. REVIEW still not a correction.

Artifacts: `artifacts/n2s-phase2-translate-extracts.json` · `artifacts/n2s-phase2-translate-panel.json`

---

## CONTRA Dual reconcile (2026-09-10)

User go: re-extract with **analog prompt** (`user=evidence`). Do **not** change `ground()` first.

| Snapshot | Prompt | Verdict | `grounded_X` |
|----------|--------|---------|--------------|
| Frozen analog (`n2s-nn-layer-patch-panel.json`) | `user=evidence` | CONTRADICTION | true |
| Exp1 recover | Schema+note | SATISFIED | false (reconnect override) |
| Analog recover | `user=evidence` | CONTRADICTION | true |

Diagnosis (`artifacts/n2s-phase2-contra-dual-reconcile.json`):

- `analog_matches_frozen`: **true**
- `prompt_explains_split`: **true**
- `ground_changed`: **false**

The mismatch was the **Schema+note recover prompt**, not two Dual engines and not a reason to rewrite `ground()`. Analog-prompt CONTRA matches frozen Dual.

Same intervention on analog extracts (CPU): **no-op** on both. Analog FOLFOX extract is empty (no quotes/outcomes) — quote-promote cannot fire. FOLFOX repair stays on Schema+note extract; Dual CONTRA control stays on analog extract.

CLI: `./scripts/run_phase2_translate.sh recover-analog`

**Do not:** change `ground()` · invent a Compute editor · treat Schema CONTRA SATISFIED as frozen Dual.

---

## Experiment 2 (same intervention on CONTRA)

Applied in Exp1 on Schema extracts: **no-op**. Frozen compute 2 lost unchanged.

Schema recover CONTRA SATISFIED is the reconnect override, **not** this intervention. Analog recover CONTRA is CONTRADICTION (matches frozen); intervention still no-op. Use analog CONTRA as the Dual control; do not use Schema CONTRA as frozen Dual.

| Expected if routing has selection value | Failure of the thesis |
|-----------------------------------------|------------------------|
| FOLFOX Dual-shaped verdict repairs; CONTRA compute loss remains | Translate “fix” also “repairs” CONTRA, or breaks CONTRA Dual, or FOLFOX only via neural patch |

Honest pairing for a later write-up (no GPU): Schema FOLFOX + quote-promote (repaired) vs analog CONTRA (CONTRADICTION, intervention no-op, 2 lost unchanged). Exp1 panel mixed Schema CONTRA in; do not treat that panel as Dual-matched.

---

## Clean pairing result (2026-09-10)

CPU panel from saved extracts. Did not overwrite Exp1. Did not change `ground()`. CLI: `./scripts/run_phase2_translate.sh panel-clean`

| Role | Snapshot | Before | After | Frozen encode/compute |
|------|----------|--------|-------|------------------------|
| FOLFOX repair | Schema+note | NOT_SATISFIED | **SATISFIED** (quote-promote) | pass / 0 lost unchanged |
| CONTRA Dual control | analog `user=evidence` | CONTRADICTION | CONTRADICTION (no-op) | fail / **2 lost unchanged** |

Thesis (`artifacts/n2s-phase2-translate-panel-clean.json`):

- `contrast_holds`: **true**
- `dual_matched_pairing`: **true**
- `locator_chose_repair_class`: **false**
- `routing_selection_value`: true only as **same-intervention contrast** (FOLFOX repaired; analog CONTRA Dual held; compute loss untouched). **Not** “`:8260` chose Translate therefore Translate is the right repair class.”

**Defensible:** a boundary Translate intervention can repair the FOLFOX analog miss and leave Dual-matched CONTRA + frozen compute loss alone.  
**Not defensible:** routing is causal localization; REVIEW is a correction; Encode/Compute editors exist.

Exp1 `n2s-phase2-translate-panel.json` stays the Schema-both record (CONTRA SATISFIED mismatch). Use the clean panel for Dual-matched contrast.

Then (later, not now): Encode editor still **does not exist** — do not manufacture one. Phase 3 matrix v1 is read-only — [TRACKB-PHASE3-MATRIX.md](./TRACKB-PHASE3-MATRIX.md).

**Compute reverse (2026-09-10):** read-only, no new editor — [TRACKB-PHASE2-COMPUTE-REVERSE.md](./TRACKB-PHASE2-COMPUTE-REVERSE.md). G5 + closed frozen-`d` do not repair FOLFOX Dual miss; Translate does not clear CONTRA 2 lost. `reverse_contrast_holds=true`. Frozen-`d` stays CLOSED.

---

## Locked

- n2n frozen-`d` editor CLOSED (L12+L16+L20 α=2 PARTIAL)
- U-C SAE / U-D
- α-chase / Dual_full Phase-7 rescore
- `:8260` GUI expansion
- auto-correct pipeline

Routing claims stay **non-causal**.
