# Phase-14 protocol — conditional / gated activation steer

**Status:** **FROZEN** 2026-08-25 — gate **14 NO**; live record in `notes/progress-notes.pdf`; soft negative on τ+α=4 gated recipe; not 12C.  
**Builds on:** Ph10 vector; Ph11 PARTIAL; Ph12A YES; Ph12B NO; Ph13 YES (soft geometry).  
**Not:** Phase-12C blanket adaptive α; new steering vector; SAE; reopening Ph11–13 gates.

## Live record (brief)

| Case | Role | cos@1.00 | STEER | Base X → final X | Notes |
|------|------|----------|-------|------------------|-------|
| BC14_E1 | course | −0.699 | ON | true→true | margin 4.38→2.88; no flip |
| BC14_E2 | course | −0.397 | OFF | true→true | outside τ |
| BC14_E3 | course | −0.497 | OFF | true→true | outside τ |
| BC14_C1 | contra | −0.495 | OFF | false→false | extract missed X (gold CONTRADICTION) |
| BC14_C2 | contra | −0.547 | OFF | false→false | extract missed X |
| BC14_N1 | no-fail | −0.130 | OFF | false→false | OK |

- `contra_all_steer_off`: **True** · `n1_ok`: **True** · `utility_flips`: **0**  
- Gate **NO** (no course flip under STEER ON).  
- Soft: Ph13 τ band barely transferred (1/3 course ON); α=4 insufficient on the one ON case.  
Artifact: `artifacts/phase14-gated-panel.json`.

## Question

> Can a **pre-declared geometry gate** (cosine of baseline repair-commit hidden to the frozen Ph10 direction @ layer 1.00) decide **when** to apply the same Ph10 activation steer (α=4), so that course false-X cases still get help while **true-contradiction controls are not steered**?

**Why now:** Ph12B showed unguarded α=4 can suppress BC12_C1. Ph13 showed course vs contradiction commits occupy **non-overlapping** cos@1.00 bands on the discovery slice. Phase-14 tests whether that geometry supports a **conditional** intervention — the intended scientific conclusion of the Ph9–13 arc (not open-ended strength control).

## Frozen inheritance

| Item | Value |
|------|--------|
| Model | `Qwen/Qwen2.5-7B-Instruct` |
| Vector | Ph10 BC_E2−BC_E1 @ layers 0.75 / 1.00 |
| Steer α when gate ON | **4** (same as Ph11 primary; not α panel) |
| When gate OFF | **no** activation steer (baseline repair commit stands) |
| Power | Prefer **240 W** (`nvidia-smi -pl 240`) |

## Pre-declared gate (frozen before any 14 live)

From Ph13 discovery (`artifacts/phase13-geometry-panel.json`):

| Class | cos@1.00 range |
|-------|----------------|
| course false-X | [−0.724, −0.700] |
| true contradiction | [−0.679, −0.554] |
| Mid-gap | **τ = −0.690** |

**Rule (runtime — gold not used):**

1. Baseline repair + commit trace (no steer) → record `cos_1.00`.  
2. If `cos_1.00` **≤ τ** (−0.690) → **STEER ON** (course-like / more aligned with Ph10 false-X direction).  
3. If `cos_1.00` **> τ** → **STEER OFF**.  
4. If no usable commit → **STEER OFF**; case unscored for utility.

τ is the midpoint of the Ph13 gap. Soft pilot threshold — **not** a fitted universal law. Soft claim must say τ was fixed from Ph13 before 14 held-out.

## Held-out set (new wording — not in Ph9–13 panels)

`data/heldout-phase14.json` — **draft**; wait wording accept.

| ID | Role | Gold |
|----|------|------|
| BC14_E1–E3 | generalization_target (course response→progression) | SATISFIED |
| BC14_C1–C2 | control_contradiction | CONTRADICTION |
| BC14_N1 | control_no_failure | NOT_SATISFIED |

**Not reused:** BC11_*, BC12_*, BC_E1/E2/C1 texts (those informed vector / Ph13 bands).

## Per-case protocol

1. Extract → analyze → verify → repair prompt (same as Ph9B/10).  
2. Baseline: `generate_trace` → commit; record X, margin, `cos_1.00`.  
3. Apply gate vs τ.  
4. If STEER ON: activation steer α=4 (same hooks as Ph10/11); record steered X, margin, verdict.  
5. If STEER OFF: final = baseline.  
6. Unload between cases. Prefer 240 W.

## Gate (14)

**14 YES** only if all hold:

1. **Control safety (contradiction):** every `control_contradiction` has **STEER OFF**  
   (i.e. baseline `cos_1.00` > τ).  
   *Rationale:* Ph12B showed α=4 can kill a true contradiction when steered; do not rely on “steered but X held.”
2. **Control safety (no-failure):** BC14_N1 final X=false (STEER OFF preferred; if ON, steered X must stay false).  
3. **Utility:** ≥ **1** `generalization_target` with baseline X=true, **STEER ON**, and steered X=false.  
4. **Direction (soft report, not hard fail):** among STEER-ON course targets, steered margin ≤ baseline margin (moves toward false or flat).

**14 NO** if any contradiction control is STEER ON, or zero course flips under STEER ON, or N1 ends X=true.

### Soft metrics (report; not in YES/NO)

- Gate–gold agreement: course→ON / contra→OFF rates on held-out.  
- How many course targets were STEER OFF (geometry miss / non-transfer of τ).

## Soft claim (if YES)

On a small held-out, a **fixed** cosine gate derived from Ph13 discovery can withhold steer from contradictions while still flipping ≥1 course false-X at α=4.  
Does **not** prove adaptive α, universal geometry, or production CXR safety.

## Soft claim (if NO)

Either τ / Ph13 bands do not transfer, or gated α=4 still lacks utility, or controls enter the STEER-ON region. Soft negative on this conditional recipe — freeze; do not iterate α inside 14 without a new protocol.

## Deliverables

- `docs/PHASE14-PROTOCOL.md` (this file)  
- `data/heldout-phase14.json` (draft wording)  
- After accept: `n2s_lab/phase14_gated.py` + `run_phase14.py`  
- After live: `artifacts/phase14-gated-panel.json`  

## Out of scope

- Blanket adaptive α / margin→α maps (12C)  
- Retuning τ on the held-out after seeing results  
- New vector; multi-α panel on every case  
- Overwriting Ph1–13  

## Wait

Wording **accepted** 2026-08-25. Live started under power limit **240 W**.
