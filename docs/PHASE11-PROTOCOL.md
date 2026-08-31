# Phase-11 protocol — fixed-vector generalization

**Status:** **FROZEN** 2026-08-24 — gate **NO** (partial generalization; controls held).  
**Explorer:** `../cxr-evidence-grounding-lab-repeng/` on **:8256**  
**Builds on:** [PHASE10-PROTOCOL.md](./PHASE10-PROTOCOL.md) (frozen; gate YES)  
**Prerequisite:** `artifacts/phase10-bc-e1-panel.json` gate YES; 9B traces intact.

## Question

Does the **same** Phase-10 steering vector (BC_E2 − BC_E1 commit @ layers 0.75/1.00) at **α=4.0** generalize to **unseen** response→progression notes (gold SATISFIED) without suppressing true contradictions?

**Not:** recompute vector or α per case.

## Frozen intervention (from Phase 10)

| Item | Value |
|------|--------|
| Vector | BC_E2 false-commit − BC_E1 true-commit @ 0.75, 1.00 |
| α | **4.0** (frozen from Ph10 gate) |
| Model | `Qwen/Qwen2.5-7B-Instruct` |
| Stage | Repair path; activation_steer at commit; **no logit bias |

## Held-out set

`data/heldout-phase11.json` — **not** used in Ph9–10 vector construction.

| ID | Role | Gold |
|----|------|------|
| BC11_E1–E3 | generalization_target | SATISFIED (course) |
| BC11_C1–C2 | control_contradiction | CONTRADICTION |
| BC11_N1 | control_no_failure | NOT_SATISFIED |

## Per-case protocol

For each case: baseline repair (no steer) → steered repair (frozen vector + α=4).

Record: raw X, repair X, margin at commit, verdict, gold.

## Gate (11C)

**YES** only if all hold:

1. **Contradiction controls:** BC11_C1, BC11_C2 steered repair X = **true**.
2. **No-failure control:** BC11_N1 steered repair X = **false**.
3. **Generalization targets:** every BC11_E* with **baseline** repair X=true has **steered** X=false and verdict ≠ wrong AUTO CONTRADICTION.

If no target has baseline X=true, gate = **NO** (nothing to generalize; report only).

## Soft claim

Fixed direction from one contrast pair; n=3 targets + 3 controls. Pilot only.

## Deliverables

- `run_phase11.py` + `n2s_lab/phase11_generalization.py`
- `artifacts/phase11-generalization-panel.json`

## Out of scope

- Per-case vector or α retune
- Overwriting Phase 1–10 artifacts

## Live 11 (2026-08-24, HF 7B, frozen vector α=4)

| ID | Role | Baseline X | Steered X | Margin Δ | Verdict Δ |
|----|------|------------|-----------|----------|-----------|
| BC11_E1 | target | false | false | −2.1→−3.8 | already OK |
| BC11_E2 | target | **true** | **false** | +1.1→−0.5 | CONTR→**SAT** ✓ |
| BC11_E3 | target | **true** | **true** | +2.4→+0.9 | still CONTR ✗ |
| BC11_C1 | control | true | true | held | held ✓ |
| BC11_C2 | control | true | true | held | held ✓ |
| BC11_N1 | control | false | false | held | held ✓ |

**Gate: NO** — controls pass; **1/2** baseline-false-X targets fixed (E2 yes, E3 partial margin shift only).

**Reading:** fixed BC_E2−BC_E1 direction generalizes to at least one unseen course case (ALL/CNS) without breaking true contradictions; not universal across wording (CLL/Richter failed). Soft negative result on full generalization claim.

**Runner:** `python3 run_phase11.py` (requires Ph10 gate YES; `.venv-phase9`).

## Pilot complete — frozen

Do not re-tune α or recompute vector per case. Negative generalization gate is a valid frozen result.

## Deliverables

- `docs/PHASE11-PROTOCOL.md` (this file)
- `run_phase11.py` + `n2s_lab/phase11_generalization.py`
- `artifacts/phase11-generalization-panel.json`
- `data/heldout-phase11.json`
- `notes/progress-notes.pdf` § Phase 11
- `cxr-evidence-grounding-lab-repeng/` explorer on `:8256`
