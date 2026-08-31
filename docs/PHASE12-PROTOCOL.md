# Phase-12A protocol — margin regime (fixed vector, α panel)

**Status:** **FROZEN** 2026-08-25 — mechanism gate **YES**; live record in `notes/progress-notes.pdf` § Phase 12A.  
**Builds on:** [PHASE11-PROTOCOL.md](./PHASE11-PROTOCOL.md) (frozen; gate NO / partial)  
**Prerequisite:** Ph10 gate YES; Ph11 panel frozen; do **not** overwrite Ph1–11 artifacts.

## Question

Ph11 answered: *does the same frozen Ph10 vector at α=4 generalize?* → **partial** (E2 flip, E3 margin-only).

Ph12A asks:

> Among held-out **baseline false-X** course targets, does flip success with the **same frozen vector** track **baseline commit margin**, and does steered margin shift **monotonically** with a **pre-declared symmetric α panel**?

**Not:** E3-only retune, new vector, or claiming universal steering.

## Frozen intervention (unchanged from Ph10–11)

| Item | Value |
|------|--------|
| Vector | BC_E2 false-commit − BC_E1 true-commit @ 0.75, 1.00 |
| Model | `Qwen/Qwen2.5-7B-Instruct` |
| Stage | Repair path; activation_steer at commit; **no logit bias |

## Cases

From `data/heldout-phase11.json` (no new notes in 12A):

| ID | Role | α panel | Why |
|----|------|---------|-----|
| BC11_E2 | generalization_target | **{2, 4, 6, 8}** | baseline false-X; flipped at α=4 in Ph11 |
| BC11_E3 | generalization_target | **{2, 4, 6, 8}** | baseline false-X; margin-only at α=4 in Ph11 |
| BC11_C1 | control_contradiction | **4 only** | sanity vs Ph11 |
| BC11_C2 | control_contradiction | **4 only** | sanity vs Ph11 |
| BC11_N1 | control_no_failure | **4 only** | sanity vs Ph11 |

**Skip** BC11_E1 in 12A (baseline X already false — not in false-X regime).

## Per-case protocol

1. **Baseline** repair (no steer) once per case.
2. **Steered** repair at each α in case’s panel (same frozen vector).
3. Record: repair X, margin at commit, verdict, gold.

Unload model between loads (CLI runner).

## Gate (12A — mechanism, not universal fix)

**12A mechanism YES** only if all hold:

1. **Controls @ α=4:** BC11_C1/C2 steered X = **true**; BC11_N1 steered X = **false** (match Ph11).
2. **Margin monotone:** on BC11_E2 and BC11_E3, steered margin is **non-increasing** as α increases 2→4→6→8 (more steer toward false).
3. **Regime readout (report either way):** record baseline margin, margin at each α, first α where X flips false (if any).

**NOT** a YES gate for “fixed vector universal.” Ph11 negative generalization gate remains frozen.

## Soft claim

n=2 false-X targets; 4-point α panel; pilot mechanism only.

**Defensible:** In the tested E3 case, the Phase-10 direction continued to move the contradiction margin monotonically in the desired direction; E3 required greater intervention strength than E2 to cross the decision boundary.

**Do not claim:** all generalization failures are “margin-regime effects”; adaptive α from representation state (future study).

Ph11 α=4 partial and Ph12 strength panel are complementary. Does not reopen Ph11 universal gate.
## Deliverables

- `docs/PHASE12-PROTOCOL.md` (this file)
- `run_phase12.py` + `n2s_lab/phase12_margin_regime.py`
- `artifacts/phase12-margin-regime-panel.json`

## Out of scope

- Per-case vector rebuild
- E3-only α search
- Overwriting Phase 1–11 artifacts
- Phase-12B second held-out — see [PHASE12B-PROTOCOL.md](./PHASE12B-PROTOCOL.md) (draft)

## Runner

```bash
cd cxr-evidence-grounding-lab
.venv-phase9/bin/python run_phase12.py          # live
.venv-phase9/bin/python run_phase12.py --selftest
```

## Live 12A (2026-08-25, HF 7B, frozen vector)

| Case | Baseline margin | α=2 | α=4 | α=6 | α=8 | First flip |
|------|-----------------|-----|-----|-----|-----|------------|
| BC11_E2 | +1.125 | +0.38 X=true | **−0.50 X=false** | −1.50 | −2.38 | **α=4** |
| BC11_E3 | +2.375 | +1.75 X=true | +0.88 X=true | 0.0 X=true | **−0.75 X=false** | **α=8** |

Controls @ α=4 held (C1/C2 X=true; N1 X=false). Margins monotone on both targets.

**Gate: YES** — mechanism. Soft claim: on tested E3, same direction has leverage; greater α needed to cross zero (first flip α=8 vs E2 α=4). Does not reopen Ph11 universal gate. Not a claim that all failures are margin-regime.
