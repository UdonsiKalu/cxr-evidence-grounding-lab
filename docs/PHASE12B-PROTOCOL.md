# Phase-12B protocol — second held-out strength / margin check

**Status:** **FROZEN** 2026-08-25 — gate **NO**; live record in `notes/progress-notes.pdf` § Phase 12B.  
**Builds on:** [PHASE12-PROTOCOL.md](./PHASE12-PROTOCOL.md) (12A frozen; mechanism YES)  
**Prerequisite:** Ph10 gate YES; Ph11/12A frozen; do **not** overwrite Ph1–12A artifacts.

## Reboot / power limit (ops — not science)

**Do not reboot** to “activate” the 270 W power limit.  
`nvidia-smi -pl 270` applies **immediately**. Current limit is already **270 W**.

A reboot typically **resets** the limit to 370 W (persistence mode off). After reboot, re-run:

```bash
sudo nvidia-smi -pl 270
```

or Apply **Research 270 W** on `:8256` → **0c · GPU**.

## Question

Ph12A (on Ph11 notes) suggested: same Ph10 direction still moves E3; higher baseline margin may need higher α.

Ph12B asks on a **new** held-out set (never used in Ph9–12A):

> Among **new** baseline false-X course targets, does the **same frozen vector** still move margin toward `false`, and does first-flip α track a **pre-declared** margin band?

**Not:** adaptive α controller; new vector; reopening Ph11 universal gate.

## Pre-declared margin bands (frozen before any 12B run)

After baseline repair (same job), classify each target with baseline X=true:

| Band | Baseline margin *m* | Prediction |
|------|---------------------|------------|
| **LOW** | *m* ≤ 1.5 | Flip by **α=4** |
| **HIGH** | *m* ≥ 2.0 | **No** flip at α=4; flip by **α=8** |
| **MID** | 1.5 < *m* < 2.0 | Report only (no pass/fail for band) |

Bands chosen from Ph12A: E2 *m*=1.125 (LOW), E3 *m*=2.375 (HIGH). Soft pilot — not a fitted law.

## Frozen intervention

| Item | Value |
|------|--------|
| Vector | BC_E2 − BC_E1 @ 0.75/1.00 (same Ph10) |
| Model | `Qwen/Qwen2.5-7B-Instruct` |
| α panel (targets) | **{4, 8}** (minimal; enough to test bands) |
| α (controls) | **4** only |

## Held-out set

`data/heldout-phase12b.json` — **new** notes; not in Ph9–10 vector or Ph11/12A panels.

| ID | Role | Gold |
|----|------|------|
| BC12_E1–E3 | generalization_target (course) | SATISFIED |
| BC12_C1 | control_contradiction | CONTRADICTION |
| BC12_N1 | control_no_failure | NOT_SATISFIED |

## Per-case protocol

1. Baseline repair once.
2. Steered at each α in panel.
3. Record X, margin, verdict; assign band from baseline margin if baseline X=true.

Prefer run under **270 W** power limit (performance-only).

## Gate (12B)

**12B YES** only if all hold:

1. **Controls @ α=4:** BC12_C1 steered X=true; BC12_N1 steered X=false.
2. **Direction:** every target with baseline X=true has steered margin at α=8 **≤** baseline margin (moves toward false or flat — not worse).
3. **Band hits:** among LOW/HIGH targets (skip MID and baseline-X=false), **≥ 1** correct band prediction **and zero contradictions of the band rule** (e.g. LOW that fails α=4 flip = miss; HIGH that flips at α=4 = unexpected but not auto-fail if α=8 still OK — count as band miss).

Soft: if no target lands in LOW or HIGH, gate = **NO** (nothing to test; report only).

## Soft claim

Second held-out pilot; n≤3 targets. Does **not** prove adaptive α. Does **not** reopen Ph11 universal fixed-α=4 claim.

## Deliverables

- `docs/PHASE12B-PROTOCOL.md` (this file)
- `data/heldout-phase12b.json`
- `run_phase12b.py` + `n2s_lab/phase12b_heldout.py` (after wording accept)
- `artifacts/phase12b-heldout-panel.json` (after live)

## Out of scope

- Adaptive α / representation→α mapping
- Overwriting Phase 1–12A
- Re-using BC11_* texts

## Wait

Wording **accepted** 2026-08-25. Live complete under power limit **270 W**.

## Live 12B (2026-08-25, HF 7B, frozen vector, α∈{4,8})

| Case | Band | Base marg. | α=4 | α=8 | Band rule |
|------|------|------------|-----|-----|-----------|
| BC12_E1 | HIGH | +4.38 | +3.00 X=true | +1.13 X=true | miss (no flip @8) |
| BC12_E2 | HIGH | +3.38 | +1.75 X=true | 0.0 X=true | miss (no flip @8) |
| BC12_E3 | HIGH | +2.88 | +1.88 X=true | +0.38 X=true | miss (no flip @8) |
| BC12_C1 | — | 0.0 | **X true→false** | — | **control fail** |
| BC12_N1 | — | −15.4 | X false held | — | OK |

**Gate: NO** — (1) contradiction control BC12_C1 suppressed at α=4; (2) all three targets HIGH; direction toward false OK but **no flip by α=8**; band 0 hits / 3 misses.

**Reading:** On this second held-out, Ph12A’s “HIGH → flip by α=8” did **not** transfer. Direction still has leverage (margins fell), but α=8 was insufficient for these higher baselines (+2.9…+4.4). Stronger steer also broke the true-contradiction control — strength is not free. Soft negative on band transfer; do **not** jump to adaptive α without a new protocol.
