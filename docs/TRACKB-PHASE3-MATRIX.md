# Phase 3 — Encode × Compute × Translate matrix (v1)

**Status:** ran 2026-09-10 · **read-only** frozen artifacts · no GPU · `:8260` stays **FROZEN**  
**Forward:** [TRACKB-PHASE2-TRANSLATE-CONTRAST.md](./TRACKB-PHASE2-TRANSLATE-CONTRAST.md)  
**Reverse:** [TRACKB-PHASE2-COMPUTE-REVERSE.md](./TRACKB-PHASE2-COMPUTE-REVERSE.md)  
**Locator:** [TRACKB-LOCATOR-V1.md](./TRACKB-LOCATOR-V1.md)

**Not this:** invent an Encode editor · reopen frozen-`d` · Dual_full Phase-7 rescore · auto-correct · GUI expansion · `ground()` rewrite

Phase 1 routed. Phase 2 showed a Translate boundary repair on FOLFOX and that existing Compute tools do not reproduce it. Phase 3 puts those facts on one **case × surface** table.

```text
PHASE 1 LOCATE     done (:8260 v1)
PHASE 2 CORRECT    done (Translate contrast + Compute reverse)
PHASE 3 THESIS     ← here: matrix of probes, routes, contain, repair-if-any
```

Repair editors are **not** filled in to make three boxes look symmetric. Encode = detect only. Compute editor CLOSED. Translate repair is FOLFOX Schema extract only.

---

## Matrix (locator cases)

| Case | Encode | Compute | Translate snapshot | Route | Contain | Repair |
|------|--------|---------|--------------------|-------|---------|--------|
| **EX_TEMPORAL_FOLFOX** | pass L8 | pass (0 lost) | 7B analog wrong_AUTO | translate | REVIEW | **quote-promote → SATISFIED** |
| **EX_CONTRA** | pass L8 | fail (2 lost) | 7B analog correct_AUTO | compute | REVIEW | Translate no-op; frozen-`d` PARTIAL, **editor CLOSED** |
| **BC_E1** | unknown (no trace) | unknown | Dual_full wrong_AUTO | translate | REVIEW | **not applied** (do not rescore Phase-7) |
| **T1 / T3 / C1 / C4** | unknown | unknown | unknown | incomplete | none | none |

CLI: `./scripts/run_phase3_matrix.sh panel` (CPU).

---

## Result (2026-09-10)

Thesis (`artifacts/n2s-phase3-matrix-panel.json`):

- `matrix_assembled`: **true**
- `n_located`: **3** · `n_incomplete`: **4** · `n_boundary_translate_repair`: **1**
- `encode_editor_invented`: **false**
- `compute_editor_invented`: **false**
- `locator_chose_repair_class`: **false**
- Phase 2 forward + reverse still hold

**Defensible:** on this frozen set, only FOLFOX has a demonstrated boundary Translate repair; CONTRA is Dual-correct with uncleared n2n loss; four notes are incomplete; Encode has no editor.  
**Not defensible:** three-surface auto-correct; locator causal selection; Dual_full BC_E1 was repaired; T1 should have been traced.

---

## Locked

- n2n frozen-`d` CLOSED
- Encode editor does not exist — do not manufacture one
- U-C SAE / U-D / α-chase
- Dual_full Phase-7 rescore
- `:8260` GUI expansion
- auto-correct Detect→route→fix
