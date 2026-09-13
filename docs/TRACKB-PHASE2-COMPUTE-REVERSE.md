# Phase 2 — Compute reverse contrast (not a new editor)

**Status:** ran 2026-09-10 · **read-only** frozen artifacts · no GPU · `:8260` stays **FROZEN**  
**Forward:** [TRACKB-PHASE2-TRANSLATE-CONTRAST.md](./TRACKB-PHASE2-TRANSLATE-CONTRAST.md)  
**Not this:** invent a Compute editor · reopen frozen-`d` · α-chase · Encode editor · auto-correct · `ground()` rewrite

The Translate contrast showed a boundary intervention can repair FOLFOX Dual and leave analog CONTRA Dual + 2 lost alone. Reverse asks whether **existing Compute tools** repair that FOLFOX miss — they must not, if the two surfaces are distinct.

```text
FORWARD   Translate quote-promote → FOLFOX SATISFIED; analog CONTRA Dual held; 2 lost unchanged
REVERSE   G5 REVIEW + closed frozen-d → FOLFOX Dual still NOT_SATISFIED; CONTRA loss not cleared
```

REVIEW remains **containment**, not a correction. Frozen-`d` stays **CLOSED** (L12+L16+L20 α=2 PARTIAL).

---

## Design (no new editor)

| Tool | Already ran | FOLFOX (translate-routed) | CONTRA (compute-routed) |
|------|-------------|---------------------------|-------------------------|
| G5 n2n-lost→REVIEW | `n2s-nn-lost-review-panel.json` | 0 lost → **does not fire** | 2 lost → REVIEW contain |
| Frozen-`d` furthest | `n2s-nn-layer-patch-l12l16l20-panel.json` α=2 | 0 lost, Dual **NOT_SATISFIED** | 2→1 lost PARTIAL, Dual CONTRADICTION |
| Translate quote-promote | `n2s-phase2-translate-panel-clean.json` | Dual **SATISFIED** | Dual CONTRADICTION; **2 lost unchanged** |

**Pass if:** Compute tools do not repair FOLFOX Dual miss, and Translate does not clear CONTRA lost-steps.  
**Fail if:** G5 or frozen-`d` makes FOLFOX Dual SATISFIED, or we invent a new Compute editor to force that.

CLI: `./scripts/run_phase2_compute_reverse.sh panel` (CPU).

---

## Result (2026-09-10)

| Check | Result |
|-------|--------|
| G5 on FOLFOX | no fire (`no_lost_steps`) — Translate miss unfixed |
| Frozen-`d` α=2 on FOLFOX | Dual still **NOT_SATISFIED**; 0 lost |
| Translate on FOLFOX | **SATISFIED** (quote-promote; Schema extract) |
| G5 on CONTRA | REVIEW contain; 2 lost uncleared |
| Frozen-`d` α=2 on CONTRA | PARTIAL 2→1 lost; Dual still CONTRADICTION |
| Translate on analog CONTRA | CONTRADICTION no-op; **2 lost unchanged** |

Thesis (`artifacts/n2s-phase2-compute-reverse-panel.json`):

- `reverse_contrast_holds`: **true**
- `compute_editor_invented`: **false**
- `d_patch_rerun`: **false**
- `locator_chose_repair_class`: **false**

**Defensible:** existing Compute tools do not reproduce the FOLFOX Translate repair; Translate does not clear CONTRA n2n loss.  
**Not defensible:** we now have a Compute editor; `:8260` chose the right class; REVIEW repaired CONTRA; frozen-`d` should be reopened.

---

## Locked

- n2n frozen-`d` editor CLOSED
- U-C SAE / U-D / α-chase
- Dual_full Phase-7 rescore
- `:8260` GUI expansion
- Encode editor (does not exist — do not manufacture one)

Phase 3 Encode × Compute × Translate matrix stays later.
