# Temporal-family **test** set — freeze

**Frozen:** 2026-09-03 · `data/temporal-family-test.json` (n=12, `TFT_*`)  
**Split:** `test` · `held_out: true` · `frozen: true`  
**First scored:** 2026-09-04 · Qwen Dual_full **wrong_AUTO=0** — [TRACKA-TEMPORAL-FAMILY-FREEZE.md](./TRACKA-TEMPORAL-FAMILY-FREEZE.md)

## Rule

**Do not open case evidence** while designing G3, gate changes, or RepEng interventions.  
Design and Track B work on **`temporal-family-dev.json` only**.  
Score this set with Phase-7 Dual **only after** an intervention is frozen.

**After first score:** do not redesign from misses on this open; log and stop or open a new protocol.

## Wiring (score later)

```bash
python3 run_phase7.py --set temporal-test
python3 run_auto_contract_score.py --from-artifacts temporal-test --paths Dual_full,D_full,C_full
```

## Coverage (ids only)

| Id | Role | Expected |
|----|------|----------|
| TFT_E1…E5 | targets (response→failure family) | SATISFIED |
| TFT_U1 | uncertain | UNCERTAIN |
| TFT_T1, TFT_T2 | toxicity / not failure | NOT_SATISFIED |
| TFT_C1, TFT_C2 | true contradiction | CONTRADICTION |
| TFT_N1, TFT_N2 | no failure | NOT_SATISFIED |

No wording overlap with `TF_*` / `BC_*` in the dev set (by design).
