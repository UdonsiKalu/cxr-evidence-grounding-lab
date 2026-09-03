# Track B — DEV expand panel (pre-model wording)

**Status:** wording frozen 2026-09-03 **before** HF runs  
**File:** `data/temporal-family-dev-expand.json` (n=30, `TX_*`)  
**Not:** `temporal-family-test.json` (still sealed) · not merged into `temporal-family-dev.json`  
**G3:** untouched

## Predeclared rules

- Class A = gold SATISFIED temporal-change **and** model `contradiction.present=false`
- Class B = gold CONTRADICTION **and** model X=true
- **Fit** only if n_A ≥ 3 and n_B ≥ 2
- **Do not fit** if n_A ≤ 2 (nearly empty)
- Wording was **not** iterated against 7B/14B output

## Sequence

```bash
./.venv-phase9/bin/python run_trackb_falsex.py --selftest
./.venv-phase9/bin/python run_trackb_falsex.py --expand-sequence
```

1. HF **7B** collect on the frozen 30  
2. If enough A/B → L20 family vector + α-sweep on 7B  
3. Else **stop 7B fit** → HF **14B** collect → patch-depth (scaled layer indices) → fit semantic vector only if both correct and failing temporal-change exist
