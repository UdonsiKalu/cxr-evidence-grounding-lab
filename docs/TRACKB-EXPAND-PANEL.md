# Track B — DEV expand panel (pre-model wording)

**Status:** wording frozen **before** HF · 7B collect+fit **done** · **7B endpoint YES @ α=8** · **14B transfer once (behavioral)**  
**File:** `data/temporal-family-dev-expand.json` (n=30, `TX_*`)  
**Not:** `temporal-family-test.json` (still sealed) · not merged into `temporal-family-dev.json`  
**G3:** untouched

## Predeclared rules

- Class A = gold SATISFIED temporal-change **and** model `contradiction.present=false`
- Class B = gold CONTRADICTION **and** model X=true
- **Fit** only if n_A ≥ 3 and n_B ≥ 2
- Wording was **not** iterated against 7B/14B output

## Commands

```bash
./.venv-phase9/bin/python run_trackb_falsex.py --selftest
./.venv-phase9/bin/python run_trackb_falsex.py --expand-sequence          # initial collect+fit
./.venv-phase9/bin/python run_trackb_falsex.py --finish-7b-then-14b       # LOO/specificity + 14B transfer
```

## 7B live record — `Qwen2.5-7B-Instruct`

| Class | n | ids |
|------|---|-----|
| A (clean temporal-change, X=false) | **4** | `TX_E08`, `TX_E13`, `TX_E21`, `TX_E23` |
| B (true contradiction, X=true) | **4** | `TX_C01`–`TX_C04` |
| fail temporal (gold SATISFIED, X=true) | 20 | (rest of `TX_E*`) |

Vector: `unit(mean(A) − mean(B))` @ **L20**.

### Endpoint (LOO + specificity) — **YES @ α=8**

| Check | Result |
|-------|--------|
| LOO cosine to full | mean **0.985** (min ~0.971) |
| LOO @ α=8/16 | **4/4** folds safe, each **3/8** flips |
| Full @ α=8 | **3/8** flips; contra+nofail intact; Gaussian **0**; reverse **0** |
| Full @ α=16 | **3/8** safe; Gaussian **2/8** (weaker specificity) |

**Frozen limited claim:** α=**8** partial editor (weak-margin fails only). Not a family-wide fix.

## 14B transfer — `Qwen2.5-14B-Instruct` (once)

**Question:** Does the temporal neural→symbolic failure persist when the model changes?  
**Not:** Can 14B solve this better?

| Metric | 14B |
|--------|-----|
| Class A (X=false temporal) | **21**/24 |
| fail_temporal (X=true) | **1** (`TX_E14`, margin ≈1.34) |
| Class B | 4/4 intact |
| Matched fail/contrast useful? | **No** |
| Forced MI ladder? | **No** |

**Reading:** On this fixed DEV panel, the false-X **cluster does not persist** at 14B — almost all temporal-change notes commit correctly. That is informative for containment / scale-variant behavior. Do **not** run a full 14B MI ladder from this thin residue.

## Open

1. ~~Freeze α=8~~ — [TRACKB-ALPHA8-FREEZE.md](./TRACKB-ALPHA8-FREEZE.md)  
2. ~~Track A reconnect~~ — grounding temporal≠contradiction; Qwen Dual wrong_AUTO 5→2 on resim ([TRACKA-TEMPORAL-RECONNECT.md](./TRACKA-TEMPORAL-RECONNECT.md))  
3. Optional live Dual re-run; remaining Qwen wrongs `TF_T1`/`TF_N1`  
4. Sealed `temporal-test` only after explicit freeze decision  
5. Do **not** modify G3 from this panel alone
