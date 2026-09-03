# Track B — DEV expand panel (pre-model wording)

**Status:** wording frozen **before** HF · 7B collect + L20 family fit **done 2026-09-03**  
**File:** `data/temporal-family-dev-expand.json` (n=30, `TX_*`)  
**Not:** `temporal-family-test.json` (still sealed) · not merged into `temporal-family-dev.json`  
**G3:** untouched · **14B:** not run (7B had enough Class A)

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

## Live record — HF `Qwen2.5-7B-Instruct`

| Class | n | ids |
|------|---|-----|
| A (clean temporal-change, X=false) | **4** | `TX_E08`, `TX_E13`, `TX_E21`, `TX_E23` |
| B (true contradiction, X=true) | **4** | `TX_C01`–`TX_C04` |
| fail temporal (gold SATISFIED, X=true) | 20 | (rest of `TX_E*`) |
| nofail controls | 2 | `TX_N01`, `TX_N02` |

Vector: `unit(mean(A) − mean(B))` @ **L20** (raw ‖A−B‖ ≈ 33.8).

| α | flips / 8 eval fails | contra intact | nofail intact | safe |
|---|---------------------|---------------|---------------|------|
| 1–4 | 2 (`TX_E05`, `TX_E06`) | yes | yes | yes |
| **8–16** | **3** (+ `TX_E04`) | yes | yes | **yes** |
| 32 | 7 | **no** (`TX_C02` → false) | yes | no |

Safe α only flips **weak-margin** fails (baseline m ∈ {0, 0, 1.875}). Strong fails (m≳5) drift but stay X=true under α≤16.

## Reading

Original DEV `TF_E*` Class A was empty because that wording set was uniformly hard for 7B. The expand panel shows natural Class A **does** exist without conditioning on model output — but it is a **minority** (4/24) and the fitted direction is a **partial** editor, not a family-wide fix.

## Open (DEV only)

1. LOO / leave-one-A stability of the L20 vector  
2. Specificity (Gaussian / reverse / donor→contradiction) at best safe α  
3. Freeze candidate (likely α=8 or 16) only after that  
4. Then — and only then — score sealed `temporal-family-test.json`  
5. Do **not** modify G3
