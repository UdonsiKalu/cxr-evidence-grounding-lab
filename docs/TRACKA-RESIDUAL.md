# Track A residual — after temporal-family freeze

**Status:** options pass landed 2026-09-06 · **wrong_AUTO still 0**  
**Freeze:** [TRACKA-TEMPORAL-FAMILY-FREEZE.md](./TRACKA-TEMPORAL-FAMILY-FREEZE.md)  
**Reconnect:** [TRACKA-TEMPORAL-RECONNECT.md](./TRACKA-TEMPORAL-RECONNECT.md)

## Coverage fix (meta / predicate-name)

Path D meta spans citing `FIRST_LINE_THERAPY_FAILED` falsely looked like hard conflict.  
`ground.py`: strip predicate-label noise; meta X override. G3 untouched.

### DEV resim (frozen extractions, no LLM)

| Model | wrong_AUTO | correct_AUTO | REVIEW |
|-------|------------|--------------|--------|
| Qwen coder 32B | **0** | **5** (was 3) | **9** (was 11) |

TF_T1 / TF_N1 Dual **AUTO** on that resim.

### Live Dual confirm (2026-09-06)

`run_phase7.py --set temporal-dev --models qwen2.5-coder:32b`  
→ `phase7-temporal-dev-qwen-live-coverage.json` · `tracka-qwen-live-coverage-dual-score.json`

| Metric | Dual_full |
|--------|-----------|
| wrong_AUTO | **0** |
| correct_AUTO | **7** |
| REVIEW | **7** |
| safety_among_auto | **1.00** |

**AUTO-correct:** BC_E1, TF_E3, TF_E5, TF_U1, TF_C1, BC_C1, BC_E2  
**REVIEW:** TF_E1, TF_E2, BC11_E3, TF_E4, **TF_T1**, TF_T2, **TF_N1**

Live re-extraction still leaves TF_T1/N1 in REVIEW (resim on *frozen* extractions had cleared them). Containment holds; coverage is live-dependent.

### Sealed test score re-open (no redesign)

`--tracka-resim-test` → `tracka-temporal-test-grounding-resim.json`  
Dual_full **wrong_AUTO=0 / correct_AUTO=5 / REVIEW=7** (unchanged vs freeze). TFT_T1/T2 stay REVIEW.

| Item | Status |
|------|--------|
| False-X / temporal reconnect | Frozen |
| Meta/predicate coverage | Landed; resim helps TF_T1/N1; live still REVIEW those two |
| Live Dual confirm | Done — wrong_AUTO=0, correct_AUTO=7 |
| Sealed test | Re-scored under current ground; no redesign |
| Llama residual | Separate |

## Decision

Spine options closed for this family. Do **not** retune from sealed test. Optional stop / portfolio claim as-is.
