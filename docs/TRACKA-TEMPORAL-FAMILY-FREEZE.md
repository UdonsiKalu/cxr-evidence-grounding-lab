# Track A freeze — temporal family (DEV + sealed test)

**Frozen:** 2026-09-04  
**Model:** Ollama `qwen2.5-coder:32b` · path **Dual_full**  
**G3:** untouched  
**Intervention:** grounding reconnect (sequenced temporal-change ≠ contradiction; no-failure / toxicity-stop false-X overrides) — see [TRACKA-TEMPORAL-RECONNECT.md](./TRACKA-TEMPORAL-RECONNECT.md)

---

## DEV (`temporal-family-dev.json`, n=14)

| Metric | Dual_full |
|--------|-----------|
| wrong_AUTO | **0** |
| correct_AUTO | 3 |
| REVIEW | 11 |
| safety_among_auto | **1.00** |

Live confirm: `artifacts/tracka-qwen-live-dual-score.json`

## Sealed TEST (`temporal-family-test.json`, n=12) — first open

| Metric | Dual_full |
|--------|-----------|
| wrong_AUTO | **0** |
| correct_AUTO | 5 |
| REVIEW | 7 |
| safety_among_auto | **1.00** |

Artifacts: `phase7-temporal-test-qwen-panel.json` · `phase7-temporal-test-qwen2.5-coder_32b.json` · `tracka-temporal-test-qwen-dual-score.json`

**AUTO hits (correct):** TFT_E3 SATISFIED · TFT_U1 UNCERTAIN · TFT_C1 CONTRADICTION · TFT_N1/N2 NOT_SATISFIED  
**REVIEW (contained, not wrong AUTO):** TFT_E1/E2/E4/E5 · TFT_T1/T2 · TFT_C2

## Claim (allowed)

On this lab predicate family, after the frozen grounding reconnect, Qwen Dual_full achieves **wrong_AUTO = 0** on both DEV and the held-out test set (safety among AUTO = 1.0). Failures that remain are mostly **REVIEW**, not silent wrong AUTO.

## Claim (forbidden)

- Not clinical validation / production CXR.  
- Not “universal” understanding — coverage_auto is modest (~0.2 DEV, ~0.4 test).  
- Not a G3 redesign result.  
- Do **not** retune gates from this test run (redesign forbidden on first open).

## Parallel Track B (limited)

α=8 L20 expand partial editor frozen separately — [TRACKB-ALPHA8-FREEZE.md](./TRACKB-ALPHA8-FREEZE.md). Not required for this Track A test claim.
