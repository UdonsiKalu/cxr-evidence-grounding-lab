# Track A reconnect — temporal-change ≠ contradiction

**Status:** landed 2026-09-04  
**Does not modify G3.**  
**Sealed:** `temporal-family-test.json` not used.

## Insight from Track B

HF/Ollama failures on therapy-worked-then-failed notes often set
`contradiction.present=true` on **sequenced** outcomes (response → later failure).
That is a fidelity bug at the neural→symbolic boundary, not a true contradiction.

## Change

In `n2s_lab/ground.py`:

- **Hard simultaneous conflict** (never↔given, failed↔ongoing) still → X=true.
- **Sequenced temporal-change** (response+failure, or unknown+failure / possible→confirmed)
  → X=false even if the extractor asserted X=true.

## Measure (no LLM)

```bash
python3 run_auto_contract_score.py --selftest
python3 run_auto_contract_score.py --tracka-resim
```

Re-grounds frozen Phase-7 `temporal-dev` C_full/D_full extractions and rescored Dual_full.

### Resim result (2026-09-04)

| Model | wrong_AUTO before → after | Remaining wrongs |
|-------|---------------------------|------------------|
| Qwen coder 32B | **5 → 2** | `TF_T1`, `TF_N1` (not false-X family) |
| Llama 8B | 2 → 2 | `TF_E3` UNCERTAIN, `TF_T2` |
| Mistral | 0 → 0 | — |

False-X cluster **TF_E3 / BC11_E3 / TF_E4** cleared on Qwen Dual via grounding.

## Next

Optional live Dual re-run on Qwen. Still no sealed-test peek. G3 untouched.
