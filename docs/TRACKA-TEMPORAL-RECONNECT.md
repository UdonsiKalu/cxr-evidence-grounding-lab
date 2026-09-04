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
| Qwen coder 32B | **5 → 0** | none (false-X + TF_T1/TF_N1 cleared) |
| Llama 8B | 2 → 2 | `TF_E3` UNCERTAIN, `TF_T2` |
| Mistral | 0 → 0 | — |

False-X cluster **TF_E3 / BC11_E3 / TF_E4** cleared; toxicity-stop / no-failure overfire (`TF_T1`/`TF_N1`) cleared via no-failure-without-hard-conflict override.

## Live confirm (2026-09-04)

```bash
python3 run_phase7.py --set temporal-dev --models qwen2.5-coder:32b \
  --out phase7-temporal-dev-qwen-live-grounding.json
python3 run_auto_contract_score.py --artifact artifacts/phase7-temporal-dev-qwen2.5-coder_32b.json
```

| Path | wrong_AUTO | correct_AUTO | REVIEW | safety_among_auto |
|------|------------|--------------|--------|-------------------|
| **Dual_full** | **0** | 3 | 11 | **1.00** |

Family TF_E3/BC11_E3 AUTO-correct; TF_E4 REVIEW; TF_T1/TF_N1 Dual REVIEW (contained).

## Next

Sealed `temporal-test` only after explicit go. G3 untouched. Llama residual wrongs are a separate UNCERTAIN-shaped issue.
