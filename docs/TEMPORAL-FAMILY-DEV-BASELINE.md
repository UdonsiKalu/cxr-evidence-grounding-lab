# Temporal-family-dev baseline — Phase-7 Dual_full (unchanged)

**Status:** measured 2026-09-03 · **G3 not modified**  
**Stack:** unchanged Phase-7 Dual (`run_phase7.py --set temporal-dev`)  
**Set:** `data/temporal-family-dev.json` (n=14) — **development / diagnostic, not held-out**  
**Artifacts:** `artifacts/phase7-temporal-dev-panel.json` · `auto-contract-score-temporal-dev.json` · `auto-contract-diagnose-temporal-dev.json`

---

## Aggregate Dual_full

| Model | wrong_AUTO | correct_AUTO | REVIEW | false_contradiction_like |
|-------|------------|--------------|--------|---------------------------|
| Llama 8B | 2 | 1 | 11 | 0 |
| Mistral | **0** | 0 | **14** | 0 |
| **Qwen coder 32B** | **5** | 1 | 8 | **3** |

Interpretation (ChatGPT table):

| Outcome | Who |
|---------|-----|
| 0 wrong AUTO, several REVIEW | **Mistral** — contract contains failures via REVIEW-all |
| Several related temporal wrong AUTOs | **Qwen** — genuine **failure family** (false CONTRADICTION) |
| Mixed / other wrong AUTO types | **Llama** — 2 wrong AUTO as UNCERTAIN (not BC_E1-like X) |

---

## Qwen Dual_full wrong AUTOs (the research signal)

| ID | Gold | Dual | C / D | X on D | BC_E1-like? |
|----|------|------|-------|--------|-------------|
| **TF_E3** | SATISFIED | CONTRADICTION | C=C D=C | true | **yes** |
| **BC11_E3** | SATISFIED | CONTRADICTION | C=C D=C | true | **yes** |
| **TF_E4** | SATISFIED | CONTRADICTION | C=C D=C | true | **yes** |
| TF_T1 | NOT_SATISFIED | CONTRADICTION | C=C D=C | true | no (toxicity stop) |
| TF_N1 | NOT_SATISFIED | CONTRADICTION | C=C D=C | true | no (no-failure control) |

**Family finding:** at least **three** SATISFIED temporal cases (response→failure / confirmed progression) escape Dual_full as AUTO CONTRADICTION with `contradiction.present=true` on both paths — same shape as the original BC_E1 miss.

Note: seed **BC_E1** itself is **REVIEW** on Qwen Dual_full this run (D path REVIEW) — not wrong AUTO here. The family still shows the mechanism on TF_E3 / BC11_E3 / TF_E4.

---

## Llama Dual_full wrong AUTOs (different mechanism)

| ID | Gold | Dual | Notes |
|----|------|------|-------|
| TF_E3 | SATISFIED | UNCERTAIN | paths agree UNCERTAIN — not false-X |
| TF_T2 | NOT_SATISFIED | UNCERTAIN | paths agree UNCERTAIN |

BC_E1 / BC11_E3 often **REVIEW** via `paths_disagree` (C=CONTRADICTION, D=SATISFIED) — Dual contains false-X on one path.

---

## Next moves (ordered)

1. **Do not tighten G3 yet** until we classify Qwen’s five wrongs (3 family + 2 control overfire).  
2. Grow / freeze **`temporal-family-test.json`** (unseen) before gate redesign.  
3. Track B: probe/patch/steer on the **false-contradiction cluster** {TF_E3, BC11_E3, TF_E4} vs controls.  
4. Separately: why Dual lets TF_T1 / TF_N1 AUTO CONTRADICTION (broader reliability).

---

## Claim hygiene

Lab-scale baseline only. Development set was inspected — **not** a held-out test claim.
