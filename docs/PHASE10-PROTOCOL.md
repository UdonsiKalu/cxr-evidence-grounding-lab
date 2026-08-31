# Phase-10 protocol — activation steering (strict rep-eng)

**Status:** **FROZEN completed pilot** 2026-08-24 — 10C gate **YES** (`artifacts/phase10-bc-e1-panel.json`).  
**Claim scope:** genuine pilot-scale rep-eng (hidden steer → natural margin flip); not “found the contradiction representation.”  
**Next:** Phase-11 fixed-vector generalization (draft only; wait **go**).  
**Builds on:** [PHASE9-PROTOCOL.md](./PHASE9-PROTOCOL.md) (frozen completed pilot)  
**Simple notes:** [PHASE9-NOTES-SIMPLE.md](./PHASE9-NOTES-SIMPLE.md) · PDF: `notes/progress-notes.pdf` § Phase 10

## Why Phase 10 exists

Phase 9 completed Observe → Localize → **output-level** Intervene:

- 9B localized the split to the **repair `contradiction.present` commit** (7B margin +1.63 vs 14B −3.97).
- 9C showed **logit bias** at that commit flips X and verdict on BC_E1 with controls held.

That is a valid **behavioral causal** result. It is **not** yet representation engineering in the strict sense: we did not manipulate hidden states and ask the model to change its preference on its own.

## Scientific question (Phase 10)

> Can we change HF **Qwen2.5-7B-Instruct**’s contradiction decision on BC_E1 by intervening on **hidden activations** at the repair commit — **without** directly biasing the `true`/`false` output token?

Success = the model’s **natural** logit margin shifts toward `false` (or commit becomes `false`) after activation patch/steer only.

## Frozen inherit from Phase 9

| Item | Value |
|------|--------|
| Case | `BC_E1` (+ controls BC_C1, BC_E2) |
| Model | `Qwen/Qwen2.5-7B-Instruct` (fail proxy) |
| Prompt / schema | Unchanged from Phase 9 |
| Stage | **Repair** path after verify fail (same as 9B/9C) |
| Commit locus | `contradiction.present` boolean (step ~136 in 9B trace) |
| Reference traces | `artifacts/phase9b-bc-e1-*-trace.json` |

Do **not** overwrite Phase 1–9 citation artifacts.

## Hypothesis (from 9B)

At fractional depths **0.75** and **1.00**, 7B vs 14B hidden norms diverge strongly at commit. A steering vector derived from:

- **Contrast:** 14B commit hidden @ layer L → direction “safe” (false), or  
- **Ablation:** remove component aligned with 7B “true” commit,

might shift logits without logit forcing.

## Stages (proposed)

### 10A — Build steering vector(s)

- Load 9B traces for 7B (true commit) and 14B (false commit) at layers 0.75 / 1.00.
- For 7B-only work: difference-of-means or PCA direction between “about to commit true” vs held-out “about to commit false” steps on repair runs (BC_E1 vs BC_E2 repair if needed).
- Optional: Neuronpedia / published **7B SAE** features at commit token (aid only).

### 10B — Activation intervention at commit

During repair generation, when prefix matches `contradiction.present` commit (same detector as 9C):

- **Patch / add** `α * v` to residual stream at layer L on last token position.
- Sweep small `α` grid; **no** `logit_bias_false`, **no** `force_false_at_commit`.
- Record: commit value, logit margin (true−false), repair X, rule verdict.

### 10C — Gate (strict)

**YES** only if all hold on BC_E1:

1. Baseline repair X = true (replicate 9A).
2. Activation steer alone yields repair X = false **or** logit margin crosses below 0 without token forcing.
3. Downstream verdict SATISFIED (or REVIEW, not wrong AUTO).
4. Controls BC_C1 still X = true; BC_E2 still X = false.

If NO → report negative result; do not claim rep-eng success.

## Live 10A–10C (2026-08-24, HF 7B)

**10A steering vector:** unit direction `BC_E2_false_commit − BC_E1_true_commit` at layers **0.75** (idx 20) and **1.00** (idx 27). Raw norm deltas 44 / 285.

**10B sweep (BC_E1, no logit bias):**

| α | repair X | commit | margin (true−false) | verdict |
|---|----------|--------|---------------------|---------|
| baseline | true | true | **+1.63** | CONTRADICTION |
| 1.0 | true | true | +1.13 | CONTRADICTION |
| 2.0 | true | true | +0.75 | CONTRADICTION |
| **4.0** | **false** | **false** | **−0.13** | **SATISFIED** |
| 8.0 | false | false | −1.75 | SATISFIED |
| −4.0 | true | true | +3.00 | CONTRADICTION |

**10C controls @ α=4.0:** BC_C1 X stays true; BC_E2 X stays false.

**Gate: YES** — activation patch alone flipped symbolic X and verdict on BC_E1; margin crossed below zero; controls held.

**Runner:** `python3 run_phase10.py` (requires 9B traces; `.venv-phase9`).

**Claim scope:** one case, one model, contrast-derived steering vector; soft rep-eng pilot — not clinical/product.

## Pilot complete — do not extend Phase 10

Phase 10 is **frozen**. Do not re-sweep $\alpha$ or recompute the vector for publication polish.  
For **generalization**, use Phase 11 (same frozen vector + $\alpha$ on held-out cases).

## Out of scope

- Replacing verify/repair hygiene (main-lab track A/B)
- Ollama coder-32B / 14B HF parity with Phase 7–8 Ollama misses
- Claiming universal N→S fix
- Re-sweeping Phase 10 or per-case vector retune (use Phase 11 instead)

## Deliverables

- `docs/PHASE10-PROTOCOL.md` (this file, frozen live record)
- `run_phase10.py` + `n2s_lab/phase10_activation.py`
- `artifacts/phase10-bc-e1-panel.json`
- `artifacts/phase10-bc-e2-*-trace.json` (E2 commit hidden for 10A)
- `notes/progress-notes.pdf` and `notes/progress-notes-simple.pdf` § Phase 10

## Soft claim

One case, one model, synthetic note. Pilot for learning MI/rep-eng method — not clinical, not product.

## Related

- Phase 9 frozen: [PHASE9-PROTOCOL.md](./PHASE9-PROTOCOL.md)
- Runners: `run_phase9a.py` … `run_phase9c.py` (prerequisite traces)
