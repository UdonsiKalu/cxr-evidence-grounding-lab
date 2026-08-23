# Phase-5 protocol — verify once, then REVIEW (not UNCERTAIN)

**Status:** protocol frozen 2026-08-23; implementation started; **no live Ollama claim until run completes**.  
**Direction:** [ARCHITECTURE-DIRECTION.md](./ARCHITECTURE-DIRECTION.md)  
**Does not replace** M1–M3 or Phase-1–4 records. Does not overwrite prior artifacts.

## Soft claim (required wording)

Pilot results show that explicitly preserving contradiction and uncertainty can prevent semantic states observed in analysis from being lost before symbolic evaluation.

Do **not** claim the architecture generally beats direct LLM reasoning.

## Research question

On cases where free-text analysis already carries the gold distinction (contradiction or uncertainty), does a **verify → regenerate once → REVIEW-on-fail** gate:

1. reduce **D `representation_loss`**, and/or  
2. convert unverifiable formalizations into **REVIEW** instead of a wrong auto-verdict,

without collapsing **UNCERTAIN** (clinical uncertainty) into REVIEW?

## 1. Object

**L2 + L3 only:**

```text
formalize → VERIFY → (fail) regenerate once → VERIFY → PASS→rule | FAIL→REVIEW
```

- **UNCERTAIN** = evidence/contract uncertainty (existing rule / uncertainty field).  
- **REVIEW** = verification failed after one repair — representation not trusted.

## 2. Controls

- Frozen Phase-1–4 **A–D prompts** for Conditions A/B and for baseline C/D (no verify).  
- Verify path adds a **repair** extract pass only when verify fails (documented prompt additive; does not edit frozen A–D systems).  
- B′ **off**. Temperature **0**. Gold not fed to models.  
- Do **not** overwrite `artifacts/phase1-*.json` … `phase4-*.json`.  
- No Qdrant, Archetypes, CXR production, C4 fix, new predicates.

## 3. Data

Primary: Phase-4 held-out `C9–C11`, `U9–U12` (`data/heldout-phase4.json`).  
Optional secondary: Phase-3 held-out `C5–C8`, `U5–U8`.

Report especially rows where frozen Condition D was `representation_loss` (from prior artifacts or recompute baseline D in the same run).

## 4. Models

Same three tags as Phase-2–4:

- `llama3:8b-instruct-q4_0`  
- `mistral:instruct`  
- `qwen2.5-coder:32b`

## 5. Per-case outputs

For each case / model:

| Path | Meaning |
|------|---------|
| A, B | Frozen (reference) |
| C, D | Frozen pipeline / analysis→structure **without** verify |
| C_v, D_v | Same formalization path **with** verify → regen once → rule or **REVIEW** |

## 6. Metrics (declared before live)

**Do not** optimize a single useful-rate that rewards REVIEW-all.

1. **Safety (auto):** among cases with verdict ∉ {REVIEW}, match-gold rate for C_v and D_v.  
2. **Coverage:** fraction of cases with verdict ≠ REVIEW on C_v / D_v.  
3. **D-loss:** count of `representation_loss` on D vs D_v (among regex/analysis-has-distinction screen; human \(A_{\text{human}}\) still preferred for headlines later).  
4. **REVIEW quality (secondary):** among D rows that were `representation_loss`, did D_v become (a) gold match, (b) REVIEW, or (c) still wrong auto-label?

### Stop / falsify

- If D_v does not reduce representation_loss **and** does not convert those failures to REVIEW on ≥2/3 models → L2/L3 **weakened**; do not build CXR on it yet.  
- If D_v mostly emits REVIEW but safety among remaining auto-decisions rises → **partial support** (safer, lower coverage).  
- If D_v fixes structure (gold match) without inventing UNCERTAIN for verify-fail → **stronger support**.

## 7. What we will not do in Phase-5

Force **UNCERTAIN** on verify-fail. Mint new first-class fields per miss. Iterate B′. Fix C4. Touch production CXR. Grow N for its own sake.

## Runner

```bash
python3 run_phase5.py                  # live panel (needs Ollama)
python3 run_phase5.py --selftest       # verify logic only, no Ollama
```

Writes `artifacts/phase5-verify-panel.json` (and per-model files). Prior phase artifacts untouched.
