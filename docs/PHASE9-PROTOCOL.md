# Phase-9 protocol — BC_E1 false contradiction (representation-engineering pilot)

**Status:** **FROZEN completed pilot** 2026-08-24 — 9A/9B/9C gates YES (`artifacts/phase9c-bc-e1-panel.json`).  
**Claim scope:** 9C = **output-level** causal fix at repair commit (logit bias), **not** hidden-representation steering.  
**Simple notes:** [PHASE9-NOTES-SIMPLE.md](./PHASE9-NOTES-SIMPLE.md)  
**Next (strict rep-eng):** [PHASE10-PROTOCOL.md](./PHASE10-PROTOCOL.md) — activation intervene without logit forcing; wait **go**.  
**Direction:** [ARCHITECTURE-DIRECTION.md](./ARCHITECTURE-DIRECTION.md)  
**Motivation (citation only):** Phase-7/8 BMT Dual miss on `BC_E1` — not a Dual bug study.

## Soft claim

Soft claims only. Synthetic note. One predicate (`FIRST_LINE_THERAPY_FAILED`). Small n.  
Do **not** claim clinical validity or that intervening on activations is a product fix.

## Scientific question

Given the **same note**, **same extract schema**, and **same extract prompt**, why do some Qwen2.5 variants assert `contradiction.present=true` on temporal **response→progression**, while a peer abstains / routes safer?

**Not:** Why did Dual agree? (Dual agreeing on wrong X is expected once X is set.)

## Observation that motivates this study (do not re-run)

| Model (Ollama) | Where | Dual_full | BC_E1 outcome |
|----------------|-------|-----------|---------------|
| `qwen2.5-coder:32b` | Phase-7 BMT | safety 0.75 | AUTO CONTRADICTION |
| `qwen2.5:14b` | Phase-8 BMT | safety 0.67 | AUTO CONTRADICTION |
| `qwen2.5:32b` (instruct) | Phase-8 BMT | safety 1.00 | did not auto that miss (more REVIEW) |

**Protocol forensic (frozen):** raw extract often has `X=false`; analysis may deny conflict but regex `mentions_contradiction` FP → verify fail → **repair sets `X=true`** on spans “initial response” vs “progression in cycle 5” (course ≠ contradiction) → Dual/evidence amplify.

Artifacts (citation):  
`artifacts/phase7-bmtcart-qwen2.5-coder_32b.json`,  
`artifacts/phase8-bmtcart-qwen2.5_14b.json`,  
`artifacts/phase8-bmtcart-qwen2.5_32b.json`.

**Do not overwrite** `artifacts/phase1-*.json` … `phase8-*.json`.

## Frozen probe (all Phase-9 stages)

| Item | Value |
|------|--------|
| Case | `BC_E1` in `data/heldout-bmtcart-phase7.json` |
| Note | `DLBCL: first-line R-CHOP for four cycles with initial response, then radiographic progression in cycle 5. Discontinued R-CHOP for progression; started R-ICE as second-line therapy.` |
| Gold | `SATISFIED` |
| Prompt | `EXTRACT_SYSTEM` + `EXTRACT_SCHEMA` in `n2s_lab/neural.py` (unchanged) |
| Repair suffix | `REPAIR_EXTRACT_SUFFIX` (unchanged) |
| Analysis | `ANALYSIS_SYSTEM` (unchanged) when running verify/repair path |
| Temperature | 0 / greedy |

## Stages (gates)

```text
9A Observe  →  YES (clean X divergence)  →  9B Localize  →  YES (probe signal)  →  9C Intervene
```

Stop at the first NO. Do **not** start 9B/9C, Neuronpedia, verify hygiene, Dual fixes, or Phase-9 PDF freeze until the gate passes and the user asks.

### Phase 9A — Behavioral reproduction (HF)

**Object:** Reproduce `contradiction.present` behavior on HF weights (activation-ready), not Dual_full replay.

**Success criterion:** Clean **X divergence** across proxies on the same frozen probe — at least one model/path with false `X=true` and one with `X=false` (or safer non-assertion), recording **which stage** flips the bit:

1. Raw extract (note → JSON)
2. Analysis (for verify reference)
3. Verify + one repair (C_verified path: extract → verify vs analysis → repair)

**VRAM proxies (RTX 3090 24GB):**

| HF id | Role |
|-------|------|
| `Qwen/Qwen2.5-14B-Instruct` | Fail-class proxy (Ollama Ph8 14B miss) |
| `Qwen/Qwen2.5-7B-Instruct` | Smaller instruct proxy / optional Neuronpedia SAE later |
| Optional later | `Qwen/Qwen2.5-32B-Instruct` 4-bit if needed for safe-class contrast |

Ollama coder-32B / instruct-32B remain **behavioral citations**, not required HF loads for 9A gate.

**Outputs:** `artifacts/phase9a-bc-e1-*.json` (new only).

**Live 9A (2026-08-24):**

| HF model | Raw X | Final X | Stage that set X | Disposition |
|----------|-------|---------|------------------|-------------|
| `Qwen/Qwen2.5-7B-Instruct` | false | **true** | **repair** | AUTO CONTRADICTION |
| `Qwen/Qwen2.5-14B-Instruct` (4-bit) | false | **false** | none | REVIEW (verify fail after repair) |

Gate **YES** (clean X divergence). Note: HF 14B did **not** replay the Ollama `qwen2.5:14b` AUTO miss; the repair-path false X reproduced on **7B** instead. Ollama Ph7/8 rows remain citation-only.

**Runner:** `python3 run_phase9a.py` (uses `.venv-phase9` if present).

**Stop rule:** If no clean X divergence after the planned proxies → **stop**; report; do not proceed to 9B. (Cleared 2026-08-24.)

### Phase 9B — Representation localization (gated)

Only if 9A = YES.  
Residual / logit attribution on **repair** generation at `contradiction.present` commit step.  
Compare fail (7B) vs safe (14B) at matched fractional depths (scalar norm stats — hidden dim differs).

**Live 9B (2026-08-24):**

| HF model | commit | logit margin (true−false) | repair X |
|----------|--------|---------------------------|----------|
| `Qwen/Qwen2.5-7B-Instruct` | `present: true` | **+1.63** | true |
| `Qwen/Qwen2.5-14B-Instruct` | `present: false` | **−3.97** | false |

Cross-model at commit: max layer-norm delta **390** at depth 1.00 (scalar stats; hidden dim differs).  
Gate **YES** (behavior + logit margin separation). Artifacts: `phase9b-bc-e1-panel.json`, per-model `*-trace.json`.

**Runner:** `python3 run_phase9b.py` (requires 9A gate YES).

### Phase 9C — Causal intervention (gated)

Only if 9B = YES.  
Steer / ablate at repair `contradiction.present` commit → does symbolic X change → does downstream verdict change → controls hold?

**Live 9C (2026-08-24, HF 7B):**

| Case | Baseline repair X | Intervened X | Baseline verdict | Intervened verdict |
|------|-------------------|--------------|------------------|-------------------|
| **BC_E1** (target) | true | **false** | CONTRADICTION | **SATISFIED** |
| BC_C1 (true contradiction control) | true | true | CONTRADICTION | CONTRADICTION |
| BC_E2 (no-failure control) | false | false | NOT_SATISFIED | NOT_SATISFIED |

Intervention: `logit_bias_false` (+4.0) at `contradiction.present` commit token during repair JSON generation.  
Gate **YES** — **behavioral / output-level** causal claim: flipping commit logit flips symbolic X and rule verdict on BC_E1 without breaking controls.  
**Not claimed:** internal representation was steered (see Phase 10).

**Runner:** `python3 run_phase9c.py` (requires 9B gate YES).

## Pilot complete — do not extend Phase 9

Phase 9 is **frozen**. Do not add 9D or re-run for publication counts.  
For **strict representation engineering**, use Phase 10 (activation patch/steer without logit forcing).

## Out of scope (this protocol)

- Patching verify negation / contradiction≠course hygiene (main-lab orthogonal track A/B)
- Dual / round-trip / evidence redesign
- Overwriting Phase-1–8 citation artifacts
- Treating REVIEW as UNCERTAIN

## Related

- Lab: `../` (`cxr-evidence-grounding-lab/`)
- Safety stack UI: sibling `:8255` (replay only; not Phase-9 runner)
- Handoff: `../../cxr-agent-handoff/07-next-actions.md`
- Phase 10 (draft): [PHASE10-PROTOCOL.md](./PHASE10-PROTOCOL.md)
