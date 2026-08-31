# Phase-6 protocol — faithfulness experiments (L2b round-trip + L2c dual-path)

**Status:** protocol frozen 2026-08-23; implementation started; **no live Ollama claim until run completes**.  
**Direction:** [ARCHITECTURE-DIRECTION.md](./ARCHITECTURE-DIRECTION.md)  
**Does not replace** M1–M5 or Phase-1–5 records. Does not overwrite prior artifacts.

## Soft claim (required wording)

Pilot results (through Phase-5) show that explicitly preserving contradiction and uncertainty, plus verify → regenerate once → REVIEW, can clear measured representation loss on a small held-out slice.

Do **not** claim domain-general N→S assurance or that round-trip / dual-path *prove* faithfulness. These are experiments toward general checks.

## Research question

What **general checks** can tell us that meaning survived formalization **without** enumerating every possible thing that could have gone wrong?

> Can we demonstrate that the representation faithfully preserves the source evidence?

Phase-6 asks whether **L2b** (round-trip) and **L2c** (dual-path agreement), on top of Phase-5 **L2a** (analysis↔structure verify), improve **safety** among automatic decisions and/or convert unverifiable cases to **REVIEW** — without inventing clinical **UNCERTAIN** on verify-fail.

## Object

```text
L2a (Phase-5): formalize → VERIFY vs analysis → repair once → rule | REVIEW
L2b (Phase-6): if L2a AUTO → STRUCTURE → paraphrase → compare to source → pass | REVIEW
L2c (Phase-6): C_v and D_v (after L2a, optionally after L2b) must AGREE → else REVIEW
```

| Path | Meaning |
|------|---------|
| A–D, C_v, D_v | Frozen Phase-5 (recomputed in same run for fair pairing) |
| C_rt, D_rt | C_v / D_v after **L2b** round-trip gate |
| Dual | L2c: agree(C_v, D_v) → shared verdict; else REVIEW |
| Dual_rt | L2c on C_rt / D_rt (stricter) |

**UNCERTAIN** = clinical uncertainty from the symbolic rule only.  
**REVIEW** = unverifiable formalization / path disagreement — never force UNCERTAIN.

## Longer-term target (not Phase-6 code)

Evidence-preservation object (proposition + source + span + relationship + verify). Symbolic engine accepts claims with justification. **Not implemented** here.

## Controls

- Frozen Phase-1–4 A–D prompts; Phase-5 verify/repair prompts unchanged.  
- Round-trip paraphrase is an **additive** call only when L2a already passed (AUTO).  
- B′ **off**. Temperature **0**. Gold not fed to models.  
- Do **not** overwrite `artifacts/phase1-*.json` … `phase5-*.json`.  
- No Qdrant, Archetypes, CXR production, C4 fix, new predicates, B′ iteration.  
- No full evidence-preservation object.

## Data

Primary: Phase-4 held-out `C9–C11`, `U9–U12` (`data/heldout-phase4.json`).  
Optional secondary: Phase-3 held-out `C5–C8`, `U5–U8`.

## Models

Same three tags as Phase-2–5:

- `llama3:8b-instruct-q4_0`  
- `mistral:instruct`  
- `qwen2.5-coder:32b`

## L2b round-trip (declared)

1. Take the verified extraction (from C_v or D_v after Phase-5 gate).  
2. LLM writes a short **paraphrase** of what the JSON alone supports (no new facts; no verdict labels).  
3. Deterministic check (`verify_roundtrip`):
   - If structure asserts contradiction → paraphrase must carry contradiction cues.  
   - If structure asserts uncertainty → paraphrase must carry uncertainty cues.  
   - If source text (note for C; analysis for D) indicates the gold distinction → paraphrase must also indicate that distinction.  
4. Fail → **REVIEW** (not UNCERTAIN). Pass → keep L2a rule verdict.

## L2c dual-path (declared)

- If either path is REVIEW → Dual = REVIEW.  
- If both AUTO and verdicts **equal** → Dual = that verdict.  
- If both AUTO and verdicts **differ** → Dual = REVIEW.

## Metrics (declared before live)

1. **Safety (auto):** among verdict ∉ {REVIEW}, match-gold rate for C_rt, D_rt, Dual, Dual_rt.  
2. **Coverage:** fraction with verdict ≠ REVIEW.  
3. **Delta vs Phase-5:** compare C_v/D_v safety & coverage to C_rt/D_rt/Dual*.  
4. **Abstention quality (secondary):** REVIEW rows — warranted vs over-abstain (human later).

### Stop / falsify

- If L2b/L2c do not raise safety among remaining auto-decisions **and** do not convert additional unverifiable cases to REVIEW on ≥2/3 models → **weak support**.  
- If coverage collapses (near-all REVIEW) with little safety gain → **product-useless**; rethink checks.  
- If any path forces UNCERTAIN on round-trip/dual fail → **protocol failure**.

## What we will not do in Phase-6

Claim universality of understanding. Eliminate REVIEW. Force UNCERTAIN on check-fail. Grow ontology (L4) or wire CXR (L5). Treat round-trip as a mathematical proof of meaning preservation. Mint new first-class fields per miss.

## Runner

```bash
python3 run_phase6.py --selftest       # round-trip + dual logic only, no Ollama
python3 run_phase6.py                  # live panel (needs Ollama; long)
```

Writes `artifacts/phase6-faithfulness-panel.json` (and per-model files). Prior phase artifacts untouched.

## Related

| Doc | Role |
|-----|------|
| [ARCHITECTURE-DIRECTION.md](./ARCHITECTURE-DIRECTION.md) | Protocol-universality + evidence-preservation north star |
| [PHASE5-PROTOCOL.md](./PHASE5-PROTOCOL.md) | L2a verify + L3 REVIEW (frozen live) |
| `notes/progress-notes.pdf` | Full lab record through Phase-5 |
