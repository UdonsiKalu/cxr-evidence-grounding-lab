# Phase-8 protocol — model ladder (14B + non-coder 32B)

**Status:** live complete 2026-08-24 — `artifacts/phase8-ladder-phase4-panel.json` + `phase8-ladder-bmtcart-panel.json`.  
**Direction:** [ARCHITECTURE-DIRECTION.md](./ARCHITECTURE-DIRECTION.md)  
**Builds on:** Phase-7 full safety stack (L2a–L2d). Does **not** rewrite Phase-2–7 frozen panels.

## Soft claim

Phases 2–7 used a frozen trio (Llama 8B Q4 / Mistral Instruct / Qwen2.5-Coder 32B).
Small models often drive low Dual coverage; the coder-tuned 32B is an imperfect “large” control for clinical notes.

Phase-8 asks whether a **mid (14B)** and a **non-coder 32B** change Dual_full safety/coverage on the **same** frozen slices and prompts.

Do **not** claim domain-general assurance or that larger models remove the need for REVIEW.

## Live Dual_full (2026-08-24)

| Slice | Model | Coverage | Safety (auto) | REVIEW |
|-------|-------|----------|---------------|--------|
| Phase-4 oncology (n=7) | `qwen2.5:14b` | 0.71 | **1.00** | 2 |
| Phase-4 oncology (n=7) | `qwen2.5:32b` | 0.57 | **1.00** | 3 |
| BMT/CAR-T (n=6) | `qwen2.5:14b` | 0.50 | 0.67 | 3 |
| BMT/CAR-T (n=6) | `qwen2.5:32b` | 0.50 | **1.00** | 3 |

Reference (frozen Phase-7 Dual_full, not re-run): Qwen coder 32B safety **0.75** cov 0.67 on BMT/CAR-T — non-coder 32B improved safety to 1.00 at lower coverage (0.50).

Evidence gate: +0 new REVIEW on both new models (both slices). D-rep cleared (2→0) on both.

## Research questions

1. Does **Qwen2.5 14B** sit between 8B-class and 32B on Dual_full coverage/safety (Phase-4 oncology + Phase-7 BMT/CAR-T)?
2. Does **Qwen2.5 32B (instruct, non-coder)** improve Dual_full vs the frozen **qwen2.5-coder:32b** on the same slices (reference only — not a re-run of Phase-7)?

## Object

Same stack as Phase-7:

```text
extract → verify → round-trip → dual-path → evidence gate → AUTO | REVIEW
```

## Models (new only)

| Tag | Role |
|-----|------|
| `qwen2.5:14b` | Mid-size instruct (14B-class) |
| `qwen2.5:32b` | Non-coder 32B instruct (same family as frozen coder 32B) |

Frozen Phase-2–7 trio is **not** re-run here (citation panels stay intact). Compare new results to published Phase-6/7 tables in notes.

## Data

- Oncology: `data/heldout-phase4.json` (C9–U12, n=7)
- Cross-domain: `data/heldout-bmtcart-phase7.json` (BC_*, n=6)

## Controls

- Frozen A–D / verify / paraphrase prompts unchanged.
- B′ off. Temperature 0. Gold not fed to models.
- Do **not** overwrite `artifacts/phase1-*.json` … `phase7-*.json`.
- Outputs: `artifacts/phase8-ladder-phase4-panel.json`, `artifacts/phase8-ladder-bmtcart-panel.json` (+ per-model).

## Metrics

Dual_full / Dual_rt coverage + safety_among_auto; evidence_new_reviews; D-rep baseline→D_v.

## Runner

```bash
ollama pull qwen2.5:14b
ollama pull qwen2.5:32b
python3 run_phase8.py --selftest
python3 run_phase8.py --set phase4
python3 run_phase8.py --set bmtcart
# or both:
python3 run_phase8.py --set both
```

## Related

| Doc | Role |
|-----|------|
| [PHASE7-PROTOCOL.md](./PHASE7-PROTOCOL.md) | Stack + BMT/CAR-T |
| [PHASE6-PROTOCOL.md](./PHASE6-PROTOCOL.md) | Round-trip + dual |
