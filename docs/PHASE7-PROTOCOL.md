# Phase-7 protocol — evidence object + BMT/CAR-T cross-domain

**Status:** protocol + live BMT/CAR-T panel frozen 2026-08-24 (`artifacts/phase7-bmtcart-panel.json`; PDF record in `notes/progress-notes.pdf`).  
**Direction:** [ARCHITECTURE-DIRECTION.md](./ARCHITECTURE-DIRECTION.md)  
**Builds on:** Phase-5 (L2a verify) + Phase-6 (L2b/L2c) + **L2d evidence-preservation object**

## Soft claim

Pilot results (Phases 5–7) show verify → repair → REVIEW, optional round-trip/dual-path, and an evidence span gate can raise safety among auto-decisions on small oncology and BMT/CAR-T slices (not always Dual safety 1.0 on every model).

Do **not** claim domain-general assurance or that evidence objects *prove* faithfulness.

## Research questions

1. Does an **evidence-preservation object** (proposition + source span + verify PASS/FAIL) catch untraceable formalizations that L2a alone missed?
2. Does the **same safety stack** (L2a–L2d) behave similarly on **BMT/CAR-T hematology** wording (`data/heldout-bmtcart-phase7.json`)?

## Object

On top of Phase-6 paths, add **L2d evidence gate**:

```text
… → L2a verify → (L2b round-trip) → L2c dual-path → L2d evidence claims → rule | REVIEW
```

Paths: `C_full`, `D_full` (Phase-6 + evidence gate on AUTO verdicts only).

## Evidence object (L2d)

Each supported atom/contradiction/uncertainty becomes a claim:

- proposition (e.g. `failure_event`, `contradiction`)
- evidence span(s) from extraction
- verification: span traceable in source text → PASS else FAIL → **REVIEW**

Implementation: `n2s_lab/evidence.py`

## Controls

- Frozen Phase-1–6 A–D and verify/repair prompts unchanged.
- B′ off. Temperature 0. Gold not fed to models.
- Do **not** overwrite Phase-1–6 artifacts.
- No Qdrant, Archetypes, CXR production, pathway engine integration (notes only).

## Data

`data/heldout-bmtcart-phase7.json` — six cases (BC_C1–BC_C2, BC_U1–BC_U2, BC_E1–BC_E2).  
Hematology / BMT / CAR-T surface wording; **same predicate** `FIRST_LINE_THERAPY_FAILED`.

## Models

Same three local tags as Phase-2–6.

## Metrics

Safety, coverage, evidence-gate REVIEW count, comparison to Phase-6 on oncology slice (reference only).

## Runner

```bash
python3 run_phase7.py --selftest
python3 run_phase7.py                  # BMT/CAR-T held-out, needs Ollama
python3 run_phase7.py --set phase4     # optional oncology replay with evidence gate
```

Writes `artifacts/phase7-bmtcart-panel.json` (default).

## Related

| Doc | Role |
|-----|------|
| [PHASE6-PROTOCOL.md](./PHASE6-PROTOCOL.md) | Round-trip + dual-path |
| [PHASE5-PROTOCOL.md](./PHASE5-PROTOCOL.md) | Verify + REVIEW |
