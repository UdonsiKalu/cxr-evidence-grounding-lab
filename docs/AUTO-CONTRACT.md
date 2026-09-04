# AUTO contract (Track A) — freeze

**Status:** frozen 2026-09-03 for Track A reliability work.  
**Spine:** [RESEARCH-APPROACH.md](./RESEARCH-APPROACH.md)  
**Lab-scale only.** Not clinical product claims.

---

## Rule

```text
AUTO  ⟺  all gates pass
else  →  REVIEW
```

**AUTO** = emit a final symbolic verdict (`SATISFIED` / `NOT_SATISFIED` / `UNCERTAIN` / `CONTRADICTION`) as a trusted decision.  
**REVIEW** = do not trust the formalization; human / downstream review required.  
**REVIEW ≠ UNCERTAIN** — UNCERTAIN is a clinical/contract uncertainty state from the rule; REVIEW is a trust failure.

---

## Gates (mandatory for AUTO)

| # | Gate | Pass condition |
|---|------|----------------|
| G1 | **Verify** | Formalization verifies against analysis/evidence after ≤1 repair |
| G2 | **Dual-path agree** (when Dual path used) | C-path and D-path AUTO verdicts identical |
| G3 | **Contradiction hygiene** | Do not AUTO CONTRADICTION unless verify supports `contradiction.present` (no keyword-only force) |

**Grounding reconnect (2026-09-04, not a G3 change):** sequenced temporal-change
(response→later failure, or possible→confirmed progression) is **not** a contradiction
even if the extractor sets `contradiction.present=true`. See `n2s_lab/ground.py` and
`docs/TRACKA-TEMPORAL-RECONNECT.md`. Resim: `run_auto_contract_score.py --tracka-resim`.

Optional later (not required for v1 scoring):

| # | Gate | Pass condition |
|---|------|----------------|
| G4 | **Commit margin** | If a commit logit margin is available, \|margin\| ≥ threshold or → REVIEW |

---

## Metrics (do not collapse into one “accuracy”)

On a frozen case set, for a chosen path (e.g. `Dual_full`):

| Bucket | Definition |
|--------|------------|
| **wrong_AUTO** | disposition = AUTO **and** verdict ≠ gold |
| **correct_AUTO** | disposition = AUTO **and** verdict = gold |
| **REVIEW** | disposition = REVIEW (or verdict = REVIEW) |
| **n** | cases scored |

**Primary Track A objective:** drive **wrong_AUTO → 0**.  
**Secondary:** then reduce REVIEW without increasing wrong_AUTO.

Do **not** optimize a single useful-rate that rewards REVIEW-all.

---

## Scoring tool

```bash
python3 run_auto_contract_score.py --selftest
python3 run_auto_contract_score.py --from-artifacts phase7
python3 run_auto_contract_score.py --from-artifacts phase5
```

Writes `artifacts/auto-contract-score-*.json` without overwriting Phase 1–14 panels.

---

## Failure family (Track A breadth + Track B science)

**Development / diagnostic set (inspect during design):** `data/temporal-family-dev.json`  
Same predicate `FIRST_LINE_THERAPY_FAILED`. **Not held-out** once used to design G3 or RepEng.

**Frozen unseen test:** `data/temporal-family-test.json` (n=12, `TFT_*`) — [TEMPORAL-FAMILY-TEST.md](./TEMPORAL-FAMILY-TEST.md).  
**Do not open evidence** while designing. Score with `--set temporal-test` only after intervention freeze.

Baseline command (unchanged Phase-7 Dual stack — **do not tighten G3 first**):

```bash
python3 run_phase7.py --set temporal-dev
python3 run_auto_contract_score.py --from-artifacts temporal-dev --paths Dual_full,D_full,C_full
python3 run_auto_contract_score.py --diagnose --from-artifacts temporal-dev
```

**Baseline write-up (2026-09-03):** [TEMPORAL-FAMILY-DEV-BASELINE.md](./TEMPORAL-FAMILY-DEV-BASELINE.md) — G3 not changed.  
**Track B (DEV cluster):** [TRACKB-FALSEX-CLUSTER.md](./TRACKB-FALSEX-CLUSTER.md) · `python3 run_trackb_falsex.py`

---

## Claim hygiene

- Say: “under this contract, wrong AUTO on set S was N.”  
- Do not say: clinical safety validated; universal CXR; Track B closed.
