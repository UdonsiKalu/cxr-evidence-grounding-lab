# Advance log — N2S fidelity / RepEng / MI

**Purpose:** One place to record each advance — what we did, what happened, where the evidence lives.  
**Use for:** papers, portfolio, faculty notes, handoff.  
**Not:** chat transcript. Keep entries short; link detail docs/artifacts.

**How to add an entry (every advance):** copy the template below to the **top** of the log (newest first).

---

## Template (copy)

```markdown
### YYYY-MM-DD — short title

| | |
|--|--|
| **Track** | A reliability / B mechanism / both / infra |
| **Activity** | 1–3 bullets of what we ran or built |
| **Result** | Numbers or qualitative outcome (wrong_AUTO, gates, transfer…) |
| **Artifacts** | paths / URLs |
| **Decision** | what we will / will not do next because of this |
| **Commit** | `sha` if pushed |
```

---

## Log (newest first)

### 2026-09-03 — Temporal-family-dev Dual baseline (measure before G3)

| | |
|--|--|
| **Track** | A (measure) → informs B |
| **Activity** | Renamed set to `temporal-family-dev.json` (dev, not held-out). Ran **unchanged** Phase-7 Dual on n=14 × 3 models. Case-level diagnose; **did not** change G3. |
| **Result** | Dual_full: Mistral wrong_AUTO=**0** / REVIEW=14; Llama wrong=**2** (UNCERTAIN); **Qwen wrong=5** with false-X cluster **TF_E3, BC11_E3, TF_E4** (+ TF_T1/TF_N1 overfire). Genuine failure family, not BC_E1 alone. |
| **Artifacts** | [TEMPORAL-FAMILY-DEV-BASELINE.md](./TEMPORAL-FAMILY-DEV-BASELINE.md) · `artifacts/phase7-temporal-dev-panel.json` · `auto-contract-diagnose-temporal-dev.json` · `auto-contract-score-temporal-dev.json` |
| **Decision** | Freeze `temporal-family-test.json` before gate redesign; Track B on {TF_E3, BC11_E3, TF_E4}; G3 still deferred. |
| **Commit** | `a91d317` (wiring) · `fe3cea1` / `28e81c1` (baseline) |

### 2026-09-03 — Track A AUTO contract + scorer

| | |
|--|--|
| **Track** | A |
| **Activity** | Froze AUTO rule (AUTO iff gates pass, else REVIEW). Built `run_auto_contract_score.py` + `n2s_lab/auto_contract.py`. Scored frozen Phase-5/7 panels. |
| **Result** | Ph7 Dual_full on BMT/CAR-T: Qwen wrong_AUTO=1 (`BC_E1`); Llama/Mistral Dual_full wrong_AUTO=0. Metric = wrong_AUTO → 0. |
| **Artifacts** | [AUTO-CONTRACT.md](./AUTO-CONTRACT.md) · `artifacts/auto-contract-score-phase7.json` · `artifacts/auto-contract-score-phase5.json` |
| **Decision** | Measure on temporal family next; do not redesign G3 from BC_E1 alone. |
| **Commit** | `3db62c0` |

### 2026-09-03 — Research spine documented (Track A vs B)

| | |
|--|--|
| **Track** | infra / framing |
| **Activity** | Wrote research spine: Track A (wrong AUTO/REVIEW) vs Track B (MI causal ladder); method map; claim hygiene. Linked from READMEs. |
| **Result** | Gates **contain** failures; they do not **explain** them — two-track plan frozen in docs. |
| **Artifacts** | [RESEARCH-APPROACH.md](./RESEARCH-APPROACH.md) · pointers in ARCHITECTURE-DIRECTION, EVALUATION-JOURNEY |
| **Decision** | Build AUTO contract then measure; Track B deepens mechanism separately. |
| **Commit** | `967b6cc` (lab) · workbench README `953f8e0` |

### 2026-09-02 — Public demos + modus operandi

| | |
|--|--|
| **Track** | infra |
| **Activity** | GitHub Pages replay for grounding lab + RepEng workbench (pre-run JSON, no models). First-time Path A/B/C READMEs. |
| **Result** | Live demos: [grounding](https://udonsikalu.github.io/cxr-evidence-grounding-lab/) · [workbench](https://udonsikalu.github.io/cxr-repeng-workbench/) |
| **Artifacts** | lab `docs/` · workbench `docs/explorer.json` + `workbench.json` |
| **Decision** | Demos are replay-only; live Run stays local/GPU. |
| **Commit** | workbench Pages `43cfc79`; MO READMEs `2f1a0c0` / `dd85aba` |

### Pre-2026-09 (frozen soft pilot — summary)

| | |
|--|--|
| **Track** | A precursors + B soft pilot |
| **Activity** | Ph1–8 grounding / verify / Dual / evidence; Ph9–11 observe→localize→steer→partial transfer on BC_E1 (HF 7B). |
| **Result** | Representation loss visible (A–D); BC_E1 false X steerable at α=4 layers 20/27; Ph11 partial. |
| **Artifacts** | `docs/PHASE5`…`PHASE11` · `artifacts/phase*-*.json` · Workbench UI |
| **Decision** | Soft pilot frozen; next wave = contract + family + deeper MI. |
| **Commit** | see phase panels on `main` |

---

## Index of related docs

| Doc | Role |
|-----|------|
| [RESEARCH-APPROACH.md](./RESEARCH-APPROACH.md) | Plan / method map |
| [AUTO-CONTRACT.md](./AUTO-CONTRACT.md) | Track A gate rule |
| [TEMPORAL-FAMILY-DEV-BASELINE.md](./TEMPORAL-FAMILY-DEV-BASELINE.md) | Latest Dual baseline detail |
| [ARCHITECTURE-DIRECTION.md](./ARCHITECTURE-DIRECTION.md) | Safe protocol north star |
| [EVALUATION-JOURNEY.md](../EVALUATION-JOURNEY.md) | Early milestone narrative |
| Phase protocols `PHASE*.md` | Per-phase freeze records |
