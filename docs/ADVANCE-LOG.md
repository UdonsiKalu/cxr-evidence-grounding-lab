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

### 2026-09-03 — Track B false-X cluster ladder (DEV) — probe YES, patch YES, steer NO

| | |
|--|--|
| **Track** | B |
| **Activity** | Ran probe→localize→activation patch→Ph10 steer(α=4)→ablate(α=−4) on {TF_E3, BC11_E3, TF_E4} vs TF_C1/BC_C1 + TF_N1/BC_E2 on HF `Qwen2.5-7B-Instruct` (`.venv-phase9`, 240 W). Test set not opened. Caught a **final-layer patch confound** and re-ran the patch arm at 0.75 only. |
| **Result** | Family **reproduces on HF 7B** (3/3 false X). **Probe YES** — cos@1.00 false-X [−0.713,−0.627] vs contradiction [−0.554,−0.530], no overlap, gap 0.072, LOO 1.00 (replicates Ph13 with a wider gap). **Steer α=4: 0/3 flips** — frozen BC_E1 vector does not transfer; controls fully intact. **Ablate α=−4** raises margins (sign-consistent, underpowered). **Patch @0.75 only: 3/3 flips**, margins 6.875→−11.375, 2.75→−11.5, 9.5→−11.375 (≠ donor −14.0, so genuine recompute). Full-depth patch incl. 1.00 was **discarded** — margins were exactly the donor's, i.e. token forcing, not sufficiency. |
| **Artifacts** | [TRACKB-FALSEX-CLUSTER.md](./TRACKB-FALSEX-CLUSTER.md) · `artifacts/trackb-falsex-cluster-panel.json` · `artifacts/trackb-falsex-patch-0_75-panel.json` · `run_trackb_falsex.py` · `n2s_lab/trackb_falsex_cluster.py` |
| **Decision** | Do **not** claim steering fixes this family. Next: α sweep on cluster, refit vector on cluster (not BC_E1), layer sweep for earliest sufficient patch depth. G3 untouched; `temporal-test` stays sealed until an intervention is frozen. |
| **Commit** | _(pending push)_ |

### 2026-09-03 — Env provenance: `.venv-phase9` is the HF runtime

| | |
|--|--|
| **Track** | infra |
| **Activity** | Track B run failed on bare `python3` (`accelerate` missing). Traced the HF stack: repo-local `.venv-phase9` (created 2026-08-24, `--system-site-packages`) layers accelerate 1.14.0 + bitsandbytes 0.50.1 over system torch 2.5.1+cu121 / transformers 4.53.2. |
| **Result** | Nothing had been uninstalled — those two packages only ever lived in that venv. `cxrlabs/faiss_gpu1` also works (accelerate 1.10.0); `cxrlabs-dev/faiss_gpu1` and `cxr-migration/faiss_gpu1` are broken (no interpreter / dangling base on an unmounted drive). |
| **Artifacts** | `.venv-phase9/pyvenv.cfg` · run commands in [TRACKB-FALSEX-CLUSTER.md](./TRACKB-FALSEX-CLUSTER.md) |
| **Decision** | All HF/GPU phases run via `./.venv-phase9/bin/python`. No package installs needed. |
| **Commit** | _(pending push)_ |

### 2026-09-03 — Freeze temporal-family-test.json

| | |
|--|--|
| **Track** | A (eval harness) |
| **Activity** | Froze unseen n=12 `TFT_*` set; wired `--set temporal-test` + scorer preset; seal doc. |
| **Result** | Test sealed before Track B design on DEV. |
| **Artifacts** | `data/temporal-family-test.json` · [TEMPORAL-FAMILY-TEST.md](./TEMPORAL-FAMILY-TEST.md) |
| **Decision** | Do not peek evidence while designing; do not run Dual on test until intervention frozen. |
| **Commit** | _(pending push)_ |

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
