# Research approach — N2S fidelity (RepEng / MI)

**Status:** housekeeping freeze (2026-09-03) — documents the innovative arc before the next build wave.  
**Lab-scale only.** Not clinical product claims. Not CXR production.

**Demos (replay only, no models):**  
[Grounding Ph1–4](https://udonsikalu.github.io/cxr-evidence-grounding-lab/) · [RepEng Workbench Ph9–11](https://udonsikalu.github.io/cxr-repeng-workbench/)

**Related:** [ARCHITECTURE-DIRECTION.md](./ARCHITECTURE-DIRECTION.md) · [EVALUATION-JOURNEY.md](../EVALUATION-JOURNEY.md) · phase protocols `PHASE5`…`PHASE14`

---

## 1. Scientific problem

Language models propose meaning from messy notes. Fixed rules need precise atoms. At the **neural → symbolic** handoff, decision-critical meaning can die or be invented.

**Central failure (pilot):** case **BC_E1** — note describes therapy that later failed (not a contradiction); model commits false `contradiction.present` after repair → wrong AUTO CONTRADICTION risk.

**Two questions (do not collapse them):**

| Track | Question | “Resolved” means |
|-------|----------|------------------|
| **A — Reliability (industry)** | How do we stop a fidelity failure from becoming a wrong automatic decision? | Wrong AUTO ≈ 0; REVIEW when gates fail |
| **B — Mechanism (MI / RepEng)** | Why did the wrong symbolic commit happen, and can we causally change it? | Probe → localize → patch/steer → ablate → transfer evidence |

**Gates contain the failure; they do not explain it.** Track A is the engineering control. Track B is the research contribution. They reconnect when a mechanistic finding improves detection or recovery **without** raising wrong AUTO.

---

## 2. What is already done (frozen lab evidence)

### Track A precursors (Ph1–8)

| Phase | Innovation | Artifact / doc |
|-------|------------|----------------|
| 1–4 | Conditions **A–D** — places meaning dies (direct / analysis→verdict / structured / **representation loss**) | `artifacts/phase4-*-panel.json` · Pages demo |
| 5 | Verify → **REVIEW** (never force UNCERTAIN on verify-fail) | [PHASE5-PROTOCOL.md](./PHASE5-PROTOCOL.md) |
| 6 | Round-trip + dual-path faithfulness checks | [PHASE6-PROTOCOL.md](./PHASE6-PROTOCOL.md) |
| 7 | Evidence-preservation object + BMT/CAR-T cross-domain slice | [PHASE7-PROTOCOL.md](./PHASE7-PROTOCOL.md) |
| 8 | Model ladder (14B / 32B instruct) on same stack | [PHASE8-PROTOCOL.md](./PHASE8-PROTOCOL.md) |

North star wording: [ARCHITECTURE-DIRECTION.md](./ARCHITECTURE-DIRECTION.md) — propose → finite contract → verify/abstain.

### Track B soft pilot (Ph9–11 + Workbench)

| Phase | Innovation | Artifact / doc |
|-------|------------|----------------|
| 9A | **Observe** — HF 7B reproduces false X on BC_E1 | `phase9a-bc-e1-panel.json` |
| 9B | **Localize** — commit token; layers 0.75/1.00 (idx 20/27) | `phase9b-*-trace.json` |
| 9C | Output intervention (logit bias) — decision can flip | `phase9c-bc-e1-panel.json` |
| 10 | **RepEng / steering** — BC_E2−BC_E1 vector, α=4 frozen | `phase10-bc-e1-panel.json` · [PHASE10-PROTOCOL.md](./PHASE10-PROTOCOL.md) |
| 11 | **Generalization + controls** — partial (1/2 unseen; controls held) | `phase11-generalization-panel.json` |
| UI | Workbench v2 — Story / Layer / Commit / Why α / Gen + live Run | [cxr-repeng-workbench](https://github.com/UdonsiKalu/cxr-repeng-workbench) |

Stack choice: **HF `transformers` + custom forward hooks** (not a rewrite into TransformerLens). See lab `n2s_lab/hf_trace.py`, `hf_intervene.py`.

Ph12–14 exist as protocol/code extensions; soft pilot claim for portfolio centers on **9–11**.

---

## 3. Method map (industry-relevant MI toolkit)

Use a method when it answers a gate or transfer question — not as a checklist for show.

| Method | Status in this lab | Next use |
|--------|-------------------|----------|
| Behavior panels | **Done** Ph1–11 | Expand via failure **family** |
| **Probing** | Soft (norms / logit margin) | Explicit probe: is temporal distinction decodable at commit? |
| Localization | **Done** Ph9B | Reuse sites; refine per family |
| **Activation patching** | **Not yet** | Clean-run activations → rescue fail run; layer map |
| **RepEng / steering** | **Done** Ph10 | Refit/test on family; freeze discipline |
| **Ablation** | **Not yet** | Suppress direction → correct cases degrade? |
| **Generalization + controls** | **Partial** Ph11 | In-family ~50–200 + paraphrase; then 7B→14B |
| Intervention frameworks (pyvene / TL) | Optional later | Refactor when science stable; hooks OK now |
| SAE / circuits | Optional later | After causal evidence; not required to close Track A |

---

## 4. Resolution plan (before next “go”)

### Phase 0 — AUTO contract

Half-page rule: **AUTO only if all gates pass**; else **REVIEW**.  
Metric: wrong AUTO / REVIEW / correct AUTO on a frozen eval set.

**Frozen:** [AUTO-CONTRACT.md](./AUTO-CONTRACT.md) · scorer `run_auto_contract_score.py` · family `data/heldout-temporal-family.json`

### Phase 1 — Track A harness

Mandatory gates on the Phase-5+ path + one-command score table.  
Breadth first as a **failure family** (temporal distinction), not every clinical domain:

- therapy worked → later failed  
- stable → later progression  
- prior neg → later pos  
- no toxicity → later toxicity  
- possible → confirmed progression  
- stop for toxicity ≠ stop for progression  

### Phase 2 — Track B ladder (per family)

`behavior → probe → localize → patch → steer → ablate → held-out transfer → 14B transfer`

### Phase 3 — Reconnect

Mechanistic detector or recovery → fewer REVIEWs **without** more wrong AUTO.

---

## 5. Repos and demos

| Repo | Role |
|------|------|
| [cxr-evidence-grounding-lab](https://github.com/UdonsiKalu/cxr-evidence-grounding-lab) | Protocols, `n2s_lab/`, frozen `artifacts/`, Ph1–14 runners |
| [cxr-repeng-workbench](https://github.com/UdonsiKalu/cxr-repeng-workbench) | Operator UI + Pages replay of Ph9–11 |

Curriculum PDFs (`cxr-repeng-curriculum`, `cxr-mi-repeng-grounding`) are **study / portfolio** tracks — not this research spine.

---

## 6. Claim hygiene

**Say:** lab-scale fidelity failures; gated abstention; causal pilot on BC_E1 family; partial transfer.  
**Do not say:** clinical validation; universal understanding; production CXR replacement; “steering fixed healthcare.”

When Track A and Track B both exist in public docs, the portfolio line is:

> Safety layer prevents uncertain neural→symbolic transforms from becoming wrong AUTO; separately, MI/RepEng investigates why those commits fail and whether targeted interventions recover them under controls.
