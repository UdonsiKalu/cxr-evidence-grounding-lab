# Research approach — N2S fidelity (RepEng / MI)

**Status:** downstream rounded up (2026-09-06); upstream **U0+U1 done** (U1 null on recipe sites).  
**Lab-scale only.** Not clinical product claims. Not CXR production.

**Demos (replay only, no models):**  
[Grounding Ph1–4](https://udonsikalu.github.io/cxr-evidence-grounding-lab/) · [RepEng Workbench Ph9–11](https://udonsikalu.github.io/cxr-repeng-workbench/)

**Related:** [ARCHITECTURE-DIRECTION.md](./ARCHITECTURE-DIRECTION.md) · [EVALUATION-JOURNEY.md](../EVALUATION-JOURNEY.md) · phase protocols `PHASE5`…`PHASE14`  
**Downstream closeout:** [TRACKB-DOWNSTREAM-CLOSEOUT.md](./TRACKB-DOWNSTREAM-CLOSEOUT.md) · α=8 [TRACKB-ALPHA8-FREEZE.md](./TRACKB-ALPHA8-FREEZE.md)  
**System claim (one-pager):** [N2S-SYSTEM-CLAIM.md](./N2S-SYSTEM-CLAIM.md)  
**Living record of advances:** [ADVANCE-LOG.md](./ADVANCE-LOG.md) — append one entry per advance (activity + result + artifacts).

---

## 0. System characterization (locked 2026-09-06)

Three pieces — do not collapse them:

```text
Doctor's note
      ↓
      ↓  UPSTREAM — How does note information become / evolve as internal reps?
      ↓
Tokenization / input-token processing
      ↓
Layer-by-layer internal computation
      ↓
Formation / evolution of representations
      ↓
────────────────────────────
      ↓  DOWNSTREAM — How do those reps drive generation / structured output?
      ↓
Those representations influence generation
      ↓
Output-token generation → LLM structured output
      ↓
════════════════════════════
   NEURAL → SYMBOLIC BOUNDARY  (hard cut)
════════════════════════════
      ↓
CXR symbolic representation → rules → Decision (AUTO | REVIEW)
```

| Piece | Question | Experimental cut |
|-------|----------|------------------|
| **Upstream** | Formation / evolution of reps from the note | Prefill; cue token × layer; residual / SAE / writes; then causal |
| **Downstream** | Utilization / readout into structured neural output | Commit / generation (`contradiction.present`); steer / SAE-at-commit |
| **Boundary** | Faithful conversion into CXR symbols + gates | Track A: extract → ground → Dual → mismatch → AUTO/REVIEW |

Upstream vs downstream is a useful **partition** (formation → utilization), **not** a physical line inside the transformer — the residual stream evolves continuously. The neural→symbolic boundary **is** a genuine external cut.

**Map:** Track B MI straddles upstream + downstream. Track A owns the boundary. Fidelity = does decision-critical meaning survive note → reps → structured output → symbols?

**Status:** Downstream + boundary **rounded up**. Upstream **U0/U1/U1b null** closed for freeze-`v`×cue. **U-A program started** — multi-token formation map; see [TRACKB-UPSTREAM-PROGRAM.md](./TRACKB-UPSTREAM-PROGRAM.md).

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

### Downstream eval workbench + expand editor (2026-09)

| Piece | Status | Doc / surface |
|-------|--------|----------------|
| Expand L20 **α=8** limited editor | **Frozen** | [TRACKB-ALPHA8-FREEZE.md](./TRACKB-ALPHA8-FREEZE.md) |
| N2S Eval UI `:8257` | Evaluate → Deep dive → Intervene → SAE | `cxr-n2s-eval-workbench/` |
| Chanin L20 SAE pilot | Implemented; claim-hygiene | [TRACKB-SAE-PILOT.md](./TRACKB-SAE-PILOT.md) |
| Downstream roundup | **Closed for expansion** | [TRACKB-DOWNSTREAM-CLOSEOUT.md](./TRACKB-DOWNSTREAM-CLOSEOUT.md) |
| CLI implement sketches | W01–W08 Python Minimal patterns | `docs/N2S-CLI-Walkthrough.pdf` |

---

## 3. Method map (industry-relevant MI toolkit)

Use a method when it answers a gate or transfer question — not as a checklist for show.

| Method | Status in this lab | Next use |
|--------|-------------------|----------|
| Behavior panels | **Done** Ph1–11 | Expand via failure **family** |
| **Probing** | **Done** — false-X vs contradiction separable at commit (gap 0.072, LOO 1.00, n=5) | Scale n; hold-out probe |
| Localization | **Done** Ph9B | Reuse sites; refine per family |
| **Activation patching** | **Done** — earliest sufficient **L20** (3/3); L4–L16 0/3; not class-selective (destroys true X) | Family-level steer at L20; see [TRACKB-FALSEX-CLUSTER.md](./TRACKB-FALSEX-CLUSTER.md) |
| **RepEng / steering** | **Frozen** expand α=8 partial editor | Do not expand α/family; reuse **v** for upstream |
| **Ablation** | **Done** α=−4 — sign-consistent, underpowered | Only if upstream needs sign checks |
| **Generalization + controls** | **Partial** Ph11 + α=8 freeze controls | In-family later; not blocking upstream |
| Intervention frameworks (pyvene / TL) | Optional later | Refactor when science stable; hooks OK now |
| **SAE** | **Pilot done** (Chanin L20 @ commit) | Reuse encode on **prefill** sites next |
| **Upstream prefill** | **U0+U1+U1b null** | Program: [TRACKB-UPSTREAM-PROGRAM.md](./TRACKB-UPSTREAM-PROGRAM.md) U-A→U-D |
| **U-A multi-token map** | **Done** | L24 strongest; cos(d,v)≈0.03 |
| **U-B component ablate** | **Null @ L24 zero** | — |
| **U-B2 mean @ L20** | **Null → pause** | Do not open U-C/D from null |
| **Circuits** | C0 scaffold only | After upstream has a causal story |

---

## 4. Resolution plan

### Done — downstream roundup (2026-09-06)

Stop expanding workbench downstream MI unless a new fidelity question requires it. Boundary (Track A) remains the fidelity judge.

### Done — upstream U0+U1 (2026-09-06)

U0 trace/readout; U1 `prefill_position_steer` ±α=8 @ L20 — **no commit X flip** on temporal or contra recipe sites.

### Done — options pass (2026-09-06)

Live Dual confirm (wrong_AUTO=0, correct_AUTO=7); sealed `--tracka-resim-test` unchanged 0/5/7; U1b null @ L16. See [TRACKA-RESIDUAL.md](./TRACKA-RESIDUAL.md) · [TRACKB-UPSTREAM-PREFILL.md](./TRACKB-UPSTREAM-PREFILL.md).

### Done — U-B2 L20 mean-ablate (2026-09-06)

Also **null**. Ablation family paused. See [TRACKB-UPSTREAM-PROGRAM.md](./TRACKB-UPSTREAM-PROGRAM.md).

### Next — wait go

**Upstream ablation family left alone** — portfolio package frozen: [TRACKB-UPSTREAM-PORTFOLIO.md](./TRACKB-UPSTREAM-PORTFOLIO.md). **U-A paraphrase gen panel:** soft+strong YES — [TRACKB-UPSTREAM-UA-GENERALIZE.md](./TRACKB-UPSTREAM-UA-GENERALIZE.md). Do not open U-C SAE or U-D circuits from null. Optional next = causal follow-up only with explicit go. No α·v reopen.

### Still open (wait go; not blocking)

- U-B → U-C → U-D only after prior exit. No upstream browser until science stable.  
- Phase 0 AUTO contract harness breadth — parallel Track A engineering.  
- Llama residual UNCERTAIN-shaped — separate.

### Reconnect (unchanged north star)

Mechanistic detector or recovery → fewer REVIEWs **without** more wrong AUTO.

---

## 5. Repos and demos

| Repo | Role |
|------|------|
| [cxr-evidence-grounding-lab](https://github.com/UdonsiKalu/cxr-evidence-grounding-lab) | Protocols, `n2s_lab/`, frozen `artifacts/`, Ph1–14 runners |
| [cxr-repeng-workbench](https://github.com/UdonsiKalu/cxr-repeng-workbench) | Operator UI + Pages replay of Ph9–11 |
| `cxr-n2s-eval-workbench` | Live Evaluate / Deep dive / Intervene / SAE (:8257) + CLI walkthrough |

Curriculum PDFs (`cxr-repeng-curriculum`, `cxr-mi-repeng-grounding`) are **study / portfolio** tracks — not this research spine.

---

## 6. Claim hygiene

**Say:** lab-scale fidelity failures; gated abstention; causal pilot on BC_E1 / expand family; partial α=8 editor; SAE pilot without English labels; downstream utilization characterized; upstream formation next.  
**Do not say:** clinical validation; universal understanding; production CXR replacement; “steering fixed healthcare”; “found the temporality neuron”; upstream already done.

When Track A and Track B both exist in public docs, the portfolio line is:

> Safety layer prevents uncertain neural→symbolic transforms from becoming wrong AUTO; separately, MI/RepEng investigates why those commits fail and whether targeted interventions recover them under controls — first at utilization (downstream), then at formation (upstream).
