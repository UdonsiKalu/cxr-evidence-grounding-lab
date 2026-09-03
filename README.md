# Neural-to-Symbolic Evidence Grounding Lab

A small, hands-on lab for people new to **neuro-symbolic (NeSy)** AI.

**Try the findings demo (no install, no GPU):**  
https://udonsikalu.github.io/cxr-evidence-grounding-lab/

**Companion (Phase 9–11 steering):** [cxr-repeng-workbench](https://github.com/UdonsiKalu/cxr-repeng-workbench) · [public demo](https://udonsikalu.github.io/cxr-repeng-workbench/)

**Research spine (Track A safety + Track B MI/RepEng):** [docs/RESEARCH-APPROACH.md](./docs/RESEARCH-APPROACH.md) — problem, frozen evidence, method map, next plan.  
**Track A contract:** [docs/AUTO-CONTRACT.md](./docs/AUTO-CONTRACT.md) · `python3 run_auto_contract_score.py --from-artifacts phase7`

---

## Modus operandi (first time here)

Read this once, then pick a path. **Start with the public demo** unless you already know you need live models.

### What this repo is

| Piece | Role |
|-------|------|
| **This lab** | Phases **1–8** — where meaning dies at the **neural → symbolic** boundary (conditions A–D on synthetic notes) |
| **[RepEng Workbench](https://github.com/UdonsiKalu/cxr-repeng-workbench)** | Phases **9–11** — observe, localize, and **steer** a specific failure (BC_E1 false contradiction) |

Same research arc, two repos: **grounding first**, then **mechanistic intervention**.

### Path A — Watch only (recommended first visit)

**No install. No GPU. No models run.**

1. Open **https://udonsikalu.github.io/cxr-evidence-grounding-lab/**
2. Leave artifact on **Phase-4 held-out (C9–U12)** and click **Load**
3. Pick a case in the sidebar (e.g. **C9** or **U10**)
4. Compare columns **A / B / C / D** — verdict, gold match, loss stage
5. Read the note and the extract/ground JSON under each condition

**You are done** when you see that loss is not one thing: direct verdict (A), analysis→verdict (B), and analysis→structure (D) fail differently.

Then continue to the workbench demo: **https://udonsikalu.github.io/cxr-repeng-workbench/** → tab **1 · Story** → case **BC_E1**.

### Path B — Local replay (still no models)

```bash
git clone https://github.com/UdonsiKalu/cxr-evidence-grounding-lab.git
cd cxr-evidence-grounding-lab
python3 scripts/prepare-github-pages.py   # optional; docs/ already in repo
cd docs && python3 -m http.server 8765
# → http://127.0.0.1:8765/
```

Or replay frozen JSON from the CLI:

```bash
python3 run_experiment.py --mode mock
python3 run_phase5.py --selftest    # Phase 5 verify stack
python3 run_phase6.py --selftest    # faithfulness checks
```

### Path C — Live runs (your machine, optional)

**Requirements:** Python 3, [Ollama](https://ollama.com) on `127.0.0.1:11434`, default model `llama3:8b-instruct-q4_0`

```bash
git clone https://github.com/UdonsiKalu/cxr-evidence-grounding-lab.git
cd cxr-evidence-grounding-lab
python3 server.py
# → http://127.0.0.1:8253/
```

| Goal | Command |
|------|---------|
| Live extract + baseline vs pipeline | `python3 run_experiment.py --mode live` |
| Phase 5 verify → REVIEW | `python3 run_phase5.py` |
| Phase 6 round-trip / dual-path | `python3 run_phase6.py` |
| Phase 7 BMT/CAR-T held-out | `python3 run_phase7.py --set bmtcart` |
| Phase 8 model ladder | `python3 run_phase8.py --set both` |

Phases **9–14** (HF + GPU) run from this repo’s scripts; the **Workbench UI** is in the sibling [cxr-repeng-workbench](https://github.com/UdonsiKalu/cxr-repeng-workbench) repo.

### What not to expect

- Not real patient data, not CXR production, not a clinical accuracy claim  
- Public demos **replay frozen JSON only** — they never call your GPU or Ollama  
- Do not expose a public tunnel to your local Ollama for these demos  

---

## What is this about?

Language models are good at reading messy text. Rules and logic are good at being **precise and checkable**.  
Neuro-symbolic systems try to use **both**: the model reads the note; a fixed rule decides the outcome.

The hard part is the handoff:

> When the model “understands” something in English, does that meaning survive when we turn it into a formal structure the rule can use?

That handoff is the **neural → symbolic** boundary. This lab makes it visible.

---

## The toy problem (one rule only)

We ask a single yes/no-style clinical question (synthetic notes, not real patients):

> Did **first-line therapy fail**?

A fixed symbolic rule needs four facts (plus contradiction):

| Atom | Meaning in plain English |
|------|---------------------------|
| **A** | A first-line therapy is identified |
| **B** | It was actually given / attempted |
| **C** | A failure event happened |
| **D** | That failure is of *that* first-line regimen |
| **X** | The note contradicts itself about those facts |

The rule never “wings it.” Same atoms in → same verdict out: satisfied, not satisfied, uncertain, or contradiction.

We compare two paths on the **same** notes:

1. **Ask the LLM directly** for a verdict  
2. **Extract → ground into atoms → apply the rule**

If (1) and (2) disagree, something important was lost at the boundary.

---

## Start here: watch the findings

Open the [public demo](https://udonsikalu.github.io/cxr-evidence-grounding-lab/).

You will see four **conditions** (A–D). They are not four medical rules — they are four *places meaning can die*:

| Condition | Plain meaning |
|-----------|----------------|
| **A** | Direct LLM verdict (no structure) |
| **B** | Model writes an analysis, then gives a final answer — does the answer keep what the analysis said? |
| **C** | Full pipeline: extract → atoms → fixed rule |
| **D** | Analysis had the distinction, but the **structured form** dropped it (**representation loss**) |

Pick a case, read the note, and compare the columns. That is the whole teaching point.

The demo only **replays frozen results**. It does not run models on anyone’s computer.

---

## What we found (lab-scale, short)

- Putting **contradiction** and **uncertainty** into the structured form helps the rule keep states the raw LLM often narrates and then discards.  
- Loss is **not one thing**: A, B, and D fail in different ways.  
- On held-out notes, **representation loss (D)** still showed up across local models — including a larger one that looked clean on the discovery set.  
- This is a **small synthetic lab**, not a clinical product and not a claim of production accuracy.

**Guides:** [simple illustrated PDF](./notes/progress-notes-simple.pdf) · [full progress notes](./notes/progress-notes.pdf) · [EVALUATION-JOURNEY.md](./EVALUATION-JOURNEY.md)  
**Direction (safe protocol / verify → REVIEW):** [docs/ARCHITECTURE-DIRECTION.md](./docs/ARCHITECTURE-DIRECTION.md) · [docs/PHASE5-PROTOCOL.md](./docs/PHASE5-PROTOCOL.md) · [docs/PHASE6-PROTOCOL.md](./docs/PHASE6-PROTOCOL.md)

---

## Run locally (optional)

Needs Python 3. Live LLM runs need [Ollama](https://ollama.com) on your machine (`127.0.0.1:11434`).

```bash
git clone https://github.com/UdonsiKalu/cxr-evidence-grounding-lab.git
cd cxr-evidence-grounding-lab

# UI — baseline vs pipeline on the locked 20 notes
python3 server.py
# → http://127.0.0.1:8253/
```

Sibling UIs (separate folders; share this lab’s `artifacts/` + `n2s_lab/`):

| Port | Folder | Role |
|------|--------|------|
| **8253** | this repo | M1 demo |
| **8254** | `../cxr-evidence-grounding-lab-panel/` | Conditions A–D (Phase 1–4) |
| **8255** | `../cxr-evidence-grounding-lab-safety/` | Phase 5–7 safety stack (verify / Dual / evidence) |
| **8256** | `../cxr-evidence-grounding-lab-repeng/` | Phase 9–11 rep-eng explorer (replay) · [public demo](https://udonsikalu.github.io/cxr-repeng-workbench/) |

| Mode | Command | Needs Ollama? |
|------|---------|----------------|
| Browse frozen / mock path | `python3 run_experiment.py --mode mock` | No |
| Live extract + baseline | `python3 run_experiment.py --mode live` | Yes |
| Replay Pages demo locally | `cd docs && python3 -m http.server 8765` | No |
| Phase-5 verify selftest | `python3 run_phase5.py --selftest` | No |
| Phase-5 live (verify→REVIEW) | `python3 run_phase5.py` | Yes |
| Phase-6 faithfulness selftest | `python3 run_phase6.py --selftest` | No |
| Phase-6 live (round-trip + dual-path) | `python3 run_phase6.py` | Yes |
| Phase-7 evidence + BMT/CAR-T selftest | `python3 run_phase7.py --selftest` | No |
| Phase-7 live (BMT/CAR-T held-out) | `python3 run_phase7.py --set bmtcart` | Yes |
| Phase-7 temporal-family **dev** baseline | `python3 run_phase7.py --set temporal-dev` | Yes |
| Phase-7 oncology replay + evidence | `python3 run_phase7.py --set phase4` | Yes |
| Phase-8 ladder selftest | `python3 run_phase8.py --selftest` | No |
| Phase-8 ladder (14B + non-coder 32B) | `python3 run_phase8.py --set both` | Yes |
| Phase-9A selftest | `python3 run_phase9a.py --selftest` | No |
| Phase-9A BC_E1 HF reproduce | `.venv-phase9/bin/python run_phase9a.py` | HF download |
| Phase-9B BC_E1 localization | `.venv-phase9/bin/python run_phase9b.py` | HF; requires 9A YES |
| Phase-9C BC_E1 intervention | `.venv-phase9/bin/python run_phase9c.py` | HF; requires 9B YES |
| Phase-10 activation steer | `.venv-phase9/bin/python run_phase10.py` | HF; requires 9B YES |
| Phase-11 generalization | `.venv-phase9/bin/python run_phase11.py` | HF; requires Ph10 YES |
| Rep-eng explorer UI | `cd ../cxr-evidence-grounding-lab-repeng && python3 server.py` | No (replay JSON) |

Default live model: `llama3:8b-instruct-q4_0` (override with `N2S_OLLAMA_MODEL`).

Phase-5 adds **L2 verification + L3 REVIEW** (never force UNCERTAIN on verify-fail). See [docs/PHASE5-PROTOCOL.md](./docs/PHASE5-PROTOCOL.md). Does not overwrite Phase-1–4 artifacts.

Phase-6 adds **L2b round-trip** and **L2c dual-path agreement** on top of Phase-5 (experiments toward general faithfulness checks — not a proof). See [docs/PHASE6-PROTOCOL.md](./docs/PHASE6-PROTOCOL.md). Does not overwrite Phase-1–5 artifacts.

Phase-7 adds **L2d evidence-preservation object** (claim + source span + verify) and a **BMT/CAR-T cross-domain** held-out slice (`data/heldout-bmtcart-phase7.json`). See [docs/PHASE7-PROTOCOL.md](./docs/PHASE7-PROTOCOL.md). Does not overwrite Phase-1–6 artifacts.

Phase-8 adds a **model ladder** (`qwen2.5:14b` + non-coder `qwen2.5:32b`) on the same Phase-7 stack / frozen slices. See [docs/PHASE8-PROTOCOL.md](./docs/PHASE8-PROTOCOL.md). Does not overwrite Phase-1–7 citation panels.

Phase-9 is a **representation-engineering pilot** on BC_E1 false `contradiction.present` (HF activations; not Dual fix). **Frozen completed pilot** — see [docs/PHASE9-NOTES-SIMPLE.md](./docs/PHASE9-NOTES-SIMPLE.md) and [docs/PHASE9-PROTOCOL.md](./docs/PHASE9-PROTOCOL.md). **Phase-10 frozen:** activation steer gate YES — [docs/PHASE10-PROTOCOL.md](./docs/PHASE10-PROTOCOL.md). **Phase-11 frozen:** fixed-vector generalization gate NO (partial) — [docs/PHASE11-PROTOCOL.md](./docs/PHASE11-PROTOCOL.md). **Rep-eng explorer:** `../cxr-evidence-grounding-lab-repeng/` on **:8256**.
