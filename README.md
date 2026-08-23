# Neural-to-Symbolic Evidence Grounding Lab

A small, hands-on lab for people new to **neuro-symbolic (NeSy)** AI.

**Try the findings demo (no install, no GPU):**  
https://udonsikalu.github.io/cxr-evidence-grounding-lab/

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

| Mode | Command | Needs Ollama? |
|------|---------|----------------|
| Browse frozen / mock path | `python3 run_experiment.py --mode mock` | No |
| Live extract + baseline | `python3 run_experiment.py --mode live` | Yes |
| Replay Pages demo locally | `cd docs && python3 -m http.server 8765` | No |

Default live model: `llama3:8b-instruct-q4_0` (override with `N2S_OLLAMA_MODEL`).
