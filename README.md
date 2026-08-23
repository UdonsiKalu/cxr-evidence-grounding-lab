# Neural-to-Symbolic Evidence Grounding Lab

Isolated Milestone 1–3 mock-up. **Does not modify** Claim Studio, rehearsal UI, `cxrlabs-dev`, or `claim_analysis_tools`.

**Question:** Can a system reliably determine whether evidence expressed in natural language satisfies a formally defined condition?

This folder tests that on **one** predicate, **~20** hand-authored snippets, and a **direct-LLM baseline** so the structured pipeline can be compared rather than merely demonstrated.

**Write-up notes (evaluation journey):** [EVALUATION-JOURNEY.md](./EVALUATION-JOURNEY.md) · **PDF:** [notes/progress-notes.pdf](./notes/progress-notes.pdf)

| App | Port / URL |
|-----|------|
| **This lab (local)** | **8253** |
| **Diagnostic panel (local)** | **8254** (`../cxr-evidence-grounding-lab-panel/`) |
| **Public findings demo** | [docs/](./docs/) — **artifact replay only** (GitHub Pages; no live inference) |
| Reasoning Desk lab | 8252 |
| Rehearsal UI | 8251 |
| Claim Studio | 8250 |

### Public demo (GitHub Pages, no GPU)

Frozen Phase 1–4 results in the browser — same A–D / loss-stage explainer, **no** calls to your machine or Ollama.

```bash
python3 scripts/prepare-github-pages.py
cd docs && python3 -m http.server 8765   # http://127.0.0.1:8765/
```

On GitHub: **Settings → Pages → Deploy from branch → `/docs`**. Details: [docs/README.md](./docs/README.md).

## Milestone 1 scope

- Predicate: `FIRST_LINE_THERAPY_FAILED`
- Categories: explicit, implicit, temporal, uncertain, conflicting (4 each)
- UI: original evidence → neural extraction → grounding atoms → symbolic predicate → rule result
- Experiment runner: expected vs **direct LLM** vs **pipeline**; where they differ
- **Not in M1:** Qdrant, CXR kernel, Archetypes, Next.js, Docker, more predicates, accuracy tuning

## Milestone 2 (contradiction slice)

Same 20 cases and the same symbolic rule. The extractor must now always fill `contradiction.present` as **true** (with two incompatible spans) or **false** (explicit none). Grounding sets **X** from that field. Compare whether the pipeline can emit CONTRADICTION on C1–C4 while the direct LLM still will not.

## Milestone 2.1 (U4 hygiene)

Leave **C4** failing (temporal/identity class). Prompt + grounding: unresolved which-line / mixed / pending is **uncertainty**, not X. Success = C1–C3 still CONTRADICTION, U4 not X.

## Milestone 3 (uncertainty slice)

Same 20 cases and the same symbolic rule. The extractor must now always fill `uncertainty.present` as **true** (cue + why) or **false** (explicit none). Grounding keeps implicated atoms unknown so the rule can fire UNCERTAIN. Success = U1–U4 pipeline UNCERTAIN, C1–C3 still CONTRADICTION, U4 X false, C4 still in the failure set.

## Optional Phase-1 (removable)

Transition diagnostic: where meaning is lost (Conditions A–D on C1–C4 + U1–U4). See [docs/PHASE1-DIAGNOSTIC.md](./docs/PHASE1-DIAGNOSTIC.md). Does not change M1–M3. Undo by deleting that doc, `run_phase1_diagnostic.py`, `n2s_lab/phase1_diagnostic.py`, and `artifacts/phase1-*.json`.

```bash
python3 run_phase1_diagnostic.py
```

## Optional Phase-2 (same prompts, model panel)

Same A–D conditions and C1–C4+U1–U4 gold. **No** per-model prompt tuning. Local panel: Llama 8B Q4 (control), Mistral Instruct, Qwen 2.5 Coder 32B.

```bash
N2S_OLLAMA_TIMEOUT=600 python3 run_phase2_panel.py
```

Writes `artifacts/phase2-model-panel.json`. Unseen/held-out cases are **not** in this runner.

## Optional B′ ablation (frozen B kept)

Same 8 Phase-2 analyses. Mapping prompt on the verdict step only. Does not overwrite Phase-1/2 artifacts.

```bash
python3 run_bprime_ablation.py
```

## Optional Phase-3 (held-out C/U, frozen A–D)

New wording C5–C8 / U5–U8 in `data/heldout-phase3.json`. B′ is not the default.

```bash
N2S_OLLAMA_TIMEOUT=600 python3 run_phase3_heldout.py
```

## Optional Phase-4 (2nd held-out C/U, frozen A–D)

New wording C9–C11 / U9–U12 in `data/heldout-phase4.json`. No C4/C8 analog. B′ is not the default.

```bash
N2S_OLLAMA_TIMEOUT=600 python3 run_phase4.py
```

See [docs/PHASE4-PROTOCOL.md](./docs/PHASE4-PROTOCOL.md).

## Run

```bash
cd /home/udonsi-kalu/staging/cxr-evidence-grounding-lab

# Instant: heuristic extractor + no LLM (symbolic path still real)
python3 run_experiment.py --mode mock

# Live: Ollama extract + Ollama direct baseline (needs :11434)
python3 run_experiment.py --mode live

# UI
python3 server.py
```

Open **http://127.0.0.1:8253/**

Default model: `llama3:8b-instruct-q4_0` (override with `N2S_OLLAMA_MODEL`).

## Mock vs functional

| Component | M1 status |
|-----------|-----------|
| Formal predicate + symbolic rule | **Functional** (deterministic Python) |
| Grounding (extraction → atoms) | **Functional** (deterministic Python) |
| Experiment harness + gold labels | **Functional** |
| Direct LLM baseline | **Functional** if Ollama is up; skipped in `--mode mock` |
| Neural extraction | **Functional** if Ollama is up; **heuristic mock** otherwise |
| UI | **Functional** viewer/runner |
| CXR analyzer / Qdrant / Archetypes | **Not used** (intentionally) |

`--mode mock` is for inspecting the symbolic boundary without waiting on the model. It is **not** a research result.

## Success criterion (version 1)

Visually show the same formal requirement, very different natural-language expressions, neural interpretation, and a controlled symbolic representation — and mark succeed / fail / uncertain / contradiction.

Do **not** optimize accuracy yet. The point is to make the neural-to-symbolic boundary observable, and to see whether the pipeline disagrees with a direct LLM on the same 20 statements.

## Undo

```bash
rm -rf /home/udonsi-kalu/staging/cxr-evidence-grounding-lab
```

`:8250` / `:8251` / `:8252` are unaffected.
