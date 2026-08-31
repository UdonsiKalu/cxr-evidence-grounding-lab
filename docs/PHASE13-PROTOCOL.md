# Phase-13 protocol — representation geometry (course false-X vs contradiction)

**Status:** **FROZEN** 2026-08-25 — gate **13 YES** (soft); live record in `notes/progress-notes.pdf`; first attempt OOM’d then fixed (KV-cache / `logits_to_keep=1` / pin 7B / `del model`); no adaptive α.  
**Builds on:** Ph10 vector; Ph11/12A/12B frozen.  
**Not:** Phase-12C adaptive α; α retune; new steering vector.

## Live record (brief)

| Class | n usable | cos@1.00 range |
|-------|----------|----------------|
| course false-X | 5/5 | [−0.724, −0.700] |
| true contradiction | 4/4 | [−0.679, −0.554] |
| overlap | **no** (gap ≈ 0.021) | |
| LOO probe | **0.889** | |
| Gate | **YES** (soft separability) | |

Artifact: `artifacts/phase13-geometry-panel.json`. Soft claim only — pilot geometry, not adaptive α.

## Thermal ops (performance-only)

270 W still reached **~86°C** at 100% util (fan ~81%, power pegged at limit).  
Before Ph13 live: set **`sudo nvidia-smi -pl 240`** (Research profile updated to 240 W on `:8256`).  
Expect longer wall time. If still ≥85°C, next step is case airflow / Quiet **200 W**, not science retune.

## Question

> At the repair `contradiction.present` commit, do **false-X course** targets and **true-contradiction** controls occupy **separable** regions in hidden space (relative to the frozen Ph10 steering direction and/or commit activations)?

**Why now:** Ph12B showed the same vector moves course margins but can suppress a true contradiction at α=4. Adaptive strength is premature until we know whether the two classes are geometrically separable.

## Frozen inheritance

| Item | Value |
|------|--------|
| Model | `Qwen/Qwen2.5-7B-Instruct` |
| Layers | 0.75, 1.00 (idx 20/27) |
| Vector | Ph10 BC_E2−BC_E1 (unit direction for cosine only) |
| Stage | Baseline repair path (**no** activation steer) |
| Power | Prefer **240 W** |

## Case sets (existing notes only — no new wording)

**Course false-X pool** (prior baseline repair X=true; gold SATISFIED course):

- BC11_E2, BC11_E3  
- BC12_E1, BC12_E2, BC12_E3  

**True-contradiction pool** (gold CONTRADICTION):

- BC11_C1, BC11_C2  
- BC12_C1  
- BC_C1 (Phase-7/10 control, if loadable)

Skip cases that fail to produce a commit token.

## Per-case protocol

1. Build repair prompt (same as Ph9B/Inspect).  
2. `generate_trace` (greedy); find `contradiction.present` commit.  
3. Record: commit value, margin, ‖h‖ @ 0.75/1.00, **cosine(h, Ph10 steer direction)** @ each layer.  
4. Unload between cases (thermal).

## Analysis (pre-declared)

1. **Cosine-to-vector:** distributions for course vs contradiction at layers 20/27.  
2. **Separability heuristic (soft):** ranges of cosine @ 1.00 for the two classes — report overlap / gap.  
3. **Optional 2-feature probe:** leave-one-out linear probe on `[cos_0.75, cos_1.00]` if both classes have ≥3 usable commits — accuracy vs chance (0.5); **soft only**.

## Gate (13)

**13 YES (observational)** only if:

1. ≥3 course and ≥3 contradiction cases yield commit hiddens.  
2. Soft separability: either (a) cosine@1.00 class means differ with **non-overlapping** min–max ranges, **or** (b) LOO probe accuracy ≥ 0.75 when n_scored ≥ 6.

**NOT** a claim that we found “the contradiction representation.”

## Soft claim

Pilot geometry on small n. Motivates whether **conditional** intervention is plausible later — does **not** authorize 12C adaptive α.

## Deliverables

- `docs/PHASE13-PROTOCOL.md` (this file)  
- `n2s_lab/phase13_geometry.py` + `run_phase13.py`  
- `artifacts/phase13-geometry-panel.json`  

## Out of scope

- Adaptive α / strength controllers  
- Overwriting Ph1–12B  
- SAE / Neuronpedia  

## Runner

```bash
cd cxr-evidence-grounding-lab
# Prefer: sudo nvidia-smi -pl 240
.venv-phase9/bin/python run_phase13.py --selftest
.venv-phase9/bin/python run_phase13.py
```
