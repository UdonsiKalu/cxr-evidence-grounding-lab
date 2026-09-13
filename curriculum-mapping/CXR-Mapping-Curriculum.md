# CXR Dual Translate Mapping Curriculum

**Status:** Frozen 2026-09-11 — this pass is **stopped and canonized**.  
**Track UI:** http://127.0.0.1:8264/  
**Workbench:** map viewer http://127.0.0.1:8263/  
**Not this:** locator `:8260` expansion · new Translate cell G11 · `ground()` rewrite · auto-correct · Encode/Compute editor

## North-star (this track)

> By MAP-09 you can explain, in plain language, what Dual grounding is, what we tried to do, what the nested Translate map actually is, how to replay it on a note, and what the evidence does **not** support.

**Four mastery levels:** Intuition · Pipeline · Code · Results.

## Honest claim (memorize)

V1 demonstrated **mapping**, not **selection**. Nested G3–G9 wrappers are one Translate class (deeper includes earlier). Not independent A vs B vs C. Not a generic corrector for every LLM failure. `locator_chose_repair_class=false`. Not auto-correct.

## What this curriculum is

A beginner path through **one lab investigation** (2026-09-11): can Dual-shaped misses on a clinical-note question be grouped and patched without breaking other notes?

It is **not** the whole N2S history (M1–Ph14 / RepEng). That lives in `cxr-repeng-curriculum` (`:8262`) and `notes/progress-notes.pdf`.

## Order

| ID | Module | You should leave able to |
|----|--------|--------------------------|
| MAP-00 | Orientation | Say the claim and where the files live |
| MAP-01 | Aim and background | State the original question vs what Dual is |
| MAP-02 | Backend pipeline | Walk extract → ground → rule |
| MAP-03 | Three surfaces | Why this pass is Translate-only |
| MAP-04 | Filling extracts | Why Goal 2 existed; load a frozen extract |
| MAP-05 | Nested cells | Name G3–G9 and what each edits |
| MAP-06 | Four notes | Replay I2, C4, T1, T2 |
| MAP-07 | Leftovers and stop | Why leftover 6 is not G11 |
| MAP-08 | Mapping ≠ selection | Read `:8263`; defend the walk-back |
| MAP-09 | Replicate and ceiling | Run the book; list what we did not achieve |

## Companion lab

All code and artifacts: `cxr-evidence-grounding-lab/`

Python (typical): `../cxrlabs/faiss_gpu1/bin/python`

Design document: `docs/TRACKB-FAILURE-INTERVENTION-MAP.md`
