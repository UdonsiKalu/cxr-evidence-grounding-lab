# MAP-03 — Three surfaces (why this pass is Translate)

**Prerequisite:** MAP-02 · **Next:** [MAP-04](./04-filling-extracts.md)

## Theory — Encode, Compute, Translate

A Dual miss can be *talked about* in three places. That is a **diagnostic partition**, not proof of where the error “first emerged.”

| Surface | Rough question | What we had in this lab |
|---------|----------------|-------------------------|
| **Encode** | Did the note land in the wrong representation? | L8 readout on two notes; **no Encode editor** |
| **Compute** | Did layer-to-layer computation lose the fact? | Lost-steps; frozen-`d` patch **CLOSED**; G5 REVIEW = contain, not repair |
| **Translate** | Did the extract+`ground` drop meaning the model already wrote? | **This is where the only demonstrated Dual repair lived** |

Phase 1–3 parked that story in `docs/TRACKB-PHASE1-3-PORTFOLIO.md`. Locator `:8260` **routes** known notes to a surface label. It does **not** choose a repair class. `locator_chose_repair_class=false`.

This mapping pass therefore stayed on **Translate**: copy or wrap extract-relabel functions. We did not invent an Encode editor to fill a matrix. We did not reopen frozen-`d`.

## Why “several approaches per note” still one class

On `:8263` you see G3 then G4 then G6… That looks like several approaches. They are **nested**: G9 includes G8 includes … includes G3. Choosing G9 is not choosing a different *kind* of intervention than G4. It is a **deeper stack of the same kind** (edit the extract, then `ground()`).

Independent classes would be things like: Translate patch vs hidden-state patch vs “leave it / REVIEW.” We only had the first.

## GUI

Open `:8260` only to remember it exists. Do not add cases. This curriculum’s live instrument is `:8263`.

## Backend — locator is frozen

```bash
cd ~/staging/cxr-evidence-grounding-lab
sed -n '1,12p' docs/TRACKB-LOCATOR-V1.md
```

Do **not** run GPU frozen-`d` patches as part of this track.

## Week 1 — send-back

One sentence: why mapping here is Translate-only. One sentence: why nested G3–G9 is not Encode vs Compute vs Translate selection.
