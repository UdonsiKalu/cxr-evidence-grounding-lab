# MAP-00 — Orientation

**Parent:** CXR Dual Translate Mapping Curriculum  
**Next:** [MAP-01](./01-aim-and-background.md)  
**Duration:** 1 session

## Theory — say this out loud

This track is about **one question on synthetic clinical notes**:

> Did first-line therapy fail?

A language model reads the note and fills a JSON **extract**. A **fixed rule** (not the model) then says SATISFIED, NOT_SATISFIED, UNCERTAIN, or CONTRADICTION.

In 2026-09-11 we asked: when the Dual path is wrong, can we **group** those mistakes and **patch the extract** so the same rule becomes right — without flipping notes that were already right?

**What we got:** a nested map of Translate patches (G3 through G9), shown per note in a GUI.  
**What we did not get:** a machine that automatically corrects every failure in a large language model.

**Soft claim:** mapping yes (partial). One-size-fits-all no. Selection among independent editors no. Auto-correct no.

## GUI — two ports, do not mix them

| Port | What it is |
|------|------------|
| **8264** | This curriculum (reading) |
| **8263** | Map viewer (replay the frozen ladder on 108 notes) |
| **8260** | Frozen locator. Routing only. Do not expand it for this track |

1. Open http://127.0.0.1:8263/
2. Read the yellow banner. It should say mapping, not selection.
3. Click **T1** (already correct) and **T2** (still miss). You will use those in MAP-06.

Start the viewer if it is down:

```bash
cd ~/staging/cxr-evidence-grounding-lab
./scripts/run_map_viewer.sh
```

## Backend — trees

```bash
cd ~/staging/cxr-evidence-grounding-lab
ls n2s_lab/n2s_g{2,3,4,6,7,8,9,10}*.py n2s_lab/n2s_v1_selection_contrast.py n2s_lab/n2s_map_viewer.py
ls artifacts/n2s-g2-analog-fill-extracts.json artifacts/n2s-v1-selection-contrast.json
ls curriculum-mapping/modules
```

**Call stack you will defend:**

```text
frozen Dual extract (Goal 2 JSON)
    -> optional nested relabel G3..G9  (copy/wrap; not a new model)
    -> ground()                         (unchanged)
    -> evaluate_rule()                  (unchanged)
    -> verdict vs gold
```

## Week 1 — send-back

Three lines: (1) the Dual question, (2) mapping vs auto-correct, (3) which GUI is `:8263`.
