# MAP-04 — Filling Dual extracts (Goal 2)

**Prerequisite:** MAP-03 · **Next:** [MAP-05](./05-nested-cells.md)

## Theory — the empty-extract problem

Batch 0 Dual analog (user = note only) often produced **thin extracts**: almost no quotes, polarities unknown. Quote-promote then had nothing to copy. P-bind (failure sitting in quotes but not in outcome polarity) was **0/7**.

Goal 2 kept Dual analog (`user=evidence`) but put the extract **schema in the system prompt**. Same 7B, CPU/GPU as the original fill. Result: thin extracts **91 → 0** on n=108. That unblocked mapping. It is **not** Schema+note (a different prompt family we already knew changes extracts).

We **reuse** `artifacts/n2s-g2-analog-fill-extracts.json`. Do not overwrite it. Do not re-run the 7B fill unless you are deliberately reproducing Goal 2 (slow, GPU).

## GUI

`:8263` always starts from those Dual-fill extracts. Filter **Raw Dual-wrong**: 30 of 108 miss gold before any G3–G9 patch.

## Backend — inspect one frozen extract (no GPU)

```bash
cd ~/staging/cxr-evidence-grounding-lab
../cxrlabs/faiss_gpu1/bin/python - <<'PY'
import json
from n2s_lab.n2s_g2_analog_fill import EXTRACT_PATH
from n2s_lab.types import Extraction

bundle = json.loads(EXTRACT_PATH.read_text())
print("n extracts", len(bundle["extracts"]))
rec = bundle["extracts"]["I2"]
print("parse_ok", rec["parse_ok"])
ex = Extraction.from_dict(rec["extract"])
print("admin", ex.administration_status)
print("quotes", ex.quotes)
print("notes", (ex.notes or "")[:200])
print("outcomes", [(o.polarity, o.text[:60]) for o in ex.outcome_statements])
PY
```

I2 often comes out `administration_status=planned` even though the note is platinum-refractory and considering a next line. That is the **shape** Goal 9 later patches. You are looking at the raw Dual fill, not G9 yet.

Optional live fill (only if you must reproduce Goal 2; not required for this curriculum):

```bash
# Slow. GPU. Do not overwrite if you only wanted to read.
# ./scripts/run_g2_analog_fill.sh
ls -l artifacts/n2s-g2-analog-fill-extracts.json
```

## Result (Goal 2)

| Metric | After Dual-fill |
|--------|-----------------|
| n | 108 |
| thin extracts | 0 (was 91 on analog-empty) |
| Quote-promote still leaky | P-neg still fired until Goal 3 gated |

## Week 1 — send-back

Why analog-empty blocked mapping. What Goal 2 changed (schema in system, user still the note). I2 admin status you observed.
