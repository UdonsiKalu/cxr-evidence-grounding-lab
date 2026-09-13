# MAP-02 — Backend pipeline (extract, ground, rule)

**Prerequisite:** MAP-01 · **Next:** [MAP-03](./03-three-surfaces.md)

This module is the **implementation** you must be able to replay without a GPU.

## Theory — three functions, three jobs

```text
note text
   |  (language model, already run; we reuse frozen JSON)
   v
Extraction  (quotes, admin status, outcome polarities, uncertainty, contradiction cues)
   |  ground()
   v
Grounding   (atoms A B C D, flag X)
   |  evaluate_rule()
   v
Verdict     SATISFIED | NOT_SATISFIED | UNCERTAIN | CONTRADICTION
```

- `ground()` **does not call a model.** It is a deterministic map from extract fields to atoms. We **did not rewrite it** in this pass.
- `evaluate_rule()` is also frozen. Short version:

```text
if X: CONTRADICTION
else if any of A,B,C,D is false: NOT_SATISFIED
else if any of A,B,C,D is unknown: UNCERTAIN
else: SATISFIED
```

Patches in this investigation edit the **extract** (Translate), then call the same `ground()` and the same rule.

## GUI

On `:8263`, pick **T1**. The **Atoms** column is A B C D X after each cell. On T1 they never change: the raw extract already grounds to SATISFIED.

## Backend — run this (CPU)

From the lab root. Uses the **frozen Goal 2 fill extracts**, not a live 7B call.

```bash
cd ~/staging/cxr-evidence-grounding-lab
../cxrlabs/faiss_gpu1/bin/python - <<'PY'
import json
from n2s_lab.n2s_g2_analog_fill import EXTRACT_PATH
from n2s_lab.ground import ground
from n2s_lab.predicate import evaluate_rule, PREDICATE_FORMULA
from n2s_lab.types import Extraction

print(PREDICATE_FORMULA)
raw = json.loads(EXTRACT_PATH.read_text())["extracts"]["T1"]["extract"]
ex = Extraction.from_dict(raw)
g = ground(ex)
r = evaluate_rule(g)
print("admin", ex.administration_status)
print("outcomes", [(o.text[:40], o.polarity) for o in ex.outcome_statements])
print("atoms", g.to_dict())
print("verdict", r.verdict.value, "rule", r.fired_rule)
PY
```

You should see T1 SATISFIED with A=B=C=D true and X false.

Read the rule text:

```bash
sed -n '14,45p' n2s_lab/predicate.py
```

Read that `ground` is the boundary and does not call a model:

```bash
sed -n '1,8p' n2s_lab/ground.py
```

## Files

| File | Role |
|------|------|
| `n2s_lab/types.py` | `Extraction`, `Grounding`, `Verdict` |
| `n2s_lab/ground.py` | extract → atoms |
| `n2s_lab/predicate.py` | atoms → verdict |
| `n2s_lab/n2s_phase2_translate.py` | `atoms_payload()` helper + later relabels |
| `artifacts/n2s-g2-analog-fill-extracts.json` | frozen Dual-fill extracts n=108 |

## Week 1 — send-back

Paste your T1 verdict line. Name the three stages extract / ground / rule.
