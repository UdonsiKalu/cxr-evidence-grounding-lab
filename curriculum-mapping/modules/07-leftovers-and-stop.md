# MAP-07 — Leftovers and why we stopped

**Prerequisite:** MAP-06 · **Next:** [MAP-08](./08-mapping-not-selection.md)

## Theory — leftover is not a homework list

After G9, **6 / 108** Dual-fill notes still miss gold:

| ID | Shape (Goal 10 tag) | Why not a new Translate cell |
|----|---------------------|------------------------------|
| T2 | too-early | one-off; failure has not happened |
| U2, U6 | may-have | weak pair; BC_U2 already matches |
| TX_E03 | held-for-progression | one-off |
| BC12_C1 | missing pole / response-only mix | not a clean recurring phenotype |
| BC14_C1 | naive ≠ never | one-off; do not stretch `ground()` to fit `naive` |

Goal 10 tagged these shapes on **every** note (including already-correct ones). Recurrence was too weak to justify G11. `v1_freeze_recommended=true`.

Chasing miss→0 by writing an ID-fitted patch for T2 would **not** be mapping. It would be overfitting.

## Results (Goal 10)

Artifact: `artifacts/n2s-g10-leftover-census.json`

```bash
cd ~/staging/cxr-evidence-grounding-lab
../cxrlabs/faiss_gpu1/bin/python - <<'PY'
import json
from pathlib import Path
p = json.loads(Path("artifacts/n2s-g10-leftover-census.json").read_text())
f = p["finding"]
print("n_miss", f["n_miss"])
print("ids_miss", f["ids_miss"])
print("new cell justified?", f["shapes_that_justify_new_translate_cell"])
print("v1_freeze_recommended", f["v1_freeze_recommended"])
PY
```

If the JSON schema uses nested lists, open the file and find the six leftover ids. The map viewer filter **Still miss after G9** is the same six: T2, U2, TX_E03, U6, BC12_C1, BC14_C1.

## Stop rule (canon)

- Do not add G11 leftover repairs.
- Do not rewrite `ground()` to absorb naive / too-early.
- Do not jump to CAA / frozen-`d` / Encode editor to “finish” Dual miss 6.
- The map is frozen: six nested cells + leftover census + viewer.

## Week 1 — send-back

Why T2 should stay unrepaired. Why leftover 6 is a freeze, not a sprint backlog.
