# Dual mapping — code walkthrough

Same job as `docs/N2S-CLI-Walkthrough.pdf`: **Intuition → copy-paste Python → send-back**. CPU only. No GPU fill. Does not rewrite `ground()`.

Lab root: `~/staging/cxr-evidence-grounding-lab`  
Python: `../cxrlabs/faiss_gpu1/bin/python`

Not this PDF: `mapping-notes-simple.pdf` (GUI walk), `CXR-Mapping-Curriculum.pdf` (MAP-00…09), `progress-notes.pdf` (M1–Ph14).

---

## C0 — Path setup

**Intuition.** Every later snippet assumes the lab is on `sys.path` and Dual-fill extracts already exist.

```bash
cd ~/staging/cxr-evidence-grounding-lab
ls artifacts/n2s-g2-analog-fill-extracts.json n2s_lab/ground.py n2s_lab/predicate.py
```

**Send-back:** `C0 done` plus the `ls` output.

---

## C1 — Extract, ground, rule (no patch)

**Intuition.** The model is not in this loop. Frozen JSON → `Extraction` → `ground()` → `evaluate_rule()`.

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
print("verdict", r.verdict.value, r.fired_rule)
print(g.to_dict())
PY
```

Expect T1 SATISFIED, A=B=C=D true, X false.

**Send-back:** paste the verdict line.

---

## C2 — Nested ladder (four teaching notes)

**Intuition.** `_apply(ex, "G9")` runs G3…G9. Deeper includes earlier.

```bash
cd ~/staging/cxr-evidence-grounding-lab
../cxrlabs/faiss_gpu1/bin/python curriculum-mapping/snippets/replay_ladder.py I2 C4 T1 T2
```

Expect:

- I2: miss until G9 (`planned` → `given`)
- C4: CONTRADICTION from G4; selector G8
- T1: match at `none`
- T2: unrepaired UNCERTAIN

**Send-back:** four lines, one per id.

---

## C3 — Leftover freeze (read-only)

**Intuition.** Goal 10 did not add a cell. It tagged leftover shapes.

```bash
cd ~/staging/cxr-evidence-grounding-lab
../cxrlabs/faiss_gpu1/bin/python - <<'PY'
import json
from pathlib import Path
f = json.loads(Path("artifacts/n2s-g10-leftover-census.json").read_text())["finding"]
print("n_miss", f["n_miss"])
print("ids_miss", f["ids_miss"])
print("new cell?", f["shapes_that_justify_new_translate_cell"])
print("freeze", f["v1_freeze_recommended"])
PY
```

**Send-back:** the six ids and `shapes_that_justify_new_translate_cell=[]`.

---

## C4 — Viewer selftest + live page

**Intuition.** `:8263` is the same ladder as C2, rendered.

```bash
cd ~/staging/cxr-evidence-grounding-lab
../cxrlabs/faiss_gpu1/bin/python map_viewer_server.py --selftest
./scripts/run_map_viewer.sh
# open http://127.0.0.1:8263/
```

Optional curriculum UI:

```bash
./scripts/run_mapping_curriculum.sh
# open http://127.0.0.1:8264/
```

**Do not run** (not this walkthrough): `./scripts/run_g2_analog_fill.sh` (GPU 7B), frozen-`d` patches, Dual_full rescore.

**Send-back:** `C4 done` plus HTTP 200 on `/` or selftest ok.

---

## Claim hygiene

Say: mapping on Dual extracts, nested Translate, measured collateral.  
Never say: auto-correct every LLM failure; `:8260` chose the repair class; G3–G9 are six independent editors.
