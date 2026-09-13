# MAP-05 — Nested Translate cells (G3–G9)

**Prerequisite:** MAP-04 · **Next:** [MAP-06](./06-worked-examples.md)

## Theory — one class, six depths

Each “cell” is a **function** that takes an `Extraction` and returns a possibly edited `Extraction`. Then `ground()` and the rule run as usual.

Deeper functions **call the previous ones**. G9 = G8 + admin-blocked. G8 = G7 + same-time mix. And so on. There is no G5 repair cell (Goal 5 was a **census** only).

| Cell | Plain job | What it typically edits |
|------|-----------|-------------------------|
| **none** | Raw Dual-fill extract | nothing |
| **G3** | Gated failure promote | asserted, non-negated failure words → failure polarity (not “no progression”) |
| **G4** | Mixed-span | simultaneous incompatible 1L claims → `contradiction.present` |
| **G6** | Never-quote | copy never / no-prior quotes into contradiction cues |
| **G7** | Restage | clear hedge once failure is confirmed; undo stable-then-progress X |
| **G8** | Same-time mix | undo notes-promote on uncertain; continue-now response → ongoing |
| **G9** | Admin-blocked | planned-after-refractory → given; fail vs never-dispensed cues |

**Do not** treat these as six products you could ship independently. Nested means G6 still runs G3 and G4.

## Implementation (where to read)

All live wrappers sit in `n2s_lab/n2s_phase2_translate.py` (copied/wrapped, not in-place mutation of G3 when G4 was added). The ladder order is declared in `n2s_lab/n2s_v1_selection_contrast.py` as `CELLS`.

```bash
cd ~/staging/cxr-evidence-grounding-lab
rg -n "^def relabel_failure_polarity" n2s_lab/n2s_phase2_translate.py
sed -n '50,58p' n2s_lab/n2s_v1_selection_contrast.py
```

Apply **one** depth the same way the GUI does:

```bash
../cxrlabs/faiss_gpu1/bin/python - <<'PY'
import json
from n2s_lab.n2s_g2_analog_fill import EXTRACT_PATH
from n2s_lab.n2s_phase2_translate import atoms_payload
from n2s_lab.n2s_v1_selection_contrast import _apply
from n2s_lab.types import Extraction

ex = Extraction.from_dict(
    json.loads(EXTRACT_PATH.read_text())["extracts"]["I2"]["extract"]
)
for name in ("none", "G3", "G9"):
    after, meta = _apply(ex, name)
    p = atoms_payload(after)
    print(name, p["verdict"], "admin", after.administration_status, "kind", meta.get("kind"))
PY
```

Expect I2: none/G3 still NOT_SATISFIED (planned); G9 SATISFIED (`admin=given`).

## Results after the stack (n=108 Dual-fill)

| After | Design P-bind match | n=108 Dual-wrong |
|-------|---------------------|------------------|
| raw (none) | — | 30 |
| G4 | 2/7 repaired path | 27 miss |
| G6 | — | 19 |
| G7 | 3/7 | 12 |
| G8 | 3/7 | 8 |
| G9 | **4/7 repaired; 7/7 match** | **6** |

P-neg leak **0** after G3. Design P-xspan **10/10** after G6. No new flips on G6–G9 vs the previous cell.

Panels (do not overwrite): `artifacts/n2s-g3-gated-promote-panel.json` … `n2s-g9-admin-blocked-panel.json`.

## Week 1 — send-back

Name the six cells. Say in one line why G9 is not a second editor class. Paste I2 none vs G9 verdicts from the snippet.
