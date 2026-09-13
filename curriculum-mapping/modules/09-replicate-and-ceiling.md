# MAP-09 — Replicate, and the claim ceiling

**Prerequisite:** MAP-08  
**Unlock:** you can teach MAP-00…08 without inflating auto-correct.

## Replicate book (CPU, frozen artifacts)

From `cxr-evidence-grounding-lab`. Typical Python: `../cxrlabs/faiss_gpu1/bin/python`.

**1. Self-tests (no GPU)**

```bash
cd ~/staging/cxr-evidence-grounding-lab
../cxrlabs/faiss_gpu1/bin/python -m n2s_lab.n2s_map_viewer
../cxrlabs/faiss_gpu1/bin/python map_viewer_server.py --selftest
```

**2. Ladder on four teaching notes**

```bash
../cxrlabs/faiss_gpu1/bin/python curriculum-mapping/snippets/replay_ladder.py I2 C4 T1 T2
```

**3. Viewer**

```bash
./scripts/run_map_viewer.sh
# open http://127.0.0.1:8263/
```

**4. Curriculum UI**

```bash
./scripts/run_mapping_curriculum.sh
# open http://127.0.0.1:8264/
```

**5. Optional: rebuild this PDF**

```bash
./curriculum-mapping/modules/build-mapping-pdf.sh
```

**Do not need, for this track:** re-extracting 108 notes with 7B; Dual_full rescore; frozen-`d`; CAA; rewriting `ground()`.

## Files that *are* the canon

| Path | Role |
|------|------|
| `docs/TRACKB-FAILURE-INTERVENTION-MAP.md` | Design + cell results |
| `artifacts/n2s-g2-analog-fill-extracts.json` | Frozen Dual-fill extracts |
| `n2s_lab/n2s_phase2_translate.py` | Nested relabels |
| `n2s_lab/ground.py` / `predicate.py` | Unchanged boundary |
| `artifacts/n2s-g10-leftover-census.json` | Why no G11 |
| `artifacts/n2s-v1-selection-contrast.json` | Mapping vs apply-all |
| `static/map-viewer.html` | `:8263` |
| `curriculum-mapping/` | This track |

## Claim ceiling — say / never say

**Say**

- Dual-shaped extract failures can be grouped.
- Nested Translate relabels repaired a measured subset with controlled collateral on P-neg / P-xspan / P-bind.
- One-size-fits-all is the wrong default (T1 needs nothing; T2 should not be forced).
- This is a **partial positive** on mapping.

**Never say**

- We automatically correct every LLM failure mode.
- `:8260` chose the right repair class.
- G3–G9 are six independent editors.
- Encode/Compute editors exist in this pass.
- Leftover 6 means the map is unfinished homework.

## What would be a *new* investigation (not this PDF)

A second **independent** class (or principled abstain as a chosen non-repair), then A vs B vs abstain. Encode none today; Compute editor CLOSED. That is a new question. This curriculum stops.

## Week 1 — send-back

List three commands you ran. Quote the honest claim in one sentence. Name one claim you will not make.
