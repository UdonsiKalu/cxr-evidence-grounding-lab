# MAP-08 — Mapping is not selection (and the viewer)

**Prerequisite:** MAP-07 · **Next:** [MAP-09](./09-replicate-and-ceiling.md)

## Theory — two words people mix up

**Mapping:** this extract *shape* is associated with this *kind of patch*, and we measured collateral.

**Selection:** at run time, choose among **independent** interventions (A vs B vs abstain), and show that the choice beats “always apply everything.”

We ran a contrast on the frozen stack (`n2s_lab/n2s_v1_selection_contrast.py`):

| Arm | Miss / 108 |
|-----|------------|
| none (raw Dual-fill) | 30 |
| extract-signature selector | 6 |
| always G9 (apply-all) | 6 |
| oracle (shallowest match) | upper bound; 79.6% agreement with selector |

The selector chose G9 on **2 / 108** notes (not “always G9 in disguise”). Miss 6 = apply-all 6. That is **stack-depth routing from extract phrases**, not Encode vs Compute vs Translate, and not A vs B vs C vs abstain.

Walk-back (same day): nested wrappers are still **one class**. V1 demonstrated mapping, not class selection.

## GUI — canonization workbench

http://127.0.0.1:8263/

```bash
cd ~/staging/cxr-evidence-grounding-lab
./scripts/run_map_viewer.sh
```

Use filters:

- **All** — 108 notes
- **Raw Dual-wrong** — 30
- **Still miss after G9** — 6 leftovers
- **Design P-bind** — the bind arm
- **Selector ≠ none** — notes whose extract signature is not `none`

Blue row = signature depth. Dashed outline = oracle. Blue bar = that cell changed the extract.

Frozen locator `:8260` is a different product (routing). This viewer is not an expansion of it.

## Backend — meta API

```bash
curl -sS http://127.0.0.1:8263/api/meta | python3 -c "import json,sys; m=json.load(sys.stdin); print(m['claim']); print(m['n_cases'], m['n_miss_none'], m['n_miss_apply_all'])"
```

Panel JSON: `artifacts/n2s-map-viewer-panel.json`  
Contrast JSON: `artifacts/n2s-v1-selection-contrast.json`

## Week 1 — send-back

In two sentences: what the selector measured, and why we still say “not selection.”
