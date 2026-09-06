# Thin upstream prefill pilot — cue × layer formation (U0)

**Status:** U0+U1 done · U1 **null** on recipe sites · **Not** a framework  
**Module:** `n2s_lab/n2s_upstream_prefill.py`  
**Stack:** HF `transformers` + forward hooks · run via **faiss_gpu1**

## Quest

> How does information in the doctor's note become / evolve as internal representations **during prefill** (before commit utilization)?

U0 = alignment. U1 = causal probe at cue positions → commit X.

## Ladder

| Step | Name | What |
|------|------|------|
| **U0** | Prefill trace | One forward; residual @ layers `{8,12,16,20,24}` on cue token positions; proj / cos onto frozen expand **v**; Chanin L20 SAE encode |
| **U0 readout** | Interpret | `readout` — findings + U1 site recipe (CPU) |
| **U1** | Site patch | Prefill-position ±α·v at recipe sites → repair commit X / margin |
| **U2** | Workbench (optional) | Only after U1 is stable |

## Run

```bash
cd cxr-evidence-grounding-lab
./scripts/run_upstream_prefill.sh trace
./scripts/run_upstream_prefill.sh readout
./scripts/run_upstream_prefill.sh patch --alpha 8
```

Needs ~14 GiB free (HF 7B) for `trace` / `patch`. Unload Ollama 32B first if needed.

## Artifacts

| File | Content |
|------|---------|
| `artifacts/n2s-upstream-prefill-trace.json` | U0 cue×layer proj/cos_v; SAE top-k @ L20 |
| `artifacts/n2s-upstream-u0-readout.json` | Findings + U1 recipe |
| `artifacts/n2s-upstream-prefill-patch.json` | U1 arms: baseline / +v / −v → X, margin |

## U0 findings (first live run)

- Commit-fitted **v** is **weak** at note-body cues (|cos|≲0.05 @ L20).  
- Best trajectory: **second FOLFOX** on temporal note (cos rises L12→L20).  
- SAE feats **165 / 9204** off at prefill; **1196** on but not selective.  
- First FOLFOX temporal−contra Δ≈0 — early drug mention is not a class separator.

## U1 result (first live run, α=8 @ L20)

| Note | Sites | baseline X | +v X | −v X | Effect |
|------|-------|------------|------|------|--------|
| EX_TEMPORAL_FOLFOX | last FOLFOX | false | false | false | no flip; margin ~flat |
| EX_CONTRA | failed + responsive | true | true | true | no flip; control held |

**Read:** prefill ±α·v at these cue sites does **not** move commit X on this pair. Matches U0 (weak cue alignment with commit-v). Do **not** claim upstream editor; stop or try different sites/model — not α=32.

## U1b — different sites/layer (not α-chase)

**Hypothesis:** temporal meaning forms on **outcome-language** cues mid-network (`progression` / `failure` @ **L16**), not last drug-name @ L20.

```bash
./scripts/run_upstream_prefill.sh patch --hypothesis u1b --alpha 8
```

Artifact: `artifacts/n2s-upstream-prefill-patch-u1b.json`  
Same α=8 freeze; only sites/layer change.

### U1b result (2026-09-06, α=8 @ L16)

| Note | Sites | baseline X | +v X | −v X | Effect |
|------|-------|------------|------|------|--------|
| EX_TEMPORAL_FOLFOX | progression + failure | false | false | false | no flip; margin ~flat |
| EX_CONTRA | failed + responsive + addendum | true | true | true | no flip; control held |

**Read:** U1b also **null** — different sites/layer, same α, still no commit X edit. Stop upstream α/site squeeze for this pair.

## Claim hygiene

**Say:** prefill cue×layer alignment; U1/U1b tested whether those sites causally affect commit X (both null on this pair).  
**Do not say:** we found the temporality circuit; formation fully characterized; upstream steer works like commit α=8.
