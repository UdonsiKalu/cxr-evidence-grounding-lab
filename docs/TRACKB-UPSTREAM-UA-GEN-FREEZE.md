# Track B freeze — U-A held-out paraphrase generalization

**Status:** **FROZEN 2026-09-06**  
**Scope:** correlational lexical generalization of frozen U-A formation direction `d`  
**Model:** HF `Qwen/Qwen2.5-7B-Instruct`  
**Not this:** causal editor · U-C SAE · U-D circuits · L20/L24 ablation thrash reopen

**Protocol:** [TRACKB-UPSTREAM-UA-GENERALIZE.md](./TRACKB-UPSTREAM-UA-GENERALIZE.md)  
**Portfolio close:** [TRACKB-UPSTREAM-PORTFOLIO.md](./TRACKB-UPSTREAM-PORTFOLIO.md)  
**Program:** [TRACKB-UPSTREAM-PROGRAM.md](./TRACKB-UPSTREAM-PROGRAM.md)

---

## Claim (allowed)

On the discovery pair **EX_TEMPORAL_FOLFOX** / **EX_CONTRA**, the frozen per-layer direction

```text
d = unit( μ_T − μ_C )
```

(baked once; cached in `artifacts/n2s-upstream-ua-directions.pt`) **separates** held-out temporal-change vs same-time contradiction notes at the U-A best layer (**L24**) on `data/heldout-ua-paraphrase.json`:

| Gate | Result (2026-09-06 panel) |
|------|---------------------------|
| Soft: mean_T > mean_C | **YES** — ≈58.4 > ≈23.5 (sep ≈34.9) |
| Strong: min_T > max_C | **YES** — ≈45.5 > ≈29.0 |
| Anchors | temporal ≈71.4 · contra ≈9.4 |

This is a **soft pilot** on a small held-out set (near + far paraphrases). It supports **correlational lexical generalization** of the formation map.

## Claim (forbidden)

- Do **not** say a temporality / contradiction **neuron** or **circuit** was found.  
- Do **not** say `d` is a causal prefill editor (U-B / U-B2 ablation at top sites remains **null**).  
- Do **not** rebuild `d` from held-out text and call that the freeze.  
- Do **not** reopen U-C SAE / U-D / live ablate thrash from this YES.  
- Do **not** claim production clinical reliability.

## Artifacts (frozen record)

| Path | Role |
|------|------|
| `data/heldout-ua-paraphrase.json` | Held-out panel wording |
| `artifacts/n2s-upstream-ua-directions.pt` | Frozen `d` (discovery pair only) |
| `artifacts/n2s-upstream-ua-map.json` | U-A discovery map |
| `artifacts/n2s-upstream-ua-gen-panel.json` | Gen panel scores + soft_gate |
| `artifacts/n2s-upstream-ua-gen-readout.json` | Human readout |
| `n2s_lab/n2s_upstream_ua_generalize.py` | Runner |
| `scripts/run_upstream_ua_gen.sh` | CLI |

## Reproduce

```bash
cd cxr-evidence-grounding-lab
./scripts/run_upstream_ua_gen.sh panel   # uses cached directions.pt
./scripts/run_upstream_ua_gen.sh readout
```

Prefer `../cxrlabs/faiss_gpu1/bin/python`. Do **not** pass `--rebuild-directions` unless intentionally rebaking discovery `d`.

## Next (not frozen here)

Optional **causal** follow-up only with explicit go — still a **new** question, not more of the paused U-B ablation family.
