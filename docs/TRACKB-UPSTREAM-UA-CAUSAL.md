# Upstream U-A causal — intervene along frozen formation direction `d`

**Status:** protocol locked 2026-09-06 · **first live panels NULL/weak** (commit + prefill @ L24) · correlational `d` ≠ causal editor  
**Prior:** [TRACKB-UPSTREAM-UA-GEN-FREEZE.md](./TRACKB-UPSTREAM-UA-GEN-FREEZE.md) · [TRACKB-UPSTREAM-PORTFOLIO.md](./TRACKB-UPSTREAM-PORTFOLIO.md)

**First live (2026-09-06):** α∈{1,2,4,8} · ±d · gaussian · EX_TEMPORAL_FOLFOX + EX_CONTRA.

| Site | Soft gate | Notes |
|------|-----------|-------|
| **commit** `activation_steer` @ L24 | **NULL/weak** | X unchanged; margins ~noise (≤0.4); contra stays X=true |
| **prefill** note-body @ L24 | **NULL/weak** | X unchanged; margins flat; contra stays X=true |

Temporal baseline already **X=false** (dull for false-X clearing). Still no dose-dependent margin editor. Artifacts: `n2s-upstream-ua-causal-commit-panel.json` · `n2s-upstream-ua-causal-prefill-panel.json`.

---

## Question

> Does intervening directly along the validated frozen U-A direction  
> `d = unit(μ_T − μ_C)`  
> change the model’s downstream contradiction commit (**X**) or logit **margin** in a **dose-dependent** way?

This treats **`d` as the causal object**. It is **not** ablating U-A top sites (U-B/U-B2 — already null).

---

## Design

| Piece | Choice |
|-------|--------|
| **Vector** | Frozen `artifacts/n2s-upstream-ua-directions.pt` **L24** only — do not rebuild from held-out |
| **Model** | HF `Qwen/Qwen2.5-7B-Instruct` |
| **Site (primary)** | **Commit-token** `activation_steer`: `h ← h + α·d` at `contradiction.present` commit @ **L24** |
| **Site (optional)** | Prefill note-body positions @ L24 (`prefill_position_steer`) — formation-faithful contrast |
| **Path** | Extract (`EXTRACT_SYSTEM` + evidence) — same X/margin readout as U-B |
| **Notes** | `EX_TEMPORAL_FOLFOX` + `EX_CONTRA` (true-contra control) |
| **α grid** | `1, 2, 4, 8` |
| **Arms** | `baseline` · `+α·d` · `−α·d` · `gaussian_unit @ α` (seed `GAUSS_SEED`) |

### Soft gates (pilot)

| Gate | Pass if |
|------|---------|
| **Dose (temporal)** | `+α·d` moves X/margin toward non-contradiction vs baseline in a monotone-ish α trend (report even if partial) |
| **Sign** | `−α·d` does **not** help the same way as `+α·d` (or moves opposite) |
| **Specificity** | gaussian @ α does **not** match `+α·d` |
| **Control** | `EX_CONTRA` stays `X=true` under `+α·d` (no true→false wreck) |

Null / partial is an allowed honest outcome.

---

## Run

```bash
cd cxr-evidence-grounding-lab
./scripts/run_upstream_ua_causal.sh panel          # commit @ L24
./scripts/run_upstream_ua_causal.sh panel --site prefill
./scripts/run_upstream_ua_causal.sh readout
```

Artifacts: `n2s-upstream-ua-causal-panel.json` · `n2s-upstream-ua-causal-readout.json`

---

## Claim hygiene

| Say | Do not say |
|-----|------------|
| Commit-time steer along frozen U-A `d` did / did not move X/margin dose-dependently | “Found temporality circuit” |
| Distinct from top-site ablation null (U-B) | Reopen U-C SAE / U-D from this |
| Pilot on two notes + α grid | Production editor / family-wide fix |
| Prefill site is optional contrast | Same as expand L20 α=8 freeze-`v` claim |

---

## Not this

- U-B/U-B2 top-site ablation thrash  
- U-C SAE / U-D circuits  
- Rebuild `d` from paraphrases  
- Sealed-test redesign  
- Confusing this with Track B expand **v** @ L20 (`TRACKB-ALPHA8-FREEZE`)
