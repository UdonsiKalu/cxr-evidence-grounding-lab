# Where SAE & circuits apply (N2S map)

**Status:** orientation 2026-09-06 · not a new science go  
**Point:** SAE/circuits are **microscopes on a causal object that already bites** — not a way to invent causality after nulls.

---

## Two stacks (do not mix)

```text
UPSTREAM (formation / prefill)          DOWNSTREAM (utilization / commit)
  U-A map d @ L24                         expand v @ L20 (α=8 FREEZE — partial editor)
  gen YES (correlational)                 SAE Chanin L20 on v  ← already piloted
  U-B ablate NULL                         Circuit write-audit on v/SAE ← natural next
  α·d steer NULL
  U-C SAE / U-D  ← LOCKED off from nulls
```

| Tool | Downstream (warranted today) | Upstream (not warranted now) |
|------|------------------------------|------------------------------|
| **SAE** | Decompose **frozen expand `v`** @ L20 into sparse feats; top-k steer + EX_CONTRA control (`TRACKB-SAE-PILOT`, `:8257` SAE button) | U-C: sparse-decompose **`d`** / U-A sites — only if a **causal** lever on `d` or sites exists |
| **Circuits** | C0 write-audit + **C1 L20 MLP patch soft_gate YES** (`TRACKB-CIRCUIT-PILOT`) | U-D: token→head→token on formation path — only if U-B/U-C bite |

---

## Scenario that **warrants** SAE

You already have (or just found) a **direction or site that moves X/margin** with controls.

**Canonical lab scenario (Downstream — already true):**

1. Expand `v` @ L20, α=8 flips some temporal false-X; gaussian/reverse fail; true-contra holds.  
2. **Then SAE:** which sparse Chanin L20 features align with `v` and, when steered, reproduce (part of) that flip?  
3. Ranking alone is **not** enough — causal top-k arms are the warrant check.

**Would warrant Upstream SAE (U-C) — we do not have this yet:**

- α·`d` or a prefill patch **dose-dependently** moves X/margin with sign + gaussian controls, **or**  
- a small U-B-style site set actually flips X (not null).  
Then: encode at those sites / along `d`, clamp top feats, re-measure X.

---

## Scenario that **warrants** circuits

After SAE (or a dense direction) you have **1–few features / components that causally matter**.

**Canonical lab scenario (Downstream — next thin step):**

1. SAE named candidate feat ids that track `v` / move X on a temporal note.  
2. **Then circuit C0:** at `contradiction.present` commit, which L16/L20/L24 **attn vs MLP** writes project onto `v` and those SAE decoder dirs (Δ temporal vs contra)?  
3. **C1** only if C0 yields a shortlist — activation-patch those writes; metric = X/margin.

**Would warrant Upstream U-D — we do not have this yet:**

- U-C (or U-B) leaves a **small** causal site set.  
Then sketch token→component→token with controls. Null U-B + null α·d → **no U-D**.

---

## Scenario that does **not** warrant them

| Situation | Why skip |
|-----------|----------|
| U-B/U-B2 null + α·d null | No causal object to sparsify / circuit |
| Correlational map / gen YES alone | Localization ≠ control |
| “See what’s inside” curiosity after null | Portfolio thrash; claim hygiene fails |
| Track A Dual disagree | Prompt/extract/grounding first — not SAE |

---

## Practical “where to click / run”

| Want | Where |
|------|--------|
| Downstream SAE (live) | `:8257` **SAE features** |
| Downstream SAE (CLI) | `./scripts/run_sae_pilot.sh score\|causal` |
| Downstream circuit C0 | `./scripts/run_circuit_pilot.sh audit` |
| Downstream circuit C1 | `./scripts/run_circuit_pilot.sh patch` |
| Downstream circuit C2 | `./scripts/run_circuit_pilot.sh path` |
| Downstream circuit freeze | [TRACKB-CIRCUIT-C01-FREEZE.md](./TRACKB-CIRCUIT-C01-FREEZE.md) |
| Upstream score only | `:8258` (correlational) |
| Upstream U-C/U-D | **Do not** — leave alone (null ablate + null d-steer; Downstream C0–C2 does not reopen) |

---

## Bottom line

- **Yes, there is a warrant scenario** — but it is **Downstream**: SAE decomposes the **α=8 expand editor `v`**; circuits ask **who writes that signal at commit**.  
- **Upstream** SAE/circuits are the same *idea*, deferred until formation has a **causal bite**. Right now Upstream’s honest claim stops at: map + gen, no editor.
