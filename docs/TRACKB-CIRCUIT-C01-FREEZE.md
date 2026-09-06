# Track B freeze — Downstream circuit C0 + C1 (+ C2)

**Status:** **FROZEN 2026-09-06**  
**Scope:** Downstream utilization circuit ladder on expand Class A/B @ commit  
**Model:** HF `Qwen/Qwen2.5-7B-Instruct`  
**Not this:** Upstream U-C SAE · U-D · full temporality circuit · production CXR

**Pilot:** [TRACKB-CIRCUIT-PILOT.md](./TRACKB-CIRCUIT-PILOT.md)  
**Where map:** [TRACKB-SAE-CIRCUIT-WHERE.md](./TRACKB-SAE-CIRCUIT-WHERE.md)  
**System claim:** [N2S-SYSTEM-CLAIM.md](./N2S-SYSTEM-CLAIM.md)

---

## Claim (allowed)

On expand fit Class A (`TX_E08`, `TX_E13`) vs Class B (`TX_C01`, `TX_C02`):

### C0 — write audit

Top |Δ(A−B)| write along frozen expand `v` at `contradiction.present` commit:

**L20 / mlp / frozen_v** ≈ **+6.75** (then L20/mlp SAE 165 / 9204; L20/attn).

### C1 — causal site

Commit-time patch of **L20 MLP**:

| Result | Detail |
|--------|--------|
| Soft gate | **YES** |
| B←A mean | mean Δmargin ≈ **−3.125** (beats L20 attn / L16 mlp controls); no B→X=false flip |
| A←B mean | **X false→true** on both TX_E08 and TX_E13; mean Δmargin ≈ **+8.1** |

**Say:** L20 MLP last-token write at commit is a **causal site** for repair X / margin.

### C2 — path restrict

| Soft gate | **LOCAL_SUFFICIENT** |
|-----------|----------------------|
| B L20←A Δmargin | ≈ **−3.125** (C1 replay) |
| B L16←A Δmargin | ≈ **+0.25** (weak feeder alone) |
| B path stack (L16 zero + L20←A) | ≈ **−3.875** (≈ L20-only) |
| B L24 mlp zero | ≈ **−0.06** (post control null) |
| A X-flips L20←B / L16←B | **2 / 0** (L20 flips both; L16 flips none) |

**Say:** at this grain the C1 effect is **local to L20 MLP**; L16 MLP is not the necessary feeder.  
**Do not say:** multi-hop temporality circuit / named clinical path.

## Claim (forbidden)

- Do **not** say the full circuit was found.  
- Do **not** name a “temporality neuron” or head.  
- Do **not** reopen Upstream U-C SAE / U-D from this Downstream hit.  
- Do **not** α-chase expand-`v` or redesign sealed temporal-test from these panels.  
- Do **not** claim clinical / production reliability.

## Artifacts (frozen record)

| Path | Role |
|------|------|
| `artifacts/n2s-circuit-write-audit.json` | C0 ranked write map |
| `artifacts/n2s-circuit-c1-l20-mlp-patch.json` | C1 panel + soft_gate |
| `artifacts/n2s-circuit-c2-path-restrict.json` | C2 path panel + soft_gate |
| `n2s_lab/n2s_circuit_pilot.py` | `audit` / `patch` / `path` |
| `scripts/run_circuit_pilot.sh` | CLI |

## Reproduce

```bash
cd cxr-evidence-grounding-lab
./scripts/run_circuit_pilot.sh audit --max-per-class 2
./scripts/run_circuit_pilot.sh patch --max-per-class 2
./scripts/run_circuit_pilot.sh path --max-per-class 2
```

Prefer `../cxrlabs/faiss_gpu1/bin/python`. Needs ~14 GiB free GPU.

## Leave Upstream alone (priority 3)

Ablation (U-B/U-B2) and α·`d` steer are **null**. This Downstream C0–C2 freeze does **not** warrant Upstream U-C SAE or U-D. Formation map + gen remain correlational only.
