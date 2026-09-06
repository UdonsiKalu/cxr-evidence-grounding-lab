# Thin circuit pilot — who writes into L20 along v / SAE features?

**Status:** C0–C1 **FROZEN** · C2 path runnable · [TRACKB-CIRCUIT-C01-FREEZE.md](./TRACKB-CIRCUIT-C01-FREEZE.md)  
**Module:** `n2s_lab/n2s_circuit_pilot.py`  
**Stack:** HF `transformers` + forward hooks (same as Intervene / SAE) · run via **faiss_gpu1**

## Quest

After SAE ranked features that align with frozen

`v = unit(mean_A − mean_B)` @ L20,

ask the circuit question:

> Which **components** (layer × attn vs MLP) write into the residual stream along `v` (and top SAE decoder directions) at the `contradiction.present` commit — differently for temporal/A vs contradiction/B?

Success is **not** “we drew a full circuit diagram.”  
Success is a **ranked write map** + one controlled patch that moves X / margin in the predicted direction.

## Ladder (thin)

| Step | Name | What |
|------|------|------|
| **C0** | Write audit | At commit, project attn-out & mlp-out @ layers {16,20,24} onto `v` + SAE top feats; Δ(A−B) |
| **C1** | Site patch | Zero/replace high-Δ site (L20 MLP) at commit; metric = X / margin; attn + L16 controls |
| **C2** | Path / edge (optional) | Only if C1 finds a small site set — restrict path; else stop |
| **C3** | Workbench (optional) | Expose C0 table after SAE — only after C0 is stable |

**Do not** install TransformerLens for this pilot. TL remains study material in `cxr-mi-repeng-grounding` A03–A06.

## Relation to prior work

| Prior | Role |
|-------|------|
| Expand `v` @ L20 | Causal steer site (α=8 freeze) |
| SAE Chanin L20 | Sparse features aligned with `v` (165 / 9204 / 1196…) |
| Track A Dual / mismatch | Behavioral containment — unchanged |
| Ph9B localization | Related layer contrast — this pilot is family/`v`-specific write audit |

## Run (C0)

```bash
cd cxr-evidence-grounding-lab
./scripts/run_circuit_pilot.sh audit --max-per-class 2
# or
../cxrlabs/faiss_gpu1/bin/python -m n2s_lab.n2s_circuit_pilot audit --max-per-class 2
```

Needs ~14 GiB free (HF 7B). Uses frozen Class A/B ids from expand fit + SAE top from `n2s-sae-pilot-scores.json` when present.

**C0 result (live):** top |Δ(A−B)| = **L20/mlp/frozen_v** ≈ +6.75; then L20/mlp SAE 165 / 9204; L20/attn/frozen_v.

## Run (C1)

```bash
./scripts/run_circuit_pilot.sh patch --max-per-class 2
```

Arms at `contradiction.present` commit:

| Arm | What |
|-----|------|
| `baseline` | no intervention |
| `l20_mlp_zero` | zero L20 MLP last-token write |
| `l20_mlp_replace_A_mean` | replace with mean Class A L20 MLP write (**primary**) |
| `l20_mlp_replace_B_mean` | replace with mean Class B write (secondary) |
| `l20_attn_zero` | control — same layer, attn |
| `l16_mlp_zero` | control — other layer, mlp |

Soft gate (Class B): primary mean Δmargin < −1 and (X flips false on ≥1 B case **or** primary beats controls).

### C1 live result (2026-09-06)

| Arm (Class B mean Δmargin vs baseline) | TX_C01 / TX_C02 |
|----------------------------------------|-----------------|
| `l20_mlp_replace_A_mean` (**primary**) | −3.75 / −2.50 → **mean −3.125** |
| `l20_mlp_zero` | −1.375 / −0.75 |
| `l20_attn_zero` | −0.25 / +1.75 |
| `l16_mlp_zero` | −0.125 / −0.875 |

- Soft gate: **YES** (primary beats controls; no B→X=false flip).
- Bidirectional: Class A `l20_mlp_replace_B_mean` flips **X false→true** on both TX_E08 and TX_E13 (margins −8.1→+0.25, −5.9→+1.9); mean Δmargin ≈ **+8.1**.
- Claim: L20 MLP write at commit is a **causal site** for X/margin (site-level). **Not** a full temporality circuit.
- **Freeze:** [TRACKB-CIRCUIT-C01-FREEZE.md](./TRACKB-CIRCUIT-C01-FREEZE.md)

## Run (C2)

```bash
./scripts/run_circuit_pilot.sh path --max-per-class 2
```

Path-restrict arms: L20 replace (replay) · L16 replace (feeder alone) · L16 zero + L20 replace (stack) · L24 mlp zero (post control) · A←B at L20 vs L16.

Soft gate: `LOCAL_SUFFICIENT` vs `FEEDER_MATTERS` vs `NULL/weak`.

### C2 live result (2026-09-06)

Soft gate: **LOCAL_SUFFICIENT** — B L20←A Δ≈−3.1; L16←A Δ≈+0.25; path stack Δ≈−3.9; L24 zero null. L20 MLP is locally sufficient at this grain.

## Artifacts

| File | Content |
|------|---------|
| `artifacts/n2s-circuit-write-audit.json` | per-layer attn/mlp projection onto `v` / SAE dirs; Δ(A−B) |
| `artifacts/n2s-circuit-c1-l20-mlp-patch.json` | C1 panel rows + soft_gate |
| `artifacts/n2s-circuit-c2-path-restrict.json` | C2 path panel + soft_gate |

## Claim hygiene

**Say:** write-audit candidates; C1 shows L20 MLP commit write is causally involved in X/margin (soft_gate YES; A←B flips X); C2 is path-restrict only.  
**Do not say:** we found the full circuit; head H does “temporality”; production CXR understanding.  
**Leave Upstream alone:** U-C/U-D stay locked.
