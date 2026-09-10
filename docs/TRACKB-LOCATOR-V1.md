# Locator v1 — diagnostic routing (not causal localization)

**Status:** **FROZEN v1** 2026-09-10 · diagnostic routing workbench · CPU · stop GUI  
**Walked:** CONTRA → compute · FOLFOX → translate · BC_E1 → translate · T1 incomplete  
**GUI:** http://127.0.0.1:8260/  
**Not this:** causal first-emergence · neural repair · n2n frozen-`d` editor · α-chase · U-C SAE / U-D · Dual_full rescore · silent auto-fix

`:8260` **routes known cases** to an encode-, compute-, or translate-**associated** first-break using frozen probes. It does **not** establish where a failure first emerged.

```text
Failed / known case
       │
       ▼
   OBSERVATIONS
       ├── Encode probe (L8 class-correct, correlational)
       ├── Compute trace (lost-steps, correlational)
       └── Translate Dual snapshot
       │
       ▼
FIRST-BREAK ROUTING  (not causal diagnosis)
       │
       ▼
Evidence + probe type + strength (weak/moderate/strong — never a %)
       │
       ▼
AUTOMATED RESPONSE  (containment, not correction)
       ├── REVIEW
       └── ABSTAIN
```

Repair editors are **future**. Do not manufacture them to fill the three boxes.

---

## What you can claim

| Stage | Observe | Claim |
|-------|---------|--------|
| Encode | L8 class-correct on frozen `d` | Encode probe/readout |
| Compute | lost-step pattern on correlational trace | Compute-associated routing signal |
| Translate | Dual snapshot / mismatch | Translation-associated routing signal |
| Response | REVIEW / ABSTAIN | Containment |
| Repair | no generally safe editor | **Not established** |

**Evidence strength** is qualitative (`none` / `moderate` in v1). Never a calibrated %. **Strong** is reserved for a later causal intervention.

Encode is **formation (L8)**, not L24.

## Automated response (containment)

Writes `artifacts/n2s-locator-fixes.json`. Does **not** patch hidden states.

| First-break routing | Action |
|---------------------|--------|
| encode | **ABSTAIN** |
| compute | Dual **REVIEW** (`n2n_layer_loss`) |
| translate | Dual **REVIEW** (`dual_wrong_AUTO`) |
| incomplete | none |

## Frozen demo (selftest)

| Note | Routing | Probe type |
|------|---------|------------|
| EX_CONTRA | **compute** | correlational_trace |
| EX_TEMPORAL_FOLFOX | **translate** | dual_analog (7B) |
| BC_E1 | **translate** | dual_comparison (frozen Dual_full) |
| T1 / T3 / C1 / C4 | **incomplete** | none |

## Run

```bash
cd cxr-evidence-grounding-lab
python3 locator_server.py
# http://127.0.0.1:8260/
```

## Later (not v1) — do not continue GUI here

**v1 frozen.** Stop adding cases, ports, or editors to `:8260`.

v2: given routing, can an intervention **repair** that surface (causal verify + invariants).  
v3: does first-break routing beat blind intervention choice.  
Need contrasts (translate-only vs compute) and real editors. Do not start from this GUI.
