# Track B — false-X cluster (temporal-family-dev)

**Status:** **live ladder run 2026-09-03** — probe **YES**, mid-depth patch **YES**, steer transfer **NO** at frozen α=4  
**Env:** `.venv-phase9/bin/python` (accelerate 1.14.0; the Ph9–14 environment) · GPU 240 W  
**DEV only:** `data/temporal-family-dev.json`  
**Sealed:** `data/temporal-family-test.json` — **do not open evidence** while designing interventions.

## Cluster

| Role | Ids |
|------|-----|
| False-X targets (Qwen Dual_full family) | `TF_E3`, `BC11_E3`, `TF_E4` |
| Contradiction controls | `TF_C1`, `BC_C1` |
| No-failure controls | `TF_N1`, `BC_E2` |

## Ladder

`behavior → probe/localize → activation patch → Ph10 steer (α=4) → ablate (α=−4)`

| Step | What |
|------|------|
| Behavior | HF `Qwen2.5-7B-Instruct` repair path; record `contradiction.present` + margin |
| Probe / localize | Cosine to Ph10 BC_E2−BC_E1 direction @ layers 0.75 / 1.00; LOO 1-NN |
| Patch | Replace commit hiddens with donor `BC_E2` (clean false commit) |
| Steer | Frozen Ph10 unit vectors @ α=4 |
| Ablate | α=−4 on targets (direction flip check) |

**Not:** G3 redesign · scoring `temporal-test` · TransformerLens rewrite.

## Commands

Use the Ph9–14 venv — bare `python3` lacks `accelerate` and fails at model load.

```bash
./.venv-phase9/bin/python run_trackb_falsex.py --selftest
./.venv-phase9/bin/python run_trackb_falsex.py                    # full ladder
./.venv-phase9/bin/python run_trackb_falsex.py --patch-frac 0.75  # depth-restricted patch only
```

## Gates (soft)

- **Probe YES** if ≥2 usable false-X and ≥2 contradiction commits, and (non-overlapping cos@1.00 **or** LOO ≥ 0.75).
- **Causal YES** if ≥1 baseline false-X target flips to false under patch **or** steer.
- **Patch-depth YES** additionally requires patched margins **≠** donor margin (see confound below).

---

## Live record (2026-09-03)

### Behavior — family reproduces on HF 7B

All three targets commit false `contradiction.present` on `Qwen2.5-7B-Instruct`, so the
Qwen-coder-32B Ollama failure family transfers to the HF stack.

| Case | class | baseline X | margin | verdict | cos@1.00 |
|------|-------|-----------|--------|---------|----------|
| TF_E3 | false-X | true | 6.875 | CONTRADICTION | −0.687 |
| BC11_E3 | false-X | true | 2.75 | CONTRADICTION | −0.713 |
| TF_E4 | false-X | true | 9.5 | UNCERTAIN | −0.627 |
| TF_C1 | contradiction | true | 12.375 | CONTRADICTION | −0.530 |
| BC_C1 | contradiction | true | 13.375 | CONTRADICTION | −0.554 |
| TF_N1 | no-failure | false | −12.375 | NOT_SATISFIED | −0.276 |
| BC_E2 | no-failure | false | −14.0 | NOT_SATISFIED | −0.170 |

### Probe — **YES** (replicates Ph13 on a new cluster)

cos@1.00 false-X `[−0.713, −0.627]` vs contradiction `[−0.554, −0.530]`; **no overlap**,
gap **0.072**, LOO 1-NN **1.00** (n=5 — small). Ph13 saw the same ordering with gap 0.021 / LOO 0.889.

### Steer α=4 — **NO transfer** (honest negative)

Frozen Ph10 BC_E2−BC_E1 vector flips **0/3**. Margins move only slightly in the right
direction; controls fully intact.

| Case | margin base → steered | X |
|------|----------------------|---|
| TF_E3 | 6.875 → 6.0 | true → true |
| BC11_E3 | 2.75 → 1.25 | true → true |
| TF_E4 | 9.5 → 8.25 | true → true |
| TF_C1 / BC_C1 | 12.375 → 11.5 / 13.375 → 12.25 | stay true ✔ |
| TF_N1 / BC_E2 | −12.375 → −13.375 / −14.0 → −14.875 | stay false ✔ |

### Ablate α=−4 — sign-consistent, underpowered

Margins move **up** (6.875→7.75, 2.75→4.25, 9.5→10.5). The direction is causally relevant
but |α|=4 is far too weak to cross zero on this cluster.

### Patch — depth matters

**Confound found:** the first pass patched fractions 0.75 **and 1.00**. Fraction 1.00 is the
final layer, so writing the donor hidden there sets the commit-token logits to the donor's
**by construction** — every patched margin came back as exactly `−14.0`, BC_E2's own margin.
That arm is equivalent to token forcing and is **not** evidence of sufficiency.

Re-run at **0.75 only**, leaving the remaining blocks free to recompute:

| Case | margin base → patched | X | verdict |
|------|----------------------|---|---------|
| TF_E3 | 6.875 → **−11.375** | true → false | SATISFIED |
| BC11_E3 | 2.75 → **−11.5** | true → false | SATISFIED |
| TF_E4 | 9.5 → **−11.375** | true → false | UNCERTAIN |

Patched margins differ from the donor's `−14.0`, so the network genuinely recomputed:
**mid-depth donor activation is sufficient to flip the commit downstream** (3/3, soft, n=3).

### Reading

The temporal distinction is **decodable and manipulable at layer 0.75**, but the frozen Ph10
steering direction is the wrong instrument for this cluster at α=4 — too weak, and fitted on
BC_E1. Patch (sufficiency) succeeds where steer (targeted edit) fails.

### Artifacts

`artifacts/trackb-falsex-cluster-panel.json` · `artifacts/trackb-falsex-patch-0_75-panel.json`
· logs under `artifacts/logs/`

### Open

1. α sweep on this cluster (Ph10 grid reached 32; only α=4 tested here).
2. Refit a steering vector **on this cluster** rather than reusing the BC_E1 vector.
3. Layer sweep for patch (0.25 / 0.5) to find the earliest sufficient depth.
4. Only after an intervention is frozen: score `temporal-family-test.json`.

## Claim hygiene

Say: lab-scale causal pilot on DEV false-X cluster; probe separable; mid-depth patch sufficient; frozen steer does not transfer at α=4.  
Do not say: Track B closed; clinical safety; test-set validated; "we found the contradiction neuron."
