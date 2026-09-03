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
./.venv-phase9/bin/python run_trackb_falsex.py                    # original ladder
./.venv-phase9/bin/python run_trackb_falsex.py --patch-frac 0.75
./.venv-phase9/bin/python run_trackb_falsex.py --patch-depth-sweep  # ChatGPT order step 1
./.venv-phase9/bin/python run_trackb_falsex.py --family-vector      # step 3 (may not fit)
./.venv-phase9/bin/python run_trackb_falsex.py --expand-sequence    # expand panel 7B→(14B if needed)
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

### Patch-depth sweep (absolute layers) — earliest sufficient **L20**

Qwen2.5-7B has **28** blocks. Coarse sweep L4 / L8 / L12 / L16 / L20 / L24 (final L27 excluded).
Donor = `BC_E2` only (**not** `TF_N1` — Qwen Dual wrong_AUTO).

| Layer | flips | patched margins |
|-------|-------|-----------------|
| L4 | 0/3 | 7.0, 3.0, 9.5 |
| L8 | 0/3 | 6.875, 2.875, 9.625 |
| L12 | 0/3 | 6.875, 2.75, 9.625 |
| L16 | 0/3 | 7.25, 2.25, 9.75 |
| **L20** | **3/3** | **−11.375, −11.5, −11.375** (≠ donor −14.0) |
| L24 | 3/3 | −12.625, −12.625, −11.625 |

Rescue **starts at L20** (same site as Ph9B 0.75 / idx 20). Earlier layers do not carry a sufficient donor state.

### Specificity controls @ L20 — mixed

| Control | Result | Read |
|---------|--------|------|
| Gaussian matched-norm → false-X | **0/3** flips | not “any vector works” |
| True-contradiction (`TF_C1`) → false-X | **0/3** flips (margins **up**) | contradiction state does not rescue |
| Reverse: `TF_E3` → `BC_E2` | **induces** X=true (margin −14 → +4) | false-X state is causally sufficient both ways |
| `BC_E2` → `TF_C1` / `BC_C1` | **destroys** true X (both flipped false) | donor is a general push-to-false, not class-selective |
| `BC_E2` → self | stays false | sanity |

Soft specificity gate **failed** because contradictions were not preserved. Patch at L20 is **causally real** but **not a contradiction-preserving temporal editor**.

### Frozen BC_E1 α-sweep — transfer failure at α=4 was **magnitude**

| α | flips | mean Δmargin | contra stay | nofail `BC_E2` |
|---|-------|--------------|-------------|----------------|
| 1 | 0/3 | −0.21 | yes | yes |
| 2 | 0/3 | −0.50 | yes | yes |
| 4 | 0/3 | −1.21 | yes | yes |
| 8 | **1/3** (`BC11_E3`) | −2.54 | yes | yes |
| 16 | **1/3** (`BC11_E3`) | −5.96 | yes | yes |
| 32 | **3/3** | −14.75 | **no** (`TF_C1` flipped) | yes |

BC_E1 captured part of a shared direction; α=4 was underpowered. α=8–16 is the only band that flips a target **without** breaking contradiction controls (only the easiest case, `BC11_E3`, margin 2.75). α=32 is a sledgehammer.

### Family-level L20 vector — **could not fit** (empty class A)

Attempted `unit(mean(A) − mean(B))` at L20 with:

- **Class A** = gold SATISFIED temporal-change **and** HF commit X=false (`TF_E1`, `TF_E2`, `TF_E5`)
- **Class B** = gold CONTRADICTION **and** HF commit X=true (`TF_C1`, `BC_C1`)
- Failures `{TF_E3, BC11_E3, TF_E4}` **eval-only** (not in the contrast). `TF_N1` excluded.

| Case | intended | HF X | margin | kept |
|------|----------|------|--------|------|
| TF_E1 | class A | **true** | 4.125 | no |
| TF_E2 | class A | **true** | 3.375 | no |
| TF_E5 | class A | **true** | 5.25 | no |
| TF_C1 / BC_C1 | class B | true | 12.375 / 13.375 | yes |
| BC_E2 | nofail control | false | −14.0 | not in contrast |

**Class A kept = ∅.** On HF `Qwen2.5-7B-Instruct`, *every* gold-SATISFIED temporal-change note we have in DEV commits false `contradiction.present`. The “three-case cluster” is the whole SATISFIED family, not a subset. Fitting `mean(3 failures) − BC_E2` was **not** done (that is the overfit contrast we rejected).

No α-sweep. No intervention frozen. Test set still sealed.

**Update 2026-09-03:** expand panel (`TX_*`, n=30) found **n_A=4**, fitted L20 family vector, safe α≤16 with 3/8 weak-fail flips — see [TRACKB-EXPAND-PANEL.md](./TRACKB-EXPAND-PANEL.md). Original `TF_*` Class A emptiness stands; expand did not rewrite those notes.

### Reading

Localize: L20 is the earliest sufficient patch site.  
Characterize old intervention: Ph10 vector is the right *sign*, wrong *strength/specificity* for the family.  
Semantic family direction: **blocked on current DEV** — no clean temporal-change commits to average. L20 patch remains causal but not class-selective.

### Artifacts

`artifacts/trackb-falsex-cluster-panel.json` · `artifacts/trackb-falsex-patch-0_75-panel.json` · `artifacts/trackb-falsex-patch-depth-panel.json` · `artifacts/trackb-falsex-bce1-alpha-panel.json` · `artifacts/trackb-falsex-family-l20-panel.json`

### Open

1. ~~Need clean class A~~ — expand panel has n_A=4 on 7B; see [TRACKB-EXPAND-PANEL.md](./TRACKB-EXPAND-PANEL.md).
2. ~~LOO + specificity~~ — **7B endpoint YES @ α=8** (LOO-stable partial editor); 14B transfer: failure cluster does **not** persist (1/24 fail) — no 14B MI ladder.
3. Freeze limited α=8 claim vs Track A reconnect; then optionally score `temporal-family-test.json`. Do not modify G3.

## Claim hygiene

Say: lab-scale causal pilot on DEV false-X cluster; probe separable; mid-depth patch sufficient; frozen steer does not transfer at α=4.  
Do not say: Track B closed; clinical safety; test-set validated; "we found the contradiction neuron."
