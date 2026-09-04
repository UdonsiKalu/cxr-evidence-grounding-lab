# Track B freeze — expand L20 α=8 limited claim

**Status:** **FROZEN 2026-09-04**  
**Scope:** DEV expand panel only (`data/temporal-family-dev-expand.json`)  
**Sealed:** `temporal-family-test.json` — still not scored  
**G3:** untouched

---

## Claim (allowed)

On HF `Qwen2.5-7B-Instruct`, at absolute layer **L20**, the semantic direction

```text
v = unit( mean(Class A) − mean(Class B) )
```

with

- **Class A** = gold SATISFIED temporal-change **and** commit `contradiction.present=false`
- **Class B** = gold CONTRADICTION **and** commit X=true

at strength **α = 8**:

- is **LOO-stable** (mean cos≈0.985; 4/4 folds safe),
- **beats** matched Gaussian noise and the reverse vector,
- flips **3/8** weak-margin false-X fails while preserving contradiction + nofail controls.

This is a **partial editor**, not a family-wide fix.

## Claim (forbidden)

- Do **not** say steering fixes temporal-change false-X in general.
- Do **not** say α=16/32 is the freeze (α=16 weaker specificity; α=32 destroys controls).
- Do **not** claim 14B transfer of the editor — 14B was a **behavioral** transfer check only.

## 14B transfer (same frozen wording)

HF `Qwen2.5-14B-Instruct` on the same expand DEV: fail_temporal **1/24**.  
The false-X **cluster does not persist**. No 14B MI ladder was run.

## Artifacts

- `artifacts/trackb-expand-7b-endpoint.json`
- `artifacts/trackb-expand-14b-transfer.json`
- `artifacts/trackb-expand-finish7b-then-14b-summary.json`
- [TRACKB-EXPAND-PANEL.md](./TRACKB-EXPAND-PANEL.md)

## Reconnect

Track B stops here for this family. Track A contains remaining wrong_AUTO (esp. Qwen Dual) via protocol gates / grounding — not by more 7B squeezing.
