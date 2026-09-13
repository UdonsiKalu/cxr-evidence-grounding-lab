# MAP-06 — Four notes walked (I2, C4, T1, T2)

**Prerequisite:** MAP-05 · **Next:** [MAP-07](./07-leftovers-and-stop.md)

Theory without a note is fog. Run the same ladder the GUI uses.

## Theory — how to read a ladder

For each depth: verdict, match vs gold, atoms, whether the extract actually changed.

- **Signature depth** (selector): phrase patterns in the extract that *would* pick that cell. Not a trained router.
- **Oracle**: shallowest depth that already matches gold. Upper bound, not a deployed selector.

If selector is `none` and oracle is `none`, the raw Dual extract was already right (T1).  
If oracle is `unrepaired`, no cell in the stack matches gold (T2).

## Backend — one command

```bash
cd ~/staging/cxr-evidence-grounding-lab
../cxrlabs/faiss_gpu1/bin/python curriculum-mapping/snippets/replay_ladder.py I2 C4 T1 T2
```

## What you should see

### I2 — implied failure, admin token wrong

Note: platinum-refractory, now considering next line. Gold **SATISFIED**.

- none: NOT_SATISFIED, `admin=planned`, B false
- G3: C and D become true (failure language), still planned so still miss
- G9: `planned → given`, B true, SATISFIED. Selector and oracle both G9

This is P-bind/P-impl: the clinic language was there; the extract parked “planned.”

### C4 — two 1L stories at once

Gold **CONTRADICTION**.

- none/G3: SATISFIED (the rule believed a clean failure)
- G4: X true → CONTRADICTION. **Oracle = G4**
- G8: still CONTRADICTION, but may restage an outcome polarity. **Selector = G8** (deeper signature even though gold already matched)

This is why nested ≠ independent choice: G8 still includes G4.

### T1 — already correct

Gold **SATISFIED**. Response then later progression. Raw extract already A=B=C=D true. Every cell no-op. Selector `none`, oracle `none`.

### T2 — leftover (too early)

Gold **NOT_SATISFIED** (just started cycle 1; no failure event). Dual stays **UNCERTAIN**. No cell changes it. Oracle `unrepaired`. Tag `S-too-early`. Force-fitting a “failure” here would be the one-size-fits-all mistake.

## GUI

On `:8263`, click the same four ids. Confirm the table matches the CLI (verdicts and “what this cell changed”).

## Week 1 — send-back

Four lines: I2 / C4 / T1 / T2 — gold, first matching cell (or unrepaired), one-sentence why.
