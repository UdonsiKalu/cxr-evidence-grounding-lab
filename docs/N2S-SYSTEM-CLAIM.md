# N2S system claim — September 2026 freeze snapshot

**Lab-scale only.** Not clinical validation. Not production CXR.  
**Audience:** visitor / faculty / portfolio one-pager.  
**Detail:** [RESEARCH-APPROACH.md](./RESEARCH-APPROACH.md) §0 · [TRACKB-DOWNSTREAM-CLOSEOUT.md](./TRACKB-DOWNSTREAM-CLOSEOUT.md) · [TRACKB-UPSTREAM-PREFILL.md](./TRACKB-UPSTREAM-PREFILL.md) · [TRACKB-UPSTREAM-PORTFOLIO.md](./TRACKB-UPSTREAM-PORTFOLIO.md) · [TRACKB-UPSTREAM-UA-GENERALIZE.md](./TRACKB-UPSTREAM-UA-GENERALIZE.md) · [TRACKB-UPSTREAM-UA-GEN-FREEZE.md](./TRACKB-UPSTREAM-UA-GEN-FREEZE.md) · [TRACKA-TEMPORAL-FAMILY-FREEZE.md](./TRACKA-TEMPORAL-FAMILY-FREEZE.md) · [TRACKB-ALPHA8-FREEZE.md](./TRACKB-ALPHA8-FREEZE.md)

---

## The picture (three pieces)

```text
Doctor's note
      ↓  UPSTREAM — formation of internal reps (prefill)
      ↓  DOWNSTREAM — utilization → structured LLM output
════════════════════════════
   NEURAL → SYMBOLIC BOUNDARY
════════════════════════════
      ↓  CXR atoms / Dual / gates → AUTO | REVIEW
```

Upstream vs downstream = experimental partition (formation → utilization).  
Neural→symbolic = hard external cut (Track A).

---

## What we claim (allowed)

### Boundary — Track A

On the temporal-family lab predicate (`FIRST_LINE_THERAPY_FAILED`), after grounding reconnect (temporal-change ≠ contradiction):

- Ollama **Qwen Dual_full** achieves **wrong_AUTO = 0** on DEV and sealed TEST (safety among AUTO = 1.0).  
- Remaining misses are mostly **REVIEW** (containment), not silent wrong AUTO.  
- Coverage: meta/predicate grounding → DEV resim correct_AUTO 3→5; live Dual confirm **correct_AUTO=7**, wrong_AUTO=0 (TF_T1/N1 still REVIEW on live re-extract).  
- Sealed test re-open score-only: Dual still wrong_AUTO=0 / 5 / 7.  
- G3 untouched; sealed test not used for redesign.

### Downstream — Track B utilization

- Frozen expand **v** @ L20, **α=8**: **partial** editor — selective flip on some temporal false-X; **no flip** on true-contradiction controls; beats gaussian/reverse in the freeze panel.  
- SAE (Chanin L20): decompose **v** + top-k steers — pilot only; ranking ≠ “temporality neuron.”  
- **Circuit C0–C2 (frozen):** top write **L20/mlp** along `v`; C1 soft_gate **YES** (causal site); C2 **LOCAL_SUFFICIENT** (L16 feeder weak). **Not** a full circuit — [TRACKB-CIRCUIT-C01-FREEZE.md](./TRACKB-CIRCUIT-C01-FREEZE.md).  
- Workbench `:8257` + CLI implement sketches ship Evaluate → Deep dive → Intervene → SAE.

### Upstream — formation (U0 / U1 / U-A / U-B)

- U0: note-body cue × layer residuals; commit-fitted **v** is **weak** at cues.  
- U1/U1b: prefill-position ±α·v at recipe / L16 outcome sites (**no** commit X flip).  
- **U-A:** multi-token class-mean Δ map — strongest separation **L24**; cos(d, freeze-v)≈0.03 (formation ≠ commit editor).  
- **U-A gen:** frozen `d` separates held-out near/far paraphrases @ L24 (**soft+strong YES** on small panel) — correlational lexical generalization only.  
- **U-A causal-d:** steer along frozen `d` @ L24 (commit + prefill) — **NULL/weak** (no dose-dependent X/margin editor on FOLFOX/CONTRA panel).  
- **U-B / U-B2:** component ablation at top sites — **null** (no X flip). Ablation family **paused**.  
- **Workbench `:8258`:** live score vs frozen U-A `d` + optional custom-pair map (correlational demos). Not on `:8257`.  
- **Null causal** at recipe ablation sites **and** α·d steer — do **not** claim an upstream editor or circuit.

---

## What we do not claim

- Clinical / production safety.  
- Steering fixed healthcare or all temporal false-X.  
- α=16/32 freeze; 14B MI editor transfer.  
- Upstream circuit / named clinical feature / “found temporality neuron.”  
- Live ablate / U-C SAE / U-D circuits from the U-B/U-B2 nulls **or** from Downstream C0–C2.  
- Universal coverage (AUTO rate remains modest).

---

## Status / next

| Stream | Status |
|--------|--------|
| Downstream MI | Rounded up — stop expanding SAE/circuits/α |
| Upstream U0+U1+U1b | Done — null; no α-chase |
| Upstream U-A | Done — L24 formation map (correlational) |
| Upstream U-B/U-B2 | Done — ablation **null**; family **paused** |
| Upstream GUI `:8258` | Live score + custom pair demos OK |
| Downstream circuit C0–C2 | **FROZEN** L20 MLP causal site (+ path restrict) — [TRACKB-CIRCUIT-C01-FREEZE.md](./TRACKB-CIRCUIT-C01-FREEZE.md) |
| Upstream U-A gen | **FROZEN** soft+strong YES @ L24 — [TRACKB-UPSTREAM-UA-GEN-FREEZE.md](./TRACKB-UPSTREAM-UA-GEN-FREEZE.md) |
| Upstream U-A causal-d | First panel **NULL/weak** (commit+prefill @ L24) — [TRACKB-UPSTREAM-UA-CAUSAL.md](./TRACKB-UPSTREAM-UA-CAUSAL.md) |
| Track A temporal family | **Frozen** wrong_AUTO=0 DEV+test |
| Residual Track A | Live Dual confirm **done** (wrong_AUTO=0, correct_AUTO=7); TF_T1·N1 still REVIEW live |
| Sealed test | Re-scored under current ground (`--tracka-resim-test`); unchanged 0/5/7; do not redesign |
| **Next** | Circuit ladder frozen; **leave Upstream U-C/U-D alone**; no α-chase / sealed redesign |

**Portfolio line:** Track A contains uncertain neural→symbolic transforms via REVIEW; Track B shows a limited commit-time editor **and** a frozen L20 MLP causal site (not a full circuit); Upstream shows a correlational late-layer formation map that generalizes to held-out paraphrases but does **not** yield a causal editor via top-site ablation **or** α·d steer at L24 (see [TRACKB-UPSTREAM-PORTFOLIO.md](./TRACKB-UPSTREAM-PORTFOLIO.md) · [TRACKB-CIRCUIT-C01-FREEZE.md](./TRACKB-CIRCUIT-C01-FREEZE.md)).
