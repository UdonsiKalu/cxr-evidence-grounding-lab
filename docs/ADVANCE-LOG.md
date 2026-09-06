# Advance log — N2S fidelity / RepEng / MI

**Purpose:** One place to record each advance — what we did, what happened, where the evidence lives.  
**Use for:** papers, portfolio, faculty notes, handoff.  
**Not:** chat transcript. Keep entries short; link detail docs/artifacts.

**How to add an entry (every advance):** copy the template below to the **top** of the log (newest first).

---

## Template (copy)

```markdown
### YYYY-MM-DD — short title

| | |
|--|--|
| **Track** | A reliability / B mechanism / both / infra |
| **Activity** | 1–3 bullets of what we ran or built |
| **Result** | Numbers or qualitative outcome (wrong_AUTO, gates, transfer…) |
| **Artifacts** | paths / URLs |
| **Decision** | what we will / will not do next because of this |
| **Commit** | `sha` if pushed |
```

---

## Log (newest first)

### 2026-09-06 — Circuit C0–C2 freeze + leave Upstream alone

| | |
|--|--|
| **Track** | B / Downstream circuit + portfolio hygiene |
| **Activity** | Freeze C0+C1; run C2 path restrict (L16 feeder vs L20); lock Upstream U-C/U-D leave-alone in PROGRAM / SYSTEM-CLAIM. |
| **Result** | C1 soft_gate YES retained; C2 **LOCAL_SUFFICIENT** (L16 alone weak; stack ≈ L20). |
| **Artifacts** | `TRACKB-CIRCUIT-C01-FREEZE.md` · `n2s-circuit-c2-path-restrict.json` · CIRCUIT-PILOT / SAE-CIRCUIT-WHERE / SYSTEM-CLAIM |
| **Decision** | Circuit ladder frozen at site+local path. Do not reopen Upstream U-C/U-D. Stop C3 unless new Q. |
| **Commit** | (this push) |

### 2026-09-06 — Circuit C1 L20 MLP causal patch (soft_gate YES)

| | |
|--|--|
| **Track** | B / Downstream circuit |
| **Activity** | C0 top site L20/mlp; commit-time zero/replace MLP (+ attn & L16 controls) on Class A/B expand cases. |
| **Result** | Soft gate **YES** — B←A mean Δmargin ≈ −3.1 (beats controls); no B X-flip. A←B flips X false→true on both TX_E08/E13 (Δmargin ≈ +8.1). |
| **Artifacts** | `TRACKB-CIRCUIT-PILOT.md` · `n2s_circuit_pilot.py` (`patch`) · `n2s-circuit-c1-l20-mlp-patch.json` |
| **Decision** | L20 MLP commit write is a causal site (not full circuit). Optional C2 path restrict wait-go; do not reopen Upstream U-C/U-D from this. |
| **Commit** | (local) |

### 2026-09-06 — U-A causal-d intervene panel (commit+prefill NULL)

| | |
|--|--|
| **Track** | B / new causal Q |
| **Activity** | Protocol + runner; steer along frozen U-A `d` @ L24 (commit + prefill note-body); α=1/2/4/8; ±d + gaussian; FOLFOX+CONTRA. |
| **Result** | **NULL/weak** both sites — no X flip; margins flat/noise; contra control held. Distinct from U-B (direction vs top-site ablate); both null families. |
| **Artifacts** | `TRACKB-UPSTREAM-UA-CAUSAL.md` · `n2s_upstream_ua_causal.py` · `*-commit-panel.json` · `*-prefill-panel.json` |
| **Decision** | Frozen `d` remains correlational. Do not claim causal editor. Do not reopen U-C/U-D/ablate thrash from this null. |
| **Commit** | (local) |

### 2026-09-06 — Freeze U-A paraphrase generalization (soft+strong YES)

| | |
|--|--|
| **Track** | B / freeze |
| **Activity** | Locked freeze claim for held-out paraphrase gen of frozen U-A `d` @ L24. |
| **Result** | Soft+strong YES frozen; correlational only; ablation branch still closed. |
| **Artifacts** | `docs/TRACKB-UPSTREAM-UA-GEN-FREEZE.md` · panel/readout/directions |
| **Decision** | Do not rebuild `d` from held-out; do not reopen U-C/U-D/ablate from this YES. |
| **Commit** | (this freeze commit) |

### 2026-09-06 — U-A held-out paraphrase generalization (soft+strong YES)

| | |
|--|--|
| **Track** | B / new Q |
| **Activity** | Froze protocol + held-out JSON; ran panel scoring frozen U-A `d` on near/far paraphrases (not rebuild `d`; not ablation). |
| **Result** | L24 soft+strong **YES** — held-out mean_T≈58.4 > mean_C≈23.5; min_T≈45.5 > max_C≈29.0. Anchors 71.4 vs 9.4. |
| **Artifacts** | `docs/TRACKB-UPSTREAM-UA-GENERALIZE.md` · `data/heldout-ua-paraphrase.json` · `n2s-upstream-ua-gen-panel.json` · `n2s_upstream_ua_generalize.py` |
| **Decision** | Correlational lexical gen holds on this small panel. Still **not** causal editor; do **not** reopen U-C/U-D/ablate thrash. Optional next = causal Q only with explicit go. |
| **Commit** | (local) |

### 2026-09-06 — Upstream portfolio package (summary + diagram + plain English)

| | |
|--|--|
| **Track** | B / portfolio |
| **Activity** | Froze one Upstream write-up: summary (U-A / U-B·U-B2 / :8258 demos), correlation≠control diagram, plain-English portfolio line; leave-branch lock. |
| **Result** | Package shipped; science branch left alone. Next technical Q deferred = held-out paraphrase / lexical generalization (not more L20/L24 ablate). |
| **Artifacts** | `docs/TRACKB-UPSTREAM-PORTFOLIO.md` |
| **Decision** | **Do not** reopen U-C SAE / U-D circuits / live ablate thrash. Portfolio package is the close for this Upstream ablation family. |
| **Commit** | (local) |

### 2026-09-06 — Upstream pause + :8258 demos (Score / custom pair)

| | |
|--|--|
| **Track** | B / infra |
| **Activity** | Dedicated `:8258` live workbench; Score FOLFOX temporal + EX_CONTRA vs frozen U-A `d`; custom-pair map (short progression vs no-new-lesions). Claim lock in `N2S-SYSTEM-CLAIM.md` + PROGRAM. |
| **Result** | FOLFOX temporal L24 `|μ·d|≈71`; EX_CONTRA L24 `|μ·d|≈18.5` (top Disease/remains); custom pair weaker sep (~25.6 vs lab ~62). Ablation family remains null/paused. |
| **Artifacts** | `n2s-upstream-live-runs/` · `upstream_server.py` · `n2s_upstream_live.py` |
| **Decision** | **Pause** Upstream science for portfolio/write-up. **Do not** open U-C SAE / U-D circuits / live ablate thrash. |
| **Commit** | (local) |

### 2026-09-06 — Upstream U-B2 L20 mean-ablate (null → pause)

| | |
|--|--|
| **Track** | B |
| **Activity** | `PrefillComponentAblateSpec` mean mode; U-B2 `--layer 20 --mode mean` on U-A L20 top-5. |
| **Result** | **Null** — no X flip; |Δmargin|<1. Same story as U-B. |
| **Artifacts** | `n2s-upstream-ub2-patch.json` · `n2s-upstream-ub2-readout.json` |
| **Decision** | **Pause** ablation family; do not open U-C/U-D from null. New question only on explicit go. |
| **Commit** | (local) |

### 2026-09-06 — Upstream U-B component ablation (null)

| | |
|--|--|
| **Track** | B |
| **Activity** | `PrefillComponentZeroSpec` in `hf_intervene`; `n2s_upstream_ub_patch` zero-ablate attn/mlp/resid @ U-A L24 top-5 → extract X/margin. |
| **Result** | **Null** — no X flip either note; |Δmargin|≤0.5. Shortlist empty. |
| **Artifacts** | `n2s-upstream-ub-patch.json` · `n2s-upstream-ub-readout.json` · `TRACKB-UPSTREAM-PROGRAM.md` |
| **Decision** | Do not jump to circuits/SAE. Optional U-B2 (L20 / mean-ablate / cross-patch) wait-go. |
| **Commit** | (local) |

### 2026-09-06 — Upstream program + U-A multi-token map

| | |
|--|--|
| **Track** | B |
| **Activity** | Froze `TRACKB-UPSTREAM-PROGRAM.md` (U-A→U-D). Implemented `n2s_upstream_ua_map`; live map on FOLFOX temporal vs contra. |
| **Result** | Separation grows with depth; strongest **L24** \|\|μ_T−μ_C\|\|≈62; **cos(d, freeze-v)≈0.03** everywhere. Temporal top sites = disease/imaging/lesions language; contra = Disease/failed/remains. U-A exit met. |
| **Artifacts** | `TRACKB-UPSTREAM-PROGRAM.md` · `n2s-upstream-ua-map.json` · `n2s-upstream-ua-readout.json` |
| **Decision** | No browser yet. Next = U-B path-patch top sites (attn/MLP) — wait go. Not α·v. |
| **Commit** | (local) |

### 2026-09-06 — Options pass: live Dual + sealed resim + U1b

| | |
|--|--|
| **Track** | both |
| **Activity** | (1) Live Qwen Dual temporal-dev confirm. (2) `--tracka-resim-test` sealed score re-open (no redesign). (3) U1b outcome cues @ L16 α=8. (4) Write-up polish. |
| **Result** | Live Dual **wrong_AUTO=0**, correct_AUTO **7**, REVIEW **7** (TF_T1/N1 still REVIEW live). Sealed resim unchanged **0/5/7**. U1b **null** (no X flip). |
| **Artifacts** | `phase7-temporal-dev-qwen-live-coverage.json` · `tracka-qwen-live-coverage-dual-score.json` · `tracka-temporal-test-grounding-resim.json` · `n2s-upstream-prefill-patch-u1b.json` |
| **Decision** | Options closed for this family. No α-chase. Sealed test not used for redesign. |
| **Commit** | (local) |

### 2026-09-06 — Track A TF_T1/N1 coverage (meta/predicate grounding)

| | |
|--|--|
| **Track** | A |
| **Activity** | Fixed Path-D meta false-X: strip predicate-label noise in `_clinical_blob_for_conflict`; meta contradiction override no longer blocked by predicate-name “failure”; unit tests; `--tracka-resim` on frozen Phase-7 temporal-dev. G3 untouched. |
| **Result** | Qwen Dual_full **wrong_AUTO=0**, **correct_AUTO 3→5**, **REVIEW 11→9**; TF_T1 + TF_N1 Dual **AUTO NOT_SATISFIED** (match gold). Llama wrong_AUTO=2 unchanged; Mistral all REVIEW. |
| **Artifacts** | `n2s_lab/ground.py` · `artifacts/tracka-temporal-dev-grounding-resim.json` · `docs/TRACKA-RESIDUAL.md` |
| **Decision** | Coverage win on DEV resim. Optional live Dual confirm wait-go; do not retune sealed test. |
| **Commit** | (local) |

### 2026-09-06 — System claim write-up + Track A residual assess

| | |
|--|--|
| **Track** | both |
| **Activity** | Wrote `N2S-SYSTEM-CLAIM.md`; updated newcomer tour + PDF (frozen claims crib). Assessed Track A residual: wrong_AUTO already 0; TF_T1/N1 = REVIEW coverage, not fire — `TRACKA-RESIDUAL.md`. No live Dual re-run; no gate redesign. |
| **Result** | Portfolio/visitor claim locked. Track A residual = optional coverage wait-go. |
| **Artifacts** | `docs/N2S-SYSTEM-CLAIM.md` · `docs/TRACKA-RESIDUAL.md` · `N2S-Workbench-Newcomer-Tour.pdf` |
| **Decision** | Stop MI expand. Coverage experiments only on explicit go. |
| **Commit** | (local) |

### 2026-09-06 — Upstream U0 readout + U1 prefill patch (null)

| | |
|--|--|
| **Track** | B |
| **Activity** | Wrote U0 readout; added `prefill_position_steer` in `hf_intervene.py`; ran U1 ±α=8 @ L20 on recipe sites (last FOLFOX temporal; failed+responsive contra). |
| **Result** | U0: commit-v weak at cues. U1: **no X flip** on either note; margins ~unchanged; true contra stays X=true. Negative causal result for these sites. |
| **Artifacts** | `n2s-upstream-u0-readout.json` · `n2s-upstream-prefill-patch.json` · `TRACKB-UPSTREAM-PREFILL.md` |
| **Decision** | Do not claim upstream editor. Stop α chase. Optional later: different cues / layers / notes — not blocking. Downstream α=8 claim unchanged. |
| **Commit** | (local) |

### 2026-09-06 — Upstream U0 prefill_trace

| | |
|--|--|
| **Track** | B |
| **Activity** | Implemented `n2s_upstream_prefill.py` + `run_upstream_prefill.sh` + `TRACKB-UPSTREAM-PREFILL.md`. Ran U0 on EX_TEMPORAL_FOLFOX + EX_CONTRA (note-body cues only; EXTRACT_SYSTEM hits excluded). |
| **Result** | Artifact `n2s-upstream-prefill-trace.json` — cue×layer proj/cos_v + L20 SAE top-k; shared-cue L20 contrast primarily FOLFOX. Observational only (no patch). |
| **Artifacts** | `artifacts/n2s-upstream-prefill-trace.json` · `docs/TRACKB-UPSTREAM-PREFILL.md` |
| **Decision** | Next = interpret U0; U1 patch→commit only if coherent. Do not expand SAE/circuits yet. |
| **Commit** | (local) |

### 2026-09-06 — Downstream roundup + three-piece system map

| | |
|--|--|
| **Track** | both |
| **Activity** | Locked upstream / downstream / neural→symbolic characterization in `RESEARCH-APPROACH.md` §0. Wrote `TRACKB-DOWNSTREAM-CLOSEOUT.md`. Updated newcomer tour. CLI walkthrough already has implement sketches (W01–W08). |
| **Result** | Downstream utilization + boundary **closed for expansion**; next = thin upstream prefill (`prefill_trace`). α=8 + SAE pilots unchanged; sealed test still sealed. |
| **Artifacts** | `docs/TRACKB-DOWNSTREAM-CLOSEOUT.md` · `docs/RESEARCH-APPROACH.md` · `cxr-n2s-eval-workbench/docs/N2S-Workbench-Newcomer-Tour.pdf` |
| **Decision** | Do not expand SAE/circuits/α before first upstream prefill slice. Reuse HF hooks + frozen **v** / Chanin L20. |
| **Commit** | (local) |

### 2026-09-04 — Sealed temporal-test: Qwen Dual wrong_AUTO=0

| | |
|--|--|
| **Track** | A |
| **Activity** | First open of `temporal-family-test.json` after DEV freeze. Live Dual on `qwen2.5-coder:32b` only. Scored AUTO contract. **No redesign** from results. G3 untouched. |
| **Result** | Dual_full n=12: **wrong_AUTO=0**, correct_AUTO=5, REVIEW=7, safety_among_auto=**1.00**. Matches DEV live claim. |
| **Artifacts** | `artifacts/tracka-temporal-test-qwen-dual-score.json` · `phase7-temporal-test-qwen-*.json` · [TRACKA-TEMPORAL-FAMILY-FREEZE.md](./TRACKA-TEMPORAL-FAMILY-FREEZE.md) |
| **Decision** | Track A temporal-family claim frozen (DEV+test). Do not retune from this open. Optional later: Llama residual / coverage improvement without raising wrong_AUTO. |
| **Commit** | (this) |

### 2026-09-04 — Live Qwen Dual: wrong_AUTO=0 after grounding reconnect

| | |
|--|--|
| **Track** | A |
| **Activity** | Extended grounding overrides (no-failure / toxicity-stop false X; “no progression” ≠ failure). Live `run_phase7.py --set temporal-dev --models qwen2.5-coder:32b`. Scored Dual_full under AUTO contract. G3 untouched; test sealed. |
| **Result** | **Dual_full wrong_AUTO=0** / correct_AUTO=3 / REVIEW=11 / safety_among_auto=**1.0**. False-X family live-confirmed (TF_E3/BC11_E3 AUTO SATISFIED; TF_E4 REVIEW). TF_T1/TF_N1 Dual **REVIEW** (contained) even when D_full alone still errs. Resim also Qwen wrong_AUTO **0**. |
| **Artifacts** | `artifacts/phase7-temporal-dev-qwen-live-grounding.json` · `artifacts/phase7-temporal-dev-qwen2.5-coder_32b.json` (live overwrite) · `artifacts/tracka-qwen-live-dual-score.json` · `artifacts/tracka-temporal-dev-grounding-resim.json` |
| **Decision** | Track A containment for this DEV family is **live-confirmed** on Qwen Dual. Next: optional Llama residual / sealed-test decision. No G3 redesign. |
| **Commit** | (this) |

### 2026-09-04 — Freeze α=8 Track B claim + Track A temporal grounding reconnect

| | |
|--|--|
| **Track** | both |
| **Activity** | (1) Froze limited Track B claim @ α=8 (`TRACKB-ALPHA8-FREEZE.md`). (2) Track A reconnect **without G3 change**: `ground.py` treats sequenced temporal-change (response→failure / possible→confirmed) as X=false even if extractor sets X; fixed substring traps (`continue`∈`Discontinued`). Resim Dual_full from frozen Phase-7 temporal-dev extractions (no LLM). |
| **Result** | Qwen Dual_full wrong_AUTO **5→2**; false-X family **TF_E3 / BC11_E3 / TF_E4 cleared**; remaining wrongs = `TF_T1`/`TF_N1` (control overfire, different shape). Llama/Mistral unchanged pattern (2 / 0 wrong). True-contradiction controls stay REVIEW. |
| **Artifacts** | `docs/TRACKB-ALPHA8-FREEZE.md` · `docs/TRACKA-TEMPORAL-RECONNECT.md` · `artifacts/tracka-temporal-dev-grounding-resim.json` |
| **Decision** | Track B family closed as limited claim. Track A primary path = Qwen Dual containment via grounding; optional live Dual re-run later. Test sealed; G3 untouched. |
| **Commit** | (this) |

### 2026-09-03 — 7B endpoint YES (α=8); 14B transfer: failure does not persist

| | |
|--|--|
| **Track** | B |
| **Activity** | Finished 7B LOO Class-A + steer specificity @ α=8/16 on expand L20 vector. Then one HF 14B behavioral collect on the **same** frozen expand DEV. Question = does the temporal N2S failure persist across models? **Not** “can 14B solve better?” No forced MI ladder. |
| **Result** | **7B endpoint YES @ α=8** — mean LOO cos≈0.985, 4/4 LOO folds safe with 3/8 flips; full beats Gaussian (3 vs 0); reverse 0 flips. α=16 also safe but Gaussian gets 2/8 (weaker specificity). **14B:** Class A=21/24, fail_temporal=**1** (`TX_E14` only), Class B=4 intact → matched contrast **not useful**; **MI ladder not run**. |
| **Artifacts** | `artifacts/trackb-expand-7b-endpoint.json` · `artifacts/trackb-expand-14b-transfer.json` · `artifacts/trackb-expand-finish7b-then-14b-summary.json` · [TRACKB-EXPAND-PANEL.md](./TRACKB-EXPAND-PANEL.md) |
| **Decision** | Freeze limited 7B claim at **α=8** (partial editor). 14B mostly avoids false X on this panel — supports **Track A containment / scale-variant behavior**, not a 14B steer program. Test sealed; G3 untouched. |
| **Commit** | `be6180e` (+ results docs) |

### 2026-09-03 — Expand panel: natural Class A on 7B; L20 family vector safe at α≤16

| | |
|--|--|
| **Track** | B |
| **Activity** | Froze `temporal-family-dev-expand.json` (n=30, wording pre-HF). Ran `--expand-sequence`: HF 7B collect → Class A/B thresholds → L20 `unit(mean(A)−mean(B))` α-sweep. Predeclared: fit only if n_A≥3 and n_B≥2. Test sealed; G3 untouched. **Did not** escalate to 14B (7B had enough A). |
| **Result** | **Class A = 4** (`TX_E08/E13/E21/E23`); **Class B = 4**; fail_temporal = 20/24. Safe flips: α=1–4 → 2/8 (`TX_E05/E06`); α=8–16 → **3/8** (+`TX_E04`) with contra+nofail intact. α=32 → 7/8 but **destroys `TX_C02`**. Hard fails (m≳5) only move partially under safe α. |
| **Artifacts** | `artifacts/trackb-expand-collect-*.json` · `artifacts/trackb-expand-fit-*-L20.json` · `artifacts/trackb-expand-sequence-summary.json` · [TRACKB-EXPAND-PANEL.md](./TRACKB-EXPAND-PANEL.md) |
| **Decision** | Prefer **α=8 or 16** as candidate safe family edit (not 32). Next: LOO Class-A stability + specificity before freeze; still **no** `temporal-test` score. 14B deferred. |
| **Commit** | `ef311bd` / `852fee3` (+ follow-up for results docs) |

### 2026-09-03 — Family L20 vector: empty class A (cannot fit)

| | |
|--|--|
| **Track** | B |
| **Activity** | Attempted semantic contrast at L20: mean(clean temporal-change, X=false) − mean(true contradiction, X=true). Candidates A=`TF_E1/E2/E5`; B=`TF_C1/BC_C1`. Did **not** put {TF_E3, BC11_E3, TF_E4} or `TF_N1` in the contrast. |
| **Result** | **Class A = ∅.** HF 7B commits false X on **all** gold-SATISFIED temporal-change notes collected (E1 m=4.125, E2 m=3.375, E5 m=5.25, plus original cluster). Class B intact. **No vector fitted, no α-sweep.** Gate NO. |
| **Artifacts** | `artifacts/trackb-falsex-family-l20-panel.json` · `n2s_lab/trackb_family_vector.py` · [TRACKB-FALSEX-CLUSTER.md](./TRACKB-FALSEX-CLUSTER.md) |
| **Decision** | Do not fall back to fail−`BC_E2`. Need new clean class-A notes (or a model that commits some SATISFIED cases as X=false) before a family-level editor. G3 untouched; test sealed. |
| **Commit** | `3f80b78` |

### 2026-09-03 — Frozen BC_E1 α-sweep: α=4 miss was magnitude

| | |
|--|--|
| **Track** | B |
| **Activity** | Swept frozen Ph10 BC_E2−BC_E1 vector at α ∈ {0,1,2,4,8,16,32} on DEV false-X cluster + controls. Did **not** replace the vector. |
| **Result** | α=1–4: 0/3 flips, margins drift down, controls intact. **α=8 and 16: 1/3** (`BC11_E3` only) with contradictions preserved. **α=32: 3/3 flips but `TF_C1` destroyed**. Transfer failure at the frozen α=4 was **magnitude**, not a dead direction — but there is no α that flips the whole family without harming true contradiction. |
| **Artifacts** | `artifacts/trackb-falsex-bce1-alpha-panel.json` · [TRACKB-FALSEX-CLUSTER.md](./TRACKB-FALSEX-CLUSTER.md) |
| **Decision** | Keep Ph10 vector as a characterized baseline. Next: family-level **semantic** vector at L20 (temporal-change vs contradiction), not 3-fail−`BC_E2`. No G3; test sealed. |
| **Commit** | `20eb5da` |

### 2026-09-03 — Patch-depth sweep: earliest sufficient L20; specificity mixed

| | |
|--|--|
| **Track** | B |
| **Activity** | Absolute-layer patch (L4/8/12/16/20/24) donor `BC_E2` → {TF_E3, BC11_E3, TF_E4}. Specificity @ L20: Gaussian, contradiction-donor, reverse, donor→contradiction, donor→self. `TF_N1` not used as donor. |
| **Result** | L4–L16: **0/3** flips. **L20: 3/3** (margins ≈ −11.4, ≠ donor −14). L24: 3/3. Gaussian **0/3**; `TF_C1` donor **0/3**. Reverse induces false X on `BC_E2`. **Donor into true contradictions flips both to false** — causal but not class-selective. |
| **Artifacts** | `artifacts/trackb-falsex-patch-depth-panel.json` · `n2s_lab/trackb_patch_depth.py` |
| **Decision** | Fit any new steer at **L20**. Do not treat L20 patch as a safe contradiction-preserving editor. |
| **Commit** | `20eb5da` |

### 2026-09-03 — Track B false-X cluster ladder (DEV) — probe YES, patch YES, steer NO

| | |
|--|--|
| **Track** | B |
| **Activity** | Ran probe→localize→activation patch→Ph10 steer(α=4)→ablate(α=−4) on {TF_E3, BC11_E3, TF_E4} vs TF_C1/BC_C1 + TF_N1/BC_E2 on HF `Qwen2.5-7B-Instruct` (`.venv-phase9`, 240 W). Test set not opened. Caught a **final-layer patch confound** and re-ran the patch arm at 0.75 only. |
| **Result** | Family **reproduces on HF 7B** (3/3 false X). **Probe YES** — cos@1.00 false-X [−0.713,−0.627] vs contradiction [−0.554,−0.530], no overlap, gap 0.072, LOO 1.00 (replicates Ph13 with a wider gap). **Steer α=4: 0/3 flips** — frozen BC_E1 vector does not transfer; controls fully intact. **Ablate α=−4** raises margins (sign-consistent, underpowered). **Patch @0.75 only: 3/3 flips**, margins 6.875→−11.375, 2.75→−11.5, 9.5→−11.375 (≠ donor −14.0, so genuine recompute). Full-depth patch incl. 1.00 was **discarded** — margins were exactly the donor's, i.e. token forcing, not sufficiency. |
| **Artifacts** | [TRACKB-FALSEX-CLUSTER.md](./TRACKB-FALSEX-CLUSTER.md) · `artifacts/trackb-falsex-cluster-panel.json` · `artifacts/trackb-falsex-patch-0_75-panel.json` · `run_trackb_falsex.py` · `n2s_lab/trackb_falsex_cluster.py` |
| **Decision** | Do **not** claim steering fixes this family. Next: α sweep on cluster, refit vector on cluster (not BC_E1), layer sweep for earliest sufficient patch depth. G3 untouched; `temporal-test` stays sealed until an intervention is frozen. |
| **Commit** | `be0e222` |

### 2026-09-03 — Env provenance: `.venv-phase9` is the HF runtime

| | |
|--|--|
| **Track** | infra |
| **Activity** | Track B run failed on bare `python3` (`accelerate` missing). Traced the HF stack: repo-local `.venv-phase9` (created 2026-08-24, `--system-site-packages`) layers accelerate 1.14.0 + bitsandbytes 0.50.1 over system torch 2.5.1+cu121 / transformers 4.53.2. |
| **Result** | Nothing had been uninstalled — those two packages only ever lived in that venv. `cxrlabs/faiss_gpu1` also works (accelerate 1.10.0); `cxrlabs-dev/faiss_gpu1` and `cxr-migration/faiss_gpu1` are broken (no interpreter / dangling base on an unmounted drive). |
| **Artifacts** | `.venv-phase9/pyvenv.cfg` · run commands in [TRACKB-FALSEX-CLUSTER.md](./TRACKB-FALSEX-CLUSTER.md) |
| **Decision** | All HF/GPU phases run via `./.venv-phase9/bin/python`. No package installs needed. |
| **Commit** | `be0e222` |

### 2026-09-03 — Freeze temporal-family-test.json

| | |
|--|--|
| **Track** | A (eval harness) |
| **Activity** | Froze unseen n=12 `TFT_*` set; wired `--set temporal-test` + scorer preset; seal doc. |
| **Result** | Test sealed before Track B design on DEV. |
| **Artifacts** | `data/temporal-family-test.json` · [TEMPORAL-FAMILY-TEST.md](./TEMPORAL-FAMILY-TEST.md) |
| **Decision** | Do not peek evidence while designing; do not run Dual on test until intervention frozen. |
| **Commit** | `be0e222` |

### 2026-09-03 — Temporal-family-dev Dual baseline (measure before G3)

| | |
|--|--|
| **Track** | A (measure) → informs B |
| **Activity** | Renamed set to `temporal-family-dev.json` (dev, not held-out). Ran **unchanged** Phase-7 Dual on n=14 × 3 models. Case-level diagnose; **did not** change G3. |
| **Result** | Dual_full: Mistral wrong_AUTO=**0** / REVIEW=14; Llama wrong=**2** (UNCERTAIN); **Qwen wrong=5** with false-X cluster **TF_E3, BC11_E3, TF_E4** (+ TF_T1/TF_N1 overfire). Genuine failure family, not BC_E1 alone. |
| **Artifacts** | [TEMPORAL-FAMILY-DEV-BASELINE.md](./TEMPORAL-FAMILY-DEV-BASELINE.md) · `artifacts/phase7-temporal-dev-panel.json` · `auto-contract-diagnose-temporal-dev.json` · `auto-contract-score-temporal-dev.json` |
| **Decision** | Freeze `temporal-family-test.json` before gate redesign; Track B on {TF_E3, BC11_E3, TF_E4}; G3 still deferred. |
| **Commit** | `a91d317` (wiring) · `fe3cea1` / `28e81c1` (baseline) |

### 2026-09-03 — Track A AUTO contract + scorer

| | |
|--|--|
| **Track** | A |
| **Activity** | Froze AUTO rule (AUTO iff gates pass, else REVIEW). Built `run_auto_contract_score.py` + `n2s_lab/auto_contract.py`. Scored frozen Phase-5/7 panels. |
| **Result** | Ph7 Dual_full on BMT/CAR-T: Qwen wrong_AUTO=1 (`BC_E1`); Llama/Mistral Dual_full wrong_AUTO=0. Metric = wrong_AUTO → 0. |
| **Artifacts** | [AUTO-CONTRACT.md](./AUTO-CONTRACT.md) · `artifacts/auto-contract-score-phase7.json` · `artifacts/auto-contract-score-phase5.json` |
| **Decision** | Measure on temporal family next; do not redesign G3 from BC_E1 alone. |
| **Commit** | `3db62c0` |

### 2026-09-03 — Research spine documented (Track A vs B)

| | |
|--|--|
| **Track** | infra / framing |
| **Activity** | Wrote research spine: Track A (wrong AUTO/REVIEW) vs Track B (MI causal ladder); method map; claim hygiene. Linked from READMEs. |
| **Result** | Gates **contain** failures; they do not **explain** them — two-track plan frozen in docs. |
| **Artifacts** | [RESEARCH-APPROACH.md](./RESEARCH-APPROACH.md) · pointers in ARCHITECTURE-DIRECTION, EVALUATION-JOURNEY |
| **Decision** | Build AUTO contract then measure; Track B deepens mechanism separately. |
| **Commit** | `967b6cc` (lab) · workbench README `953f8e0` |

### 2026-09-02 — Public demos + modus operandi

| | |
|--|--|
| **Track** | infra |
| **Activity** | GitHub Pages replay for grounding lab + RepEng workbench (pre-run JSON, no models). First-time Path A/B/C READMEs. |
| **Result** | Live demos: [grounding](https://udonsikalu.github.io/cxr-evidence-grounding-lab/) · [workbench](https://udonsikalu.github.io/cxr-repeng-workbench/) |
| **Artifacts** | lab `docs/` · workbench `docs/explorer.json` + `workbench.json` |
| **Decision** | Demos are replay-only; live Run stays local/GPU. |
| **Commit** | workbench Pages `43cfc79`; MO READMEs `2f1a0c0` / `dd85aba` |

### Pre-2026-09 (frozen soft pilot — summary)

| | |
|--|--|
| **Track** | A precursors + B soft pilot |
| **Activity** | Ph1–8 grounding / verify / Dual / evidence; Ph9–11 observe→localize→steer→partial transfer on BC_E1 (HF 7B). |
| **Result** | Representation loss visible (A–D); BC_E1 false X steerable at α=4 layers 20/27; Ph11 partial. |
| **Artifacts** | `docs/PHASE5`…`PHASE11` · `artifacts/phase*-*.json` · Workbench UI |
| **Decision** | Soft pilot frozen; next wave = contract + family + deeper MI. |
| **Commit** | see phase panels on `main` |

---

## Index of related docs

| Doc | Role |
|-----|------|
| [RESEARCH-APPROACH.md](./RESEARCH-APPROACH.md) | Plan / method map |
| [AUTO-CONTRACT.md](./AUTO-CONTRACT.md) | Track A gate rule |
| [TEMPORAL-FAMILY-DEV-BASELINE.md](./TEMPORAL-FAMILY-DEV-BASELINE.md) | Latest Dual baseline detail |
| [ARCHITECTURE-DIRECTION.md](./ARCHITECTURE-DIRECTION.md) | Safe protocol north star |
| [EVALUATION-JOURNEY.md](../EVALUATION-JOURNEY.md) | Early milestone narrative |
| Phase protocols `PHASE*.md` | Per-phase freeze records |
