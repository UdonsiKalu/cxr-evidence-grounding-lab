# HF determinism (Ph9–14 / Workbench)

**Problem:** Greedy (`do_sample=False`) still allowed GPU bf16 / TF32 / cuDNN
nondeterminism so commit margins jittered (±0.13 around zero) and LIVE BC_E1
could disagree with frozen Ph10.

**Fix (2026-08-28):** `n2s_lab/determinism.py` + call sites in `hf_client`,
`hf_intervene`, `hf_trace`, Workbench `server.py` / `workbench.py`.

| Knob | Value |
|------|--------|
| Seed | `N2S_HF_SEED` (default **0**) |
| `CUBLAS_WORKSPACE_CONFIG` | `:4096:8` |
| cuDNN deterministic | on; benchmark off |
| TF32 | off |
| `use_deterministic_algorithms` | `warn_only=True` |
| Sampling | `do_sample=False` only |

## Verify

```bash
cd ~/staging/cxr-evidence-grounding-lab
.venv-phase9/bin/python run_determinism_check.py
# → artifacts/determinism-check-bc-e1.json  (expect "match": true)
```

Then Workbench **0 · Run → baseline → steered** twice on BC_E1; margins must match.

## Rollback

Snapshot:

`staging/.backups/hf-determinism-20260828T0106Z/`

```bash
cp -a ~/staging/.backups/hf-determinism-20260828T0106Z/n2s_lab/* \
  ~/staging/cxr-evidence-grounding-lab/n2s_lab/
cp ~/staging/.backups/hf-determinism-20260828T0106Z/repeng-static/index.html \
  ~/staging/cxr-evidence-grounding-lab-repeng/static/
cp ~/staging/.backups/hf-determinism-20260828T0106Z/workbench.py \
  ~/staging/cxr-evidence-grounding-lab-repeng/workbench.py
# restore server.py from git if needed; remove n2s_lab/determinism.py
systemctl --user restart n2s-repeng-8256
```

Lab git branch/commit: see `git log` on `cxr-evidence-grounding-lab` after ship.
