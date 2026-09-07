# Thin SAE pilot — decompose frozen L20 expand direction

**Status:** implemented · **Not** a framework  
**Module:** `n2s_lab/n2s_sae_pilot.py`

## Quest

Decompose `v = unit(mean_A − mean_B)` @ L20 into sparse features that may carry the temporal-change vs contradiction effect. Causal test on FOLFOX temporal + true-contra. **No English labels.**

## SAE choice

**Reuse** `chanind/qwen2.5-7B-it-layer-20-saes` (`pile/matryoshka/k-100`, JumpReLU, `blocks.20.hook_resid_post` on `Qwen/Qwen2.5-7B-Instruct`).

Train only if `hook_sanity.pass_heuristic` fails hard.

## Run

Use the **faiss_gpu1** venv (CUDA / HF stack). Do **not** use bare system `python3` for GPU phases.

```bash
cd cxr-evidence-grounding-lab

# preferred wrapper (resolves ../cxrlabs/faiss_gpu1)
./scripts/run_sae_pilot.sh score
./scripts/run_sae_pilot.sh causal --top-k 3 --alpha 8 --notes TX_E01,TX_C01

# or explicit
../cxrlabs/faiss_gpu1/bin/python -m n2s_lab.n2s_sae_pilot score
```

Override interpreter: `CXR_PYTHON=/path/to/python ./scripts/run_sae_pilot.sh …`

Fallback if faiss_gpu1 missing: lab `.venv-phase9` (same torch 2.5.1+cu121 stack).

No `sae_lens` install required — weights load via `safetensors` + JumpReLU encode in `n2s_sae_pilot.py`.

## Artifacts

| File | Content |
|------|---------|
| `artifacts/n2s-sae-pilot-scores.json` | sanity, ranked feat_id / Δ / cos_to_v / r_X / r_margin |
| `artifacts/n2s-sae-pilot-causal.json` | arms table per note |

## Success

A feature or small top-k group that distinguishes A/B, aligns with `v`, tracks X/margin, and moves temporal X without destroying true-contra — with random feature control not mimicking the effect.

Negative result + top-k group fail = stop (do not expand).

## Workbench UI

Same pilot is exposed on **port 8257** as **SAE features** (`POST /api/sae`) after Evaluate.

```bash
cd cxr-n2s-eval-workbench
../cxrlabs/faiss_gpu1/bin/python server.py
# open http://127.0.0.1:8257/ → Evaluate → SAE features
```
