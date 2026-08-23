# Optional Phase-1 — transition diagnostic (removable)

**Status:** optional research experiment. Does **not** replace Milestones 1–3.

## Purpose (plain language)

M2/M3 showed that putting *contradiction* and *uncertainty* into the structured
form helps the symbolic rule. Phase-1 asks a sharper question:

> When something goes wrong, **where** did the important meaning disappear?

We look at four places:

1. The model never notices the issue in its written analysis  
2. The analysis notices it, but the final yes/no-style answer ignores it  
3. The analysis notices it, but the structured form drops it  
4. The structured form is right, but the fixed rule mishandles it  

## How to run

```bash
cd /home/udonsi-kalu/staging/cxr-evidence-grounding-lab
python3 run_phase1_diagnostic.py
python3 run_phase1_diagnostic.py --model mistral:instruct --out phase2-mistral_instruct.json
```

Default cases: `C1–C4` and `U1–U4`. Writes `artifacts/phase1-transition-c-u.json`.

## Phase-2 — same A–D, model panel (prompts frozen)

```bash
python3 run_phase2_panel.py
# optional: N2S_OLLAMA_TIMEOUT=600 python3 run_phase2_panel.py
```

Default panel: control `llama3:8b-instruct-q4_0` (reuses Phase-1 artifact),
`mistral:instruct`, `qwen2.5-coder:32b`.
Writes `artifacts/phase2-model-panel.json` + per-model `phase2-*.json`.
No per-model prompt tuning. Unseen cases come **after** this panel freezes.

**Live panel (2026-08-17):** match gold A/B/C/D — Llama 1/3/7/4, Mistral 7/3/7/5, Qwen32B 4/7/8/8. D `representation_loss` 3 / 2 / 0. B analysis→answer mismatch still on all three. This n=8 slice cannot decide research-problem vs capability; next is unseen cases.

## B′ ablation (frozen B kept)

```bash
python3 run_bprime_ablation.py
```

Reuses Phase-2 analyses. Writes `artifacts/bprime-ablation.json`. Does **not** overwrite Phase-1/2 artifacts.

**Live (2026-08-17):** B / B′ match — Llama 3/3, Mistral 3/4, Qwen 7/8. Mismatch 3/3, 4/3, 1/0. Mapping prompt did not fix sub-10B contradiction collapse. Do not iterate on these 8 notes.

## Phase-3 — held-out C5–C8 / U5–U8 (frozen A–D)

```bash
N2S_OLLAMA_TIMEOUT=600 python3 run_phase3_heldout.py
```

Writes `artifacts/phase3-heldout-panel.json`. Does **not** overwrite Phase-1/2. Does **not** use B′.

**Live (2026-08-17):** A/B/C/D — Llama 1/4/7/6, Mistral 7/4/7/6, Qwen32B 3/6/8/6. D `representation_loss` 1 / 2 / 1 (Qwen now has D-loss on unseen). B mismatch on all three.

## Phase-4 — held-out C9–C11 / U9–U12 (frozen A–D)

```bash
N2S_OLLAMA_TIMEOUT=600 python3 run_phase4.py
```

Writes `artifacts/phase4-heldout-panel.json`. Does **not** overwrite Phase-1/2/3. Does **not** use B′. No C4/C8 analog.

**Live (2026-08-17):** A/B/C/D — Llama 2/4/6/5, Mistral 7/4/7/4, Qwen32B 2/5/6/6. D `representation_loss` 1 / 1 / 1 (C11, C11, C10). B mismatch on all three. Regex screen ≠ human \(A_{\text{human}}\) headline.

## How to undo (backward compatible)

Leave the main lab alone. To drop this direction:

```bash
rm -f run_phase1_diagnostic.py
rm -f n2s_lab/phase1_diagnostic.py
rm -f artifacts/phase1-*.json
```

Also remove the Phase-1 helpers in `n2s_lab/neural.py` / `chat_text` in
`ollama_client.py` only if you want a full revert; `run_experiment.py` never
depends on them.

## What we checked about Ollama

Structured calls use `format: "json"` plus a schema **in the prompt**, then a
parser. That is **not** true constrained decoding against our fields. Free-text
analysis for Conditions B/D uses plain generation with **no** `format: json`.
