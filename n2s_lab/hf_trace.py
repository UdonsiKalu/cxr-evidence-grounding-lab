"""HF forward hooks for Phase-9B activation / logit traces."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch

from .hf_client import load_model
from .determinism import configure_determinism, reseeds_before_generate


@dataclass
class StepTrace:
    token_id: int
    token_text: str
    logits_top5: list[dict[str, Any]]
    logit_true: float | None = None
    logit_false: float | None = None
    layer_hidden: dict[str, list[float]] = field(default_factory=dict)


@dataclass
class GenerationTrace:
    prompt: str
    prompt_token_count: int
    steps: list[StepTrace]
    full_text: str


def _model_layers(model: Any) -> list[Any]:
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return list(model.model.layers)
    if hasattr(model, "transformer") and hasattr(model.transformer, "h"):
        return list(model.transformer.h)
    raise RuntimeError("Unsupported model architecture for layer hooks")


def _layer_count(model: Any) -> int:
    return len(_model_layers(model))


def _fraction_layers(model: Any, fractions: tuple[float, ...]) -> list[int]:
    n = _layer_count(model)
    out: list[int] = []
    for f in fractions:
        idx = min(n - 1, max(0, int(round(f * (n - 1)))))
        if idx not in out:
            out.append(idx)
    return sorted(out)


def _top_logits(logits: torch.Tensor, tokenizer: Any, k: int = 5) -> list[dict[str, Any]]:
    vals, ids = torch.topk(logits, k)
    out = []
    for v, i in zip(vals.tolist(), ids.tolist()):
        out.append({"id": i, "token": tokenizer.decode([i]), "logit": v})
    return out


def _logit_for_ids(logits: torch.Tensor, tokenizer: Any, token_strs: tuple[str, ...]) -> float | None:
    for ts in token_strs:
        ids = tokenizer.encode(ts, add_special_tokens=False)
        if ids:
            return float(logits[ids[0]].item())
    return None


def generate_trace(
    *,
    system: str | None,
    user: str,
    model_id: str,
    max_new_tokens: int = 600,
    layer_fractions: tuple[float, ...] = (0.25, 0.5, 0.75, 1.0),
    stop_on_contradiction_commit: bool = False,
) -> GenerationTrace:
    """Greedy decode with per-step hidden states at fractional depths.

    Uses KV cache so activation grows with new tokens only (full-seq recompute
    OOMs on long repair prompts on 24GB).
    """
    configure_determinism()
    model, tokenizer = load_model(model_id)
    reseeds_before_generate()
    messages: list[dict[str, str]] = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": user})
    prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )

    layer_idxs = _fraction_layers(model, layer_fractions)
    frac_by_idx = {
        idx: f"{f:.2f}" for idx, f in zip(layer_idxs, layer_fractions, strict=False)
    }
    layers = _model_layers(model)
    captured: dict[str, torch.Tensor] = {}

    def _make_hook(layer_idx: int, frac_label: str):
        def hook(_module, _inp, out):
            hs = out[0] if isinstance(out, tuple) else out
            captured[frac_label] = hs[0, -1, :].detach().float().cpu()

        return hook

    handles = [
        layers[i].register_forward_hook(_make_hook(i, frac_by_idx[i])) for i in layer_idxs
    ]
    steps: list[StepTrace] = []

    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(model.device)
    generated = input_ids.clone()
    past_key_values = None

    try:
        with torch.inference_mode():
            for _ in range(max_new_tokens):
                captured.clear()
                if past_key_values is None:
                    out = model(
                        input_ids=generated,
                        use_cache=True,
                        logits_to_keep=1,
                    )
                else:
                    out = model(
                        input_ids=generated[:, -1:],
                        past_key_values=past_key_values,
                        use_cache=True,
                        logits_to_keep=1,
                    )
                past_key_values = out.past_key_values
                logits = out.logits[0, -1, :]
                next_id = int(torch.argmax(logits).item())
                tok_text = tokenizer.decode([next_id])
                step = StepTrace(
                    token_id=next_id,
                    token_text=tok_text,
                    logits_top5=_top_logits(logits, tokenizer),
                    logit_true=_logit_for_ids(logits, tokenizer, (" true", "true")),
                    logit_false=_logit_for_ids(logits, tokenizer, (" false", "false")),
                )
                for frac, vec in captured.items():
                    step.layer_hidden[frac] = vec.tolist()
                steps.append(step)
                generated = torch.cat(
                    [generated, torch.tensor([[next_id]], device=model.device)], dim=1
                )
                if next_id == tokenizer.eos_token_id:
                    break
                if stop_on_contradiction_commit:
                    commit_i, _ = find_present_commit_step(steps)
                    if commit_i is not None:
                        break
    finally:
        for h in handles:
            h.remove()
        del past_key_values
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    full = tokenizer.decode(generated[0], skip_special_tokens=True)
    return GenerationTrace(
        prompt=prompt,
        prompt_token_count=int(input_ids.shape[1]),
        steps=steps,
        full_text=full,
    )


def find_present_commit_step(steps: list[StepTrace]) -> tuple[int | None, str | None]:
    """Find step emitting true/false for contradiction.present (not uncertainty.present)."""
    text = ""
    for i, s in enumerate(steps):
        text += s.token_text
        # Look for contradiction block then present key
        low = text.lower()
        if "contradiction" not in low:
            continue
        # After contradiction key, find present then boolean
        idx = low.rfind("contradiction")
        tail = text[idx:]
        if "present" not in tail.lower():
            continue
        tok = s.token_text.strip().lower()
        if tok in ("true", "false"):
            return i, tok
    return None, None


def find_step_indices(steps: list[StepTrace], *needles: str) -> list[int]:
    """Return step indices whose cumulative text contains any needle (case-insensitive)."""
    text = ""
    hits: list[int] = []
    for i, s in enumerate(steps):
        text += s.token_text
        low = text.lower()
        for n in needles:
            if n.lower() in low:
                hits.append(i)
                break
    return hits
