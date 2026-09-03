"""Greedy generation with optional logit intervention at contradiction.present commit."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Literal

import torch

from .hf_client import load_model
from .determinism import configure_determinism, reseeds_before_generate
from .hf_trace import (
    GenerationTrace,
    StepTrace,
    _fraction_layers,
    _logit_for_ids,
    _model_layers,
    _top_logits,
)

InterventionKind = Literal[
    "none",
    "logit_bias_false",
    "force_false_at_commit",
    "activation_steer",
    "activation_patch",
]


@dataclass
class ActivationSteerSpec:
    """Unit vectors keyed by layer index (not fraction label)."""

    vectors_by_layer: dict[int, torch.Tensor]
    alpha: float = 1.0


@dataclass
class ActivationPatchSpec:
    """Replace last-token hidden at commit with donor activations (by layer idx)."""

    vectors_by_layer: dict[int, torch.Tensor]


def _at_contradiction_present_commit(prefix_text: str) -> bool:
    low = prefix_text.lower()
    if "contradiction" not in low:
        return False
    tail = prefix_text[low.rfind("contradiction") :]
    if "present" not in tail.lower():
        return False
    return bool(re.search(r'present"\s*:\s*$', tail) or re.search(r"present'\s*:\s*$", tail))


def generate_intervened(
    *,
    system: str | None,
    user: str,
    model_id: str,
    max_new_tokens: int = 600,
    intervention: InterventionKind = "none",
    logit_bias: float = 4.0,
    layer_fractions: tuple[float, ...] = (0.25, 0.5, 0.75, 1.0),
    layer_indices: tuple[int, ...] | None = None,
    activation_steer: ActivationSteerSpec | None = None,
    activation_patch: ActivationPatchSpec | None = None,
) -> tuple[GenerationTrace, list[int]]:
    """Greedy decode; intervene only at contradiction.present boolean commit.

    If ``layer_indices`` is set, hook those absolute block indices (labels ``L{idx}``)
    instead of fractional depths — preferred for patch-depth writeups.
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

    layers = _model_layers(model)
    n_layers = len(layers)
    if layer_indices is not None:
        layer_idxs = sorted({i for i in layer_indices if 0 <= i < n_layers})
        if not layer_idxs:
            raise ValueError(f"no valid layer_indices in 0..{n_layers - 1}")
        frac_by_idx = {idx: f"L{idx}" for idx in layer_idxs}
    else:
        layer_idxs = _fraction_layers(model, layer_fractions)
        frac_by_idx = {
            idx: f"{f:.2f}" for idx, f in zip(layer_idxs, layer_fractions, strict=False)
        }
    captured: dict[str, torch.Tensor] = {}

    def _make_hook(layer_idx: int, frac_label: str):
        def hook(_module, _inp, out):
            hs = out[0] if isinstance(out, tuple) else out
            captured[frac_label] = hs[0, -1, :].detach().float().cpu()

        return hook

    handles = [
        layers[i].register_forward_hook(_make_hook(i, frac_by_idx[i])) for i in layer_idxs
    ]
    steer_handles: list[Any] = []
    steer_active = {"on": False}

    def _make_steer_hook(layer_idx: int, vec: torch.Tensor):
        def hook(_module, _inp, out):
            if not steer_active["on"]:
                return
            hs = out[0] if isinstance(out, tuple) else out
            modified = hs.clone()
            v = vec.to(device=modified.device, dtype=modified.dtype)
            modified[0, -1, :] = modified[0, -1, :] + (
                (activation_steer.alpha if activation_steer else 0.0) * v
            )
            if isinstance(out, tuple):
                return (modified,) + out[1:]
            return modified

        return hook

    def _make_patch_hook(layer_idx: int, vec: torch.Tensor):
        def hook(_module, _inp, out):
            if not steer_active["on"]:
                return
            hs = out[0] if isinstance(out, tuple) else out
            modified = hs.clone()
            v = vec.to(device=modified.device, dtype=modified.dtype)
            modified[0, -1, :] = v
            if isinstance(out, tuple):
                return (modified,) + out[1:]
            return modified

        return hook

    if intervention == "activation_steer" and activation_steer is not None:
        for layer_idx, vec in activation_steer.vectors_by_layer.items():
            if 0 <= layer_idx < len(layers):
                steer_handles.append(
                    layers[layer_idx].register_forward_hook(_make_steer_hook(layer_idx, vec))
                )
    if intervention == "activation_patch" and activation_patch is not None:
        for layer_idx, vec in activation_patch.vectors_by_layer.items():
            if 0 <= layer_idx < len(layers):
                steer_handles.append(
                    layers[layer_idx].register_forward_hook(_make_patch_hook(layer_idx, vec))
                )

    steps: list[StepTrace] = []
    intervened_steps: list[int] = []

    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(model.device)
    generated = input_ids.clone()
    gen_prefix = ""
    past_key_values = None

    false_ids = tokenizer.encode(" false", add_special_tokens=False) or tokenizer.encode(
        "false", add_special_tokens=False
    )
    true_ids = tokenizer.encode(" true", add_special_tokens=False) or tokenizer.encode(
        "true", add_special_tokens=False
    )

    try:
        with torch.inference_mode():
            for step_i in range(max_new_tokens):
                captured.clear()
                at_commit = _at_contradiction_present_commit(gen_prefix)
                use_activation = at_commit and (
                    (
                        intervention == "activation_steer"
                        and activation_steer is not None
                    )
                    or (
                        intervention == "activation_patch"
                        and activation_patch is not None
                    )
                )
                steer_active["on"] = use_activation

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
                steer_active["on"] = False

                intervened = use_activation
                if at_commit and intervention == "force_false_at_commit" and false_ids:
                    next_id = false_ids[0]
                    intervened = True
                elif at_commit and intervention == "logit_bias_false":
                    modified = logits.clone()
                    for tid in false_ids:
                        modified[tid] = modified[tid] + logit_bias
                    for tid in true_ids:
                        modified[tid] = modified[tid] - logit_bias
                    next_id = int(torch.argmax(modified).item())
                    intervened = True
                else:
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
                if intervened:
                    intervened_steps.append(step_i)
                gen_prefix += tok_text
                generated = torch.cat(
                    [generated, torch.tensor([[next_id]], device=model.device)], dim=1
                )
                if next_id == tokenizer.eos_token_id:
                    break
    finally:
        for h in handles:
            h.remove()
        for h in steer_handles:
            h.remove()
        del past_key_values
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    full = tokenizer.decode(generated[0], skip_special_tokens=True)
    trace = GenerationTrace(
        prompt=prompt,
        prompt_token_count=int(input_ids.shape[1]),
        steps=steps,
        full_text=full,
    )
    return trace, intervened_steps
