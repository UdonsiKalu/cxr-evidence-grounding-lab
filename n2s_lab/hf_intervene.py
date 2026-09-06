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
    "prefill_position_steer",
    "prefill_component_zero",
    "prefill_component_ablate",
    "commit_component",
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


@dataclass
class PrefillPositionSteerSpec:
    """Add alpha·v at absolute prompt token positions during the first (prefill) forward."""

    positions: list[int]
    vectors_by_layer: dict[int, torch.Tensor]
    alpha: float = 1.0


@dataclass
class PrefillComponentZeroSpec:
    """Zero attn or mlp (or block residual) outputs at positions on the prefill forward.

    Used by upstream U-B: which component write at U-A sites matters for commit X.
    """

    layer: int
    positions: list[int]
    component: Literal["attn", "mlp", "resid"]


@dataclass
class PrefillComponentAblateSpec:
    """Zero or mean-ablate attn/mlp/resid at positions on the prefill forward (U-B / U-B2)."""

    layer: int
    positions: list[int]
    component: Literal["attn", "mlp", "resid"]
    mode: Literal["zero", "mean"] = "zero"
    note_token_start: int | None = None
    note_token_end: int | None = None


@dataclass
class CommitComponentSpec:
    """At contradiction.present commit, zero or replace last-token attn/mlp/resid write.

    Used by circuit C1 (e.g. L20 MLP causal patch).
    """

    layer: int
    component: Literal["attn", "mlp", "resid"]
    mode: Literal["zero", "replace"] = "zero"
    fill: torch.Tensor | None = None


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
    prefill_steer: PrefillPositionSteerSpec | None = None,
    prefill_component_zero: PrefillComponentZeroSpec | None = None,
    prefill_component_ablate: PrefillComponentAblateSpec | None = None,
    commit_component: CommitComponentSpec | None = None,
    commit_components: list[CommitComponentSpec] | None = None,
) -> tuple[GenerationTrace, list[int]]:
    """Greedy decode; intervene at contradiction.present commit and/or prefill positions.

    If ``layer_indices`` is set, hook those absolute block indices (labels ``L{idx}``)
    instead of fractional depths — preferred for patch-depth writeups.

    ``prefill_steer`` applies on the first forward (full prompt) at absolute token
    positions — upstream U1. Commit steers still use last-token only when active.

    ``prefill_component_zero`` zeros attn/mlp/resid writes at positions on prefill — U-B.
    ``prefill_component_ablate`` zero or mean-ablates component writes — U-B / U-B2.
    ``commit_component`` / ``commit_components`` zero/replace attn/mlp/resid last-token
    write(s) at commit — circuit C1 / C2 path restrict.
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

    commit_specs: list[CommitComponentSpec] = []
    if commit_components:
        commit_specs.extend(commit_components)
    elif commit_component is not None:
        commit_specs.append(commit_component)

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

    def _component_module(layer_idx: int, component: str):
        block = layers[layer_idx]
        if component == "attn":
            return getattr(block, "self_attn", None)
        if component == "mlp":
            return getattr(block, "mlp", None)
        return block

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

    def _make_prefill_pos_hook(layer_idx: int, vec: torch.Tensor, positions: list[int], alpha: float):
        pos_set = sorted({int(p) for p in positions if p >= 0})
        need_len = (max(pos_set) + 1) if pos_set else 0

        def hook(_module, _inp, out):
            hs = out[0] if isinstance(out, tuple) else out
            # Only the initial full-prompt forward has these absolute positions.
            if hs.shape[1] < need_len:
                return
            modified = hs.clone()
            v = vec.to(device=modified.device, dtype=modified.dtype)
            for pos in pos_set:
                modified[0, pos, :] = modified[0, pos, :] + alpha * v
            if isinstance(out, tuple):
                return (modified,) + out[1:]
            return modified

        return hook

    def _make_prefill_component_fill_hook(
        positions: list[int],
        fill: torch.Tensor | None,
    ):
        """Replace position rows with ``fill`` (or zeros if fill is None)."""
        pos_set = sorted({int(p) for p in positions if p >= 0})
        need_len = (max(pos_set) + 1) if pos_set else 0

        def hook(_module, _inp, out):
            hs = out[0] if isinstance(out, tuple) else out
            if hs.dim() < 3 or hs.shape[1] < need_len:
                return
            modified = hs.clone()
            for pos in pos_set:
                if pos < modified.shape[1]:
                    if fill is None:
                        modified[0, pos, :] = 0
                    else:
                        modified[0, pos, :] = fill.to(
                            device=modified.device, dtype=modified.dtype
                        )
            if isinstance(out, tuple):
                return (modified,) + out[1:]
            return modified

        return hook

    def _make_commit_component_hook(fill: torch.Tensor | None):
        """Zero or replace last-token component write when steer_active."""

        def hook(_module, _inp, out):
            if not steer_active["on"]:
                return
            hs = out[0] if isinstance(out, tuple) else out
            modified = hs.clone()
            if fill is None:
                modified[0, -1, :] = 0
            else:
                modified[0, -1, :] = fill.to(
                    device=modified.device, dtype=modified.dtype
                )
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
    if intervention == "commit_component" and commit_specs:
        for spec in commit_specs:
            L = int(spec.layer)
            if not (0 <= L < len(layers)):
                raise ValueError(f"commit_component layer {L} out of range")
            target = _component_module(L, spec.component)
            if target is None:
                raise RuntimeError(f"block L{L} has no component {spec.component}")
            fill = None if spec.mode == "zero" else spec.fill
            if spec.mode == "replace" and fill is None:
                raise ValueError("commit_component replace mode requires fill tensor")
            steer_handles.append(
                target.register_forward_hook(_make_commit_component_hook(fill))
            )
    if intervention == "prefill_position_steer" and prefill_steer is not None:
        for layer_idx, vec in prefill_steer.vectors_by_layer.items():
            if 0 <= layer_idx < len(layers):
                steer_handles.append(
                    layers[layer_idx].register_forward_hook(
                        _make_prefill_pos_hook(
                            layer_idx,
                            vec,
                            prefill_steer.positions,
                            float(prefill_steer.alpha),
                        )
                    )
                )
    if intervention == "prefill_component_zero" and prefill_component_zero is not None:
        L = int(prefill_component_zero.layer)
        if 0 <= L < len(layers):
            target = _component_module(L, prefill_component_zero.component)
            if target is None:
                raise ValueError(
                    f"block L{L} has no component {prefill_component_zero.component}"
                )
            steer_handles.append(
                target.register_forward_hook(
                    _make_prefill_component_fill_hook(
                        prefill_component_zero.positions, fill=None
                    )
                )
            )

    # input_ids needed before mean-capture for ablate
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(model.device)

    if intervention == "prefill_component_ablate" and prefill_component_ablate is not None:
        L = int(prefill_component_ablate.layer)
        if not (0 <= L < len(layers)):
            raise ValueError(f"invalid ablate layer {L}")
        target = _component_module(L, prefill_component_ablate.component)
        if target is None:
            raise ValueError(
                f"block L{L} has no component {prefill_component_ablate.component}"
            )
        fill_vec: torch.Tensor | None = None
        if prefill_component_ablate.mode == "mean":
            t0 = prefill_component_ablate.note_token_start
            t1 = prefill_component_ablate.note_token_end
            if t0 is None or t1 is None or t1 <= t0:
                raise ValueError("mean ablate requires note_token_start/end")
            bucket: dict[str, torch.Tensor] = {}

            def _capture_mean(_m, _inp, out):
                hs = out[0] if isinstance(out, tuple) else out
                # full prefill only
                if hs.dim() >= 3 and hs.shape[1] >= t1:
                    bucket["mean"] = hs[0, t0:t1, :].detach().mean(dim=0).float().cpu()

            cap_h = target.register_forward_hook(_capture_mean)
            try:
                with torch.inference_mode():
                    model(input_ids=input_ids, use_cache=False)
            finally:
                cap_h.remove()
            if "mean" not in bucket:
                raise RuntimeError("failed to capture component mean over note body")
            fill_vec = bucket["mean"]
        steer_handles.append(
            target.register_forward_hook(
                _make_prefill_component_fill_hook(
                    prefill_component_ablate.positions, fill=fill_vec
                )
            )
        )

    steps: list[StepTrace] = []
    intervened_steps: list[int] = []

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
                    or (
                        intervention == "commit_component"
                        and bool(commit_specs)
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
