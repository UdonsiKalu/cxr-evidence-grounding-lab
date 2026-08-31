"""HuggingFace chat helpers for Phase-9 (activation-ready). Not used by Phase-1–8 Ollama runners."""

from __future__ import annotations

import gc
import re
from typing import Any

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from .determinism import configure_determinism, reseeds_before_generate
from .ollama_client import parse_json_object

_LOADED: dict[str, Any] = {"model_id": None, "model": None, "tokenizer": None}
_DET_STATUS: dict[str, Any] | None = None


def determinism_status() -> dict[str, Any] | None:
    return _DET_STATUS


def unload_model() -> None:
    """Drop the cached HF model. Callers must also `del` any local model refs."""
    mid = _LOADED.get("model_id")
    if _LOADED.get("model") is not None:
        del _LOADED["model"]
        _LOADED["model"] = None
    if _LOADED.get("tokenizer") is not None:
        del _LOADED["tokenizer"]
        _LOADED["tokenizer"] = None
    _LOADED["model_id"] = None
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
    if mid:
        print(f"[hf] unloaded {mid}")


def load_model(
    model_id: str,
    *,
    load_in_4bit: bool | None = None,
) -> tuple[Any, Any]:
    """Load one HF causal LM; unloads any previously loaded model first."""
    global _DET_STATUS
    # Configure before first CUDA alloc in this process when possible.
    _DET_STATUS = configure_determinism()

    if _LOADED.get("model_id") == model_id and _LOADED.get("model") is not None:
        return _LOADED["model"], _LOADED["tokenizer"]

    unload_model()
    _DET_STATUS = configure_determinism()

    if load_in_4bit is None:
        # 14B+ needs 4-bit on 24GB; 7B fits bf16.
        load_in_4bit = any(x in model_id for x in ("14B", "32B", "72B"))

    print(f"[hf] loading {model_id} (4bit={load_in_4bit}) …")
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    kwargs: dict[str, Any] = {
        "trust_remote_code": True,
    }
    if load_in_4bit:
        # Multi-GPU / offload OK for large quantized models.
        kwargs["device_map"] = "auto"
        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )
    else:
        # Pin 7B fully on cuda:0. device_map="auto" can CPU-offload under
        # fragmentation and then OOM when layers bounce back on long prompts.
        kwargs["device_map"] = {"": 0}
        kwargs["torch_dtype"] = torch.bfloat16

    model = AutoModelForCausalLM.from_pretrained(model_id, **kwargs)
    model.eval()
    _LOADED["model_id"] = model_id
    _LOADED["model"] = model
    _LOADED["tokenizer"] = tokenizer
    if torch.cuda.is_available():
        free, total = torch.cuda.mem_get_info()
        print(f"[hf] loaded; CUDA free={free/1e9:.1f}G / total={total/1e9:.1f}G")
    return model, tokenizer


def _generate(
    *,
    system: str | None,
    user: str,
    model_id: str,
    max_new_tokens: int,
    force_json: bool,
) -> str:
    model, tokenizer = load_model(model_id)
    messages: list[dict[str, str]] = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": user})

    prompt = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    inputs = tokenizer(prompt, return_tensors="pt")
    inputs = {k: v.to(model.device) for k, v in inputs.items()}

    # Explicit greedy only — no temperature / top_p sampling path.
    gen_kwargs: dict[str, Any] = {
        "max_new_tokens": max_new_tokens,
        "do_sample": False,
    }
    reseeds_before_generate()

    with torch.inference_mode():
        out = model.generate(**inputs, **gen_kwargs)
    new_tokens = out[0, inputs["input_ids"].shape[-1] :]
    text = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
    if force_json:
        # Strip markdown fences if present.
        fence = re.search(r"```(?:json)?\s*([\s\S]*?)```", text)
        if fence:
            text = fence.group(1).strip()
    return text


def chat_text(
    prompt: str,
    *,
    system: str | None = None,
    model_id: str,
    max_new_tokens: int = 500,
) -> str:
    return _generate(
        system=system,
        user=prompt,
        model_id=model_id,
        max_new_tokens=max_new_tokens,
        force_json=False,
    )


def chat_json(
    prompt: str,
    *,
    system: str | None = None,
    model_id: str,
    max_new_tokens: int = 600,
) -> dict[str, Any]:
    """Ask for JSON via prompt; parse like Ollama path (not constrained decoding)."""
    user = prompt
    if "JSON" not in (system or "") and "json" not in prompt.lower():
        user = prompt + "\n\nReturn a single JSON object only."
    text = _generate(
        system=system,
        user=user,
        model_id=model_id,
        max_new_tokens=max_new_tokens,
        force_json=True,
    )
    try:
        return parse_json_object(text or "{}")
    except ValueError:
        # One retry with a sharper instruction.
        text2 = _generate(
            system=(system or "") + "\nReturn ONLY valid JSON. No markdown.",
            user=user,
            model_id=model_id,
            max_new_tokens=max_new_tokens,
            force_json=True,
        )
        return parse_json_object(text2 or "{}")
