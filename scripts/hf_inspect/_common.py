"""Shared load + HF hooks. Used by look / intervene / process. Not a CLI."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Literal

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

DEFAULT_MODEL = "Qwen/Qwen2.5-7B-Instruct"
SAE_DIR = Path(
    "/home/udonsi-kalu/.cache/huggingface/hub/"
    "models--chanind--qwen2.5-7B-it-layer-20-saes/snapshots/"
    "db8c88b400e36e603d0563abcf8d289e5eff5687/pile/matryoshka/k-100"
)
RETRIEVED = (
    "Retrieved snippet: FOLFOX is oxaliplatin plus 5-FU. "
    "A note that says the course ended and surveillance only means FOLFOX has stopped."
)
PROMPT_A = (
    "Progress note: FOLFOX completed last month. Patient is now on observation. "
    "Has FOLFOX stopped? Answer yes or no."
)
PROMPT_B = (
    "Progress note: FOLFOX completed last month. Patient continues FOLFOX this cycle. "
    "Has FOLFOX stopped? Answer yes or no."
)
PROMPT_TEST = (
    "Progress note: Oxaliplatin/5-FU course ended in March. Surveillance only. "
    "Has FOLFOX stopped? Answer yes or no."
)

Site = Literal["block", "attn", "mlp"]


def unwrap(out: Any) -> torch.Tensor:
    t = out[0] if isinstance(out, tuple) else out
    return t


def device(model: Any) -> torch.device:
    return next(model.parameters()).device


def blocks(model: Any) -> list[Any]:
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        return list(model.model.layers)
    if hasattr(model, "transformer") and hasattr(model.transformer, "h"):
        return list(model.transformer.h)
    raise RuntimeError("Need model.model.layers or model.transformer.h")


def module(block: Any, site: Site) -> Any:
    if site == "block":
        return block
    if site == "attn":
        m = getattr(block, "self_attn", None) or getattr(block, "attn", None)
        if m is None:
            raise RuntimeError("no attn module on this block")
        return m
    m = getattr(block, "mlp", None)
    if m is None:
        raise RuntimeError("no mlp module on this block")
    return m


def load_offline(model_id: str):
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    tok = AutoTokenizer.from_pretrained(model_id, local_files_only=True)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        local_files_only=True,
        torch_dtype=torch.float16,
        device_map="auto",
    )
    model.eval()
    return model, tok


def encode(tok, text: str, dev: torch.device) -> dict[str, torch.Tensor]:
    ids = tok(text, return_tensors="pt")
    return {k: v.to(dev) for k, v in ids.items()}


def full_residual(
    model,
    ids: dict[str, torch.Tensor],
    layer: int,
    site: Site = "block",
) -> torch.Tensor:
    captured: dict[str, torch.Tensor] = {}

    def hook(_m, _inp, out):
        hs = unwrap(out)
        captured["h"] = hs[0].detach().float().cpu()

    handle = module(blocks(model)[layer], site).register_forward_hook(hook)
    try:
        with torch.inference_mode():
            model(**ids)
    finally:
        handle.remove()
    return captured["h"]


def last_residual(
    model,
    ids: dict[str, torch.Tensor],
    layer: int,
    site: Site = "block",
) -> torch.Tensor:
    return full_residual(model, ids, layer, site)[-1]


def greedy(
    model,
    tok,
    ids: dict[str, torch.Tensor],
    max_new: int,
    on_layer: int | None = None,
    site: Site = "block",
    add: torch.Tensor | None = None,
    replace: torch.Tensor | None = None,
    zero_last: bool = False,
    logit_bias: dict[int, float] | None = None,
) -> str:
    handles = []
    if on_layer is not None and (add is not None or replace is not None or zero_last):

        def hook(_m, _inp, out):
            hs = unwrap(out)
            modified = hs.clone()
            if zero_last:
                modified[0, -1, :] = 0
            elif replace is not None:
                modified[0, -1, :] = replace.to(
                    device=modified.device, dtype=modified.dtype
                )
            elif add is not None:
                modified[0, -1, :] = modified[0, -1, :] + add.to(
                    device=modified.device, dtype=modified.dtype
                )
            if isinstance(out, tuple):
                return (modified,) + out[1:]
            return modified

        handles.append(module(blocks(model)[on_layer], site).register_forward_hook(hook))

    generated = ids["input_ids"].clone()
    past = None
    try:
        with torch.inference_mode():
            for _ in range(max_new):
                if past is None:
                    out = model(input_ids=generated, use_cache=True)
                else:
                    out = model(
                        input_ids=generated[:, -1:],
                        past_key_values=past,
                        use_cache=True,
                    )
                past = out.past_key_values
                logits = out.logits[0, -1, :].clone()
                if logit_bias:
                    for tid, val in logit_bias.items():
                        logits[tid] = logits[tid] + val
                next_id = int(logits.argmax())
                generated = torch.cat(
                    [generated, torch.tensor([[next_id]], device=generated.device)],
                    dim=1,
                )
                if next_id == tok.eos_token_id:
                    break
    finally:
        for h in handles:
            h.remove()
    new_ids = generated[0, ids["input_ids"].shape[1] :]
    return tok.decode(new_ids, skip_special_tokens=True)


def token_ids(tok, pieces: tuple[str, ...]) -> list[int]:
    out: list[int] = []
    for p in pieces:
        ids = tok.encode(p, add_special_tokens=False)
        if ids:
            out.append(ids[0])
    return out


def unembed_top(model, h: torch.Tensor, tok, k: int = 5) -> list[tuple[str, float]]:
    dev = device(model)
    x = h.to(device=dev, dtype=next(model.parameters()).dtype)
    x = x.unsqueeze(0).unsqueeze(0)
    with torch.inference_mode():
        x = model.model.norm(x)
        logits = model.lm_head(x)[0, 0].float()
    vals, idxs = torch.topk(logits, k)
    return [(tok.decode([int(i)]), float(v)) for v, i in zip(vals, idxs)]


def top_logits(model, tok, ids: dict[str, torch.Tensor], k: int = 8) -> list[tuple[str, float]]:
    with torch.inference_mode():
        logits = model(**ids).logits[0, -1, :]
    vals, idxs = torch.topk(logits.float(), k)
    return [(tok.decode([int(i)]), float(v)) for v, i in zip(vals, idxs)]


def boot(model_id: str, layer: int):
    print(f"load {model_id} (local only) …")
    model, tok = load_offline(model_id)
    n = len(blocks(model))
    print(f"blocks={n}  hidden={model.config.hidden_size}  using layer {layer}")
    if not 0 <= layer < n:
        raise SystemExit(f"layer must be 0..{n-1}")
    return model, tok
