#!/usr/bin/env python3
"""Bucket 2 — INTERVENE. Write the residual or logits. Qwen weights stay frozen.

  steer patch mlp_zero attn_zero resid_zero logit_bias
"""

from __future__ import annotations

import argparse
from typing import Literal

from _common import (
    DEFAULT_MODEL,
    PROMPT_A,
    PROMPT_B,
    PROMPT_TEST,
    boot,
    device,
    encode,
    greedy,
    last_residual,
    token_ids,
)


def cmd_steer(model, tok, layer: int, alpha: float, max_new: int) -> None:
    dev = device(model)
    h_a = last_residual(model, encode(tok, PROMPT_A, dev), layer)
    h_b = last_residual(model, encode(tok, PROMPT_B, dev), layer)
    direction = (h_a - h_b) / ((h_a - h_b).norm() + 1e-8)
    ids = encode(tok, PROMPT_TEST, dev)
    base = greedy(model, tok, ids, max_new)
    steered = greedy(model, tok, ids, max_new, on_layer=layer, add=alpha * direction)
    print(f"steer L{layer} alpha={alpha}")
    print("BASE   ", repr(base))
    print("STEER  ", repr(steered))


def cmd_patch(model, tok, layer: int, max_new: int) -> None:
    dev = device(model)
    donor = last_residual(model, encode(tok, PROMPT_A, dev), layer)
    ids = encode(tok, PROMPT_B, dev)
    base = greedy(model, tok, ids, max_new)
    patched = greedy(model, tok, ids, max_new, on_layer=layer, replace=donor)
    print(f"patch L{layer}: write A's last-token vector into B's forward")
    print("BASE B ", repr(base))
    print("PATCH  ", repr(patched))


def cmd_zero(
    model, tok, layer: int, max_new: int, site: Literal["block", "attn", "mlp"]
) -> None:
    ids = encode(tok, PROMPT_TEST, device(model))
    base = greedy(model, tok, ids, max_new)
    zapped = greedy(
        model, tok, ids, max_new, on_layer=layer, site=site, zero_last=True
    )
    print(f"zero L{layer} {site} last-token write")
    print("BASE   ", repr(base))
    print("ZERO   ", repr(zapped))


def cmd_logit_bias(model, tok, max_new: int) -> None:
    ids = encode(tok, PROMPT_TEST, device(model))
    yes_ids = token_ids(tok, (" Yes", "Yes", " yes"))
    no_ids = token_ids(tok, (" No", "No", " no"))
    bias = {i: 4.0 for i in yes_ids} | {i: -4.0 for i in no_ids}
    print("BASE ", repr(greedy(model, tok, ids, max_new)))
    print("BIAS yes+4 no-4 ", repr(greedy(model, tok, ids, max_new, logit_bias=bias)))


MODES = (
    "steer",
    "patch",
    "mlp_zero",
    "attn_zero",
    "resid_zero",
    "logit_bias",
)


def runners(model, tok, args):
    return {
        "steer": lambda: cmd_steer(model, tok, args.layer, args.alpha, args.max_new),
        "patch": lambda: cmd_patch(model, tok, args.layer, args.max_new),
        "mlp_zero": lambda: cmd_zero(model, tok, args.layer, args.max_new, "mlp"),
        "attn_zero": lambda: cmd_zero(model, tok, args.layer, args.max_new, "attn"),
        "resid_zero": lambda: cmd_zero(model, tok, args.layer, args.max_new, "block"),
        "logit_bias": lambda: cmd_logit_bias(model, tok, args.max_new),
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--mode", default="steer", choices=[*MODES, "all"])
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument("--layer", type=int, default=20)
    p.add_argument("--alpha", type=float, default=8.0)
    p.add_argument("--max-new", type=int, default=24)
    args = p.parse_args()
    model, tok = boot(args.model, args.layer)
    run = runners(model, tok, args)
    if args.mode == "all":
        for name, fn in run.items():
            print(f"\n===== intervene/{name} =====")
            fn()
    else:
        run[args.mode]()


if __name__ == "__main__":
    main()
