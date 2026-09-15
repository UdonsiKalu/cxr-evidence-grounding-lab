#!/usr/bin/env python3
"""Bucket 1 — LOOK. Read activations. Do not write the forward.

  capture sites positions layer_sweep difference logits logit_lens sae
"""

from __future__ import annotations

import argparse

import torch

from _common import (
    PROMPT_A,
    PROMPT_B,
    PROMPT_TEST,
    SAE_DIR,
    DEFAULT_MODEL,
    boot,
    blocks,
    device,
    encode,
    full_residual,
    last_residual,
    top_logits,
    unembed_top,
)


def cmd_capture(model, tok, layer: int, text: str) -> None:
    ids = encode(tok, text, device(model))
    ntok = int(ids["input_ids"].shape[1])
    h = last_residual(model, ids, layer)
    print(f"prompt tokens: {ntok}")
    print(f"shape last-token: (1, {ntok}, {h.numel()})  [printing last token only]")
    print(f"layer {layer} last-token L2: {float(h.norm()):.4f}")
    print("first 10:", [round(x, 5) for x in h[:10].tolist()])


def cmd_sites(model, tok, layer: int, text: str) -> None:
    ids = encode(tok, text, device(model))
    for site in ("block", "attn", "mlp"):
        h = last_residual(model, ids, layer, site=site)  # type: ignore[arg-type]
        print(f"L{layer} {site:5s}  dim={h.numel()}  L2={float(h.norm()):.4f}")


def cmd_positions(model, tok, layer: int, text: str) -> None:
    ids = encode(tok, text, device(model))
    seq = full_residual(model, ids, layer)
    pieces = tok.convert_ids_to_tokens(ids["input_ids"][0].tolist())
    print(f"layer {layer}  tokens={len(pieces)}  dim={seq.shape[1]}")
    for i, (tok_s, row) in enumerate(zip(pieces, seq)):
        mark = "  <- last" if i == len(pieces) - 1 else ""
        print(f"  [{i:3d}] L2={float(row.norm()):7.3f}  {tok_s}{mark}")


def cmd_layer_sweep(model, tok, text: str) -> None:
    ids = encode(tok, text, device(model))
    n = len(blocks(model))
    idxs = sorted({0, n // 4, n // 2, (3 * n) // 4, n - 1, 20} & set(range(n)))
    print(f"last-token L2 by layer for {text!r}")
    for i in idxs:
        h = last_residual(model, ids, i)
        print(f"  L{i:2d}  L2={float(h.norm()):7.3f}  dim={h.numel()}")


def cmd_difference(model, tok, layer: int) -> None:
    dev = device(model)
    h_a = last_residual(model, encode(tok, PROMPT_A, dev), layer)
    h_b = last_residual(model, encode(tok, PROMPT_B, dev), layer)
    h_t = last_residual(model, encode(tok, PROMPT_TEST, dev), layer)
    delta = h_a - h_b
    direction = delta / (delta.norm() + 1e-8)

    def score(h):
        return float(torch.dot(h / (h.norm() + 1e-8), direction))

    print(f"layer {layer}  dim {h_a.numel()}")
    print(f"||h_A|| {float(h_a.norm()):.4f}  ||h_B|| {float(h_b.norm()):.4f}")
    print(f"||A-B|| {float(delta.norm()):.4f}")
    print(f"cos to (A-B)  A={score(h_a):+.4f}  B={score(h_b):+.4f}  test={score(h_t):+.4f}")
    print("test closer to A if score > 0, to B if score < 0")


def cmd_logits(model, tok, text: str) -> None:
    ids = encode(tok, text, device(model))
    print("next-token logits (top 8):")
    for tok_s, val in top_logits(model, tok, ids):
        print(f"  {val:+8.3f}  {tok_s!r}")


def cmd_logit_lens(model, tok, text: str) -> None:
    ids = encode(tok, text, device(model))
    n = len(blocks(model))
    idxs = sorted({0, n // 2, 20, n - 1} & set(range(n)))
    print("logit lens: decode last-token residual as if it were final")
    for i in idxs:
        h = last_residual(model, ids, i)
        tops = unembed_top(model, h, tok)
        bits = ", ".join(f"{t!r}:{v:.2f}" for t, v in tops)
        print(f"  L{i:2d}  {bits}")


def cmd_sae(model, tok, layer: int) -> None:
    if layer != 20:
        print("note: Chanin SAE was trained on L20; encoding another layer is off-label")
    w_path = SAE_DIR / "sae_weights.safetensors"
    if not w_path.is_file():
        print("SAE weights not on disk:", w_path)
        return
    from safetensors import safe_open

    with safe_open(str(w_path), framework="pt", device="cpu") as f:
        W_enc = f.get_tensor("W_enc").float()
        b_enc = f.get_tensor("b_enc").float()
        b_dec = f.get_tensor("b_dec").float()
        thresh = f.get_tensor("threshold").float()
    h = last_residual(model, encode(tok, PROMPT_TEST, device(model)), layer)
    pre = (h - b_dec) @ W_enc + b_enc
    feats = pre * (pre > thresh)
    nz = int((feats > 0).sum())
    topv, topi = torch.topk(feats, 8)
    print(f"SAE d_in={h.numel()} d_sae={feats.numel()} nonzero={nz}")
    print("top feature ids:", [(int(i), round(float(v), 4)) for v, i in zip(topv, topi)])


MODES = (
    "capture",
    "sites",
    "positions",
    "layer_sweep",
    "difference",
    "logits",
    "logit_lens",
    "sae",
)


def runners(model, tok, args):
    return {
        "capture": lambda: cmd_capture(model, tok, args.layer, args.text),
        "sites": lambda: cmd_sites(model, tok, args.layer, args.text),
        "positions": lambda: cmd_positions(model, tok, args.layer, args.text),
        "layer_sweep": lambda: cmd_layer_sweep(model, tok, args.text),
        "difference": lambda: cmd_difference(model, tok, args.layer),
        "logits": lambda: cmd_logits(model, tok, PROMPT_TEST),
        "logit_lens": lambda: cmd_logit_lens(model, tok, PROMPT_TEST),
        "sae": lambda: cmd_sae(model, tok, args.layer),
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--mode", default="difference", choices=[*MODES, "all"])
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument("--layer", type=int, default=20)
    p.add_argument("--text", default="The residual stream")
    args = p.parse_args()
    model, tok = boot(args.model, args.layer)
    run = runners(model, tok, args)
    if args.mode == "all":
        for name, fn in run.items():
            print(f"\n===== look/{name} =====")
            fn()
    else:
        run[args.mode]()


if __name__ == "__main__":
    main()
