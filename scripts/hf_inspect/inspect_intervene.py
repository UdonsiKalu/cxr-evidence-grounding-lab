#!/usr/bin/env python3
"""Thin dispatcher. Prefer look.py, intervene.py, or process.py.

Kept so old `--mode` commands still work.
"""

from __future__ import annotations

import argparse

import intervene as intervene_mod
import look as look_mod
import process as process_mod
from _common import DEFAULT_MODEL, boot


def main() -> None:
    look_modes = look_mod.MODES
    intervene_modes = intervene_mod.MODES
    process_modes = process_mod.MODES
    all_modes = (*look_modes, *intervene_modes, *process_modes)

    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--mode", default="difference", choices=[*all_modes, "all"])
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument("--layer", type=int, default=20)
    p.add_argument("--alpha", type=float, default=8.0)
    p.add_argument("--max-new", type=int, default=24)
    p.add_argument("--text", default="The residual stream")
    args = p.parse_args()

    model, tok = boot(args.model, args.layer)
    look_r = look_mod.runners(model, tok, args)
    int_r = intervene_mod.runners(model, tok, args)
    proc_r = process_mod.runners(model, tok, args)

    if args.mode == "all":
        for bucket, run in (
            ("look", look_r),
            ("intervene", int_r),
            ("process", proc_r),
        ):
            for name, fn in run.items():
                print(f"\n===== {bucket}/{name} =====")
                fn()
        return

    if args.mode in look_r:
        look_r[args.mode]()
    elif args.mode in int_r:
        int_r[args.mode]()
    else:
        proc_r[args.mode]()


if __name__ == "__main__":
    main()
