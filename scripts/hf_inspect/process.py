#!/usr/bin/env python3
"""Bucket 3 — PROCESS. Text around the model: scaffold, retrieve, rules, gate.

  scaffold retrieve constrain symbolic verifier tool gate
"""

from __future__ import annotations

import argparse
import json
import re

from _common import (
    DEFAULT_MODEL,
    PROMPT_TEST,
    RETRIEVED,
    boot,
    device,
    encode,
    greedy,
)


def cmd_scaffold(model, tok, max_new: int) -> None:
    bare = PROMPT_TEST
    scaffolded = (
        "You may only answer yes or no. Quote nothing. "
        "If the note says completed/ended/surveillance, answer yes.\n\n"
        + PROMPT_TEST
    )
    dev = device(model)
    print("BARE     ", repr(greedy(model, tok, encode(tok, bare, dev), max_new)))
    print("SCAFFOLD ", repr(greedy(model, tok, encode(tok, scaffolded, dev), max_new)))


def cmd_retrieve(model, tok, max_new: int) -> None:
    dev = device(model)
    bare = greedy(model, tok, encode(tok, PROMPT_TEST, dev), max_new)
    with_hit = greedy(
        model, tok, encode(tok, RETRIEVED + "\n\n" + PROMPT_TEST, dev), max_new
    )
    print("BARE      ", repr(bare))
    print("RETRIEVED ", repr(with_hit))


def cmd_constrain(model, tok, max_new: int) -> None:
    schema = (
        'Reply with JSON only: {"answer":"yes|no","span":"quote from the note"}\n\n'
        + PROMPT_TEST
    )
    out = greedy(model, tok, encode(tok, schema, device(model)), max_new)
    print("OUTPUT ", repr(out))
    try:
        start, end = out.find("{"), out.rfind("}")
        obj = json.loads(out[start : end + 1] if start >= 0 and end > start else out)
        print("PARSE  ok", obj)
    except json.JSONDecodeError as exc:
        print("PARSE  fail", exc)


def cmd_symbolic(model, tok, max_new: int) -> None:
    note = PROMPT_TEST
    out = greedy(model, tok, encode(tok, note, device(model)), max_new)
    stopped_cues = bool(re.search(r"ended|completed|stopped|surveillance", note, re.I))
    says_no = bool(re.search(r"\bno\b", out, re.I))
    flag = stopped_cues and says_no
    print("OUTPUT ", repr(out))
    print(f"note has stop-cue: {stopped_cues}  output has no: {says_no}")
    print("SYMBOLIC GATE:", "REVIEW (output fights the note)" if flag else "pass")


def cmd_verifier(model, tok, max_new: int) -> None:
    note = PROMPT_TEST
    out = greedy(model, tok, encode(tok, note, device(model)), max_new)
    note_l, out_l = note.lower(), out.lower()
    overlap = [w for w in re.findall(r"[a-z0-9]+", out_l) if len(w) > 3 and w in note_l]
    print("OUTPUT ", repr(out))
    print("content words also in note:", overlap[:12])
    print("VERIFIER:", "grounded-ish" if overlap else "no overlap — treat as ungrounded")


def cmd_tool(model, tok, max_new: int) -> None:
    note = PROMPT_TEST
    ended = bool(re.search(r"ended|completed|stopped|surveillance", note, re.I))
    tool_answer = "yes" if ended else "unknown"
    print("TOOL extracted stop-cue:", ended, "=>", tool_answer)
    wrapped = (
        f"A date/med tool already extracted: FOLFOX_stopped={tool_answer}. "
        "Do not contradict the tool. " + note
    )
    print("LLM+TOOL ", repr(greedy(model, tok, encode(tok, wrapped, device(model)), max_new)))


def cmd_gate(model, tok, max_new: int) -> None:
    note = PROMPT_TEST
    out = greedy(model, tok, encode(tok, note, device(model)), max_new)
    stop = bool(re.search(r"ended|completed|surveillance", note, re.I))
    says_no = bool(re.search(r"\bno\b", out, re.I))
    review = stop and says_no
    print("OUTPUT ", repr(out))
    print("POLICY GATE:", "REVIEW (do not release)" if review else "release")


MODES = (
    "scaffold",
    "retrieve",
    "constrain",
    "symbolic",
    "verifier",
    "tool",
    "gate",
)


def runners(model, tok, args):
    return {
        "scaffold": lambda: cmd_scaffold(model, tok, args.max_new),
        "retrieve": lambda: cmd_retrieve(model, tok, args.max_new),
        "constrain": lambda: cmd_constrain(model, tok, args.max_new),
        "symbolic": lambda: cmd_symbolic(model, tok, args.max_new),
        "verifier": lambda: cmd_verifier(model, tok, args.max_new),
        "tool": lambda: cmd_tool(model, tok, args.max_new),
        "gate": lambda: cmd_gate(model, tok, args.max_new),
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--mode", default="scaffold", choices=[*MODES, "all"])
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument("--layer", type=int, default=20)
    p.add_argument("--max-new", type=int, default=24)
    args = p.parse_args()
    model, tok = boot(args.model, args.layer)
    run = runners(model, tok, args)
    if args.mode == "all":
        for name, fn in run.items():
            print(f"\n===== process/{name} =====")
            fn()
    else:
        run[args.mode]()


if __name__ == "__main__":
    main()
