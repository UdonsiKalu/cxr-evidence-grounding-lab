#!/usr/bin/env python3
"""Run the Milestone 1 experiment (mock or live) and print a comparison table."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from n2s_lab.experiment import format_table, run_experiment  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="N2S evidence-grounding lab — Milestone 1")
    parser.add_argument("--mode", choices=("mock", "live"), default="mock")
    parser.add_argument("--ids", default=None, help="comma-separated case ids (default: all 20)")
    parser.add_argument("--out", default=None, help="filename under artifacts/")
    args = parser.parse_args()
    ids = [x.strip() for x in args.ids.split(",")] if args.ids else None
    report = run_experiment(mode=args.mode, ids=ids, out_name=args.out)
    print(format_table(report))
    artifact = report.get("_artifact") or f"artifacts/{args.mode}-latest.json"
    print(f"\nwrote {artifact}")


if __name__ == "__main__":
    main()
