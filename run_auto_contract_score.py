#!/usr/bin/env python3
"""Score frozen panels under the Track A AUTO contract (wrong_AUTO / REVIEW / correct_AUTO).

Does not overwrite Phase 1–14 experiment panels. See docs/AUTO-CONTRACT.md.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from n2s_lab.auto_contract import (  # noqa: E402
    load_temporal_family,
    score_panel_or_report,
    score_preset,
    selftest,
    write_score_artifact,
)
from n2s_lab.paths import ARTIFACTS_DIR  # noqa: E402


def _print_summary(payload: dict) -> None:
    print(f"contract: {payload.get('contract')}")
    print(f"preset/source: {payload.get('preset') or payload.get('source')}")
    for block in payload.get("results") or [payload]:
        if block.get("error"):
            print(f"  ERROR {block['error']}")
            continue
        if block.get("kind") == "panel":
            print(f"  panel {block.get('source')}")
            for m in block.get("models") or []:
                if m.get("error"):
                    print(f"    {m.get('model')}: ERROR {m['error']}")
                    continue
                print(f"    model {m.get('model')}")
                for path, pdata in (m.get("paths") or {}).items():
                    s = pdata["summary"]
                    print(
                        f"      {path}: n={s['n']} wrong_AUTO={s['wrong_AUTO']} "
                        f"correct_AUTO={s['correct_AUTO']} REVIEW={s['REVIEW']} "
                        f"safety_among_auto={s['safety_among_auto']} "
                        f"wrong_ids={s['wrong_AUTO_ids']}"
                    )
        elif "paths" in block:
            print(f"  report {block.get('source')} model={block.get('model')}")
            for path, pdata in block["paths"].items():
                s = pdata["summary"]
                print(
                    f"    {path}: n={s['n']} wrong_AUTO={s['wrong_AUTO']} "
                    f"correct_AUTO={s['correct_AUTO']} REVIEW={s['REVIEW']} "
                    f"wrong_ids={s['wrong_AUTO_ids']}"
                )


def main() -> None:
    parser = argparse.ArgumentParser(description="AUTO contract score (Track A)")
    parser.add_argument("--selftest", action="store_true")
    parser.add_argument(
        "--from-artifacts",
        choices=("phase5", "phase6", "phase7"),
        help="score frozen panel preset",
    )
    parser.add_argument(
        "--artifact",
        help="path to a panel or per-model report JSON",
    )
    parser.add_argument(
        "--paths",
        default="Dual_full,D_full,C_full,D_v,C_v",
        help="comma-separated condition paths to score",
    )
    parser.add_argument(
        "--list-family",
        action="store_true",
        help="print temporal failure-family case ids",
    )
    parser.add_argument("--out", default="", help="artifact filename under artifacts/")
    args = parser.parse_args()

    if args.selftest:
        selftest()
        fam = load_temporal_family()
        assert len(fam.get("cases") or []) >= 10
        print(f"temporal family cases: {len(fam['cases'])}")
        return

    if args.list_family:
        fam = load_temporal_family()
        print(f"family={fam.get('family')} n={len(fam['cases'])}")
        for c in fam["cases"]:
            print(f"  {c['id']:12} {c['expected']:16} {c.get('subtype')}")
        return

    paths = tuple(x.strip() for x in args.paths.split(",") if x.strip())

    if args.from_artifacts:
        payload = score_preset(args.from_artifacts, paths=paths)
        out_name = args.out or f"auto-contract-score-{args.from_artifacts}.json"
        out = write_score_artifact(payload, out_name)
        _print_summary(payload)
        print(f"\nwrote {out.relative_to(ROOT)}")
        return

    if args.artifact:
        path = Path(args.artifact)
        if not path.is_file():
            path = ARTIFACTS_DIR / args.artifact
        if not path.is_file():
            raise SystemExit(f"missing artifact: {args.artifact}")
        payload = score_panel_or_report(path, paths=paths)
        payload["contract"] = "docs/AUTO-CONTRACT.md"
        out_name = args.out or f"auto-contract-score-{path.stem}.json"
        out = write_score_artifact(payload, out_name)
        _print_summary({"results": [payload], "contract": payload["contract"], "source": str(path)})
        print(f"\nwrote {out.relative_to(ROOT)}")
        return

    parser.print_help()
    raise SystemExit(2)


if __name__ == "__main__":
    main()
