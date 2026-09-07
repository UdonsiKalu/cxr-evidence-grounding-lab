#!/usr/bin/env python3
"""Score / diagnose frozen panels under the Track A AUTO contract.

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
    diagnose_panel_or_report,
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


def _print_diagnose(payload: dict) -> None:
    print(f"diagnose path={payload.get('score_path')} source={payload.get('source')}")
    for m in payload.get("models") or [payload]:
        if m.get("error"):
            print(f"  ERROR {m['error']}")
            continue
        print(
            f"  model={m.get('model')} n={m.get('n_rows')} "
            f"wrong_AUTO={m.get('wrong_AUTO_n')} REVIEW={m.get('REVIEW_n')} "
            f"false_X_like={m.get('false_contradiction_like_n')}"
        )
        for c in m.get("wrong_AUTO") or []:
            print(
                f"    WRONG {c['id']}: gold={c['gold']} → {c['verdict']} "
                f"C={c.get('C_full_verdict')} D={c.get('D_full_verdict')} "
                f"agreed={c.get('paths_agreed')} X_D={c.get('D_contradiction_present')} "
                f"BC_E1_like={c.get('false_contradiction_like_BC_E1')}"
            )
        for c in m.get("REVIEW") or []:
            print(
                f"    REVIEW {c['id']}: gold={c['gold']} "
                f"C={c.get('C_full_verdict')} D={c.get('D_full_verdict')} "
                f"reason={c.get('Dual_reason')}"
            )


def main() -> None:
    parser = argparse.ArgumentParser(description="AUTO contract score (Track A)")
    parser.add_argument("--selftest", action="store_true")
    parser.add_argument(
        "--from-artifacts",
        choices=("phase5", "phase6", "phase7", "temporal-dev", "temporal-test"),
        help="score frozen panel preset",
    )
    parser.add_argument(
        "--artifact",
        help="path to a panel or per-model report JSON",
    )
    parser.add_argument(
        "--diagnose",
        action="store_true",
        help="case-level wrong_AUTO/REVIEW cards (gate/path/X fields)",
    )
    parser.add_argument(
        "--paths",
        default="Dual_full,D_full,C_full,D_v,C_v",
        help="comma-separated condition paths to score",
    )
    parser.add_argument(
        "--list-family",
        action="store_true",
        help="print temporal-family-dev case ids",
    )
    parser.add_argument(
        "--tracka-resim",
        action="store_true",
        help="resim temporal-dev Dual_full after grounding temporal-change fix (no LLM)",
    )
    parser.add_argument(
        "--tracka-resim-test",
        action="store_true",
        help="score re-open: resim sealed temporal-test Dual under current grounding (no LLM, no redesign)",
    )
    parser.add_argument("--out", default="", help="artifact filename under artifacts/")
    args = parser.parse_args()

    if args.selftest:
        selftest()
        fam = load_temporal_family()
        assert len(fam.get("cases") or []) >= 10
        assert fam.get("split") == "development"
        from n2s_lab.ground import selftest_ground  # noqa: E402

        selftest_ground()
        print(f"temporal-family-dev cases: {len(fam['cases'])} split={fam.get('split')}")
        print("ground selftest OK")
        return

    if args.tracka_resim:
        from n2s_lab.tracka_temporal_resim import run_tracka_temporal_resim  # noqa: E402

        run_tracka_temporal_resim()
        return

    if args.tracka_resim_test:
        from n2s_lab.tracka_temporal_resim import (  # noqa: E402
            run_tracka_temporal_test_resim,
        )

        run_tracka_temporal_test_resim()
        return

    if args.list_family:
        fam = load_temporal_family()
        print(
            f"family={fam.get('family')} split={fam.get('split')} n={len(fam['cases'])} "
            "(DEV — not held-out)"
        )
        for c in fam["cases"]:
            print(f"  {c['id']:12} {c['expected']:16} {c.get('subtype')}")
        return

    paths = tuple(x.strip() for x in args.paths.split(",") if x.strip())

    if args.diagnose:
        if args.from_artifacts == "temporal-dev":
            path = ARTIFACTS_DIR / "phase7-temporal-dev-panel.json"
        elif args.from_artifacts == "phase7":
            path = ARTIFACTS_DIR / "phase7-bmtcart-panel.json"
        elif args.artifact:
            path = Path(args.artifact)
            if not path.is_file():
                path = ARTIFACTS_DIR / args.artifact
        else:
            raise SystemExit("--diagnose needs --from-artifacts temporal-dev|phase7 or --artifact")
        if not path.is_file():
            raise SystemExit(f"missing panel for diagnose: {path}")
        payload = diagnose_panel_or_report(path, score_path="Dual_full")
        out_name = args.out or f"auto-contract-diagnose-{path.stem}.json"
        out = write_score_artifact(payload, out_name)
        _print_diagnose(payload)
        print(f"\nwrote {out.relative_to(ROOT)}")
        return

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
