#!/usr/bin/env python3
"""Track B: false-X cluster ladder on temporal-family-dev (HF 7B).

See docs/TRACKB-FALSEX-CLUSTER.md. Does not touch temporal-family-test evidence.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from n2s_lab.paths import TEMPORAL_FAMILY_DEV_PATH, TEMPORAL_FAMILY_TEST_PATH  # noqa: E402
from n2s_lab.trackb_falsex_cluster import (  # noqa: E402
    CONTRA_CONTROLS,
    NOFAIL_CONTROLS,
    TARGETS,
    run_patch_layer_subset,
    run_trackb_falsex,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Track B false-X cluster (DEV only)")
    parser.add_argument("--selftest", action="store_true")
    parser.add_argument(
        "--patch-frac",
        default="",
        help="run ONLY the depth-restricted patch arm, e.g. 0.75 (avoids 1.00 token-forcing confound)",
    )
    args = parser.parse_args()

    if args.selftest:
        if not TEMPORAL_FAMILY_DEV_PATH.is_file():
            print("missing temporal-family-dev.json")
            sys.exit(1)
        if not TEMPORAL_FAMILY_TEST_PATH.is_file():
            print("missing temporal-family-test.json (freeze first)")
            sys.exit(1)
        p10 = ROOT / "artifacts" / "phase10-bc-e1-panel.json"
        if not p10.is_file():
            print("missing phase10 panel (need Ph10 vectors)")
            sys.exit(1)
        if not json.loads(p10.read_text()).get("gate_activation_intervention", {}).get(
            "pass"
        ):
            print("phase10 gate not YES")
            sys.exit(1)
        from n2s_lab.hf_intervene import ActivationPatchSpec, ActivationSteerSpec  # noqa: F401

        print(
            f"trackb selftest OK — targets={list(TARGETS)} "
            f"contra={list(CONTRA_CONTROLS)} nofail={list(NOFAIL_CONTROLS)} "
            f"test_sealed={TEMPORAL_FAMILY_TEST_PATH.name}"
        )
        return

    if args.patch_frac:
        fracs = tuple(float(x) for x in args.patch_frac.split(",") if x.strip())
        panel = run_patch_layer_subset(patch_fracs=fracs)
        gate = panel["gate_patch_depth"]
        print(f"\nwrote {panel['_artifact']}")
        print(f"patch-depth gate: {'YES' if gate['pass'] else 'NO'} — {gate['note']}")
        for cid, r in panel["results"].items():
            print(
                f"  {cid}: x {r['baseline_x']} -> {r['patched_x']} | "
                f"margin {r['baseline_margin']} -> {r['patched_margin']} | "
                f"verdict {r['verdict_patched']}"
            )
        if not gate["pass"]:
            sys.exit(2)
        return

    panel = run_trackb_falsex()
    gp = panel["gate_probe"]
    gc = panel["gate_causal"]
    print(f"\nwrote {panel['_artifact']}")
    print(f"probe gate: {'YES' if gp['pass'] else 'NO'} — {gp['note']}")
    print(f"  LOO={gp.get('loo_probe_acc')} overlap={gp.get('cosine_1_00_ranges_overlap')}")
    print(f"causal gate: {'YES' if gc['pass'] else 'NO'} — {gc['note']}")
    print(
        f"  steer_flips={gc['n_steer_flips']} patch_flips={gc['n_patch_flips']} "
        f"baseline_fx={gc['n_targets_baseline_false_x']}"
    )
    if not gp["pass"] and not gc["pass"]:
        sys.exit(2)


if __name__ == "__main__":
    main()
