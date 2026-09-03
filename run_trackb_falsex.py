#!/usr/bin/env python3
"""Track B: false-X cluster ladder on temporal-family-dev (HF 7B).

See docs/TRACKB-FALSEX-CLUSTER.md. Does not touch temporal-family-test evidence.

Use .venv-phase9/bin/python (bare python3 lacks accelerate).
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
        help="depth-restricted patch by fraction, e.g. 0.75",
    )
    parser.add_argument(
        "--patch-depth-sweep",
        action="store_true",
        help="absolute-layer patch sweep + specificity controls (ChatGPT order step 1)",
    )
    parser.add_argument(
        "--bce1-alpha-sweep",
        action="store_true",
        help="α-sweep frozen Ph10 BC_E1 vector on cluster (step 2, before refit)",
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
        from n2s_lab.trackb_patch_depth import DEFAULT_SWEEP_LAYERS  # noqa: F401
        from n2s_lab.trackb_bce1_alpha import SWEEP_ALPHAS  # noqa: F401

        print(
            f"trackb selftest OK — targets={list(TARGETS)} "
            f"contra={list(CONTRA_CONTROLS)} nofail={list(NOFAIL_CONTROLS)} "
            f"sweep_L={list(DEFAULT_SWEEP_LAYERS)} bce1_alphas={list(SWEEP_ALPHAS)} "
            f"test_sealed={TEMPORAL_FAMILY_TEST_PATH.name}"
        )
        return

    if args.patch_depth_sweep:
        from n2s_lab.trackb_patch_depth import run_patch_depth_sweep  # noqa: E402

        panel = run_patch_depth_sweep()
        gate = panel["gate_patch_depth_sweep"]
        print(f"\nwrote {panel['_artifact']}")
        print(f"depth gate: {'YES' if gate['pass'] else 'NO'} — {gate['note']}")
        print(f"  earliest={gate.get('earliest_sufficient_layer')} "
              f"specificity={gate.get('specificity_soft_pass')}")
        for key, block in panel["by_layer"].items():
            print(
                f"  {key}: flips={block['n_flips']}/3 "
                f"margins={block['patched_margins']} taut={block['margins_equal_donor_tautology']}"
            )
        if not gate["pass"]:
            sys.exit(2)
        return

    if args.bce1_alpha_sweep:
        from n2s_lab.trackb_bce1_alpha import run_bce1_alpha_sweep  # noqa: E402

        panel = run_bce1_alpha_sweep()
        gate = panel["gate_bce1_alpha"]
        print(f"\nwrote {panel['_artifact']}")
        print(f"BC_E1-α: {gate['note']}")
        print(f"  any_flip={gate['any_target_flip']} look={gate['transfer_failure_looks']}")
        for a, block in panel["by_alpha"].items():
            if a == "0":
                continue
            print(
                f"  α={a}: flips={block['n_target_flips']}/3 "
                f"mean_Δmargin={block.get('mean_target_margin_delta')} "
                f"contra_ok={block.get('contra_stay_true')} "
                f"nofail_ok={block.get('nofail_BC_E2_stay_false')}"
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
