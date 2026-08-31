#!/usr/bin/env python3
"""Back-to-back BC_E1 baseline+steered under locked HF determinism.

Expect identical margins / X / verdict across two runs (same process).
Usage (Phase-9 venv):
  ../cxr-evidence-grounding-lab/.venv-phase9/bin/python run_determinism_check.py
"""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from n2s_lab.determinism import configure_determinism  # noqa: E402
from n2s_lab.hf_client import unload_model  # noqa: E402
from n2s_lab.paths import ARTIFACTS_DIR  # noqa: E402
from n2s_lab.phase10_activation import (  # noqa: E402
    FAIL_MODEL,
    build_steering_vectors,
    _load_case,
    _summarize_repair_run,
)


def _pair(case, model_id, vectors, alpha=4.0):
    configure_determinism()
    base = _summarize_repair_run(case, model_id=model_id, intervention="none")
    unload_model()
    configure_determinism()
    steered = _summarize_repair_run(
        case,
        model_id=model_id,
        intervention="activation_steer",
        vectors_by_layer=vectors,
        alpha=alpha,
    )
    unload_model()
    return {
        "baseline_margin": base.get("logit_margin_true_minus_false"),
        "baseline_x": base.get("repair_final_x"),
        "baseline_verdict": base.get("verdict"),
        "steered_margin": steered.get("logit_margin_true_minus_false"),
        "steered_x": steered.get("repair_final_x"),
        "steered_verdict": steered.get("verdict"),
        "commit_step_b": base.get("commit_step"),
        "commit_step_s": steered.get("commit_step"),
    }


def main() -> int:
    det = configure_determinism()
    print("determinism:", det)
    model_id = FAIL_MODEL
    case = _load_case("BC_E1")
    print("building vectors…")
    meta = build_steering_vectors(model_id=model_id)
    vectors = meta.pop("_vectors_by_layer")
    unload_model()

    print("=== run 1 ===")
    a = _pair(case, model_id, vectors)
    print(a)
    print("=== run 2 ===")
    b = _pair(case, model_id, vectors)
    print(b)

    keys = [
        "baseline_margin",
        "baseline_x",
        "baseline_verdict",
        "steered_margin",
        "steered_x",
        "steered_verdict",
        "commit_step_b",
        "commit_step_s",
    ]
    mismatches = [k for k in keys if a.get(k) != b.get(k)]
    out = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "determinism": det,
        "run_1": a,
        "run_2": b,
        "match": len(mismatches) == 0,
        "mismatches": mismatches,
    }
    path = ARTIFACTS_DIR / "determinism-check-bc-e1.json"
    path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print("wrote", path)
    if mismatches:
        print("FAIL mismatches:", mismatches)
        return 1
    print("PASS — identical back-to-back BC_E1 baseline+steered")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
