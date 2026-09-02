"""Phase-9C: causal intervention at repair contradiction.present commit (7B)."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Any

from .ground import ground
from .hf_client import unload_model
from .hf_intervene import InterventionKind, generate_intervened
from .hf_trace import find_present_commit_step
from .neural import EXTRACT_SCHEMA, EXTRACT_SYSTEM, REPAIR_EXTRACT_SUFFIX
from .ollama_client import parse_json_object
from .paths import ARTIFACTS_DIR, HELDOUT_BMTCART_PHASE7_PATH
from .phase9a_behavioral import BC_E1_ID, _analyze_hf, _extract_hf
from .phase9b_localization import _repair_user_prompt
from .predicate import evaluate_rule
from .types import Extraction
from .verify import verify_formalization

FAIL_MODEL = "Qwen/Qwen2.5-7B-Instruct"
INTERVENTIONS: tuple[tuple[InterventionKind, float], ...] = (
    ("logit_bias_false", 4.0),
    ("force_false_at_commit", 0.0),
)

# Controls: true contradiction + clear non-contradiction course
CONTROL_CASES = ("BC_C1", "BC_E2")


def _load_case(case_id: str) -> dict[str, Any]:
    data = json.loads(HELDOUT_BMTCART_PHASE7_PATH.read_text(encoding="utf-8"))
    for case in data["cases"]:
        if case["id"] == case_id:
            return case
    raise KeyError(case_id)


def _repair_with_intervention(
    case: dict[str, Any],
    *,
    model_id: str,
    intervention: InterventionKind,
    logit_bias: float = 4.0,
) -> dict[str, Any]:
    evidence = case["evidence"]
    gold = case["expected"]
    case_id = case["id"]

    raw = _extract_hf(evidence, model_id=model_id)
    analysis = _analyze_hf(evidence, model_id=model_id)
    v1 = verify_formalization(
        analysis=analysis,
        extraction=raw,
        grounding=ground(raw).to_dict(),
        case_id=case_id,
        gold=gold,
    )

    user = _repair_user_prompt(evidence, raw, v1.reasons)
    trace, intervened_steps = generate_intervened(
        system=EXTRACT_SYSTEM,
        user=user,
        model_id=model_id,
        max_new_tokens=400,
        intervention=intervention,
        logit_bias=logit_bias,
    )

    gen_only = "".join(s.token_text for s in trace.steps)
    try:
        data = parse_json_object(gen_only)
        ex = Extraction.from_dict(data)
        final_x = bool(ex.contradiction_present)
        parse_ok = True
    except ValueError:
        ex = None
        final_x = None
        parse_ok = False

    g = ground(ex) if ex else None
    rule = evaluate_rule(g) if g else None
    commit_step, commit_val = find_present_commit_step(trace.steps)

    return {
        "case_id": case_id,
        "gold": gold,
        "model_id": model_id,
        "intervention": intervention,
        "logit_bias": logit_bias if intervention == "logit_bias_false" else None,
        "raw_x": bool(raw.contradiction_present),
        "verify_ok": v1.ok,
        "repair_final_x": final_x,
        "parse_ok": parse_ok,
        "commit_step": commit_step,
        "commit_value": commit_val,
        "intervened_steps": intervened_steps,
        "verdict": rule.verdict.value if rule else None,
        "disposition": "AUTO" if rule else None,
    }


def _gate_9c(bc_e1_block: dict[str, Any], controls: dict[str, Any]) -> dict[str, Any]:
    flipped = bc_e1_block.get("x_flipped") is True
    verdict_fixed = bc_e1_block.get("verdict_baseline") == "CONTRADICTION" and bc_e1_block.get(
        "verdict_intervened"
    ) in ("SATISFIED", "REVIEW")
    c1_ok = controls.get("BC_C1", {}).get("x_unchanged_or_correct") is True
    e2_ok = controls.get("BC_E2", {}).get("x_unchanged_or_correct") is True
    yes = flipped and verdict_fixed and c1_ok and e2_ok
    return {
        "pass": yes,
        "x_flipped_on_BC_E1": flipped,
        "verdict_fixed": verdict_fixed,
        "control_BC_C1_ok": c1_ok,
        "control_BC_E2_ok": e2_ok,
        "note": (
            "9C YES — soft causal claim on BC_E1 repair commit"
            if yes
            else "9C NO — intervention did not pass all gates"
        ),
    }


def run_phase9c(*, model_id: str = FAIL_MODEL) -> dict[str, Any]:
    bc_e1 = _load_case(BC_E1_ID)
    baseline = _repair_with_intervention(
        bc_e1, model_id=model_id, intervention="none", logit_bias=0.0
    )

    best: dict[str, Any] | None = None
    for kind, bias in INTERVENTIONS:
        if kind == "none":
            continue
        print(f"\n=== 9C · BC_E1 · {kind} bias={bias} ===")
        row = _repair_with_intervention(
            bc_e1, model_id=model_id, intervention=kind, logit_bias=bias
        )
        if row["repair_final_x"] is False and best is None:
            best = row
        unload_model()

    if best is None:
        # retry stronger bias
        print("\n=== 9C · BC_E1 · logit_bias_false bias=8.0 ===")
        best = _repair_with_intervention(
            bc_e1, model_id=model_id, intervention="logit_bias_false", logit_bias=8.0
        )
        unload_model()

    controls_out: dict[str, Any] = {}
    intervention = best["intervention"] if best else "logit_bias_false"
    bias = best.get("logit_bias") or 4.0

    for cid in CONTROL_CASES:
        print(f"\n=== 9C · control {cid} · {intervention} ===")
        case = _load_case(cid)
        base = _repair_with_intervention(
            case,
            model_id=model_id,
            intervention="none",
        )
        iv = _repair_with_intervention(
            case,
            model_id=model_id,
            intervention=intervention,
            logit_bias=bias if intervention == "logit_bias_false" else 4.0,
        )
        unload_model()
        # C1 should keep X=true; E2 should keep X=false
        if cid == "BC_C1":
            ok = iv["repair_final_x"] is True
        else:
            ok = iv["repair_final_x"] is False
        controls_out[cid] = {
            "baseline_x": base["repair_final_x"],
            "intervened_x": iv["repair_final_x"],
            "baseline_verdict": base["verdict"],
            "intervened_verdict": iv["verdict"],
            "x_unchanged_or_correct": ok,
        }

    unload_model()

    bc_e1_block = {
        "baseline": baseline,
        "intervened": best,
        "x_flipped": (
            baseline.get("repair_final_x") is True
            and best is not None
            and best.get("repair_final_x") is False
        ),
        "verdict_baseline": baseline.get("verdict"),
        "verdict_intervened": best.get("verdict") if best else None,
    }

    panel = {
        "phase": "9C",
        "protocol": "docs/PHASE9-PROTOCOL.md",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "model_id": model_id,
        "BC_E1": bc_e1_block,
        "controls": controls_out,
        "gate_causal_intervention": _gate_9c(bc_e1_block, controls_out),
        "note": "Logit intervention at repair contradiction.present commit; soft claim n=1.",
    }

    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    out = ARTIFACTS_DIR / "phase9c-bc-e1-panel.json"
    out.write_text(json.dumps(panel, indent=2), encoding="utf-8")
    panel["_artifact"] = str(out)
    return panel
