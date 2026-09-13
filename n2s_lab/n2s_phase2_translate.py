"""Phase 2 Exp1 — Translate-side FOLFOX contrast (boundary only).

Recover 7B analog extract (no hidden-state patch) → dump A–D → apply a
text-faithful failure-polarity relabel → re-ground → rule. Then the same
intervention on CONTRA.

See docs/TRACKB-PHASE2-TRANSLATE-CONTRAST.md.
REVIEW overlay is not a YES. Frozen-d patch stays closed. Dual_full not rescored.
"""

from __future__ import annotations

import argparse
import json
from copy import deepcopy
from datetime import datetime, timezone
from typing import Any

from .ground import ground, _hard_simultaneous_conflict
from .n2s_nn_layer_loss import DEFAULT_NOTES
from .paths import ARTIFACTS_DIR
from .predicate import evaluate_rule
from .types import Extraction, OutcomeStatement, Verdict

PANEL_PATH = ARTIFACTS_DIR / "n2s-phase2-translate-panel.json"
CLEAN_PANEL_PATH = ARTIFACTS_DIR / "n2s-phase2-translate-panel-clean.json"
EXTRACT_PATH = ARTIFACTS_DIR / "n2s-phase2-translate-extracts.json"
EXTRACT_ANALOG_PATH = ARTIFACTS_DIR / "n2s-phase2-translate-extracts-analog.json"
RECONCILE_PATH = ARTIFACTS_DIR / "n2s-phase2-contra-dual-reconcile.json"
PATCH_PANEL = ARTIFACTS_DIR / "n2s-nn-layer-patch-panel.json"
LOCATOR_PANEL = ARTIFACTS_DIR / "n2s-locator-panel.json"

FOLFOX_ID = "EX_TEMPORAL_FOLFOX"
CONTRA_ID = "EX_CONTRA"

FAILURE_MARKERS = (
    "progression",
    "progressed",
    "progressive",
    "treatment failure",
    "discontinued for",
    "refractory",
    "new hepatic",
    "new lesion",
    "failed",
    "failure",
)
RESPONSE_MARKERS = (
    "partial response",
    "response",
    "responsive",
    "responded",
)


def atoms_payload(ex: Extraction) -> dict[str, Any]:
    g = ground(ex)
    rule = evaluate_rule(g)
    return {
        "extract": ex.to_dict(),
        "atoms": {
            "A_first_line_identified": g.first_line_identified.value,
            "B_first_line_administered": g.first_line_administered.value,
            "C_failure_event": g.failure_event.value,
            "D_failure_of_first_line": g.failure_of_first_line.value,
            "X_contradiction": g.contradiction,
        },
        "ground_trace": list(g.trace),
        "verdict": rule.verdict.value,
        "fired_rule": rule.fired_rule,
        "rule_trace": list(rule.trace),
    }


def _has_failure_marker(text: str) -> bool:
    blob = (text or "").lower()
    return any(m in blob for m in FAILURE_MARKERS)


# Local negation of a failure marker. Predeclared from P-neg (denial of
# failure in extract text), not from listing leak note IDs. Window is the
# characters immediately before the marker; "-free" is a suffix form.
NEGATION_PREFIXES = (
    "no ",
    "not ",
    "without ",
    "never ",
    "n't ",
    "denies ",
    "denied ",
    "absent ",
    "no evidence of ",
    "negative for ",
    "lack of ",
    "lacking ",
)
_NEGATION_WINDOW = 28


def _marker_occurrence_negated(blob: str, idx: int, marker: str) -> bool:
    prefix = blob[max(0, idx - _NEGATION_WINDOW) : idx]
    if any(cue in prefix for cue in NEGATION_PREFIXES):
        return True
    after = blob[idx + len(marker) : idx + len(marker) + 6]
    return after.startswith("-free") or after.startswith(" free")


def _has_asserted_failure_marker(text: str) -> bool:
    """True iff a FAILURE_MARKER occurs that is not under local negation."""
    blob = (text or "").lower()
    for marker in FAILURE_MARKERS:
        start = 0
        while True:
            idx = blob.find(marker, start)
            if idx < 0:
                break
            if not _marker_occurrence_negated(blob, idx, marker):
                return True
            start = idx + 1
    return False


def relabel_failure_polarity_from_text(ex: Extraction) -> tuple[Extraction, dict[str, Any]]:
    """Boundary intervention: use failure already written in extract text.

    1. If an outcome's TEXT states failure, set polarity=failure.
    2. If no failure polarity exists, promote a quote/notes span that states failure
       into an outcome_statement (7B analog often parked failure in quotes).

    Does not patch hidden states. Does not flip contradiction.present.
    """
    changed_idx: list[int] = []
    new_outcomes: list[OutcomeStatement] = []
    for i, o in enumerate(ex.outcome_statements):
        pol = (o.polarity or "").lower().strip()
        if pol != "failure" and _has_failure_marker(o.text):
            new_outcomes.append(OutcomeStatement(text=o.text, polarity="failure"))
            changed_idx.append(i)
        else:
            new_outcomes.append(OutcomeStatement(text=o.text, polarity=o.polarity))
    promoted_from: str | None = None
    has_failure_polarity = any(
        (o.polarity or "").lower().strip() == "failure" for o in new_outcomes
    )
    if not has_failure_polarity:
        for q in ex.quotes:
            if _has_failure_marker(q):
                new_outcomes.append(OutcomeStatement(text=q, polarity="failure"))
                promoted_from = "quotes"
                break
        if promoted_from is None and _has_failure_marker(ex.notes):
            new_outcomes.append(OutcomeStatement(text=ex.notes, polarity="failure"))
            promoted_from = "notes"
    cues = list(ex.implicit_cues)
    blob = " ".join(o.text.lower() for o in new_outcomes)
    added_cue = False
    if (
        any(m in blob for m in FAILURE_MARKERS)
        and any(m in blob for m in RESPONSE_MARKERS)
        and "response_then_progression" not in {c.lower() for c in cues}
    ):
        cues.append("response_then_progression")
        added_cue = True
    payload = ex.to_dict()
    payload["outcome_statements"] = [
        {"text": o.text, "polarity": o.polarity} for o in new_outcomes
    ]
    payload["implicit_cues"] = cues
    out = Extraction.from_dict(payload)
    meta = {
        "kind": "failure_evidence_from_extract_text",
        "n_polarities_relabeled": len(changed_idx),
        "relabeled_outcome_idx": changed_idx,
        "promoted_failure_from": promoted_from,
        "added_response_then_progression": added_cue,
        "touched_contradiction_present": False,
        "hidden_state_patched": False,
    }
    return out, meta


def relabel_failure_polarity_from_text_gated(
    ex: Extraction,
) -> tuple[Extraction, dict[str, Any]]:
    """Quote-promote gated on asserted (non-negated) failure markers.

    Same two steps as `relabel_failure_polarity_from_text`, but a span is
    failure evidence only if `_has_asserted_failure_marker` is true.
    Does not mutate the ungated function. Does not patch hidden states.
    Does not flip contradiction.present. Does not change ground().
    """
    changed_idx: list[int] = []
    new_outcomes: list[OutcomeStatement] = []
    for i, o in enumerate(ex.outcome_statements):
        pol = (o.polarity or "").lower().strip()
        if pol != "failure" and _has_asserted_failure_marker(o.text):
            new_outcomes.append(OutcomeStatement(text=o.text, polarity="failure"))
            changed_idx.append(i)
        else:
            new_outcomes.append(OutcomeStatement(text=o.text, polarity=o.polarity))
    promoted_from: str | None = None
    has_failure_polarity = any(
        (o.polarity or "").lower().strip() == "failure" for o in new_outcomes
    )
    if not has_failure_polarity:
        for q in ex.quotes:
            if _has_asserted_failure_marker(q):
                new_outcomes.append(OutcomeStatement(text=q, polarity="failure"))
                promoted_from = "quotes"
                break
        if promoted_from is None and _has_asserted_failure_marker(ex.notes):
            new_outcomes.append(OutcomeStatement(text=ex.notes, polarity="failure"))
            promoted_from = "notes"
    cues = list(ex.implicit_cues)
    blob = " ".join(o.text.lower() for o in new_outcomes)
    added_cue = False
    if (
        _has_asserted_failure_marker(blob)
        and any(m in blob for m in RESPONSE_MARKERS)
        and "response_then_progression" not in {c.lower() for c in cues}
    ):
        cues.append("response_then_progression")
        added_cue = True
    payload = ex.to_dict()
    payload["outcome_statements"] = [
        {"text": o.text, "polarity": o.polarity} for o in new_outcomes
    ]
    payload["implicit_cues"] = cues
    out = Extraction.from_dict(payload)
    meta = {
        "kind": "failure_evidence_from_extract_text_gated",
        "gate": "asserted_failure_marker",
        "n_polarities_relabeled": len(changed_idx),
        "relabeled_outcome_idx": changed_idx,
        "promoted_failure_from": promoted_from,
        "added_response_then_progression": added_cue,
        "touched_contradiction_present": False,
        "hidden_state_patched": False,
    }
    return out, meta


def relabel_failure_polarity_from_text_gated_mixed(
    ex: Extraction,
) -> tuple[Extraction, dict[str, Any]]:
    """Goal 3 gated promote, then surface simultaneous 1L conflict.

    P-xspan is two incompatible first-line claims (failed AND
    continue/ongoing). Frozen ground() already scores that as X if
    contradiction.present is true; Dual extracts often omit the flag and
    plant response_then_progression instead. This step sets present=true
    when ground's existing hard-simultaneous check is true.

    Does not mutate the gated/ungated promote functions. Does not rewrite
    ground(). Not fitted to held-out leak IDs.
    """
    out, meta = relabel_failure_polarity_from_text_gated(ex)
    pols = [(o.polarity or "").lower().strip() for o in out.outcome_statements]
    flag = _hard_simultaneous_conflict(out, pols) and out.contradiction_present is not True
    meta = dict(meta)
    meta["kind"] = "failure_evidence_from_extract_text_gated_mixed"
    meta["gate"] = "asserted_failure_marker+simultaneous_1L"
    meta["flagged_simultaneous_1L"] = False
    if flag:
        payload = deepcopy(out.to_dict())
        payload["contradiction_present"] = True
        payload["contradiction"] = dict(payload.get("contradiction") or {})
        payload["contradiction"]["present"] = True
        out = Extraction.from_dict(payload)
        meta["flagged_simultaneous_1L"] = True
        meta["touched_contradiction_present"] = True
    return out, meta


NEVER_THERAPY_PHRASES = (
    "never received",
    "never having",
    "never receiving",
    "no prior",
    "no induction",
    "systemic-therapy naive",
    "therapy naive",
    "not been administered",
    "was not administered",
    "no cytotoxic",
)
GIVEN_THERAPY_PHRASES = ("cycles", "completed", "received", "administered")


def _span_has_never_therapy(text: str) -> bool:
    blob = (text or "").lower()
    return any(p in blob for p in NEVER_THERAPY_PHRASES)


def _span_has_given_therapy(text: str) -> bool:
    blob = (text or "").lower()
    if _span_has_never_therapy(blob):
        return False
    return any(p in blob for p in GIVEN_THERAPY_PHRASES)


def _never_vs_given_spans(ex: Extraction) -> tuple[str, str] | None:
    """Quote/notes never-therapy claim vs given/cycles claim.

    Frozen ground() conflict blob omits quotes, so never↔given parked in
    quotes is invisible unless copied into contradiction cues.
    """
    spans = list(ex.quotes) + ([ex.notes] if ex.notes else [])
    never_span = next((s for s in spans if _span_has_never_therapy(s)), None)
    if not never_span:
        return None
    given_cands = list(ex.quotes) + [o.text for o in ex.outcome_statements] + (
        [ex.notes] if ex.notes else []
    )
    given_span = next(
        (s for s in given_cands if s != never_span and _span_has_given_therapy(s)),
        None,
    )
    if given_span is None and (ex.administration_status or "").lower().strip() == "given":
        given_span = "administration_status=given"
    if not given_span:
        return None
    return never_span, given_span


def relabel_failure_polarity_from_text_gated_mixed_never(
    ex: Extraction,
) -> tuple[Extraction, dict[str, Any]]:
    """Goal 4 mixed flag, then copy never↔given from quotes into cues.

    Does not mutate Goal 3/4. Does not rewrite ground(). Predeclared from
    S-never-quote (Goal 5 census), not from leftover IDs.
    """
    out, meta = relabel_failure_polarity_from_text_gated_mixed(ex)
    meta = dict(meta)
    meta["kind"] = "failure_evidence_from_extract_text_gated_mixed_never"
    meta["gate"] = "asserted_failure_marker+simultaneous_1L+never_vs_given"
    meta["flagged_never_vs_given"] = False
    pair = _never_vs_given_spans(out)
    if pair and out.contradiction_present is not True:
        never_span, given_span = pair
        payload = deepcopy(out.to_dict())
        payload["contradiction_present"] = True
        payload["contradiction"] = {
            "present": True,
            "span_a": never_span,
            "span_b": given_span,
        }
        out = Extraction.from_dict(payload)
        meta["flagged_never_vs_given"] = True
        meta["touched_contradiction_present"] = True
        meta["never_span"] = never_span[:160]
        meta["given_span"] = given_span[:160]
    return out, meta


CONTINUE_CURRENT_PHRASES = (
    "continue current",
    "continues first-line",
    "ongoing response to first-line",
    "continue present",
    "still responding",
    "remains responsive",
    "remains sensitive",
)
CONFIRMED_FAILURE_PHRASES = (
    "clear progressive",
    "unequivocal",
    "confirmed progression",
    "confirmed progressive",
    "confirmed increasing",
    "meeting progression",
)


def _extract_blob(ex: Extraction) -> str:
    parts = list(ex.quotes) + [ex.notes] + [o.text for o in ex.outcome_statements]
    return " ".join(parts).lower()


def relabel_failure_polarity_from_text_gated_mixed_never_restage(
    ex: Extraction,
) -> tuple[Extraction, dict[str, Any]]:
    """Goal 6, then restaging confirmation vs simultaneous mix.

    1. If quotes/outcomes contain confirmed failure, clear extractor
       uncertainty.present (possible/pseudo resolved by a later scan).
    2. If Goal 4 flagged simultaneous 1L only because of stable disease
       then later progression, and the note does not say continue-current,
       revert contradiction.present (temporal restaging, not P-xspan).

    Does not mutate Goal 3/4/6. Does not rewrite ground().
    """
    orig_present = ex.contradiction_present
    out, meta = relabel_failure_polarity_from_text_gated_mixed_never(ex)
    meta = dict(meta)
    meta["kind"] = "failure_evidence_from_extract_text_gated_mixed_never_restage"
    meta["gate"] = "asserted_failure+simultaneous_1L+never_vs_given+restage"
    meta["cleared_hedge_uncertainty"] = False
    meta["reverted_stable_seq"] = False
    blob = _extract_blob(out)
    if out.uncertainty_present is True and any(
        p in blob for p in CONFIRMED_FAILURE_PHRASES
    ):
        payload = deepcopy(out.to_dict())
        payload["uncertainty_present"] = False
        payload["uncertainty_cues"] = []
        payload["uncertainty"] = {"present": False, "cue": None, "why": None}
        out = Extraction.from_dict(payload)
        meta["cleared_hedge_uncertainty"] = True
    if (
        meta.get("flagged_simultaneous_1L")
        and not meta.get("flagged_never_vs_given")
        and orig_present is not True
        and "stable disease" in blob
        and not any(p in blob for p in CONTINUE_CURRENT_PHRASES)
    ):
        payload = deepcopy(out.to_dict())
        payload["contradiction_present"] = False
        payload["contradiction"] = {"present": False, "span_a": None, "span_b": None}
        out = Extraction.from_dict(payload)
        meta["reverted_stable_seq"] = True
        meta["flagged_simultaneous_1L"] = False
        meta["touched_contradiction_present"] = True
    return out, meta


ONGOING_NOW_PHRASES = CONTINUE_CURRENT_PHRASES + ("do not change therapy",)
_FAIL_SPAN_HINTS = ("failed", "progressed", "progression", "failure")


def relabel_failure_polarity_from_text_gated_mixed_never_restage_simult(
    ex: Extraction,
) -> tuple[Extraction, dict[str, Any]]:
    """Goal 7, then same-time fail vs continue (not sequenced restaging).

    1. If the extract was already uncertain and Goal 3 only promoted failure
       from notes, revert (meta notes are not a clinical failure span).
    2. Relabel continue-now response polarities to ongoing so frozen ground()
       hard-simultaneous can fire instead of sequenced override.
    3. Copy continue-current quotes into contradiction cues when failure is
       already in the extract but present was omitted (quote-blind blob).

    Does not mutate Goal 3–7. Does not rewrite ground().
    """
    orig = Extraction.from_dict(ex.to_dict())
    orig_unc = orig.uncertainty_present is True
    out, meta = relabel_failure_polarity_from_text_gated_mixed_never_restage(ex)
    meta = dict(meta)
    meta["kind"] = "failure_evidence_from_extract_text_gated_mixed_never_restage_simult"
    meta["undid_notes_promote_on_uncertain"] = False
    meta["relabeled_ongoing_now"] = False
    meta["copied_continue_quotes"] = False
    if orig_unc and meta.get("promoted_failure_from") == "notes":
        meta["undid_notes_promote_on_uncertain"] = True
        meta["flagged_simultaneous_1L"] = False
        meta["promoted_failure_from"] = None
        meta["touched_contradiction_present"] = False
        return orig, meta
    new_outcomes: list[OutcomeStatement] = []
    changed = False
    for o in out.outcome_statements:
        pol = (o.polarity or "").lower().strip()
        if pol == "response" and any(p in o.text.lower() for p in ONGOING_NOW_PHRASES):
            new_outcomes.append(OutcomeStatement(text=o.text, polarity="ongoing"))
            changed = True
        else:
            new_outcomes.append(OutcomeStatement(text=o.text, polarity=o.polarity))
    if changed:
        payload = deepcopy(out.to_dict())
        payload["outcome_statements"] = [
            {"text": o.text, "polarity": o.polarity} for o in new_outcomes
        ]
        payload["implicit_cues"] = [
            c for c in out.implicit_cues if c.lower() != "response_then_progression"
        ]
        out = Extraction.from_dict(payload)
        meta["relabeled_ongoing_now"] = True
    blob = _extract_blob(out)
    has_fail = any(
        (o.polarity or "").lower().strip() == "failure" for o in out.outcome_statements
    ) or any(h in blob for h in _FAIL_SPAN_HINTS)
    has_cont = any(p in blob for p in ONGOING_NOW_PHRASES)
    if has_fail and has_cont and out.contradiction_present is not True:
        spans = list(out.quotes) + [o.text for o in out.outcome_statements] + (
            [out.notes] if out.notes else []
        )
        fail_span = next((s for s in spans if any(h in s.lower() for h in _FAIL_SPAN_HINTS)), None)
        cont_span = next((s for s in spans if any(p in s.lower() for p in ONGOING_NOW_PHRASES)), None)
        if fail_span and cont_span and fail_span != cont_span:
            payload = deepcopy(out.to_dict())
            payload["contradiction_present"] = True
            payload["contradiction"] = {
                "present": True,
                "span_a": fail_span,
                "span_b": cont_span,
            }
            payload["implicit_cues"] = [
                c for c in out.implicit_cues if c.lower() != "response_then_progression"
            ]
            out = Extraction.from_dict(payload)
            meta["copied_continue_quotes"] = True
            meta["touched_contradiction_present"] = True
    return out, meta


NEXT_LINE_PHRASES = (
    "now considering",
    "being considered",
    "considering",
    "considered for",
    "next line",
    "next-line",
    "second-line",
    "switching to",
)
REFRACTORY_PHRASES = ("refractory", "platinum-refractory", "platinum_refractory")
NEVER_DISPENSED_PHRASES = (
    "never dispensed",
    "not dispensed",
    "ever dispensed",
    "was not dispensed",
)


def _span_has_never_dispensed(text: str) -> bool:
    blob = (text or "").lower()
    if not any(p in blob for p in NEVER_DISPENSED_PHRASES):
        return False
    if "ever dispensed" in blob and not any(
        n in blob for n in ("no ", "not ", "never ")
    ):
        return False
    return True


def relabel_failure_polarity_from_text_gated_mixed_never_restage_simult_admin(
    ex: Extraction,
) -> tuple[Extraction, dict[str, Any]]:
    """Goal 8, then leftover S-admin-blocked (census shape, not ID-fitted).

    1. If extractor marked first-line planned but the note is refractory
       and now considering a next line, set administration_status=given
       (planned token is the next line, not an ungiven first line).
    2. Copy fail vs never-dispensed quotes into contradiction cues; if
       uncertainty.cue is that never-dispensed span, clear U (the pole
       was parked as hedge).

    Does not mutate Goal 3–8. Does not rewrite ground().
    """
    out, meta = relabel_failure_polarity_from_text_gated_mixed_never_restage_simult(ex)
    meta = dict(meta)
    meta["kind"] = (
        "failure_evidence_from_extract_text_gated_mixed_never_restage_simult_admin"
    )
    meta["relabeled_planned_next_after_refractory"] = False
    meta["copied_never_dispensed"] = False
    meta["cleared_dispense_uncertainty"] = False
    blob = _extract_blob(out)
    cues = {c.lower() for c in out.implicit_cues}
    has_refractory = any(p in blob for p in REFRACTORY_PHRASES) or (
        "platinum_refractory" in cues
    )
    has_next = any(p in blob for p in NEXT_LINE_PHRASES)
    if (
        (out.administration_status or "").lower().strip() == "planned"
        and has_refractory
        and has_next
    ):
        payload = deepcopy(out.to_dict())
        payload["administration_status"] = "given"
        out = Extraction.from_dict(payload)
        meta["relabeled_planned_next_after_refractory"] = True
    spans = list(out.quotes) + [o.text for o in out.outcome_statements] + (
        [out.notes] if out.notes else []
    )
    never_span = next((s for s in spans if _span_has_never_dispensed(s)), None)
    fail_span = next(
        (
            s
            for s in spans
            if s != never_span
            and any(h in s.lower() for h in _FAIL_SPAN_HINTS)
            and not _span_has_never_dispensed(s)
        ),
        None,
    )
    if never_span and fail_span and out.contradiction_present is not True:
        payload = deepcopy(out.to_dict())
        payload["contradiction_present"] = True
        payload["contradiction"] = {
            "present": True,
            "span_a": fail_span,
            "span_b": never_span,
        }
        out = Extraction.from_dict(payload)
        meta["copied_never_dispensed"] = True
        meta["touched_contradiction_present"] = True
    if never_span and out.uncertainty_present is True:
        u_blob = " ".join(
            [out.uncertainty_cue or ""] + list(out.uncertainty_cues)
        ).lower()
        if _span_has_never_dispensed(u_blob) or never_span.lower() in u_blob:
            payload = deepcopy(out.to_dict())
            payload["uncertainty_present"] = False
            payload["uncertainty_cues"] = []
            payload["uncertainty"] = {"present": False, "cue": None, "why": None}
            out = Extraction.from_dict(payload)
            meta["cleared_dispense_uncertainty"] = True
    return out, meta


def score_case(ex: Extraction, *, intervene: bool) -> dict[str, Any]:
    before = atoms_payload(ex)
    if not intervene:
        return {
            "intervened": False,
            "before": before,
            "after": before,
            "intervention": None,
            "verdict_flipped_to_gold": None,
        }
    after_ex, meta = relabel_failure_polarity_from_text(ex)
    after = atoms_payload(after_ex)
    return {
        "intervened": True,
        "before": before,
        "after": after,
        "intervention": meta,
        "verdict_before": before["verdict"],
        "verdict_after": after["verdict"],
        "atoms_before": before["atoms"],
        "atoms_after": after["atoms"],
    }


def _frozen_traces() -> dict[str, Any]:
    if not LOCATOR_PANEL.is_file():
        return {}
    blob = json.loads(LOCATOR_PANEL.read_text(encoding="utf-8"))
    out: dict[str, Any] = {}
    for rec in blob.get("cases") or []:
        out[rec["id"]] = {
            "encode_status": (rec.get("encode") or {}).get("status"),
            "compute_status": (rec.get("compute") or {}).get("status"),
            "n_lost_steps": (rec.get("compute") or {}).get("n_lost_steps"),
            "mean_score_d": rec.get("mean_score_d"),
            "translate_verdict": (rec.get("translate") or {}).get("verdict"),
        }
    return out


def _load_saved_extracts() -> dict[str, Any]:
    if not EXTRACT_PATH.is_file():
        return {}
    blob = json.loads(EXTRACT_PATH.read_text(encoding="utf-8"))
    return blob.get("extracts") or {}


def recover_hf_extracts(*, model_id: str, prompt: str = "schema") -> dict[str, Any]:
    """Replay 7B extract with intervention=none. Not a d-patch.

    prompt=schema: Schema + note (Exp1 recover).
    prompt=analog: user=evidence only (matches n2s_nn_layer_patch._run_extract).
    """
    from .hf_client import unload_model
    from .hf_intervene import generate_intervened
    from .neural import EXTRACT_SCHEMA, EXTRACT_SYSTEM
    from .ollama_client import parse_json_object
    from .phase10_activation import FAIL_MODEL

    if prompt not in {"schema", "analog"}:
        raise ValueError(prompt)
    mid = model_id or FAIL_MODEL
    extracts: dict[str, Any] = {}
    try:
        for nid in (FOLFOX_ID, CONTRA_ID):
            note = DEFAULT_NOTES[nid]
            user = (
                note["evidence"]
                if prompt == "analog"
                else f"Schema:\n{EXTRACT_SCHEMA}\n\nNote:\n{note['evidence']}"
            )
            trace, _iv = generate_intervened(
                system=EXTRACT_SYSTEM,
                user=user,
                model_id=mid,
                max_new_tokens=400,
                intervention="none",
                layer_indices=(12,),
            )
            gen_only = "".join(s.token_text for s in trace.steps)
            try:
                data = parse_json_object(gen_only)
                ex = Extraction.from_dict(data)
                parse_ok = True
            except ValueError:
                ex = None
                parse_ok = False
            scored = atoms_payload(ex) if ex is not None else None
            extracts[nid] = {
                "id": nid,
                "gold": note["gold"],
                "parse_ok": parse_ok,
                "prompt": prompt,
                "extract": None if ex is None else ex.to_dict(),
                "grounded": None
                if scored is None
                else {
                    "atoms": scored["atoms"],
                    "verdict": scored["verdict"],
                    "ground_trace": scored["ground_trace"],
                },
                "backend": f"hf:{mid}",
                "intervention": "none",
                "not": "n2n frozen-d patch",
            }
    finally:
        unload_model()
    dest = EXTRACT_ANALOG_PATH if prompt == "analog" else EXTRACT_PATH
    blob = {
        "ok": True,
        "kind": "n2s_phase2_translate_extracts_v1",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "model_id": mid,
        "prompt": prompt,
        "extracts": extracts,
    }
    dest.write_text(json.dumps(blob, indent=2) + "\n", encoding="utf-8")
    return extracts


def _frozen_analog_contra() -> dict[str, Any] | None:
    if not PATCH_PANEL.is_file():
        return None
    blob = json.loads(PATCH_PANEL.read_text(encoding="utf-8"))
    for row in blob.get("rows") or []:
        if row.get("case_id") == CONTRA_ID and row.get("label") == "baseline":
            return {
                "source": PATCH_PANEL.name,
                "extract_x": row.get("extract_x"),
                "grounded_X": row.get("grounded_X"),
                "verdict": row.get("verdict"),
                "prompt": "analog (user=evidence)",
                "extract_json_saved": False,
            }
    return None


def reconcile_contra_duals() -> dict[str, Any]:
    """Compare frozen analog Dual vs schema recover vs analog-prompt recover."""
    frozen = _frozen_analog_contra()
    schema_blob = (
        json.loads(EXTRACT_PATH.read_text(encoding="utf-8"))
        if EXTRACT_PATH.is_file()
        else {}
    )
    analog_blob = (
        json.loads(EXTRACT_ANALOG_PATH.read_text(encoding="utf-8"))
        if EXTRACT_ANALOG_PATH.is_file()
        else {}
    )
    schema = ((schema_blob.get("extracts") or {}).get(CONTRA_ID) or {})
    analog = ((analog_blob.get("extracts") or {}).get(CONTRA_ID) or {})
    schema_g = None
    if schema.get("parse_ok") and schema.get("extract"):
        schema_g = atoms_payload(Extraction.from_dict(schema["extract"]))
    analog_g = analog.get("grounded")
    if analog_g is None and analog.get("parse_ok") and analog.get("extract"):
        analog_g = atoms_payload(Extraction.from_dict(analog["extract"]))
    frozen_v = (frozen or {}).get("verdict")
    analog_v = (analog_g or {}).get("verdict")
    schema_v = (schema_g or {}).get("verdict")
    analog_matches_frozen = bool(
        analog_v == frozen_v
        and analog_g
        and analog_g.get("atoms", {}).get("X_contradiction") is True
        and (frozen or {}).get("grounded_X") is True
    )
    prompt_explains = analog_matches_frozen and schema_v != frozen_v
    analog_folfox = (analog_blob.get("extracts") or {}).get(FOLFOX_ID) or {}
    analog_intervene: dict[str, Any] = {}
    for nid, rec in ((FOLFOX_ID, analog_folfox), (CONTRA_ID, analog)):
        if rec.get("parse_ok") and rec.get("extract"):
            analog_intervene[nid] = score_case(
                Extraction.from_dict(rec["extract"]), intervene=True
            )
    out = {
        "ok": True,
        "kind": "n2s_phase2_contra_dual_reconcile_v1",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "protocol": "docs/TRACKB-PHASE2-TRANSLATE-CONTRAST.md",
        "frozen_analog": frozen,
        "schema_recover": {
            "prompt": "Schema+note",
            "parse_ok": schema.get("parse_ok"),
            "verdict": schema_v,
            "grounded": schema_g,
            "cues": (schema.get("extract") or {}).get("implicit_cues"),
            "polarities": [
                o.get("polarity")
                for o in ((schema.get("extract") or {}).get("outcome_statements") or [])
            ],
        },
        "analog_recover": {
            "prompt": "user=evidence",
            "parse_ok": analog.get("parse_ok"),
            "verdict": analog_v,
            "grounded": analog_g,
            "cues": (analog.get("extract") or {}).get("implicit_cues"),
            "polarities": [
                o.get("polarity")
                for o in ((analog.get("extract") or {}).get("outcome_statements") or [])
            ],
        },
        "analog_intervene": analog_intervene,
        "analog_folfox": {
            "parse_ok": analog_folfox.get("parse_ok"),
            "verdict": (analog_folfox.get("grounded") or {}).get("verdict"),
            "empty_extract": not bool(
                (analog_folfox.get("extract") or {}).get("quotes")
                or (analog_folfox.get("extract") or {}).get("outcome_statements")
                or (analog_folfox.get("extract") or {}).get("regimen_names")
            ),
            "note": (
                "Analog (user=evidence) FOLFOX extract is too thin for quote-promote. "
                "FOLFOX repair stays on Schema+note extract; Dual CONTRA control "
                "stays on analog extract."
            ),
        },
        "diagnosis": {
            "analog_matches_frozen": analog_matches_frozen,
            "prompt_explains_split": prompt_explains,
            "ground_changed": False,
            "do_not_change_ground": analog_matches_frozen,
            "note": (
                "If analog_matches_frozen: Schema+note recover was the mismatch; "
                "do not change ground(). If analog still SATISFIED: reconnect/"
                "same-day vs sequenced, still no Compute editor."
            ),
        },
        "not": [
            "n2n frozen-d",
            "Compute editor",
            "Dual_full rescore",
        ],
    }
    RECONCILE_PATH.write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
    return out


def _extraction_from_saved(nid: str, saved: dict[str, Any]) -> Extraction | None:
    rec = saved.get(nid) or {}
    if not rec.get("parse_ok"):
        return None
    data = rec.get("extract")
    if not isinstance(data, dict):
        return None
    return Extraction.from_dict(data)


def _score_cases(
    saved: dict[str, Any],
    *,
    snapshots: dict[str, str] | None = None,
) -> dict[str, Any]:
    traces = _frozen_traces()
    cases: dict[str, Any] = {}
    for nid, gold in ((FOLFOX_ID, "SATISFIED"), (CONTRA_ID, "CONTRADICTION")):
        ex = _extraction_from_saved(nid, saved)
        if ex is None:
            cases[nid] = {
                "ok": False,
                "id": nid,
                "gold": gold,
                "error": "no recovered 7B extract — run --recover-hf",
                "frozen_trace": traces.get(nid),
            }
            continue
        scored = score_case(ex, intervene=True)
        after_v = scored["verdict_after"]
        rec = {
            "ok": True,
            "id": nid,
            "gold": gold,
            "match_gold_before": scored["verdict_before"] == gold,
            "match_gold_after": after_v == gold,
            "repaired": scored["verdict_before"] != gold and after_v == gold,
            "broke_control": scored["verdict_before"] == gold and after_v != gold,
            "frozen_trace": traces.get(nid),
            "invariants": {
                "hidden_state_patched": False,
                "dual_full_rewritten": False,
                "compute_lost_steps_unchanged": True,
                "encode_unchanged": True,
                "note": "Boundary relabel only. Frozen encode/compute traces not rewritten.",
            },
            **scored,
        }
        if snapshots and nid in snapshots:
            rec["prompt_snapshot"] = snapshots[nid]
        cases[nid] = rec
    return cases


def _thesis_from_cases(cases: dict[str, Any], *, dual_matched: bool) -> dict[str, Any]:
    folfox = cases.get(FOLFOX_ID) or {}
    contra = cases.get(CONTRA_ID) or {}
    thesis: dict[str, Any] = {
        "folfox_repaired": bool(folfox.get("repaired")),
        "contra_dual_held": bool(contra.get("ok"))
        and not bool(contra.get("broke_control"))
        and contra.get("after", {}).get("atoms", {}).get("X_contradiction") is True,
        "contra_compute_untouched": True,
        "locator_chose_repair_class": False,
        "routing_selection_value": None,
        "dual_matched_pairing": dual_matched,
    }
    if folfox.get("ok") and contra.get("ok"):
        thesis["contra_verdict_unchanged"] = (
            contra.get("verdict_before") == contra.get("verdict_after")
        )
        thesis["contra_dual_held"] = bool(
            thesis["contra_verdict_unchanged"]
            and contra.get("after", {}).get("atoms", {}).get("X_contradiction")
            is True
        )
        contrast_holds = bool(
            thesis["folfox_repaired"]
            and thesis["contra_verdict_unchanged"]
            and thesis["contra_compute_untouched"]
        )
        thesis["contrast_holds"] = contrast_holds
        # Same-intervention contrast ≠ locator causal selection.
        thesis["routing_selection_value"] = bool(dual_matched and contrast_holds)
        if dual_matched:
            thesis["note"] = (
                "Dual-matched pairing: Schema FOLFOX + analog CONTRA. Contrast "
                "holds = FOLFOX repaired and analog CONTRA Dual CONTRADICTION / "
                "lost-steps unchanged. Does not mean :8260 chose the repair class."
            )
        else:
            thesis["note"] = (
                "CONTRA recover+ground may already differ from frozen Dual analog "
                "(temporal-change override). Selection value here = FOLFOX repaired "
                "and CONTRA verdict/lost-steps unchanged by this intervention."
            )
    return thesis


def build_panel(*, extracts: dict[str, Any] | None = None) -> dict[str, Any]:
    saved = extracts if extracts is not None else _load_saved_extracts()
    cases = _score_cases(saved)
    panel = {
        "ok": True,
        "kind": "n2s_phase2_translate_contrast_v1",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "protocol": "docs/TRACKB-PHASE2-TRANSLATE-CONTRAST.md",
        "intervention": "failure_evidence_from_extract_text",
        "pairing": "schema_both",
        "not": [
            "n2n frozen-d editor",
            "Dual_full Phase-7 rescore",
            "Encode/Compute editors",
            "auto-correct pipeline",
            "REVIEW-as-correction",
        ],
        "cases": cases,
        "thesis": _thesis_from_cases(cases, dual_matched=False),
        "frozen_traces_source": str(LOCATOR_PANEL.name),
    }
    PANEL_PATH.write_text(json.dumps(panel, indent=2) + "\n", encoding="utf-8")
    return panel


def _load_extract_blob(path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    blob = json.loads(path.read_text(encoding="utf-8"))
    return blob.get("extracts") or {}


def build_clean_panel() -> dict[str, Any]:
    """Schema FOLFOX repair vs analog CONTRA Dual control. Does not overwrite Exp1."""
    schema = _load_extract_blob(EXTRACT_PATH)
    analog = _load_extract_blob(EXTRACT_ANALOG_PATH)
    mixed = {
        FOLFOX_ID: schema.get(FOLFOX_ID) or {},
        CONTRA_ID: analog.get(CONTRA_ID) or {},
    }
    snapshots = {FOLFOX_ID: "schema", CONTRA_ID: "analog"}
    cases = _score_cases(mixed, snapshots=snapshots)
    panel = {
        "ok": True,
        "kind": "n2s_phase2_translate_contrast_clean_v1",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "protocol": "docs/TRACKB-PHASE2-TRANSLATE-CONTRAST.md",
        "intervention": "failure_evidence_from_extract_text",
        "pairing": {
            FOLFOX_ID: "schema+note",
            CONTRA_ID: "analog user=evidence",
        },
        "not": [
            "n2n frozen-d editor",
            "Dual_full Phase-7 rescore",
            "Encode/Compute editors",
            "auto-correct pipeline",
            "REVIEW-as-correction",
            "locator-chose-repair-class",
            "ground() rewrite",
        ],
        "cases": cases,
        "thesis": _thesis_from_cases(cases, dual_matched=True),
        "frozen_traces_source": str(LOCATOR_PANEL.name),
        "extract_sources": {
            FOLFOX_ID: str(EXTRACT_PATH.name),
            CONTRA_ID: str(EXTRACT_ANALOG_PATH.name),
        },
    }
    CLEAN_PANEL_PATH.write_text(json.dumps(panel, indent=2) + "\n", encoding="utf-8")
    return panel


def selftest() -> None:
    """CPU: typical FOLFOX analog miss vs CONTRA control."""
    folfox_miss = Extraction(
        stated_line="first",
        regimen_names=["FOLFOX"],
        administration_status="given",
        contradiction_present=False,
        outcome_statements=[
            OutcomeStatement(
                text="produced a partial response",
                polarity="response",
            ),
        ],
        quotes=[
            "First-line FOLFOX produced a partial response.",
            "FOLFOX was discontinued for treatment failure",
        ],
        implicit_cues=["response_then_progression"],
    )
    base = atoms_payload(folfox_miss)
    assert base["verdict"] == Verdict.NOT_SATISFIED.value, base
    assert base["atoms"]["C_failure_event"] == "false", base["atoms"]
    scored = score_case(folfox_miss, intervene=True)
    assert scored["verdict_after"] == Verdict.SATISFIED.value, scored
    assert scored["after"]["atoms"]["C_failure_event"] == "true"
    assert scored["intervention"]["promoted_failure_from"] == "quotes"
    assert scored["intervention"]["hidden_state_patched"] is False

    contra = Extraction(
        stated_line="first",
        regimen_names=["FOLFOX"],
        administration_status="given",
        contradiction_present=True,
        outcome_statements=[
            OutcomeStatement(
                text="Patient failed first-line FOLFOX after four cycles.",
                polarity="failure",
            ),
            OutcomeStatement(
                text="Disease remains responsive to FOLFOX; continue current regimen.",
                polarity="ongoing",
            ),
        ],
        implicit_cues=[],
    )
    c_scored = score_case(contra, intervene=True)
    assert c_scored["before"]["verdict"] == Verdict.CONTRADICTION.value
    assert c_scored["verdict_after"] == Verdict.CONTRADICTION.value
    assert c_scored["after"]["atoms"]["X_contradiction"] is True
    assert c_scored["intervention"]["touched_contradiction_present"] is False


def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 2 Translate contrast Exp1")
    parser.add_argument("--selftest", action="store_true")
    parser.add_argument("--recover-hf", action="store_true")
    parser.add_argument("--recover-hf-analog", action="store_true")
    parser.add_argument("--reconcile", action="store_true")
    parser.add_argument("--panel", action="store_true")
    parser.add_argument("--panel-clean", action="store_true")
    parser.add_argument("--model-id", default="")
    args = parser.parse_args()
    if args.selftest:
        selftest()
        print("selftest ok")
        return
    extracts = None
    if args.recover_hf:
        extracts = recover_hf_extracts(model_id=args.model_id, prompt="schema")
        print(json.dumps({k: {"parse_ok": v.get("parse_ok")} for k, v in extracts.items()}))
    if args.recover_hf_analog:
        analog = recover_hf_extracts(model_id=args.model_id, prompt="analog")
        print(
            json.dumps(
                {
                    k: {
                        "parse_ok": v.get("parse_ok"),
                        "verdict": (v.get("grounded") or {}).get("verdict"),
                    }
                    for k, v in analog.items()
                }
            )
        )
    if args.panel or args.recover_hf:
        panel = build_panel(extracts=extracts if args.recover_hf else None)
        print(json.dumps({"thesis": panel["thesis"], "artifact": str(PANEL_PATH)}, indent=2))
    if args.panel_clean:
        clean = build_clean_panel()
        print(
            json.dumps(
                {"thesis": clean["thesis"], "artifact": str(CLEAN_PANEL_PATH)},
                indent=2,
            )
        )
        return
    if args.reconcile or args.recover_hf_analog:
        rec = reconcile_contra_duals()
        print(json.dumps({"diagnosis": rec["diagnosis"], "artifact": str(RECONCILE_PATH)}, indent=2))
        return
    if args.panel or args.recover_hf:
        return
    parser.print_help()


if __name__ == "__main__":
    main()
