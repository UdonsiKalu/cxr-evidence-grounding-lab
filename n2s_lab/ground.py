"""Deterministic grounding: descriptive extraction → predicate atoms.

This is the neural-to-symbolic boundary. It does not call a model.
"""

from __future__ import annotations

from .types import Atom, Extraction, Grounding

FAILURE_POLARITIES = {"failure"}
POSITIVE_POLARITIES = {"response", "ongoing"}
IMPLICIT_FIRST_LINE_CUES = {
    "switching_to_second_line",
    "platinum_refractory",
    "now_second_line",
}
ADJUVANT_CUES = {"adjuvant_completed", "adjuvant"}
SOFT_UNCERTAINTY = {
    "questionable",
    "may_have",
    "unknown_outcome",
    "possible",
    "awaiting_scan",
    "mixed",
}
WHY_ATOMS = {
    "unresolved_line": ("A", "B", "C", "D"),
    "questionable": ("A", "B", "C", "D"),
    "may_have": ("B", "C", "D"),
    "unknown_outcome": ("C", "D"),
    "mixed": ("C", "D"),
    "pending": ("C", "D"),
    "possible": ("C", "D"),
    "awaiting_scan": ("C", "D"),
}


def _uncertainty_present(extraction: Extraction) -> bool:
    if extraction.uncertainty_present is True:
        return True
    if extraction.uncertainty_present is False:
        return False
    return bool(extraction.uncertainty_cues)


def _uncertainty_targets(extraction: Extraction) -> set[str]:
    why = (extraction.uncertainty_why or "").lower().strip()
    blob = " ".join(
        [why, extraction.uncertainty_cue, *extraction.uncertainty_cues]
    ).lower()
    targets: set[str] = set()
    if why in WHY_ATOMS:
        targets.update(WHY_ATOMS[why])
    for key, letters in WHY_ATOMS.items():
        if key in blob:
            targets.update(letters)
    if "adjuvant" in blob or "which line" in blob or "metastatic" in blob and "adjuvant" in blob:
        targets.update("A", "B", "C", "D")
    if "may have" in blob:
        targets.update("B", "C", "D")
    if not targets:
        targets.update("C", "D")
    return targets


def _apply_uncertainty(
    *,
    identified: Atom,
    administered: Atom,
    failure_event: Atom,
    failure_of_first: Atom,
    extraction: Extraction,
    trace: list[str],
) -> tuple[Atom, Atom, Atom, Atom]:
    """Keep implicated atoms unknown so the unchanged rule can fire UNCERTAIN.

    Does not change X. FALSE cannot close the predicate while the note flags
    unresolved evidence; why-mapped atoms are unknown even if they were true.
    """
    targets = _uncertainty_targets(extraction)
    trace.append(
        "U=true (extractor uncertainty.present"
        + (f", why={extraction.uncertainty_why!r}" if extraction.uncertainty_why else "")
        + f"; keep {''.join(sorted(targets))} unknown)"
    )

    def maybe_unknown(atom: Atom, letter: str) -> Atom:
        if letter not in targets and atom is not Atom.FALSE:
            return atom
        if atom is Atom.UNKNOWN:
            return atom
        trace.append(f"{letter}=unknown (uncertainty.present; was {atom.value})")
        return Atom.UNKNOWN

    return (
        maybe_unknown(identified, "A"),
        maybe_unknown(administered, "B"),
        maybe_unknown(failure_event, "C"),
        maybe_unknown(failure_of_first, "D"),
    )


def _same_event_fact_conflict(extraction: Extraction, polarities: list[str]) -> bool:
    """True when spans/outcomes look like incompatible facts, not unresolved alternatives."""
    has_failure = any(p in FAILURE_POLARITIES for p in polarities)
    has_positive = any(p in POSITIVE_POLARITIES for p in polarities)
    if has_failure and has_positive:
        return True
    blob = " ".join(
        [extraction.administration_status]
        + [c.a + " " + c.b for c in extraction.contradiction_cues]
    ).lower()
    never = any(w in blob for w in ("never", "not_given", "not given", "no prior"))
    given = any(w in blob for w in ("given", "cycles", "received", "started", "administered"))
    failed = any(w in blob for w in ("failed", "progressed", "refractory"))
    ongoing = any(w in blob for w in ("continues", "stable", "ongoing", "continue"))
    return (never and given) or (failed and ongoing)


def ground(extraction: Extraction) -> Grounding:
    trace: list[str] = []
    line = extraction.stated_line.lower().strip()
    admin = extraction.administration_status.lower().strip()
    cues = {c.lower() for c in extraction.implicit_cues}
    polarities = [o.polarity.lower().strip() for o in extraction.outcome_statements]
    has_failure = any(p in FAILURE_POLARITIES for p in polarities)
    has_positive = any(p in POSITIVE_POLARITIES for p in polarities)
    has_mixed = any(p == "mixed" for p in polarities)
    has_planned_outcome = any(p == "planned" for p in polarities)

    contradiction = False
    if extraction.contradiction_present is True:
        contradiction = True
        if extraction.contradiction_cues:
            trace.append("X=true (extractor contradiction.present=true with spans)")
        else:
            trace.append("X=true (extractor contradiction.present=true; spans missing)")
    elif extraction.contradiction_present is False:
        contradiction = False
        trace.append("X=false (extractor contradiction.present=false / none)")
    elif extraction.contradiction_cues:
        contradiction = True
        trace.append("X=true (contradiction_cues list, present omitted)")
    elif has_failure and has_positive and "temporary_hold" not in cues:
        if "response_then_progression" in cues:
            trace.append("response_then_progression cue: treat as sequenced, not X")
        else:
            contradiction = True
            trace.append("failure polarity and response/ongoing polarity both present → X=true")

    unc = {c.lower() for c in extraction.uncertainty_cues}
    if contradiction and not _same_event_fact_conflict(extraction, polarities):
        if (unc & SOFT_UNCERTAINTY) or _uncertainty_present(extraction):
            contradiction = False
            trace.append("X=false (uncertainty: unresolved alternative, not same-event conflict)")

    if line == "first" or cues & IMPLICIT_FIRST_LINE_CUES:
        identified = Atom.TRUE
        trace.append("A=true (stated first-line or implicit 1L cue)")
    elif line == "adjuvant" or cues & ADJUVANT_CUES:
        identified = Atom.FALSE
        trace.append("A=false (adjuvant / not metastatic first-line)")
    elif line == "second":
        if cues & IMPLICIT_FIRST_LINE_CUES or has_failure:
            identified = Atom.TRUE
            trace.append("A=true (later-line context implying a prior first line)")
        else:
            identified = Atom.UNKNOWN
            trace.append("A=unknown (second-line mentioned, first-line not established)")
    elif line in {"unspecified", "unknown"}:
        if cues & IMPLICIT_FIRST_LINE_CUES:
            identified = Atom.TRUE
            trace.append("A=true (implicit cue despite unspecified line)")
        elif extraction.regimen_names:
            identified = Atom.UNKNOWN
            trace.append("A=unknown (regimen named, line unspecified)")
        else:
            identified = Atom.FALSE
            trace.append("A=false (no first-line identified)")
    else:
        identified = Atom.UNKNOWN
        trace.append(f"A=unknown (stated_line={line!r})")

    if admin == "given":
        administered = Atom.TRUE
        trace.append("B=true (administered/given)")
    elif admin in {"planned", "not_given"}:
        administered = Atom.FALSE
        trace.append(f"B=false (administration_status={admin})")
    else:
        if identified is Atom.TRUE and has_failure:
            administered = Atom.TRUE
            trace.append("B=true (inferred: failure of an identified first line implies it was given)")
        elif identified is Atom.FALSE:
            administered = Atom.FALSE
            trace.append("B=false (no first-line to administer)")
        else:
            administered = Atom.UNKNOWN
            trace.append("B=unknown")

    if "temporary_hold" in cues and not has_failure:
        failure_event = Atom.FALSE
        trace.append("C=false (temporary hold, plan to resume, no progression)")
    elif has_failure:
        failure_event = Atom.TRUE
        trace.append("C=true (failure polarity)")
    elif has_mixed or extraction.uncertainty_cues:
        failure_event = Atom.UNKNOWN
        trace.append("C=unknown (mixed response or uncertainty cues)")
    elif has_positive or has_planned_outcome:
        failure_event = Atom.FALSE
        trace.append("C=false (response/ongoing/planned, no failure)")
    elif polarities:
        failure_event = Atom.UNKNOWN
        trace.append("C=unknown (outcomes present but polarity not failure/response)")
    else:
        failure_event = Atom.UNKNOWN
        trace.append("C=unknown (no outcome statements)")

    if failure_event is Atom.FALSE:
        failure_of_first = Atom.FALSE
        trace.append("D=false (no failure event)")
    elif identified is Atom.FALSE:
        failure_of_first = Atom.FALSE
        trace.append("D=false (identified line is not first-line)")
    elif failure_event is Atom.TRUE and identified is Atom.TRUE:
        failure_of_first = Atom.TRUE
        trace.append("D=true (failure attributed to identified first-line)")
    else:
        failure_of_first = Atom.UNKNOWN
        trace.append("D=unknown")

    has_uncertainty = _uncertainty_present(extraction)
    if has_uncertainty and not contradiction:
        identified, administered, failure_event, failure_of_first = _apply_uncertainty(
            identified=identified,
            administered=administered,
            failure_event=failure_event,
            failure_of_first=failure_of_first,
            extraction=extraction,
            trace=trace,
        )
    elif extraction.uncertainty_present is False:
        trace.append("U=false (extractor uncertainty.present=false / none)")

    return Grounding(
        first_line_identified=identified,
        first_line_administered=administered,
        failure_event=failure_event,
        failure_of_first_line=failure_of_first,
        contradiction=contradiction,
        uncertainty=has_uncertainty and not contradiction,
        trace=trace,
    )


def selftest_ground() -> None:
    """X comes from the first-class extract field; the rule is unchanged."""
    from .types import ContradictionCue, Extraction, OutcomeStatement

    with_spans = Extraction(
        stated_line="first",
        administration_status="given",
        contradiction_present=True,
        contradiction_cues=[
            ContradictionCue(a="first-line failed", b="continues first-line with SD")
        ],
    )
    assert ground(with_spans).contradiction is True

    explicit_none = Extraction(
        stated_line="first",
        administration_status="given",
        contradiction_present=False,
        outcome_statements=[
            OutcomeStatement(text="progressed", polarity="failure"),
            OutcomeStatement(text="response", polarity="ongoing"),
        ],
    )
    assert ground(explicit_none).contradiction is False

    unresolved_line = Extraction(
        stated_line="unknown",
        administration_status="unknown",
        contradiction_present=True,
        contradiction_cues=[
            ContradictionCue(
                a="carboplatin as first-line metastatic therapy",
                b="in the adjuvant setting",
            )
        ],
        uncertainty_cues=["questionable"],
    )
    assert ground(unresolved_line).contradiction is False

    from .predicate import evaluate_rule
    from .types import Verdict

    u4_collapsed = Extraction(
        stated_line="unspecified",
        regimen_names=["carboplatin"],
        administration_status="not_given",
        contradiction_present=False,
        uncertainty_present=True,
        uncertainty_cue="Questionable whether carboplatin was given as first-line or adjuvant",
        uncertainty_why="unresolved_line",
        uncertainty_cues=["questionable"],
        outcome_statements=[OutcomeStatement(text="no clear progression", polarity="ongoing")],
    )
    g_u4 = ground(u4_collapsed)
    assert g_u4.contradiction is False
    assert g_u4.uncertainty is True
    assert g_u4.first_line_administered is Atom.UNKNOWN
    assert evaluate_rule(g_u4).verdict is Verdict.UNCERTAIN

    u2_collapsed = Extraction(
        stated_line="unspecified",
        administration_status="not_given",
        implicit_cues=["platinum_refractory"],
        contradiction_present=False,
        uncertainty_present=True,
        uncertainty_cue="may have received platinum-based therapy",
        uncertainty_why="may_have",
        uncertainty_cues=["may_have", "unknown_outcome"],
        outcome_statements=[OutcomeStatement(text="unknown outcome", polarity="unknown")],
    )
    g_u2 = ground(u2_collapsed)
    assert g_u2.contradiction is False
    assert g_u2.first_line_administered is Atom.UNKNOWN
    assert evaluate_rule(g_u2).verdict is Verdict.UNCERTAIN

    planned = Extraction(
        stated_line="first",
        administration_status="planned",
        contradiction_present=False,
        uncertainty_present=False,
        outcome_statements=[OutcomeStatement(text="plan to start", polarity="planned")],
    )
    g_plan = ground(planned)
    assert g_plan.uncertainty is False
    assert g_plan.first_line_administered is Atom.FALSE
    assert evaluate_rule(g_plan).verdict is Verdict.NOT_SATISFIED
