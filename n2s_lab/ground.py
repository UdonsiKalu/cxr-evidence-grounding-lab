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
    """Soften *why-mapped* atoms to unknown so the rule can fire UNCERTAIN.

    Only letters in ``_uncertainty_targets`` are touched. Decisive FALSE atoms
    outside that set stay FALSE (e.g. administration_status=not_given with
    why=pending about outcomes must remain B=false → NOT_SATISFIED, not wipe
    B so Dual becomes UNCERTAIN). Does not change X.
    """
    targets = _uncertainty_targets(extraction)
    trace.append(
        "U=true (extractor uncertainty.present"
        + (f", why={extraction.uncertainty_why!r}" if extraction.uncertainty_why else "")
        + f"; keep {''.join(sorted(targets)) or '—'} unknown)"
    )

    def maybe_unknown(atom: Atom, letter: str) -> Atom:
        if letter not in targets:
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


def _blob_has_word(blob: str, words: tuple[str, ...]) -> bool:
    """Word / phrase match; avoid 'continue'∈'discontinued', 'given'∈'not_given'."""
    import re

    for w in words:
        if " " in w:
            if w in blob:
                return True
        elif re.search(rf"(?<![a-z]){re.escape(w)}(?![a-z])", blob):
            return True
    return False


def _negated_admin_claim(blob: str) -> bool:
    """True when blob asserts therapy was not given / not administered / declined."""
    import re

    patterns = (
        r"\bnot[\s-]+given\b",
        r"\bnever[\s-]+given\b",
        r"\bnot[\s-]+administered\b",
        r"\bnever[\s-]+administered\b",
        r"\bno(?:\s+\w+){0,4}\s+(?:been\s+)?administered\b",
        r"\bhas not been administered\b",
        r"\bhave not been administered\b",
        r"\bdeclined\b",
        r"\brefused\b",
        r"\bnot started\b",
        r"\bnever started\b",
        r"\bnot_given\b",
    )
    return any(re.search(p, blob) for p in patterns)


def _positive_admin_evidence(blob: str) -> bool:
    """True for positive administration evidence after stripping negated phrases."""
    import re

    cleaned = blob
    for p in (
        r"\bnot[\s-]+given\b",
        r"\bnever[\s-]+given\b",
        r"\bnot[\s-]+administered\b",
        r"\bnever[\s-]+administered\b",
        r"\bno(?:\s+\w+){0,4}\s+(?:been\s+)?administered\b",
        r"\bhas not been administered\b",
        r"\bhave not been administered\b",
        r"\bnot started\b",
        r"\bnever started\b",
        r"\bnot_given\b",
    ):
        cleaned = re.sub(p, " ", cleaned)
    return _blob_has_word(
        cleaned, ("given", "cycles", "received", "started", "administered")
    )


def _clinical_blob_for_conflict(extraction: Extraction) -> str:
    """Text used for never↔given / failed↔ongoing heuristics.

    Strip predicate-label chatter so analysis meta like
    ``FIRST_LINE_THERAPY_FAILED`` does not count as clinical ``failed``.
    """
    parts = (
        [extraction.administration_status]
        + [c.a + " " + c.b for c in extraction.contradiction_cues]
        + [o.text for o in extraction.outcome_statements]
    )
    blob = " ".join(parts).lower()
    for noise in (
        "first_line_therapy_failed",
        "first-line therapy failed",
        "formal condition",
        "formal predicate",
    ):
        blob = blob.replace(noise, " ")
    return blob


def _hard_simultaneous_conflict(extraction: Extraction, polarities: list[str]) -> bool:
    """True for incompatible *same-time* facts — not sequenced response→later failure."""
    has_failure = any(p in FAILURE_POLARITIES for p in polarities)
    has_ongoing = any(p == "ongoing" for p in polarities)
    blob = _clinical_blob_for_conflict(extraction)
    never = (
        _blob_has_word(blob, ("never", "not_given", "not given", "no prior"))
        or _negated_admin_claim(blob)
        or extraction.administration_status.lower().strip() in {"not_given", "planned"}
    )
    given = _positive_admin_evidence(blob)
    # Status planned/not_given alone is not "given"; require positive evidence beyond status token.
    if extraction.administration_status.lower().strip() in {"not_given", "planned"}:
        # Drop the status token from the given check (already in blob).
        given = _positive_admin_evidence(
            " ".join(
                [c.a + " " + c.b for c in extraction.contradiction_cues]
                + [o.text for o in extraction.outcome_statements]
            ).lower()
        )
    failed = _blob_has_word(
        blob, ("failed", "progressed", "refractory", "progression", "progressive")
    )
    # "no progression" / "without progression" are absence claims, not failure.
    if any(
        p in blob
        for p in (
            "no progression",
            "without progression",
            "not progressed",
            "no clear progression",
        )
    ):
        # Only count failure if an explicit failure polarity exists.
        failed = any(p in FAILURE_POLARITIES for p in polarities) or _blob_has_word(
            blob, ("failed", "progressed", "refractory")
        )
    ongoing = _blob_has_word(blob, ("continues", "stable", "ongoing", "continue"))
    return (
        (never and given)
        or (failed and ongoing)
        or (has_failure and has_ongoing)
    )


def _meta_contradiction_cues(extraction: Extraction) -> bool:
    """Extractor marked X but spans are commentary, not two clinical facts."""
    if not extraction.contradiction_cues:
        return False
    blob = " ".join(c.a + " " + c.b for c in extraction.contradiction_cues).lower()
    markers = (
        "no contradictory",
        "no contradiction",
        "does not support",
        "therefore, the note does not",
        "explicitly mentioned as",
        "no clear indication of a failure",
        "not support the formal condition",
        "suggests ongoing response",
    )
    return any(m in blob for m in markers)


def _false_x_without_failure(
    extraction: Extraction, polarities: list[str]
) -> bool:
    """X asserted but note has no failure polarity and no never↔given conflict.

    Covers toxicity-stop / no-failure controls where extractors invent X.
    """
    if any(p in FAILURE_POLARITIES for p in polarities):
        return False
    if _hard_simultaneous_conflict(extraction, polarities):
        return False
    return extraction.contradiction_present is True or bool(
        extraction.contradiction_cues
    )


def _sequenced_temporal_change(extraction: Extraction, polarities: list[str]) -> bool:
    """Therapy responded / possible, then later failed — temporal change, not contradiction."""
    if _hard_simultaneous_conflict(extraction, polarities):
        return False
    cues = {c.lower() for c in extraction.implicit_cues}
    has_failure = any(p in FAILURE_POLARITIES for p in polarities)
    has_response = any(p == "response" for p in polarities)
    has_unknown = any(p == "unknown" for p in polarities)
    if "response_then_progression" in cues and has_failure:
        return True
    if has_failure and has_response:
        return True
    # possible → confirmed progression (unknown then failure polarities)
    if has_failure and has_unknown:
        return True
    return False


def _same_event_fact_conflict(extraction: Extraction, polarities: list[str]) -> bool:
    """Backward-compatible name: hard simultaneous conflict only."""
    return _hard_simultaneous_conflict(extraction, polarities)


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
        # Track A reconnect (not G3): sequenced temporal-change is not a contradiction.
        if _sequenced_temporal_change(extraction, polarities):
            contradiction = False
            trace.append(
                "X=false (sequenced temporal-change; extractor contradiction overridden)"
            )
        elif _meta_contradiction_cues(extraction):
            # Meta/analysis spans are not clinical fact pairs — do not let
            # predicate-name wording (e.g. FIRST_LINE_THERAPY_FAILED) block this.
            contradiction = False
            trace.append(
                "X=false (meta contradiction cues; extractor contradiction overridden)"
            )
        elif _false_x_without_failure(extraction, polarities):
            contradiction = False
            trace.append(
                "X=false (no failure polarity / no hard conflict; extractor X overridden)"
            )
    elif extraction.contradiction_present is False:
        contradiction = False
        trace.append("X=false (extractor contradiction.present=false / none)")
    elif extraction.contradiction_cues:
        contradiction = True
        trace.append("X=true (contradiction_cues list, present omitted)")
        if _sequenced_temporal_change(extraction, polarities):
            contradiction = False
            trace.append(
                "X=false (sequenced temporal-change; cue-list contradiction overridden)"
            )
    elif has_failure and has_positive and "temporary_hold" not in cues:
        if "response_then_progression" in cues or _sequenced_temporal_change(
            extraction, polarities
        ):
            trace.append("response_then_progression / sequenced: treat as temporal, not X")
        else:
            contradiction = True
            trace.append("failure polarity and response/ongoing polarity both present → X=true")

    unc = {c.lower() for c in extraction.uncertainty_cues}
    if contradiction and not _hard_simultaneous_conflict(extraction, polarities):
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

    # Track A reconnect: response → later failure is temporal, even if extractor sets X.
    temporal_rx_fail = Extraction(
        stated_line="first",
        administration_status="given",
        contradiction_present=True,
        contradiction_cues=[
            ContradictionCue(
                a="partial response after three cycles",
                b="rising M-protein and new lesions at cycle 4",
            )
        ],
        outcome_statements=[
            OutcomeStatement(text="partial response after three cycles", polarity="response"),
            OutcomeStatement(text="rising M-protein and new lesions", polarity="failure"),
        ],
    )
    g_tmp = ground(temporal_rx_fail)
    assert g_tmp.contradiction is False
    assert any("temporal-change" in t for t in g_tmp.trace)

    possible_then_confirmed = Extraction(
        stated_line="first",
        administration_status="given",
        contradiction_present=True,
        contradiction_cues=[
            ContradictionCue(a="possible progression", b="clear progressive disease")
        ],
        outcome_statements=[
            OutcomeStatement(text="possible progression", polarity="unknown"),
            OutcomeStatement(text="clear progressive disease", polarity="failure"),
        ],
    )
    assert ground(possible_then_confirmed).contradiction is False

    # Toxicity-stop / no-failure: extractor invents X without failure polarity.
    tox_stop = Extraction(
        stated_line="first",
        administration_status="given",
        contradiction_present=True,
        contradiction_cues=[
            ContradictionCue(
                a="tumor markers falling and imaging stable",
                b="Paclitaxel stopped for grade 3 neuropathy",
            )
        ],
        outcome_statements=[
            OutcomeStatement(text="markers falling", polarity="response"),
            OutcomeStatement(text="imaging stable", polarity="ongoing"),
        ],
    )
    assert ground(tox_stop).contradiction is False

    nofail = Extraction(
        stated_line="first",
        administration_status="given",
        contradiction_present=True,
        contradiction_cues=[
            ContradictionCue(
                a="ongoing response",
                b="Plan to complete induction; no progression documented",
            )
        ],
        outcome_statements=[
            OutcomeStatement(text="ongoing response", polarity="ongoing"),
        ],
    )
    assert ground(nofail).contradiction is False

    # Path-D meta: analysis cites the predicate name — must not invent hard conflict.
    meta_predicate_name = Extraction(
        stated_line="first",
        administration_status="given",
        contradiction_present=True,
        contradiction_cues=[
            ContradictionCue(
                a="The clinical note supports identification of first-line 7+3 induction",
                b="Therefore, the note does not support the formal condition FIRST_LINE_THERAPY_FAILED.",
            )
        ],
        outcome_statements=[
            OutcomeStatement(text="Interim marrow shows ongoing response", polarity="ongoing"),
            OutcomeStatement(text="no progression documented", polarity="ongoing"),
        ],
    )
    g_meta = ground(meta_predicate_name)
    assert g_meta.contradiction is False
    assert any("meta contradiction" in t for t in g_meta.trace)

    # Path-D toxicity: meta "no clear indication of a failure" + unknown polarity.
    meta_tox = Extraction(
        stated_line="first",
        administration_status="not_given",
        contradiction_present=True,
        contradiction_cues=[
            ContradictionCue(
                a="there is no clear indication of a failure event related to progression or non-response",
                b="the tumor markers are described as falling and imaging stable, which suggests ongoing response or stable disease",
            )
        ],
        outcome_statements=[
            OutcomeStatement(text="tumor markers falling and imaging stable", polarity="ongoing"),
            OutcomeStatement(text="paclitaxel was stopped due to grade 3 neuropathy", polarity="unknown"),
        ],
        implicit_cues=["switching_to_second_line"],
    )
    assert ground(meta_tox).contradiction is False

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

    # Declined / never-given: D-path often sets U=true with why=pending (outcomes),
    # but B=false from not_given must stay decisive → NOT_SATISFIED (not UNCERTAIN).
    declined_pending_u = Extraction(
        stated_line="first",
        regimen_names=["FOLFIRINOX"],
        administration_status="not_given",
        contradiction_present=False,
        uncertainty_present=True,
        uncertainty_cue="the patient declined systemic therapy",
        uncertainty_why="pending",
        uncertainty_cues=[
            "the patient declined systemic therapy",
            "No chemotherapy has been administered",
            "pending",
        ],
        outcome_statements=[],
    )
    g_dec = ground(declined_pending_u)
    assert g_dec.first_line_administered is Atom.FALSE
    assert g_dec.uncertainty is True  # flag retained; atoms outside CD not wiped
    assert evaluate_rule(g_dec).verdict is Verdict.NOT_SATISFIED

    # Recommended + declined + "no chemo administered" is consistent, not X.
    declined_false_x = Extraction(
        stated_line="first",
        regimen_names=["FOLFIRINOX"],
        administration_status="not_given",
        contradiction_present=True,
        contradiction_cues=[
            ContradictionCue(
                a="First-line FOLFIRINOX was recommended but the patient declined systemic therapy.",
                b="No chemotherapy has been administered.",
            )
        ],
        outcome_statements=[],
    )
    g_fx = ground(declined_false_x)
    assert g_fx.contradiction is False
    assert g_fx.first_line_administered is Atom.FALSE
    assert evaluate_rule(g_fx).verdict is Verdict.NOT_SATISFIED
