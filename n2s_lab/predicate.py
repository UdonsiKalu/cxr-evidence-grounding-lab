"""Formal predicate FIRST_LINE_THERAPY_FAILED and its symbolic evaluator.

The neural side must not decide the verdict. It only fills a descriptive
extraction. Grounding maps that extraction onto these atoms. This module
maps atoms onto a four-way verdict.
"""

from __future__ import annotations

from .types import Atom, Grounding, RuleResult, Verdict

PREDICATE_ID = "FIRST_LINE_THERAPY_FAILED"

PREDICATE_TEXT = """\
FIRST_LINE_THERAPY_FAILED holds iff all of the following are true:

  A  first_line_identified     — a first-line (front-line / 1L) systemic
                                 therapy is identified, explicitly or by
                                 conventional implication
  B  first_line_administered   — that therapy was given or attempted
                                 (not merely planned)
  C  failure_event             — progression, non-response, clinician
                                 statement of failure/refractory disease,
                                 or discontinuation for intolerance
                                 without a plan to resume
  D  failure_of_first_line     — the failure event is of that first-line
                                 regimen (not only a later line; not
                                 completed adjuvant therapy with NED)

X  contradiction — mutually incompatible claims about A–D

Not sufficient: planned therapy; ongoing first-line with response or
stable disease; temporary hold with plan to resume; adjuvant completion
without metastatic first-line failure.
"""

PREDICATE_FORMULA = (
    "CONTRADICTION if X; else "
    "NOT_SATISFIED if any of A,B,C,D is false; else "
    "UNCERTAIN if any of A,B,C,D is unknown; else "
    "SATISFIED if A ∧ B ∧ C ∧ D"
)

REQUIRED_ATOMS = (
    "first_line_identified",
    "first_line_administered",
    "failure_event",
    "failure_of_first_line",
)


def evaluate_rule(grounding: Grounding) -> RuleResult:
    trace: list[str] = []
    if grounding.contradiction:
        trace.append("X is true → CONTRADICTION (short-circuit)")
        return RuleResult(
            verdict=Verdict.CONTRADICTION,
            fired_rule="X → CONTRADICTION",
            trace=trace,
        )

    atoms = {
        "A first_line_identified": grounding.first_line_identified,
        "B first_line_administered": grounding.first_line_administered,
        "C failure_event": grounding.failure_event,
        "D failure_of_first_line": grounding.failure_of_first_line,
    }
    false_names = [name for name, value in atoms.items() if value is Atom.FALSE]
    unknown_names = [name for name, value in atoms.items() if value is Atom.UNKNOWN]

    if false_names:
        trace.append("False atoms: " + ", ".join(false_names))
        return RuleResult(
            verdict=Verdict.NOT_SATISFIED,
            fired_rule="any(A,B,C,D)=false → NOT_SATISFIED",
            trace=trace,
        )
    if unknown_names:
        trace.append("Unknown atoms: " + ", ".join(unknown_names))
        return RuleResult(
            verdict=Verdict.UNCERTAIN,
            fired_rule="any(A,B,C,D)=unknown → UNCERTAIN",
            trace=trace,
        )

    trace.append("A ∧ B ∧ C ∧ D all true")
    return RuleResult(
        verdict=Verdict.SATISFIED,
        fired_rule="A ∧ B ∧ C ∧ D → SATISFIED",
        trace=trace,
    )


def selftest() -> None:
    """Tiny unit check of the symbolic rule, independent of any LLM."""
    sat = Grounding(
        first_line_identified=Atom.TRUE,
        first_line_administered=Atom.TRUE,
        failure_event=Atom.TRUE,
        failure_of_first_line=Atom.TRUE,
        contradiction=False,
    )
    assert evaluate_rule(sat).verdict is Verdict.SATISFIED

    not_sat = Grounding(
        first_line_identified=Atom.TRUE,
        first_line_administered=Atom.FALSE,
        failure_event=Atom.UNKNOWN,
        failure_of_first_line=Atom.UNKNOWN,
        contradiction=False,
    )
    assert evaluate_rule(not_sat).verdict is Verdict.NOT_SATISFIED

    unc = Grounding(
        first_line_identified=Atom.TRUE,
        first_line_administered=Atom.TRUE,
        failure_event=Atom.UNKNOWN,
        failure_of_first_line=Atom.UNKNOWN,
        contradiction=False,
    )
    assert evaluate_rule(unc).verdict is Verdict.UNCERTAIN

    contra = Grounding(
        first_line_identified=Atom.TRUE,
        first_line_administered=Atom.TRUE,
        failure_event=Atom.TRUE,
        failure_of_first_line=Atom.TRUE,
        contradiction=True,
    )
    assert evaluate_rule(contra).verdict is Verdict.CONTRADICTION
