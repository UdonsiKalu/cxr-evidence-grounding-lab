"""L2 verification: does formalization still match analysis distinctions?

Phase-5: on fail → regenerate once → on fail → REVIEW (not UNCERTAIN).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

from .phase1_diagnostic import (
    analysis_flags,
    analysis_has_distinction,
    expected_distinction,
    structure_preserves,
)
from .types import Extraction


@dataclass
class VerifyResult:
    ok: bool
    reasons: list[str] = field(default_factory=list)
    analysis_flags: dict[str, bool] = field(default_factory=dict)
    distinction: str = "other"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def verify_formalization(
    *,
    analysis: str,
    extraction: Extraction,
    grounding: dict[str, Any],
    case_id: str = "",
    gold: str = "",
) -> VerifyResult:
    """Return ok=False when analysis carries a distinction the structure drops."""
    distinction = expected_distinction(case_id, gold) if (case_id or gold) else "other"
    flags = analysis_flags(analysis)
    if distinction == "other":
        if flags["mentions_contradiction"]:
            distinction = "contradiction"
        elif flags["mentions_uncertainty"]:
            distinction = "uncertainty"

    reasons: list[str] = []
    if distinction in {"contradiction", "uncertainty"}:
        has = analysis_has_distinction(analysis, distinction)
        preserved = structure_preserves(extraction, grounding, distinction)
        if has and not preserved:
            reasons.append(
                f"analysis indicates {distinction} but formalization does not preserve it"
            )

    if extraction.contradiction_present is True:
        if not extraction.contradiction_cues or not (
            extraction.contradiction_cues[0].a and extraction.contradiction_cues[0].b
        ):
            reasons.append("contradiction.present=true but span pair missing")

    return VerifyResult(
        ok=len(reasons) == 0,
        reasons=reasons,
        analysis_flags=flags,
        distinction=distinction,
    )


def selftest_verify() -> None:
    from .types import ContradictionCue

    analysis = "The note contains mutually incompatible claims that contradict each other."
    ex_bad = Extraction(
        stated_line="first",
        contradiction_present=False,
        uncertainty_present=False,
    )
    g_bad = {
        "first_line_identified": "true",
        "first_line_administered": "true",
        "failure_event": "true",
        "failure_of_first_line": "true",
        "contradiction": False,
        "uncertainty": False,
    }
    r1 = verify_formalization(
        analysis=analysis,
        extraction=ex_bad,
        grounding=g_bad,
        case_id="C9",
        gold="CONTRADICTION",
    )
    assert not r1.ok, r1
    assert any("contradiction" in x for x in r1.reasons)

    ex_ok = Extraction(
        stated_line="first",
        contradiction_present=True,
        contradiction_cues=[ContradictionCue(a="progressed", b="responding")],
        uncertainty_present=False,
    )
    g_ok = {**g_bad, "contradiction": True}
    r2 = verify_formalization(
        analysis=analysis,
        extraction=ex_ok,
        grounding=g_ok,
        case_id="C9",
        gold="CONTRADICTION",
    )
    assert r2.ok, r2

    u_analysis = "Outcome remains uncertain; confirmatory scan is pending."
    ex_u_bad = Extraction(
        stated_line="first",
        administration_status="given",
        contradiction_present=False,
        uncertainty_present=False,
    )
    g_u_bad = {
        "first_line_identified": "true",
        "first_line_administered": "true",
        "failure_event": "true",
        "failure_of_first_line": "true",
        "contradiction": False,
        "uncertainty": False,
    }
    r3 = verify_formalization(
        analysis=u_analysis,
        extraction=ex_u_bad,
        grounding=g_u_bad,
        case_id="U9",
        gold="UNCERTAIN",
    )
    assert not r3.ok, r3
    print("verify selftest OK")


if __name__ == "__main__":
    selftest_verify()
