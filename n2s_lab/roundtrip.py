"""Phase-6 L2b: round-trip faithfulness check (structure → paraphrase ↔ source).

Does not prove faithfulness. Fail → REVIEW (not UNCERTAIN).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

from .phase1_diagnostic import (
    analysis_flags,
    analysis_has_distinction,
    expected_distinction,
)
from .types import Extraction


@dataclass
class RoundTripResult:
    ok: bool
    reasons: list[str] = field(default_factory=list)
    paraphrase_flags: dict[str, bool] = field(default_factory=dict)
    source_flags: dict[str, bool] = field(default_factory=dict)
    distinction: str = "other"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def verify_roundtrip(
    *,
    source_text: str,
    paraphrase: str,
    extraction: Extraction,
    case_id: str = "",
    gold: str = "",
) -> RoundTripResult:
    """Check that a structure→paraphrase still carries distinctions asserted by structure/source."""
    distinction = expected_distinction(case_id, gold) if (case_id or gold) else "other"
    src_flags = analysis_flags(source_text)
    para_flags = analysis_flags(paraphrase)
    reasons: list[str] = []

    if distinction == "other":
        if src_flags["mentions_contradiction"]:
            distinction = "contradiction"
        elif src_flags["mentions_uncertainty"]:
            distinction = "uncertainty"

    # Structure asserted contradiction / uncertainty must appear in the paraphrase.
    if extraction.contradiction_present is True and not para_flags["mentions_contradiction"]:
        reasons.append(
            "extraction asserts contradiction but paraphrase lacks contradiction cues"
        )
    if extraction.uncertainty_present is True and not para_flags["mentions_uncertainty"]:
        reasons.append(
            "extraction asserts uncertainty but paraphrase lacks uncertainty cues"
        )

    # Source-indicated gold distinction should survive in the paraphrase.
    if distinction in {"contradiction", "uncertainty"}:
        if analysis_has_distinction(source_text, distinction) and not analysis_has_distinction(
            paraphrase, distinction
        ):
            reasons.append(
                f"source indicates {distinction} but paraphrase does not preserve it"
            )

    return RoundTripResult(
        ok=len(reasons) == 0,
        reasons=reasons,
        paraphrase_flags=para_flags,
        source_flags=src_flags,
        distinction=distinction,
    )


def dual_path_verdict(v_c: str, v_d: str) -> dict[str, Any]:
    """L2c: C and D paths must agree when both AUTO; else REVIEW."""
    from .types import Verdict

    review = Verdict.REVIEW.value
    if v_c == review or v_d == review:
        return {
            "verdict": review,
            "disposition": "REVIEW",
            "reason": "one_or_both_paths_already_REVIEW",
            "C": v_c,
            "D": v_d,
            "agreed": False,
        }
    if v_c == v_d:
        return {
            "verdict": v_c,
            "disposition": "AUTO",
            "reason": "paths_agree",
            "C": v_c,
            "D": v_d,
            "agreed": True,
        }
    return {
        "verdict": review,
        "disposition": "REVIEW",
        "reason": "paths_disagree",
        "C": v_c,
        "D": v_d,
        "agreed": False,
    }


def selftest_roundtrip() -> None:
    from .types import ContradictionCue

    ex = Extraction(
        stated_line="first",
        contradiction_present=True,
        contradiction_cues=[ContradictionCue(a="failed", b="continue")],
        uncertainty_present=False,
    )
    source = "Assessment failed therapy. Addendum: continue first-line with stable disease."
    bad_para = "Patient is on first-line therapy with stable disease."
    r1 = verify_roundtrip(
        source_text=source,
        paraphrase=bad_para,
        extraction=ex,
        case_id="C9",
        gold="CONTRADICTION",
    )
    assert not r1.ok, r1

    good_para = (
        "The record contains conflicting claims: therapy failed versus continue "
        "first-line with stable disease."
    )
    r2 = verify_roundtrip(
        source_text=source,
        paraphrase=good_para,
        extraction=ex,
        case_id="C9",
        gold="CONTRADICTION",
    )
    assert r2.ok, r2

    d = dual_path_verdict("CONTRADICTION", "CONTRADICTION")
    assert d["agreed"] and d["verdict"] == "CONTRADICTION"
    d2 = dual_path_verdict("CONTRADICTION", "UNCERTAIN")
    assert d2["verdict"] == "REVIEW" and d2["reason"] == "paths_disagree"
    d3 = dual_path_verdict("REVIEW", "CONTRADICTION")
    assert d3["verdict"] == "REVIEW"

    print("roundtrip/dual selftest OK")


if __name__ == "__main__":
    selftest_roundtrip()
