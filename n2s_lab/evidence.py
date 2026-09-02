"""Evidence-preservation object: claims with source spans and verification.

Phase-7: every formal proposition should carry justification traceable to source text.
Fail → REVIEW (not UNCERTAIN).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

from .types import Extraction, Verdict


@dataclass
class EvidenceClaim:
    proposition: str
    status: str  # supported | unsupported | not_asserted
    source_id: str = ""
    evidence: str = ""
    relationship: str = ""
    qualifiers: dict[str, str] = field(default_factory=dict)
    verification: str = "PENDING"  # PASS | FAIL | SKIP

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class EvidenceVerifyResult:
    ok: bool
    claims: list[EvidenceClaim] = field(default_factory=list)
    reasons: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "ok": self.ok,
            "claims": [c.to_dict() for c in self.claims],
            "reasons": self.reasons,
        }


def _span_in_source(span: str, source: str) -> bool:
    if not span or not source:
        return False
    a = span.strip().lower()
    b = source.lower()
    if len(a) < 8:
        return a in b
    # Allow partial match on longer spans
    words = [w for w in a.split() if len(w) > 3]
    if not words:
        return a in b
    hits = sum(1 for w in words if w in b)
    return hits >= max(1, len(words) * 2 // 3)


def claims_from_extraction(
    extraction: Extraction,
    grounding: dict[str, Any],
    *,
    source_id: str = "note",
    source_text: str = "",
) -> list[EvidenceClaim]:
    """Build claim objects from extraction + grounding (conceptual evidence object)."""
    claims: list[EvidenceClaim] = []

    def add(prop: str, asserted: bool, evidence: str, rel: str = "") -> None:
        claims.append(
            EvidenceClaim(
                proposition=prop,
                status="supported" if asserted else "not_asserted",
                source_id=source_id,
                evidence=evidence[:240] if evidence else "",
                relationship=rel,
                qualifiers={"certainty": "definite" if asserted else "n/a"},
            )
        )

    quotes = extraction.quotes or []
    quote0 = quotes[0] if quotes else source_text[:160]

    add(
        "first_line_identified",
        grounding.get("first_line_identified") == "true",
        quote0,
        "stated_line_or_regimen",
    )
    add(
        "first_line_administered",
        grounding.get("first_line_administered") == "true",
        quote0,
        "administration_status",
    )
    add(
        "failure_event",
        grounding.get("failure_event") == "true",
        " ".join(o.text for o in extraction.outcome_statements[:2]) or quote0,
        "outcome_polarity",
    )
    add(
        "failure_of_first_line",
        grounding.get("failure_of_first_line") == "true",
        quote0,
        "failure_attributed_to_first_line",
    )

    if extraction.contradiction_present is True:
        span_a = extraction.contradiction_cues[0].a if extraction.contradiction_cues else ""
        span_b = extraction.contradiction_cues[0].b if extraction.contradiction_cues else ""
        claims.append(
            EvidenceClaim(
                proposition="contradiction",
                status="supported",
                source_id=source_id,
                evidence=f"{span_a} | {span_b}".strip(" |"),
                relationship="incompatible_spans",
                qualifiers={"certainty": "definite"},
            )
        )
    elif extraction.contradiction_present is False:
        add("contradiction_absent", True, quote0, "explicit_none")

    if extraction.uncertainty_present is True:
        cue = extraction.uncertainty_cue or quote0
        claims.append(
            EvidenceClaim(
                proposition="uncertainty",
                status="supported",
                source_id=source_id,
                evidence=cue,
                relationship=extraction.uncertainty_why or "hedged_evidence",
                qualifiers={"certainty": "hedged"},
            )
        )

    return claims


def verify_claims_against_source(
    claims: list[EvidenceClaim],
    source_text: str,
) -> EvidenceVerifyResult:
    """Check that supported claims have evidence traceable in source text."""
    reasons: list[str] = []
    for claim in claims:
        if claim.status != "supported":
            claim.verification = "SKIP"
            continue
        if not claim.evidence:
            claim.verification = "FAIL"
            reasons.append(f"{claim.proposition}: no evidence span recorded")
            continue
        if claim.proposition == "contradiction":
            parts = [p.strip() for p in claim.evidence.split("|") if p.strip()]
            ok = len(parts) >= 2 and all(_span_in_source(p, source_text) for p in parts[:2])
        else:
            ok = _span_in_source(claim.evidence, source_text)
        claim.verification = "PASS" if ok else "FAIL"
        if not ok:
            reasons.append(
                f"{claim.proposition}: evidence not traceable in source"
            )
    return EvidenceVerifyResult(ok=len(reasons) == 0, claims=claims, reasons=reasons)


def gate_verdict_with_evidence(
    *,
    verdict: str,
    extraction: Extraction,
    grounding: dict[str, Any],
    source_text: str,
    source_id: str = "note",
) -> dict[str, Any]:
    """If verdict is AUTO, require evidence claims to verify; else REVIEW."""
    if verdict == Verdict.REVIEW.value:
        return {
            "verdict": Verdict.REVIEW.value,
            "disposition": "REVIEW",
            "evidence": None,
            "skipped": "already_REVIEW",
        }
    claims = claims_from_extraction(
        extraction, grounding, source_id=source_id, source_text=source_text
    )
    ev = verify_claims_against_source(claims, source_text)
    if not ev.ok:
        return {
            "verdict": Verdict.REVIEW.value,
            "disposition": "REVIEW",
            "evidence": ev.to_dict(),
            "note": "evidence claims not traceable — REVIEW",
        }
    return {
        "verdict": verdict,
        "disposition": "AUTO",
        "evidence": ev.to_dict(),
    }


def selftest_evidence() -> None:
    ex = Extraction(
        stated_line="first",
        administration_status="given",
        outcome_statements=[],
        contradiction_present=True,
        contradiction_cues=[],
        uncertainty_present=False,
    )
    from .types import ContradictionCue

    source = "Morning note: failed first-line R-CHOP. Evening addendum: continue R-CHOP with stable disease."
    ex.contradiction_cues = [
        ContradictionCue(a="failed first-line R-CHOP", b="continue R-CHOP with stable disease")
    ]
    g = {
        "first_line_identified": "true",
        "first_line_administered": "true",
        "failure_event": "true",
        "failure_of_first_line": "true",
        "contradiction": True,
        "uncertainty": False,
    }
    claims = claims_from_extraction(ex, g, source_text=source)
    ev = verify_claims_against_source(claims, source)
    assert ev.ok, ev

    bad = verify_claims_against_source(claims, "unrelated note with no matching spans")
    assert not bad.ok, bad

    print("evidence selftest OK")


if __name__ == "__main__":
    selftest_evidence()
