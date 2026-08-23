from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any


class Verdict(str, Enum):
    SATISFIED = "SATISFIED"
    NOT_SATISFIED = "NOT_SATISFIED"
    UNCERTAIN = "UNCERTAIN"
    CONTRADICTION = "CONTRADICTION"


class Atom(str, Enum):
    TRUE = "true"
    FALSE = "false"
    UNKNOWN = "unknown"


@dataclass
class OutcomeStatement:
    text: str
    polarity: str  # failure | response | ongoing | mixed | planned | unknown


@dataclass
class ContradictionCue:
    a: str
    b: str


@dataclass
class UncertaintyField:
    present: bool | None
    cue: str = ""
    why: str = ""


def _parse_bool(value: object) -> bool | None:
    if value is None or value == "":
        return None
    if isinstance(value, bool):
        return value
    raw = str(value).strip().lower()
    if raw in {"true", "1", "yes"}:
        return True
    if raw in {"false", "0", "no", "none"}:
        return False
    return None


def _parse_contradiction(data: dict[str, Any]) -> tuple[bool | None, list[ContradictionCue]]:
    """Accept nested contradiction {present, span_a, span_b} or a cues list."""
    block = data.get("contradiction")
    if isinstance(block, dict):
        present = _parse_bool(block.get("present"))
        a = str(block.get("span_a") or block.get("a") or "").strip()
        b = str(block.get("span_b") or block.get("b") or "").strip()
        cues = [ContradictionCue(a=a, b=b)] if (a or b) else []
        if present is False:
            cues = []
        return present, cues
    cues = [
        ContradictionCue(a=str(item.get("a", "")), b=str(item.get("b", "")))
        for item in data.get("contradiction_cues") or []
        if isinstance(item, dict)
    ]
    present = _parse_bool(data.get("contradiction_present"))
    if present is None and cues:
        present = True
    return present, cues


def _parse_uncertainty(data: dict[str, Any]) -> UncertaintyField:
    """Accept nested uncertainty {present, cue, why} or flat keys / cue list."""
    block = data.get("uncertainty")
    if isinstance(block, dict):
        present = _parse_bool(block.get("present"))
        cue = str(block.get("cue") or "").strip()
        why = str(block.get("why") or "").strip()
        if present is False:
            return UncertaintyField(present=False, cue="", why="")
        return UncertaintyField(present=present, cue=cue, why=why)
    cues = [str(x) for x in data.get("uncertainty_cues") or [] if str(x).strip()]
    present = _parse_bool(data.get("uncertainty_present"))
    if present is None and cues:
        present = True
    cue = str(data.get("uncertainty_cue") or (cues[0] if cues else "")).strip()
    why = str(data.get("uncertainty_why") or "").strip()
    return UncertaintyField(present=present, cue=cue, why=why)


@dataclass
class Extraction:
    """Descriptive neural output — not a verdict."""

    stated_line: str  # first | second | adjuvant | unspecified | unknown
    regimen_names: list[str] = field(default_factory=list)
    administration_status: str = "unknown"  # given | planned | not_given | unknown
    outcome_statements: list[OutcomeStatement] = field(default_factory=list)
    implicit_cues: list[str] = field(default_factory=list)
    uncertainty_cues: list[str] = field(default_factory=list)
    uncertainty_present: bool | None = None
    uncertainty_cue: str = ""
    uncertainty_why: str = ""
    contradiction_cues: list[ContradictionCue] = field(default_factory=list)
    contradiction_present: bool | None = None
    quotes: list[str] = field(default_factory=list)
    notes: str = ""
    backend: str = "unknown"

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        first = self.contradiction_cues[0] if self.contradiction_cues else None
        payload["contradiction"] = {
            "present": self.contradiction_present,
            "span_a": first.a if first else None,
            "span_b": first.b if first else None,
        }
        payload["uncertainty"] = {
            "present": self.uncertainty_present,
            "cue": self.uncertainty_cue or None,
            "why": self.uncertainty_why or None,
        }
        return payload

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Extraction":
        outcomes = [
            OutcomeStatement(
                text=str(item.get("text", "")),
                polarity=str(item.get("polarity", "unknown")),
            )
            for item in data.get("outcome_statements") or []
        ]
        present, contradictions = _parse_contradiction(data)
        uncertainty = _parse_uncertainty(data)
        cues = [str(x) for x in data.get("uncertainty_cues") or [] if str(x).strip()]
        if uncertainty.why and uncertainty.why not in cues:
            cues.append(uncertainty.why)
        if uncertainty.cue and uncertainty.cue not in cues and uncertainty.present is not False:
            cues.append(uncertainty.cue)
        return cls(
            stated_line=str(data.get("stated_line") or "unknown"),
            regimen_names=[str(x) for x in data.get("regimen_names") or []],
            administration_status=str(data.get("administration_status") or "unknown"),
            outcome_statements=outcomes,
            implicit_cues=[str(x) for x in data.get("implicit_cues") or []],
            uncertainty_cues=cues,
            uncertainty_present=uncertainty.present,
            uncertainty_cue=uncertainty.cue,
            uncertainty_why=uncertainty.why,
            contradiction_cues=contradictions,
            contradiction_present=present,
            quotes=[str(x) for x in data.get("quotes") or []],
            notes=str(data.get("notes") or ""),
            backend=str(data.get("backend") or "unknown"),
        )


@dataclass
class Grounding:
    first_line_identified: Atom
    first_line_administered: Atom
    failure_event: Atom
    failure_of_first_line: Atom
    contradiction: bool
    uncertainty: bool = False
    trace: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "first_line_identified": self.first_line_identified.value,
            "first_line_administered": self.first_line_administered.value,
            "failure_event": self.failure_event.value,
            "failure_of_first_line": self.failure_of_first_line.value,
            "contradiction": self.contradiction,
            "uncertainty": self.uncertainty,
            "trace": list(self.trace),
        }


@dataclass
class RuleResult:
    verdict: Verdict
    fired_rule: str
    trace: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "verdict": self.verdict.value,
            "fired_rule": self.fired_rule,
            "trace": list(self.trace),
        }


@dataclass
class BaselineResult:
    verdict: Verdict
    rationale: str
    raw: str
    backend: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "verdict": self.verdict.value,
            "rationale": self.rationale,
            "raw": self.raw,
            "backend": self.backend,
        }
