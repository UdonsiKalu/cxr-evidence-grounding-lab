"""Neural components: structured extraction and direct-LLM baseline.

The extractor must describe the note. It must not output SATISFIED / etc.
The baseline *does* output a verdict — that is the comparison point.
"""

from __future__ import annotations

import re

from .ollama_client import DEFAULT_MODEL, chat_json, chat_text
from .predicate import PREDICATE_ID, PREDICATE_TEXT
from .types import BaselineResult, Extraction, OutcomeStatement, Verdict

EXTRACT_SYSTEM = """You extract structured cues from a short oncology note.
Do not decide whether a formal policy predicate holds.
Do not output SATISFIED, NOT_SATISFIED, UNCERTAIN, or CONTRADICTION as a verdict.

Contradiction is a required descriptive field, not a verdict:
- Set contradiction.present=true only when the note asserts two claims that cannot both be true about the SAME event (failed vs still responding now; never treated vs six cycles given; assessment vs reversing addendum the same day). Copy those two spans.
- Do not drop the earlier span: intake "never received systemic therapy" vs a later note of cycles given is contradiction, not an update.
- Set contradiction.present=false, span_a=null, span_b=null when there is no such pair.
- Sequenced events are NOT a contradiction (responded, then later progressed).
- Unresolved alternatives are NOT a contradiction: which line (metastatic first-line vs adjuvant), mixed response, pending scan, possible, unknown outcome. Put those in uncertainty and set contradiction.present=false.
Always fill contradiction.present as true or false. Do not omit it.

Uncertainty is a required descriptive field, not a verdict:
- Set uncertainty.present=true when the note itself flags insufficient or unresolved evidence about line, administration, or outcome. Copy one cue span and a why from the closed list.
- Examples that ARE uncertainty: possible progression awaiting confirmatory scan; may have received therapy / outcome unknown; mixed response; questionable whether metastatic first-line vs adjuvant.
- Set uncertainty.present=false, cue=null, why=null when the note is a resolved assertion (including implicit but clear failure, planned-only therapy, adjuvant completion, temporary hold with plan to resume, sequenced response-then-progression).
- Contradiction is NOT uncertainty. If contradiction.present=true, set uncertainty.present=false.
Always fill uncertainty.present as true or false. Do not omit it.
Return JSON only, matching the schema in the user message."""

EXTRACT_SCHEMA = """{
  "stated_line": "first|second|adjuvant|unspecified|unknown",
  "regimen_names": ["string"],
  "administration_status": "given|planned|not_given|unknown",
  "outcome_statements": [{"text": "short quote or paraphrase", "polarity": "failure|response|ongoing|mixed|planned|unknown"}],
  "implicit_cues": ["switching_to_second_line", "platinum_refractory", "now_second_line", "response_then_progression", "temporary_hold", "adjuvant_completed"],
  "uncertainty_cues": ["possible", "awaiting_scan", "may_have", "questionable", "mixed", "unknown_outcome"],
  "uncertainty": {
    "present": "true or false",
    "cue": "verbatim or close paraphrase of the uncertain claim, or null",
    "why": "unknown_outcome|unresolved_line|mixed|pending|possible|may_have|null"
  },
  "contradiction": {
    "present": "true or false",
    "span_a": "verbatim or close paraphrase of claim A, or null",
    "span_b": "incompatible claim B, or null"
  },
  "quotes": ["verbatim spans, max 160 chars each"],
  "notes": "one sentence, descriptive only"
}"""

BASELINE_SYSTEM = f"""You classify whether a short clinical note satisfies a formal condition.
Allowed verdicts only: SATISFIED, NOT_SATISFIED, UNCERTAIN, CONTRADICTION.
Use CONTRADICTION when the note contains mutually incompatible claims.
Use UNCERTAIN when the evidence is insufficient or mixed, not as a hedge for SATISFIED.
Return JSON only: {{"verdict": "...", "rationale": "one or two sentences"}}

Formal condition {PREDICATE_ID}:
{PREDICATE_TEXT}
"""


def extract_live(evidence: str, *, model: str = DEFAULT_MODEL) -> Extraction:
    data = chat_json(
        f"Schema:\n{EXTRACT_SCHEMA}\n\nNote:\n{evidence}",
        system=EXTRACT_SYSTEM,
        model=model,
        num_predict=600,
    )
    extraction = Extraction.from_dict(data)
    extraction.backend = f"ollama:{model}"
    return extraction


REPAIR_EXTRACT_SUFFIX = """
VERIFICATION FAILED on the previous JSON. The free-text analysis (or note) indicates a
semantic distinction that your JSON dropped. Repair the extraction so contradiction /
uncertainty fields faithfully reflect that distinction. Do not invent a policy verdict.
Do not set uncertainty.present=true merely because verification failed — only if the
source text itself is uncertain. Return corrected JSON only matching the schema.
"""


def extract_live_repair(
    evidence: str,
    *,
    model: str = DEFAULT_MODEL,
    prior: Extraction,
    verify_reasons: list[str],
) -> Extraction:
    """Phase-5: one-shot repair after verify fail (Condition C path)."""
    reasons = "; ".join(verify_reasons) or "structure/analysis mismatch"
    data = chat_json(
        f"Schema:\n{EXTRACT_SCHEMA}\n\nNote:\n{evidence}\n\n"
        f"Previous JSON:\n{prior.to_dict()}\n\nVerify reasons: {reasons}\n"
        f"{REPAIR_EXTRACT_SUFFIX}",
        system=EXTRACT_SYSTEM,
        model=model,
        num_predict=600,
    )
    extraction = Extraction.from_dict(data)
    extraction.backend = f"ollama:{model}:repair"
    return extraction


def extract_from_analysis(analysis: str, *, model: str = DEFAULT_MODEL) -> Extraction:
    """Condition D step 2: structured extraction from analysis text (not from the note)."""
    data = chat_json(
        f"Schema:\n{EXTRACT_SCHEMA}\n\nSemantic analysis to convert:\n{analysis}",
        system=ANALYSIS_TO_EXTRACT_SYSTEM,
        model=model,
        num_predict=600,
    )
    extraction = Extraction.from_dict(data)
    extraction.backend = f"ollama:{model}:from-analysis"
    return extraction


def extract_from_analysis_repair(
    analysis: str,
    *,
    model: str = DEFAULT_MODEL,
    prior: Extraction,
    verify_reasons: list[str],
) -> Extraction:
    """Phase-5: one-shot repair after verify fail (Condition D path)."""
    reasons = "; ".join(verify_reasons) or "structure/analysis mismatch"
    data = chat_json(
        f"Schema:\n{EXTRACT_SCHEMA}\n\nSemantic analysis to convert:\n{analysis}\n\n"
        f"Previous JSON:\n{prior.to_dict()}\n\nVerify reasons: {reasons}\n"
        f"{REPAIR_EXTRACT_SUFFIX}",
        system=ANALYSIS_TO_EXTRACT_SYSTEM,
        model=model,
        num_predict=600,
    )
    extraction = Extraction.from_dict(data)
    extraction.backend = f"ollama:{model}:from-analysis-repair"
    return extraction


def baseline_live(evidence: str, *, model: str = DEFAULT_MODEL) -> BaselineResult:
    data = chat_json(
        f"Note:\n{evidence}",
        system=BASELINE_SYSTEM,
        model=model,
        num_predict=250,
    )
    verdict = _parse_verdict(data.get("verdict"))
    return BaselineResult(
        verdict=verdict,
        rationale=str(data.get("rationale") or ""),
        raw=str(data),
        backend=f"ollama:{model}",
    )


# --- Optional Phase-1 diagnostic helpers (additive; M1–M3 paths do not call these) ---

ANALYSIS_SYSTEM = f"""You write a short plain-language semantic analysis of a clinical note
for the formal condition {PREDICATE_ID}.

Describe only what the note supports about:
- whether a first-line therapy is identified
- whether it was given (not merely planned)
- whether a failure event occurred
- whether that failure is of the first-line
- whether claims conflict
- whether evidence is uncertain, incomplete, or mixed

Do NOT output a final categorical verdict (do not write SATISFIED, NOT_SATISFIED,
UNCERTAIN, or CONTRADICTION as a label). Use ordinary prose. Be concrete; quote short spans.
"""

ANALYSIS_TO_EXTRACT_SYSTEM = """You convert a free-text semantic analysis of a clinical note into
structured extraction JSON. Use only information stated in the analysis (and short quotes it
contains). Do not invent facts. Do not output a predicate verdict.
Fill contradiction.present and uncertainty.present as true or false. Return JSON only."""

VERDICT_FROM_ANALYSIS_SYSTEM = f"""You read a free-text semantic analysis of a clinical note and
output only a categorical verdict for {PREDICATE_ID}.
Allowed verdicts: SATISFIED, NOT_SATISFIED, UNCERTAIN, CONTRADICTION.
Return JSON only: {{"verdict": "...", "rationale": "one sentence citing the analysis"}}
"""

# B′ ablation only. Frozen Phase-1/2 B prompt above must stay unchanged.
VERDICT_FROM_ANALYSIS_SYSTEM_B_PRIME = f"""You read a free-text semantic analysis of a clinical note and
output only a categorical verdict for {PREDICATE_ID}.
Allowed verdicts: SATISFIED, NOT_SATISFIED, UNCERTAIN, CONTRADICTION.

The analysis is prose. It was told not to emit those four labels. You must map the
prose onto one label. Use only the analysis, not an original note.

Mapping:
- If the analysis describes mutually incompatible claims about the same facts,
  output CONTRADICTION. Do not pick SATISFIED or NOT_SATISFIED from one side.
- If the analysis describes uncertainty, incompleteness, mixed findings, pending
  confirmation, or unresolved which-line/setting, output UNCERTAIN. Do not collapse
  that to SATISFIED or NOT_SATISFIED.
- Output SATISFIED or NOT_SATISFIED only when the analysis does not describe
  conflict or unresolved uncertainty.

Return JSON only: {{"verdict": "...", "rationale": "one sentence citing the analysis"}}
"""


def analyze_free_text(evidence: str, *, model: str = DEFAULT_MODEL) -> str:
    """Condition B/D step 1: unconstrained prose analysis (no format:json)."""
    return chat_text(
        f"Formal condition:\n{PREDICATE_TEXT}\n\nNote:\n{evidence}",
        system=ANALYSIS_SYSTEM,
        model=model,
        num_predict=450,
    )


def verdict_from_analysis(
    analysis: str,
    *,
    model: str = DEFAULT_MODEL,
    system: str | None = None,
    variant: str = "B",
) -> BaselineResult:
    """Condition B step 2: categorical verdict from analysis only.

    Default system is the frozen Phase-1/2 prompt. Pass
    VERDICT_FROM_ANALYSIS_SYSTEM_B_PRIME for the B′ ablation.
    """
    sys_prompt = system if system is not None else VERDICT_FROM_ANALYSIS_SYSTEM
    data = chat_json(
        f"Semantic analysis:\n{analysis}",
        system=sys_prompt,
        model=model,
        num_predict=200,
    )
    tag = "from-analysis-bprime" if variant == "B_prime" else "from-analysis"
    return BaselineResult(
        verdict=_parse_verdict(data.get("verdict")),
        rationale=str(data.get("rationale") or ""),
        raw=str(data),
        backend=f"ollama:{model}:{tag}",
    )


def extract_mock(evidence: str) -> Extraction:
    """Keyword heuristic so the symbolic path can run without a model."""
    text = evidence.lower()
    implicit: list[str] = []
    uncertainty: list[str] = []
    contradictions = []
    outcomes = []

    if "adjuvant" in text:
        stated_line = "adjuvant"
        implicit.append("adjuvant_completed")
    elif "first-line" in text or "first line" in text:
        stated_line = "first"
    elif "second-line" in text or "second line" in text:
        stated_line = "second"
    else:
        stated_line = "unspecified"

    if "plan to start" in text or "starting first-line" in text or "starting first line" in text:
        admin = "planned"
        outcomes.append(OutcomeStatement(text="planned start", polarity="planned"))
    elif re.search(r"\bnever received|never started|no prior systemic\b", text):
        admin = "not_given"
    elif re.search(
        r"\b(progressed|stopped|completed|receiving|initiated|on first-line|on first line|after carboplatin)\b",
        text,
    ):
        admin = "given"
    else:
        admin = "unknown"

    if "platinum-refractory" in text or "platinum refractory" in text:
        implicit.append("platinum_refractory")
        outcomes.append(OutcomeStatement(text="platinum-refractory", polarity="failure"))
    if "switching to second-line" in text or "switch to second-line" in text or "proceed to second-line" in text:
        implicit.append("switching_to_second_line")
    if "currently on second-line" in text:
        implicit.append("now_second_line")
    if "initial" in text and "progress" in text:
        implicit.append("response_then_progression")
    if "responded" in text and "then progressed" in text:
        implicit.append("response_then_progression")
    if "held last week" in text or "plan to resume" in text:
        implicit.append("temporary_hold")

    if re.search(r"\bprogressed|failed|failure|refractory|discontinued for intolerance|stopped after\b", text):
        outcomes.append(OutcomeStatement(text="failure language", polarity="failure"))
    if re.search(r"\bpartial response|ongoing response|stable disease|excellent ongoing|remains with no evidence|continue current\b", text):
        outcomes.append(OutcomeStatement(text="response/ongoing language", polarity="ongoing"))
    if "mixed response" in text:
        outcomes.append(OutcomeStatement(text="mixed response", polarity="mixed"))
        uncertainty.append("mixed")
    if "no evidence of progression" in text and "held" in text:
        outcomes.append(OutcomeStatement(text="no progression on hold", polarity="ongoing"))

    if "possible progression" in text or "awaiting confirmatory" in text:
        uncertainty.append("possible")
        uncertainty.append("awaiting_scan")
    if "may have received" in text or "outcome" in text and "unknown" in text:
        uncertainty.append("may_have")
        uncertainty.append("unknown_outcome")
    if "questionable whether" in text:
        uncertainty.append("questionable")

    why = ""
    cue = ""
    if "possible progression" in text or "awaiting confirmatory" in text:
        why = "pending" if "awaiting" in text else "possible"
        cue = "Possible progression" if "possible progression" in text else "awaiting confirmatory scan"
    elif "may have received" in text:
        why = "may_have"
        cue = "may have received platinum-based therapy"
    elif "outcome" in text and "unknown" in text:
        why = "unknown_outcome"
        cue = "Outcome of that treatment is unknown"
    elif "mixed response" in text:
        why = "mixed"
        cue = "Mixed response"
    elif "questionable whether" in text:
        why = "unresolved_line"
        cue = "Questionable whether carboplatin was given as first-line metastatic therapy or in the adjuvant setting"

    from .types import ContradictionCue

    parts = re.split(
        r"(?:Addendum later the same day:|Later oncology note:|Our chart review:|Later note:)",
        evidence,
        flags=re.I,
    )
    if len(parts) >= 2:
        contradictions.append(ContradictionCue(a=parts[0].strip()[:180], b=parts[1].strip()[:180]))
    elif (
        re.search(r"\bnever (received|started)\b", text)
        and re.search(r"\b(progressed|failed first-line|failed first line)\b", text)
    ):
        contradictions.append(
            ContradictionCue(a="never received/started systemic therapy", b="failed or progressed on first-line")
        )

    regimens = []
    for name in (
        "carboplatin/paclitaxel",
        "carboplatin and paclitaxel",
        "cisplatin/gemcitabine",
        "pembrolizumab",
        "folfox",
        "folfirinox",
        "nivolumab",
        "osimertinib",
        "carboplatin/pemetrexed",
        "carboplatin/etoposide",
        "topotecan",
        "docetaxel",
    ):
        if name in text:
            regimens.append(name)

    return Extraction(
        stated_line=stated_line,
        regimen_names=regimens,
        administration_status=admin,
        outcome_statements=outcomes,
        implicit_cues=sorted(set(implicit)),
        uncertainty_cues=sorted(set(uncertainty)),
        uncertainty_present=bool(uncertainty) and not contradictions,
        uncertainty_cue=cue if not contradictions else "",
        uncertainty_why=why if not contradictions else "",
        contradiction_cues=contradictions,
        contradiction_present=bool(contradictions),
        quotes=[evidence[:160]],
        notes="heuristic mock extractor",
        backend="mock-heuristic",
    )


def _parse_verdict(value: object) -> Verdict:
    raw = str(value or "").strip().upper().replace(" ", "_")
    aliases = {
        "SATISFIED": Verdict.SATISFIED,
        "TRUE": Verdict.SATISFIED,
        "YES": Verdict.SATISFIED,
        "NOT_SATISFIED": Verdict.NOT_SATISFIED,
        "UNSATISFIED": Verdict.NOT_SATISFIED,
        "FALSE": Verdict.NOT_SATISFIED,
        "NO": Verdict.NOT_SATISFIED,
        "UNCERTAIN": Verdict.UNCERTAIN,
        "UNKNOWN": Verdict.UNCERTAIN,
        "ABSTAIN": Verdict.UNCERTAIN,
        "CONTRADICTION": Verdict.CONTRADICTION,
        "CONFLICT": Verdict.CONTRADICTION,
        "CONFLICTING": Verdict.CONTRADICTION,
    }
    if raw in aliases:
        return aliases[raw]
    return Verdict.UNCERTAIN
