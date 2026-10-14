"""Scientific delivery policy for the report_research strategy."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


@dataclass(frozen=True)
class ReportPolicy:
    key: Literal["scientific_report"]
    citations_required: bool
    enumerative_coverage: bool
    scientific_delivery: bool
    network_enabled: bool = True
    max_requirements: int = 48
    max_recovery_requirements: int = 6
    max_recovery_agents: int = 3
    requirement_prompt: str = ""
    verification_prompt: str = ""
    # Scientific delivery instructions carried into worker delivery
    # contexts, gap-recovery assignments, and the finalization payload. Empty
    # means the shared prompts alone define delivery expectations.
    delivery_directive: str = ""


# Mirror the question's parts instead of diluting their depth into micro-requirements.
_SCIENTIFIC_REQUIREMENT_PROMPT = (
    "Extract every scientific sub-question, derivation, method, and assumption. Create one "
    "requirement per part or sub-question the question lists — never split a single part into "
    "micro-requirements — and name in the acceptance criteria the specific quantities, "
    "equations, mechanisms, or named entities that part demands."
)

# Specificity gate distilled from adaptive_htn's measured FSR failure modes:
# on-topic-but-generic prose, uncommitted hedging, and specific results that
# the gathered findings contained but the deliverable dropped or genericized.
_SCIENTIFIC_VERIFICATION_PROMPT = (
    "First reopen load-bearing upstream identities, organism/system choices, methods and model "
    "assumptions against the public question. Separate given facts from worker hypotheses; prior "
    "agreement is not independent validation. Correct affected downstream conclusions when a "
    "premise fails. Recomputing arithmetic or weighted rankings does not validate their premises. "
    "Check sub-question coverage, equations, methods, assumptions, and concrete claims. Apply a "
    "specificity gate — a claim is unmet when it is GENERIC (it could be written from a textbook "
    "alone and would read identically for any other instance of the same topic), UNCOMMITTED "
    "(the evidence establishes a definite value, choice, or mechanism but the text avoids "
    "stating it), or DROPPED FROM FINDINGS "
    "(a specific value, named entity, equation, or conclusion present in the gathered evidence "
    "is missing or survives only as a generic paraphrase). Resolve every gene locus, cell line, "
    "protein, reagent, and vector in the exact organism and system the question names; a match "
    "in the wrong species is a wrong answer, not a partial one. For requested computations, "
    "check dimensions, boundary conditions, numerical convergence and reproducibility. "
    "Distinguish executed calculations from proposed code."
)

_SCIENTIFIC_DELIVERY_DIRECTIVE = (
    "Develop every requested part fully and correctly; do not omit a part, compress it to a "
    "one-line assertion, or leave it generic. Depth over breadth: add no unrequested "
    "background, literature review, or Markdown table (use a table only where the question "
    "explicitly asks for one). End every sub-question with a specific, evidence-supported "
    "conclusion tied to this problem. State a definite answer when justified; when constraints "
    "are insufficient or evidence conflicts, give a conditional result or the supported "
    "alternatives, identify the missing assumption or evidence, and flag the uncertainty. "
    "Do not force a unique answer. Mark unsupported parts as uncovered while preserving "
    "supported partial results. Write out every "
    "derivation step and equation the question asks for, line by line, in the canonical form "
    "the question or the field uses; never compress a requested step to 'it follows that' and "
    "never leave a requested equation implied. Carry every specific "
    "result: each exact value, named entity, equation, mechanism, and derivation established in "
    "the gathered evidence, unless refuted by premise checks, must appear in the deliverable, never softened into a generic "
    "restatement. Show each intermediate quantity with its unit on the way to a final value, and "
    "use every numeric quantity the question supplies (or state explicitly why one is not "
    "needed). When the question requests numerical work, include reproducible code, actual "
    "computed results and convergence checks. When figures are requested, include essential "
    "numerical data and figure descriptions in the report itself; file paths alone do not "
    "establish results. State computational limitations and never claim unexecuted code ran."
)

SCIENTIFIC_REPORT_POLICY = ReportPolicy(
    key="scientific_report",
    citations_required=False,
    enumerative_coverage=False,
    scientific_delivery=True,
    max_requirements=24,
    requirement_prompt=_SCIENTIFIC_REQUIREMENT_PROMPT,
    verification_prompt=_SCIENTIFIC_VERIFICATION_PROMPT,
    delivery_directive=_SCIENTIFIC_DELIVERY_DIRECTIVE,
)
