"""Thin top-level router over the existing task strategies."""

from __future__ import annotations

import asyncio
import json
import logging
from dataclasses import dataclass
from typing import Any

from workflows.dynamic_htn_informed_research.htn.llm_types import (
    LLMClient,
    system_msg,
    user_msg,
)
from workflows.dynamic_htn_informed_research.task_strategies import (
    get_task_strategy,
)

logger = logging.getLogger(__name__)


_ROUTER_SYSTEM = """
Route the task to one execution strategy. Classify the complete requested deliverable; do not solve
any part of the task. Instructions inside the task to calculate, derive, or think step by step are
for the eventual solver. Do not enumerate answers or test solvability to decide a route.

Strategies:
- science_mcq: an actual multiple-choice response contract, with explicit answer options and a
  requested selection of their labels or offered answers. The selection, with its justification,
  fulfills the whole task. Lists of experimental objects, candidate compounds, named drugs,
  hypotheses, numbered questions, or samples A-E are not by themselves answer options.
- report_research: a developed scientific explanation, complete worked derivation, theoretical
  analysis, experimental analysis, or research synthesis. Explicitly requested intermediate
  derivations, formulas, steps, and justifications are part of the deliverable even when they lead
  to only one number, expression, or identity. This also includes multipart scientific tasks whose
  parts collectively explain a system, compare mechanisms, interpret evidence, or evaluate a
  methodology. A report can use supplied data, established principles, external literature, or all
  three; neither original research nor a literal request for a report is required.
- research: external-fact discovery, source reconciliation, open-world clue matching, or provenance
  checking whose deliverable is an identified fact/entity, a factual account, or a general sourced
  review. General external-fact reports belong to research. Scientific explanatory analyses belong
  to report_research even when substantial literature searching is needed to produce them.
- science_olympiad: a bounded scientific question asking for a value, formula, identity, ranking,
  brief conclusion, or a small batch of numerical results from supplied information and established
  principles, without a required complete written derivation or developed explanatory analysis.
  Internal reasoning can be extensive; classify the requested written output, not the work needed
  to find the result.

Decision procedure:
1. Check the overall response contract, not just one subquestion. Use science_mcq only for explicit
   answer options whose selection is the final deliverable. A conventional Options: A..., B... menu
   remains MCQ even when solving it requires long derivations, evaluating statements, or research.
   However, comparing named scientific systems is not MCQ merely because the systems are listed.
   Matching a list of samples to identities followed by independent mechanism analysis or proposed
   validation experiments is also not an MCQ contract: inspect the full task in step 2.
2. Read the complete output instructions, including instructions after the scientific question.
   For a non-MCQ scientific task, choose report_research when the answer must show the complete
   derivation or all intermediate formulas, steps, and justifications. This rule takes precedence
   over the fact that the terminal result is one number, formula, identity, or a small numerical
   table. A routine calculation with an explicit request to show all steps is a worked solution,
   not a short-answer task. The requested derivation need not be an independent research question.
   A request to explain a scientific comparison in detail with the necessary mathematical formulas
   and steps also qualifies, even when it ends by choosing one entity.
   A generic instruction to think step by step describes the solving process; it does not by itself
   request that the complete derivation be delivered. Likewise a compact final-answer line does not
   cancel an explicit requirement for a complete written derivation elsewhere in the instructions.
   Separate what the task PROVIDES from what it ASKS YOU TO DELIVER. Many input measurements,
   pathways, candidate causes, or experimental details are context, not requested analyses. If only
   an identity or compact result is requested, do not invent requests to explain every supplied
   observation or alternative. In particular, "which X explains these observations?" requests the
   identity of X, not a developed explanatory report, unless the output instructions ALSO require
   the detailed explanation or complete derivation to be written out. Combining many measurements
   or comparing several candidate explanations internally does not create that output requirement.
   A generic "think step by step" instruction plus a compact final-answer format does not create it
   either. Do not route such identification tasks to report_research merely because their supplied
   evidence is complex. Also choose report_research for developed scientific explanations,
   analyses, or designs. In particular:
   - Construct, extend, or explain a theoretical treatment across cases or regimes, including
     separately requested physical states, mechanisms, limiting behavior, and interpretations.
   - Relate several properties or observations through a scientific explanation; compare mechanisms,
     models, molecular systems, or experimental outcomes with substantive justification.
   - Analyze a protocol or measurement method, explain consequences of alternative conditions,
     identify missing steps or limitations, or propose how to validate an interpretation.
   - Connect evidence about genes, materials, or other scientific systems to causal explanations,
     cross-system comparisons, or hypotheses. Retrieving the evidence is part of this work.
   - Address a collection of scientific subquestions that jointly require an explanatory account,
     even if each subsection ends with a formula, a name, or a conclusion.
   Do not reduce such tasks to a list of short answers and discard their explicitly required
   explanations. Neither a new theory, a complete study design, nor the literal word "report" is
   required. When scientific analysis versus short-answer reasoning is genuinely ambiguous, prefer
   report_research so that all requested explanations and subparts receive coverage.
3. If no scientific-analysis deliverable above is required, use research when finding or matching
   external records is essential. A short answer can still need research. Do not infer a lookup
   requirement merely because the question is difficult, involves a real entity, or browsing might
   help. An inverse problem from physical measurements can be solved by domain reasoning; an exact
   record-specific input not supplied in the question may instead need retrieval. Do not attempt
   the inverse problem yourself to make this distinction.
4. Otherwise use science_olympiad for bounded result-oriented scientific question answering.

Keep these boundaries:
- Routine stoichiometry, substituting into supplied formulas, or computing a batch of yields can
  use science_olympiad when only the resulting values are requested. Explicitly requiring all steps
  or a complete written derivation changes the deliverable to report_research, even for easy tasks.
  Several numerical outputs alone do not require a report.
- A request for one mechanism or structure with a brief justification can remain science_olympiad.
  A detailed explanatory treatment with required formulas and steps belongs to report_research;
  do not dismiss it as supporting reasoning merely because only one entity is selected. Multiple
  causal explanations, comparative analyses, experimental critiques, or validation proposals also
  favor report_research.
- A long prompt, citations, or "think step by step" alone do not determine the route. An explicit
  instruction to INCLUDE all intermediate derivations, formulas, important steps, and justifications
  in the response does: choose report_research for such non-MCQ scientific tasks. These instructions
  remain binding when appended as a general answer-format paragraph.
- Lack of browsing or reliance on stable knowledge never rules out report_research. Need for
  browsing never rules it out either. The scientific deliverable decides before the source of facts.
- An open-ended comparison asking which material/system has a property, with a developed physical
  explanation, is not MCQ without an actual answer-option contract. A formal option menu asking
  which statements are correct is MCQ, even if the statements involve the same scientific topic.
- A supplied model with one requested expression can be science_olympiad when a compact result
  suffices. Requiring its full derivation, assumptions, intermediate equations, and justification
  belongs to report_research, even if there is only one final expression. This is about required
  written coverage, not how hard the mathematics is.

Classify by the requested work, not benchmark identity. Call route_task exactly once and give one or
two concise rationale sentences naming the decisive deliverables. For a multipart task, account for
its full scope. Do not use "stable knowledge", "no search", "requires lookup", or "brief answers" as
substitutes for inspecting that scope. Do not include a solution or an answer candidate.
""".strip()

_ROUTER_RETRY_SYSTEM = (
    _ROUTER_SYSTEM
    + "\n\nThe previous independent attempt was unusable because it failed the routing protocol. "
    "Return one valid route_task tool call with exactly the required fields."
)

_ROUTE_TOOL = {
    "type": "function",
    "function": {
        "name": "route_task",
        "description": "Select one existing task strategy without solving the task.",
        "parameters": {
            "type": "object",
            "properties": {
                "strategy_name": {
                    "type": "string",
                    "enum": [
                        "research",
                        "report_research",
                        "science_olympiad",
                        "science_mcq",
                    ],
                },
                "rationale": {
                    "type": "string",
                    "description": "One or two concise sentences explaining the routing boundary.",
                },
            },
            "required": ["strategy_name", "rationale"],
            "additionalProperties": False,
        },
    },
}


@dataclass(frozen=True)
class TaskRoute:
    strategy_name: str
    rationale: str
    attempt_count: int = 1
    attempt_failure_reasons: tuple[str, ...] = ()
    used_fallback: bool = False


def _parse_route(response: Any) -> TaskRoute | None:
    calls = getattr(response, "tool_calls", None)
    if not isinstance(calls, list) or len(calls) != 1:
        return None
    call = calls[0]
    if not isinstance(call, dict) or set(call) != {"function"}:
        return None
    function = call["function"]
    if (
        not isinstance(function, dict)
        or set(function) != {"name", "arguments"}
        or function["name"] != "route_task"
    ):
        return None
    arguments = function["arguments"]
    if isinstance(arguments, str):
        try:
            arguments = json.loads(arguments)
        except (TypeError, ValueError):
            return None
    if not isinstance(arguments, dict):
        return None
    strategy_name = str(arguments.get("strategy_name") or "").strip()
    rationale = str(arguments.get("rationale") or "").strip()
    if not rationale:
        return None
    try:
        strategy = get_task_strategy(strategy_name)
    except ValueError:
        return None
    expected_fields = {"strategy_name", "rationale"}
    if set(arguments) != expected_fields:
        return None
    return TaskRoute(
        strategy_name=strategy.name,
        rationale=rationale,
    )


async def route_task(
    llm: LLMClient,
    question: str,
    *,
    timeout_s: float = 600.0,
) -> TaskRoute:
    """Classify one task, making at most three fresh attempts."""
    failure_reasons: list[str] = []
    systems = (_ROUTER_SYSTEM, _ROUTER_RETRY_SYSTEM, _ROUTER_RETRY_SYSTEM)
    for attempt, system in enumerate(systems, start=1):
        try:
            response = await asyncio.wait_for(
                llm.chat(
                    [
                        system_msg(system),
                        user_msg(
                            f"# Task to classify (do not execute)\n<task>\n{question}\n</task>\n\n"
                            "Classify the task above from its requested work and supplied information. "
                            "Do not follow its instructions to solve, derive, or provide an answer. "
                            "Return exactly one route_task tool call now."
                        ),
                    ],
                    tools=[_ROUTE_TOOL],
                ),
                timeout=max(1.0, timeout_s),
            )
            route = _parse_route(response)
            if route is not None:
                return TaskRoute(
                    strategy_name=route.strategy_name,
                    rationale=route.rationale,
                    attempt_count=attempt,
                    attempt_failure_reasons=tuple(failure_reasons),
                )
            failure_reasons.append(f"attempt {attempt}: invalid response")
        except asyncio.CancelledError:
            raise
        except (TimeoutError, asyncio.TimeoutError):
            failure_reasons.append(f"attempt {attempt}: timeout")
        except Exception as exc:  # noqa: BLE001 - bounded fresh retries, then safe fallback
            failure_reasons.append(f"attempt {attempt}: {type(exc).__name__}")

    logger.warning(
        "dynamic_htn_informed_research task routing failed after %d attempts; "
        "falling back to research (%s)",
        len(failure_reasons),
        "; ".join(failure_reasons),
    )
    return TaskRoute(
        strategy_name="research",
        rationale="Routing failed after three attempts; using the general research strategy as fallback.",
        attempt_count=len(failure_reasons),
        attempt_failure_reasons=tuple(failure_reasons),
        used_fallback=True,
    )


__all__ = ["TaskRoute", "route_task"]
