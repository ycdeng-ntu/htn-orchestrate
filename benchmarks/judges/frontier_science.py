"""FrontierScience judge — research (rubric) + olympiad (answer-match)."""

from __future__ import annotations

import logging
import os
import re
from urllib.parse import urlparse

from openai import AsyncOpenAI

from benchmarks.judges._common import (
    Verdict,
    _build_judge_kwargs,
    _judge_call_with_effort_fallback,
)

logger = logging.getLogger(__name__)

_FS_DEFAULT_JUDGE_MODEL = "openai/gpt-5.4"     # olympiad answer matching
_FS_RESEARCH_JUDGE_MODEL = "openai/gpt-5.4"  # user-selected FSR judge
_FS_DEFAULT_REASONING_EFFORT = "high"        # gpt-5 thinking on
_FS_PASS_THRESHOLD = 7.0
_FS_JUDGE_TRIALS = 1


def _profile_judge_client(provider: str) -> AsyncOpenAI:
    """Resolve credentials at call time without changing shared judge settings."""
    if provider == "openrouter":
        api_key = os.environ.get("OPENROUTER_API_KEY")
        base_url = os.environ.get("OPENROUTER_BASE_URL") or "https://openrouter.ai/api/v1"
    elif provider == "openai":
        # FSR's OpenAI-compatible GPT-5.4 endpoint is the NEW_API account.
        # Select credentials and endpoint together; never mix provider pairs.
        prefix = "NEW_API" if os.environ.get("NEW_API_API_KEY") else "OPENAI"
        api_key = os.environ.get(prefix + "_API_KEY")
        base_url = os.environ.get(prefix + "_BASE_URL")
        if prefix == "NEW_API" and not base_url:
            raise ValueError("NEW_API_BASE_URL is required with NEW_API_API_KEY")
    else:
        raise ValueError("FSR profile judge provider must be openrouter or openai")
    if not api_key:
        raise ValueError(f"Missing API key for the configured FSR judge provider ({provider})")
    if base_url:
        base_url = base_url.rstrip("/").removesuffix("/chat/completions")
    return AsyncOpenAI(api_key=api_key, base_url=base_url)


async def _call_rubric_judge_once(
    prompt: str, *, judge_config: dict[str, str] | None = None,
) -> float | None:
    """One rubric judge call expecting a ``VERDICT: X`` line at the end.

    ``max_completion_tokens`` is set high (64K) so chain-of-thought
    over ~10 rubric items has room before the verdict.

    Retry at the requested effort only; FSR defaults to GPT-5.4/high.
    """
    config = judge_config or {}
    kwargs = _build_judge_kwargs(
        prompt, max_completion_tokens=65536,
        default_model=config.get("model") or _FS_RESEARCH_JUDGE_MODEL,
        default_reasoning_effort=config.get("reasoning_effort") or _FS_DEFAULT_REASONING_EFFORT,
    )
    if config.get("provider"):
        async with _profile_judge_client(config["provider"]) as client:
            content = await _judge_call_with_effort_fallback(
                kwargs, label="Rubric", client=client, allow_effort_fallback=False,
            )
    else:
        content = await _judge_call_with_effort_fallback(
            kwargs, label="Rubric", allow_effort_fallback=False,
        )
    return _parse_rubric_score(content)


def _parse_rubric_score(content: str) -> float | None:
    """Local parser for Appendix B's final-line verdict contract."""
    if not content or not content.strip():
        return None
    final_line = content.strip().splitlines()[-1].strip()
    match = re.fullmatch(r"VERDICT:[ \t]*([0-9]+(?:\.[0-9]+)?)", final_line)
    if match is not None:
        score = float(match.group(1))
        if 0 <= score <= 10:
            return score
    logger.warning("Rubric judge output has no valid final 0-10 verdict")
    return None


# ─── FrontierScience prompt (from paper Appendix B) ───────────────────────

# Source: https://arxiv.org/abs/2601.21165, Appendix B (Research).
# Preserve the printed "attemped" typo; normalize the typeset score placeholder
# to <total_points> and use the local {question} template parameter.
JUDGE_PROMPT_FS = """You are grading a science exam.

You will be given the problem, attempted answer, and a rubric to grade the answer. The rubric will total up to 10 points.

Evaluate the attemped answer against the provided rubric. Pay close attention to detail and grade it strictly, but fairly. Only evaluate against the rubric, as you yourself should not make any judgements (e.g., even if you think the answer is correct but rubric is wrong, you should treat the rubric as the gold standard). Return the absolute total number of points earned (it can be a decimal based on the rubric). ***

The problem: {question}

***

The rubric: {rubric}

***

The attempted answer: {answer}

***

First, think step-by-step about each rubric item. Explain your reasoning for each rubric item. Then, tally the points up and write VERDICT: <total_points> in the last line of your response, no other text. For example, VERDICT: 2.5 or VERDICT: 8.""".strip()


# Olympiad split — direct answer matching (vs Research split's rubric scoring).
# Prompt follows FrontierScience paper Appendix B.
JUDGE_PROMPT_FS_OLYMPIAD = (
    "You are grading an attempted answer to a science olympiad problem. "
    "You will be given the problem, attempted answer, and reference answer. "
    "Evaluate the solution against the provided reference solution, ensuring "
    "it is complete and matches the reference solution. Pay close attention "
    "to detail and grade it strictly, but fairly.\n\n"
    "The reference answer is either a single number or expression in latex "
    "formatting, a chemical formula, a compound name, or a phrase referring "
    "to a specific name, entity, or method.\n\n"
    "Mark the attempted answer as correct if it fully matches the reference "
    "answer or is otherwise equivalent (e.g., an equivalent algebraic "
    "expression, a numerical number within 1 decimal place rounding of the "
    "reference answer (e.g., 6.69 ≈ 6.7), an equivalent name for a "
    "compound/formula, equivalent when accounting for units, etc.). Mark it "
    "as incorrect if it is not equivalent to the reference answer.\n"
    "***\n"
    "The problem: {problem}\n"
    "***\n"
    "The reference answer: {reference_answer}\n"
    "***\n"
    "The attempted answer: {answer}\n"
    "***\n"
    "First, think step-by-step about whether the attempted answer matches "
    "the reference answer. If the attempted answer is correct, write "
    '"VERDICT: CORRECT" in the last line of your response, with no other '
    'text or formatting. If it is incorrect, write "VERDICT: INCORRECT".'
)


async def verify_frontier_science(
    question: str, target: str, predicted: str,
) -> Verdict:
    """Grade one response using the same rubric path as the numeric scorer."""
    verdict, _score = await score_frontier_science(question, target, predicted)
    return verdict


def _olympiad_judge_model() -> str:
    # The configured llm-hub gateway registers GPT-5.4 without OpenRouter's namespace.
    endpoint = os.environ.get("JUDGE_BASE_URL") or os.environ.get("OPENAI_BASE_URL", "")
    if urlparse(endpoint).hostname == "llm-hub.apodex.app":
        return "gpt-5.4"
    return _FS_DEFAULT_JUDGE_MODEL


async def _call_olympiad_verdict_once(prompt: str) -> Verdict:
    """One Olympiad-style judge call expecting ``VERDICT: CORRECT|INCORRECT``."""
    kwargs = _build_judge_kwargs(
        prompt, max_completion_tokens=32768,
        default_model=_olympiad_judge_model(),
        default_reasoning_effort=_FS_DEFAULT_REASONING_EFFORT,
    )
    content = await _judge_call_with_effort_fallback(
        kwargs, label="Olympiad", allow_effort_fallback=False,
    )
    if not content:
        return "NOT_ATTEMPTED"
    # Walk lines bottom-up looking for the VERDICT marker.
    for line in reversed(content.splitlines()):
        up = line.strip().upper()
        if "VERDICT" not in up:
            continue
        if "INCORRECT" in up:
            return "INCORRECT"
        if "CORRECT" in up:
            return "CORRECT"
    logger.warning("Olympiad judge output missing VERDICT: %r", content[-200:])
    return "NOT_ATTEMPTED"


async def verify_frontier_science_olympiad(
    question: str, target: str, predicted: str,
) -> Verdict:
    """Grade FrontierScience-Olympiad answers via direct answer matching.

    Binary CORRECT/INCORRECT; majority vote over ``_FS_JUDGE_TRIALS`` calls.
    """
    if not predicted or not predicted.strip():
        return "INCORRECT"
    n_trials = _FS_JUDGE_TRIALS
    prompt = JUDGE_PROMPT_FS_OLYMPIAD.format(
        problem=question, reference_answer=target, answer=predicted,
    )
    verdicts: list[Verdict] = []
    for _ in range(n_trials):
        v = await _call_olympiad_verdict_once(prompt)
        if v != "NOT_ATTEMPTED":
            verdicts.append(v)
    if not verdicts:
        logger.warning("Olympiad judge: all %d trial(s) failed to parse", n_trials)
        return "NOT_ATTEMPTED"
    correct = sum(1 for v in verdicts if v == "CORRECT")
    logger.info(
        "Olympiad judge: trials=%d correct=%d/%d → %s",
        len(verdicts), correct, len(verdicts),
        "CORRECT" if correct * 2 > len(verdicts) else "INCORRECT",
    )
    return "CORRECT" if correct * 2 > len(verdicts) else "INCORRECT"


async def score_frontier_science(
    question: str, target: str, predicted: str,
    *, judge_config: dict[str, str] | None = None,
) -> tuple[Verdict, float | None]:
    """Same as ``verify_frontier_science`` plus the raw rubric score (0-10)."""
    if not predicted or not predicted.strip():
        return "INCORRECT", 0.0

    # The paper averages independent evaluation trials, not repeated grades of
    # one fixed answer followed by thresholding their mean. Grade this response
    # once; runner-level repeated solves are separate observations.
    prompt = JUDGE_PROMPT_FS.format(
        question=question, rubric=target, answer=predicted,
    )
    score = (
        await _call_rubric_judge_once(prompt, judge_config=judge_config)
        if judge_config else await _call_rubric_judge_once(prompt)
    )
    if score is None:
        return "NOT_ATTEMPTED", None
    return ("CORRECT" if score >= _FS_PASS_THRESHOLD else "INCORRECT"), score


__all__ = [
    "JUDGE_PROMPT_FS",
    "JUDGE_PROMPT_FS_OLYMPIAD",
    "score_frontier_science",
    "verify_frontier_science",
    "verify_frontier_science_olympiad",
]
