"""Opt-in FSR scoring protocol from the 2026-09-21 Agent Team campaign.

Frozen source: results/2026-09-21_frontierscience_no_browsing/judge_source/
benchmarks/judges/{frontier_science,_common}.py in yaolong/AgentHarness.
Keep the default FSR protocol separate. No solver consumes this module.
"""
from __future__ import annotations

import asyncio
import logging
import re

from benchmarks.judges._common import Verdict, _build_judge_kwargs, _effort_chain
from benchmarks.judges.frontier_science import _profile_judge_client

logger = logging.getLogger(__name__)
VARIANT = "agent_team_20260921"

JUDGE_PROMPT_FS = 'You are grading a science exam. You will be given the problem, attempted answer, and a rubric to grade the answer. The rubric will total up to 10 points. Evaluate the attempted answer against the provided rubric. Pay close attention to detail and grade it strictly, but fairly. Only evaluate against the rubric, as you yourself should not make any judgements (e.g., even if you think the answer is correct but rubric is wrong, you should treat the rubric as the gold standard). Return the absolute total number of points earned (it can be a decimal based on the rubric). First, think step-by-step about each rubric item. Explain your reasoning for each rubric item. Then, tally the points up and write VERDICT: {{total_points}} in the last line of your response, no other text. For example, VERDICT: 2.5 or VERDICT: 8.\n\nProblem:\n{question}\n\nRubric:\n{rubric}\n\nAttempted Answer:\n{answer}'


async def score_frontier_science_team(
    question: str, target: str, predicted: str,
    *, judge_config: dict[str, str] | None = None,
) -> tuple[Verdict, float | None]:
    """One grade, threshold 7; high/medium/low, two attempts per effort.

    Match the campaign's prompt, request, completion acceptance and parser.
    SDK retry/timeout defaults are retained. Close each attempt's client.
    """
    if not predicted or not predicted.strip():
        return "INCORRECT", 0.0
    config = judge_config or {}
    kwargs = _build_judge_kwargs(
        JUDGE_PROMPT_FS.format(question=question, rubric=target, answer=predicted),
        max_completion_tokens=65536,
        default_model=config.get("model") or "openai/gpt-5.5",
        default_reasoning_effort=config.get("reasoning_effort") or "high",
    )
    for effort in _effort_chain(kwargs["model"], kwargs.get("reasoning_effort", "")):
        if effort:
            kwargs["reasoning_effort"] = effort
        else:
            kwargs.pop("reasoning_effort", None)
        for _ in range(2):
            try:
                async with _profile_judge_client(config.get("provider", "openrouter")) as client:
                    response = await client.chat.completions.create(**kwargs)
            except Exception as exc:  # noqa: BLE001
                logger.warning("Agent Team rubric judge request error: %s", type(exc).__name__)
                await asyncio.sleep(1.0)
                continue
            choices = getattr(response, "choices", None) or []
            choice = choices[0] if choices else None
            raw = getattr(getattr(choice, "message", None), "content", None)
            content = raw.strip() if isinstance(raw, str) else ""
            if content and getattr(choice, "finish_reason", None) == "stop":
                match = re.search(r"VERDICT:\s*(\d+(?:\.\d+)?)", content)
                if match is None:
                    return "NOT_ATTEMPTED", None
                score = float(match.group(1))
                return ("CORRECT" if score >= 7.0 else "INCORRECT"), score
            await asyncio.sleep(1.0)
    return "NOT_ATTEMPTED", None


async def verify_frontier_science_team(question: str, target: str, predicted: str) -> Verdict:
    verdict, _ = await score_frontier_science_team(question, target, predicted)
    return verdict
