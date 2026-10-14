"""BrowseComp judges (English + Chinese).

Prompts follow the published BrowseComp / BrowseComp-ZH templates.
Chinese variant pins ``gpt-4o`` with the standard sampling params.
"""

from __future__ import annotations

from benchmarks.judges._common import judge_parameters

import asyncio
import logging
import os
import re

from benchmarks.judges._common import (
    Verdict,
    _build_judge_kwargs,
    _judge_call_with_effort_fallback,
    _make_client,
)

logger = logging.getLogger(__name__)


JUDGE_PROMPT_BROWSECOMP_OFFICIAL = """Judge whether the following [response] to [question] is correct or not based on the precise and unambiguous [correct_answer] below.

[question]: {question}

[response]: {response}

Your judgement must be in the format and criteria specified below:

extracted_final_answer: The final exact answer extracted from the [response]. Put the extracted answer as 'None' if there is no exact, final answer to extract from the response.

[correct_answer]: {correct_answer}

reasoning: Explain why the extracted_final_answer is correct or incorrect based on [correct_answer], focusing only on if there are meaningful differences between [correct_answer] and the extracted_final_answer. Do not comment on any background to the problem, do not attempt to solve the problem, do not argue for any answer different than [correct_answer], focus only on whether the answers match.

correct: Answer 'yes' if extracted_final_answer matches the [correct_answer] given above, or is within a small margin of error for numerical problems. Answer 'no' otherwise, i.e. if there if there is any inconsistency, ambiguity, non-equivalency, or if the extracted answer is incorrect.

confidence: The extracted confidence score between 0|\\%| and 100|\\%| from [response]. Put 100 if there is no confidence score available."""


JUDGE_PROMPT_BC_ZH = """根据以下精确且明确的[response]，判断以下对[question]的[correct_answer]是否正确。

[question]:  {question}

[response]:  {response}

您的判断必须符合以下指定的格式和标准：

extracted_final_answer: 从[response]中提取的最终准确答案。如果无法从答案中提取出准确的最终答案，则将提取的答案填写为"None"。

[correct_answer]: {correct_answer}

reasoning: 根据[correct_answer]解释提取的最终答案正确或错误的原因， 仅关注[correct_answer]和提取的最终答案之间是否存在有意义的差异。请勿评论问题的任何背景，请勿尝试解决问题，请勿争论任何与[correct_answer]不同的答案，仅关注答案是否匹配。

correct: 如果提取的最终答案与上面给出的[correct_answer]相符，或者在数值问题的误差范围内，则回答"yes"。否则，例如，如果存在任何不一致、歧义、不等同，或者提取的答案不正确，则回答"no"。

confidence: 从[response]中提取的置信度分数，介于0% 到100% 之间。如果没有可用的置信度分数，则填写100%。

"""


JUDGE_PROMPT_BC_SEMANTIC = """Judge whether the following [response] to [question] is correct based on the [correct_answer].

[question]: {question}

[response]: {response}

[correct_answer]: {correct_answer}

Judge by SEMANTIC EQUIVALENCE, not string match. Answer 'yes' when the response identifies the same entity, person, work, event, or object as [correct_answer], even if it differs ONLY by one or more of:
- company legal suffixes (Ltd, Inc, LLC, plc) — e.g. "Acme Corporation" matches "Acme Corporation Inc."
- honorifics or titles (Dr., Mr., Prof.) — e.g. "Jane Roberts" matches "Dr. Jane Roberts"
- a middle name present in one but not the other — e.g. "Mary Johnson" matches "Mary Anne Johnson"
- a stage name or common name vs the full legal name of the same real person
- a leading definite/indefinite article ("The", "A", "An")
- capitalization, punctuation, straight vs curly quotes/apostrophes, or trailing punctuation
- accents, diacritics, or Unicode normalization form — e.g. "Jose" matches "José"
- a subtitle: if the response gives the fuller title of the same work (main title plus subtitle), that is CORRECT
- a widely-used abbreviation that UNIQUELY identifies the same event — e.g. "Games 2044" matches "Games 2044: Winter Edition"
- an appended location or explanatory phrase when the core answer is fully present — e.g. "Blue Note" matches "Blue Note in Springfield"
- a leading or partitive/possessive wrapper around the full [correct_answer] — e.g. "X" matches "a piece of X" / "part of X" / "a fragment of X"
- minor transcription differences such as singular/plural of a title token

Answer 'no' when the extracted answer is a genuinely different or contradictory answer, is missing, or is of insufficient granularity (a broader/less specific term where a more specific one was required — e.g. a general category name where a distinct sub-answer was asked for).

The equivalence above covers only surface form (suffix, honorific, article, accent, punctuation, subtitle, abbreviation, appended location) — extra or missing tokens that do NOT change who/what is referred to. It does NOT license accepting a DIFFERENT referent. In particular, answer 'no' when:
- the response names a DIFFERENT entity (one that shares no naming tokens with [correct_answer]) that is merely related to, contained by, or associated with it — e.g. the name of a physical object vs the name of the work it contains, a product vs its maker, a book vs its author. This does NOT apply when the response literally contains the full [correct_answer] and only wraps it in a possessive/partitive phrase (see the "a piece of X" case above) — that is 'yes'.
- the response is a person's name whose distinctive tokens CONFLICT with [correct_answer]: a given name or surname in the response contradicts the corresponding part of [correct_answer]. E.g. if [correct_answer] is "Ann Baker Smith" and the response is "Tom Smith", the given name "Tom" conflicts and only the common surname "Smith" overlaps — that is 'no'.

Do NOT use the rule above to reject a name that is simply MORE or LESS complete but consistent: a response that contains every token of [correct_answer]'s name and merely adds more (a fuller form of the same person — e.g. "Sam" -> "Sam Rivera", "Ann Smith" -> "Ann Beatrice Smith") is the SAME person and is 'yes'; likewise a shorter common/stage name that conflicts with no token of [correct_answer] (e.g. "Robert Charles Hale-Owen" -> "Robert Owen") is 'yes'. Reject only on a genuine CONFLICT in a distinctive token, not on added or omitted tokens.
When you are unsure whether the response and [correct_answer] denote the same real-world referent, answer 'no'.

Your judgement must be in the format and criteria specified below:

extracted_final_answer: The final exact answer extracted from the [response]. Put 'None' if there is no exact, final answer to extract.

reasoning: Explain only whether the extracted_final_answer refers to the same thing as [correct_answer] under the equivalence rules above. Do not attempt to solve the problem or argue for a different answer.

correct: Answer 'yes' or 'no' per the criteria above.

confidence: The extracted confidence score between 0|\\%| and 100|\\%| from [response]. Put 100 if there is no confidence score available.
""".strip()


_BC_JUDGE_MODEL = "gpt-4.1-2025-04-14"
_BC_ZH_JUDGE_MODEL = "gpt-4o"


async def _call_browsecomp_official_once(
    prompt: str, *, model: str | None = None, reasoning_effort: str | None = None,
) -> Verdict:
    """One judge call. Parses ``correct: yes|no`` from the structured output."""
    kwargs = _build_judge_kwargs(
        prompt, max_completion_tokens=16384, model_override=model or _BC_JUDGE_MODEL,
        default_reasoning_effort=reasoning_effort,
    )
    content = await _judge_call_with_effort_fallback(kwargs, label="BrowseComp")
    if not content:
        return "NOT_ATTEMPTED"
    m = re.search(r"correct\s*:\s*(yes|no)\b", content, re.IGNORECASE)
    if not m:
        logger.warning(
            "BrowseComp judge missing 'correct: yes|no' line: %r",
            content[-300:],
        )
        return "NOT_ATTEMPTED"
    return "CORRECT" if m.group(1).lower() == "yes" else "INCORRECT"


async def verify_browsecomp(
    question: str, target: str, predicted: str, *, model: str | None = None,
) -> Verdict:
    """Grade BrowseComp answers; 10x retry on NOT_ATTEMPTED."""
    prompt = JUDGE_PROMPT_BROWSECOMP_OFFICIAL.format(
        question=question, correct_answer=target, response=predicted,
    )
    for attempt in range(1, 11):
        verdict = await _call_browsecomp_official_once(prompt, **({"model": model} if model else {}))
        if verdict != "NOT_ATTEMPTED":
            return verdict
        if attempt < 10:
            logger.info(
                "BrowseComp judge NOT_ATTEMPTED — retrying %d/10", attempt,
            )
            await asyncio.sleep(5.0)
    return "NOT_ATTEMPTED"


async def _call_browsecomp_zh_once(prompt: str) -> Verdict:
    """One BC-ZH judge call."""
    try:
        resp = await _make_client().chat.completions.create(
            **judge_parameters(_BC_ZH_JUDGE_MODEL),
            messages=[
                {"role": "system", "content": "you are a helpful assistant!"},
                {"role": "user", "content": prompt},
            ],
            temperature=0.6,
            top_p=0.95,
            max_completion_tokens=16384,
        )
        content = (resp.choices[0].message.content or "").strip()
    except Exception as exc:
        logger.error("BC-ZH judge call error: %s", exc)
        return "NOT_ATTEMPTED"
    if not content:
        return "NOT_ATTEMPTED"
    m = re.search(r"correct\s*:\s*(yes|no)\b", content, re.IGNORECASE)
    if not m:
        logger.warning(
            "BC-ZH judge missing 'correct: yes|no' line: %r", content[-300:],
        )
        return "NOT_ATTEMPTED"
    return "CORRECT" if m.group(1).lower() == "yes" else "INCORRECT"


async def verify_browsecomp_zh(
    question: str, target: str, predicted: str,
) -> Verdict:
    """Grade BrowseComp-ZH answers; 10x retry on NOT_ATTEMPTED."""
    prompt = JUDGE_PROMPT_BC_ZH.format(
        question=question, response=predicted, correct_answer=target,
    )
    for attempt in range(1, 11):
        verdict = await _call_browsecomp_zh_once(prompt)
        if verdict != "NOT_ATTEMPTED":
            return verdict
        if attempt < 10:
            logger.info(
                "BC-ZH judge NOT_ATTEMPTED — retrying %d/10", attempt,
            )
            await asyncio.sleep(5.0)
    return "NOT_ATTEMPTED"


async def verify_browsecomp_semantic(
    question: str, target: str, predicted: str,
) -> Verdict:
    """Grade BrowseComp answers by SEMANTIC EQUIVALENCE via the lenient
    ``JUDGE_PROMPT_BC_SEMANTIC`` variant. Marks answers correct when they
    identify the same entity/work/event as the ground truth but differ only
    by boundary/format (company suffix, honorific, middle name, stage name,
    article, punctuation, diacritics, subtitle, uniquely-identifying
    abbreviation, appended location, minor transcription typo).

    Reuses the OFFICIAL prompt's ``correct: yes|no`` parser. This is a
    false-negative audit tool, NOT the official benchmark judge — use only via
    ``variant="semantic"`` and rejudge BOTH compared systems with it.
    """
    prompt = JUDGE_PROMPT_BC_SEMANTIC.format(
        question=question, correct_answer=target, response=predicted,
    )
    for attempt in range(1, 11):
        verdict = await _call_browsecomp_official_once(
            prompt,
            model="gpt-5.4",
            reasoning_effort=os.environ.get("BC200_JUDGE_REASONING_EFFORT"),
        )
        if verdict != "NOT_ATTEMPTED":
            return verdict
        if attempt < 10:
            logger.info(
                "BrowseComp semantic judge NOT_ATTEMPTED — retrying %d/10",
                attempt,
            )
            await asyncio.sleep(5.0)
    return "NOT_ATTEMPTED"


__all__ = [
    "JUDGE_PROMPT_BC_ZH",
    "JUDGE_PROMPT_BC_SEMANTIC",
    "verify_browsecomp_semantic",
    "JUDGE_PROMPT_BROWSECOMP_OFFICIAL",
    "verify_browsecomp",
    "verify_browsecomp_zh",
]
