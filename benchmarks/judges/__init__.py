"""LLM-as-judge dispatch for benchmark answer grading.

Each family lives in its own module under ``benchmarks/judges/``:
    - hle.py            → verify_hle (also covers superchem_text)
    - browsecomp.py     → verify_browsecomp, verify_browsecomp_zh
    - frontier_science.py → verify_frontier_science(_olympiad), score_frontier_science
    - deepsearchqa.py   → verify_deepsearchqa
    - xbench.py         → verify_xbench
    - widesearch.py     → score_widesearch (returns (verdict, f1_by_item))

Shared infra (LLM client, retry helpers, types) is in _common.py.

Benchmark profiles select model, reasoning effort and a credential provider.
Settings are scoped to each scoring call; family defaults apply to direct calls
without a benchmark profile. Credentials remain in environment variables.

Adding a new benchmark's judge:
    1. Create benchmarks/judges/<family>.py implementing
       ``async def verify_<name>(question, target, predicted) -> Verdict``.
    2. Register it in JUDGE_REGISTRY below.
"""

from __future__ import annotations

from functools import partial

from benchmarks.judges._common import JudgeFn, Verdict
from benchmarks.judges.browsecomp import (
    JUDGE_PROMPT_BC_ZH,
    JUDGE_PROMPT_BROWSECOMP_OFFICIAL,
    verify_browsecomp,
    verify_browsecomp_semantic,
    verify_browsecomp_zh,
)
from benchmarks.judges.deepsearchqa import (
    JUDGE_PROMPT_DEEPSEARCHQA,
    score_deepsearchqa,
    verify_deepsearchqa,
)
from benchmarks.judges.frontier_science import (
    JUDGE_PROMPT_FS,
    JUDGE_PROMPT_FS_OLYMPIAD,
    score_frontier_science,
    verify_frontier_science,
    verify_frontier_science_olympiad,
)
from benchmarks.judges.frontier_science_team import (
    VARIANT as FSR_TEAM_VARIANT,
    score_frontier_science_team,
    verify_frontier_science_team,
)
from benchmarks.judges.hle import (
    JUDGE_PROMPT_HLE,
    verify_hle,
)
from benchmarks.judges.widesearch import score_widesearch
from benchmarks.judges.xbench import verify_xbench
from benchmarks.judges.prl_bench import score_prl

JUDGE_REGISTRY: dict[str, JudgeFn] = {
    "browsecomp":                  verify_browsecomp,
    "browsecomp_200":              partial(verify_browsecomp, model="gpt-5.4"),
    "browsecomp_zh":               verify_browsecomp_zh,
    "xbench_dr_202510":            verify_xbench,
    "frontier_science_research":   verify_frontier_science,
    "frontier_science_olympiad":   verify_frontier_science_olympiad,
    "hle_text":                    verify_hle,
    "deepsearchqa":                verify_deepsearchqa,
    # SUPERChem-Text is single-letter MCQ (A-J); HLE's extract-final-answer
    # + correct: yes|no schema handles it cleanly — judge pulls the chosen
    # letter from the model's natural-language reply and matches it against
    # the gold letter in ground_truth.
    "superchem_text":              partial(verify_hle, model="gpt-5.4"),
    # WideSearch uses score_widesearch (structural row/item F1 + judge LLM
    # for column alignment). It returns (verdict, f1_by_item) so it is
    # dispatched via a separate score_* path in the runner, NOT through
    # JUDGE_REGISTRY's standard Verdict-returning dispatch.
}

# This audit variant is deliberately separate from the official default.
JUDGE_VARIANTS: dict[str, dict[str, JudgeFn]] = {
    "browsecomp_200": {"semantic": verify_browsecomp_semantic},
    "frontier_science_research": {FSR_TEAM_VARIANT: verify_frontier_science_team},
}
DEFERRED_BENCHMARKS = frozenset({"deepresearch_bench_ii"})


def has_judge(benchmark: str) -> bool:
    """True if ``benchmark`` has an LLM judge configured (rubric or binary)."""
    return benchmark in JUDGE_REGISTRY or benchmark in SCORE_REGISTRY


async def verify_answer(
    benchmark: str, question: str, target: str, predicted: str,
    *, variant: str | None = None,
) -> Verdict:
    """Dispatch to the registered judge. Raises ``KeyError`` if unregistered."""
    if variant:
        fn = JUDGE_VARIANTS.get(benchmark, {}).get(variant)
        if fn is None:
            raise KeyError(f"No judge variant {variant!r} for {benchmark!r}")
        return await fn(question, target, predicted)
    fn = JUDGE_REGISTRY.get(benchmark)
    if fn is None:
        raise KeyError(
            f"No LLM judge registered for benchmark {benchmark!r}. "
            f"Registered: {sorted(JUDGE_REGISTRY)}"
        )
    return await fn(question, target, predicted)


# Benchmarks whose judge surfaces a raw rubric value (0-10 for FS,
# F1 0-1 for WideSearch). Stored in result.json's ``rubric_score`` field.
# WideSearch has no binary verify_* sibling — it only exists here, so
# SCORE_REGISTRY is also the dispatch root for it.
SCORE_REGISTRY: dict[str, JudgeFn] = {
    "prl_bench":                score_prl,
    "deepsearchqa":              score_deepsearchqa,
    "frontier_science_research": score_frontier_science,
    "widesearch":                score_widesearch,
}


async def _score_answer_impl(
    benchmark: str, question: str, target: str, predicted: str,
    *, variant: str | None = None, judge_config: dict[str, str] | None = None,
    details: dict | None = None,
) -> tuple[Verdict, float | None]:
    """Verdict plus optional raw rubric score (``None`` for binary judges)."""
    if benchmark == "frontier_science_research" and variant == FSR_TEAM_VARIANT:
        return await score_frontier_science_team(
            question, target, predicted, judge_config=judge_config,
        )
    if variant:
        return await verify_answer(benchmark, question, target, predicted, variant=variant), None
    if (fn := SCORE_REGISTRY.get(benchmark)) is not None:
        if benchmark == "prl_bench":
            return await fn(question, target, predicted, judge_config=judge_config, details=details)
        if benchmark == "frontier_science_research" and judge_config:
            return await fn(question, target, predicted, judge_config=judge_config)
        return await fn(question, target, predicted)
    return await verify_answer(benchmark, question, target, predicted), None


__all__ = [
    "JUDGE_PROMPT_BC_ZH",
    "JUDGE_PROMPT_BROWSECOMP_OFFICIAL",
    "JUDGE_PROMPT_DEEPSEARCHQA",
    "JUDGE_PROMPT_FS",
    "JUDGE_PROMPT_FS_OLYMPIAD",
    "JUDGE_PROMPT_HLE",
    "JUDGE_REGISTRY",
    "JUDGE_VARIANTS",
    "DEFERRED_BENCHMARKS",
    "JudgeFn",
    "SCORE_REGISTRY",
    "Verdict",
    "has_judge",
    "score_answer",
    "score_deepsearchqa",
    "score_frontier_science",
    "score_widesearch",
    "verify_answer",
    "verify_browsecomp",
    "verify_browsecomp_zh",
    "verify_deepsearchqa",
    "verify_frontier_science",
    "verify_frontier_science_olympiad",
    "verify_hle",
    "verify_xbench",
]


async def score_answer(
    benchmark: str, question: str, target: str, predicted: str,
    *, variant: str | None = None, judge_config: dict | None = None,
    details: dict | None = None,
) -> tuple[Verdict, float | None]:
    """Each scoring call owns its profile; concurrent benchmarks never share settings."""
    from benchmarks.judges._common import judge_scope
    with judge_scope(judge_config):
        variant = variant or (judge_config or {}).get("variant")
        if variant == "official":
            variant = None
        return await _score_answer_impl(benchmark, question, target, predicted,
                                        variant=variant, judge_config=judge_config, details=details)
