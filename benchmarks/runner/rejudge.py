#!/usr/bin/env python3
"""Re-judge existing trial result.json files in place.

Backfills the ``rubric_score`` field (and refreshes ``is_correct`` /
``reward``) for trials that were scored before the field was tracked.
The agent's prediction is reused from result.json — only the judge is
re-run, so this is much cheaper than a full restart.

Idempotent: successfully graded trials are skipped unless ``--force``.
Failed or missing grades are retried even when ``rubric_score`` is null.
For FSR only, ``--judge-provider`` changes the scoring provider while retaining
the resolved model and reasoning effort; successful grades still skip by default.

Usage:
    uv run python -m benchmarks.runner.rejudge <out_dir> [--concurrency N]

``<out_dir>`` is the directory you passed as ``--out`` to run_subprocess.
Single-run and multi-run layouts are both supported.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from agent_harness.core.atomic_files import atomic_write_text

from benchmarks.core.registry import load_questions  # noqa: E402
from benchmarks.runner.run_one_task import _score_answer, load_evaluation_question  # noqa: E402
from benchmarks.runner.judge_config import resolve_judge_config  # noqa: E402
from benchmarks.judges import DEFERRED_BENCHMARKS  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-7s %(name)s — %(message)s",
)
logger = logging.getLogger("rejudge")


def _collect_results(out_dir: Path) -> list[Path]:
    """Find result.json files under both single-run and multi-run layouts."""
    return sorted(out_dir.glob("**/trials/*/result.json"))


def _validate_judge_provider(benchmark: str, provider: str | None) -> None:
    if provider is None:
        return
    if benchmark != "frontier_science_research":
        raise ValueError("--judge-provider is only supported for frontier_science_research")
    if provider not in {"openai", "openrouter"}:
        raise ValueError("--judge-provider must be openai or openrouter")


async def _rejudge_one(
    path: Path,
    benchmark: str,
    questions: dict[str, object],
    sem: asyncio.Semaphore,
    force: bool,
    judge_variant: str | None = None,
    judge_provider: str | None = None,
) -> str:
    _validate_judge_provider(benchmark, judge_provider)
    r = json.loads(path.read_text(encoding="utf-8"))
    if benchmark in DEFERRED_BENCHMARKS:
        r.update(is_correct=None, reward=None, rubric_score=None,
                 judge_method="not_graded", scoring_error=None)
        _atomic_write(path, r)
        return "not_graded"
    if (not force and (isinstance(r.get("is_correct"), bool) or
                      (r.get("judge_method") == "rubric" and isinstance(r.get("rubric_score"), (int, float))))
            and not r.get("scoring_error") and "rubric_score" in r):
        return "skip"
    pred = (r.get("predicted_answer") or "").strip()
    if not pred:
        r["rubric_score"] = None
        _atomic_write(path, r)
        return "empty"
    # result.json's ``question`` is truncated to 200 chars for display; reload
    # the full question + ground truth from the dataset so the judge sees the
    # same payload that the original run did.
    task_dir = path.parents[2] / "tasks" / r["question_id"]
    if (task_dir / "tests" / "evaluation.json").exists():
        question = load_evaluation_question(task_dir, benchmark, r["question_id"])
    else:
        q = questions.get(r["question_id"])
        question = ({"question": q.question, "ground_truth": q.ground_truth,
                     "answer_type": q.answer_type} if q else None)
    if question is None:
        logger.error("[%s] question_id not in dataset — skipping", path.parent.name)
        return "fail"

    variant = judge_variant or r.get("judge_variant") or question.get("judge_variant")
    if variant == "official":
        variant = None
    async with sem:
        judge_config = None
        judge_details = {}
        try:
            judge_config = resolve_judge_config(
                benchmark, task_dir, saved_config=r.get("judge_config"),
            )
            if judge_provider is not None:
                if judge_config is None:
                    raise ValueError(
                        "FSR provider override requires saved judge_config or an explicit "
                        "benchmark profile snapshot; refusing to guess model or reasoning effort"
                    )
                judge_config = {**judge_config, "provider": judge_provider}
            reward, method, error, score = await _score_answer(
                pred, question["ground_truth"], question["question"],
                question["answer_type"], r["question_id"], benchmark,
                judge_variant=variant, judge_config=judge_config,
                **({"judge_details": judge_details} if benchmark == "prl_bench" else {}),
            )
        except Exception as exc:
            reward, method, score = None, "scoring_failed", None
            error = f"{type(exc).__name__}: {exc}"

    r["rubric_score"] = score
    r["is_correct"] = None if reward is None else reward == 1
    r["reward"] = reward
    r["judge_method"] = method
    r["scoring_error"] = error
    r["judge_variant"] = variant or "official"
    r["judge_config"] = judge_config
    if benchmark == "prl_bench":
        r["judge_details"] = judge_details
        r["judge_variant"] = "local_equal_criterion_binary_v1"
    _atomic_write(path, r)
    return "fail" if error else "ok"


def _atomic_write(path: Path, obj: dict) -> None:
    atomic_write_text(path, json.dumps(obj, indent=2, ensure_ascii=False))


async def _amain(args: argparse.Namespace) -> int:
    out_dir = Path(args.out_dir).resolve()
    cfg = json.loads((out_dir / "config.json").read_text(encoding="utf-8"))
    benchmark = cfg["benchmark"]
    judge_provider = getattr(args, "judge_provider", None)
    _validate_judge_provider(benchmark, judge_provider)
    results = _collect_results(out_dir)
    needs_dataset = any(not (p.parents[2] / "tasks" / p.parent.name
                            / "tests" / "evaluation.json").exists() for p in results)
    questions = ({q.id: q for q in load_questions(benchmark)}
                 if needs_dataset and benchmark not in DEFERRED_BENCHMARKS else {})
    logger.info(
        "Rejudging %d trial(s) under %s (benchmark=%s, concurrency=%d, force=%s)",
        len(results), out_dir, benchmark, args.concurrency, args.force,
    )

    sem = asyncio.Semaphore(args.concurrency)
    outcomes = await asyncio.gather(
        *(_rejudge_one(p, benchmark, questions, sem, args.force, args.judge_variant, judge_provider)
          for p in results),
    )
    counts = {k: outcomes.count(k) for k in ("ok", "skip", "empty", "fail", "not_graded")}
    logger.info(
        "Done: %(ok)d updated, %(skip)d skipped, %(empty)d empty-pred, "
        "%(fail)d failed, %(not_graded)d not graded",
        counts,
    )
    return 0 if counts["fail"] == 0 else 1


def main() -> int:
    from dotenv import load_dotenv
    load_dotenv()
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("out_dir", help="The --out dir passed to run_subprocess")
    p.add_argument("--concurrency", type=int, default=10,
                   help="Parallel judge calls (default: 10)")
    p.add_argument("--force", action="store_true",
                   help="Re-judge even if rubric_score key already present")
    p.add_argument("--judge-variant", default=None,
                   help="Override saved judge prompt variant (requires --force for scored trials)")
    p.add_argument("--judge-provider", choices=("openai", "openrouter"), default=None,
                   help="FSR only: override provider, preserving the resolved model and reasoning effort")
    args = p.parse_args()
    try:
        return asyncio.run(_amain(args))
    except ValueError as exc:
        p.error(str(exc))


if __name__ == "__main__":
    raise SystemExit(main())
