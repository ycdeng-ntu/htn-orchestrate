#!/usr/bin/env python3
"""Single-question worker — spawned by run_harbor_subprocess.py.

Usage:
    uv run python benchmarks/run_one_task.py <qid> <out_dir> [<benchmark>]

Argv:
    qid       : question id (matches trials/<qid>/ dir under out_dir)
    out_dir   : path to the run's output dir (contains trials/, tasks/)
    benchmark : dataset key (e.g. "browsecomp", "widesearch"),
                defaults to "browsecomp"

Exit codes:
    0  — success, result.json written
    1  — fatal bootstrap or dataset error (no result.json)
    2  — question id not found in dataset

This script is self-contained: each invocation is a fresh Python process
with its own kernel + agent. Master relies on SIGKILL for timeouts
because sync-in-async tool calls (e2b sandbox) cannot be cancelled via
asyncio.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import sys
import time
from pathlib import Path
from unittest.mock import MagicMock

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scorer"))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-7s %(name)s — %(message)s",
)
from agent_harness.core.atomic_files import atomic_write_text  # noqa: E402

logger = logging.getLogger("run_one_task")


# ── Scoring ───────────────────────────────────────────────────────────

from benchmarks.judges import DEFERRED_BENCHMARKS, has_judge, score_answer  # noqa: E402
from benchmarks.runner.judge_config import resolve_judge_config  # noqa: E402

_QUESTION_TIMEOUT = 28800  # 8h per question (pipeline + scoring)


async def _score_answer(
    predicted: str,
    ground_truth: str,
    question: str,
    answer_type: str,
    q_id: str,
    benchmark: str,
    *, judge_variant: str | None = None, judge_config: dict[str, str] | None = None,
    judge_details: dict | None = None,
) -> tuple[int | None, str, str | None, float | None]:
    """LLM judge only. Returns ``(reward, judge_method, scoring_error, rubric_score)``."""
    if benchmark in DEFERRED_BENCHMARKS:
        return None, "not_graded", None, None
    if not predicted or not predicted.strip() or predicted.strip() == "<ANSWER_NOT_FOUND>":
        if benchmark == "prl_bench":
            return None, "empty", "No completed physics report to grade", None
        logger.info("[%s] SCORE: empty prediction → WRONG", q_id)
        return 0, "empty", None, None

    if not has_judge(benchmark):
        msg = f"no judge registered for benchmark {benchmark!r}"
        logger.error("[%s] SCORE: %s → NOT_GRADED", q_id, msg)
        return None, "no_judge", msg, None

    logger.info(
        "[%s] SCORE: calling %s judge | pred=%r gt=%r",
        q_id, benchmark, predicted[:50], ground_truth[:50],
    )
    try:
        settings = {"judge_config": judge_config} if judge_config else {}
        if benchmark == "prl_bench":
            settings["details"] = judge_details
        verdict, rubric_score = await score_answer(
            benchmark, question, ground_truth, predicted,
            variant=judge_variant, **settings,
        )
    except Exception as exc:  # noqa: BLE001
        detail = f"{type(exc).__name__}: {exc}"
        logger.error(
            "[%s] SCORE: judge raised — NOT_GRADED (%s). NEEDS MANUAL REVIEW.",
            q_id, detail,
        )
        return None, "scoring_failed", detail, None

    score_tag = f" rubric_score={rubric_score:.2f}" if rubric_score is not None else ""
    if verdict == "SCORED" and benchmark == "prl_bench" and rubric_score is not None:
        return None, "rubric", None, rubric_score
    if verdict == "CORRECT":
        logger.info("[%s] SCORE: llm=CORRECT%s", q_id, score_tag)
        return 1, "llm", None, rubric_score
    if verdict == "INCORRECT":
        logger.info("[%s] SCORE: llm=WRONG%s", q_id, score_tag)
        return 0, "llm", None, rubric_score

    logger.error(
        "[%s] SCORE: judge NOT_ATTEMPTED after retries → NOT_GRADED. "
        "NEEDS MANUAL REVIEW.", q_id,
    )
    return None, "scoring_failed", "judge NOT_ATTEMPTED after retries", rubric_score


# ── Single question runner ────────────────────────────────────────────

async def run_one_question(
    agent,
    question: dict,
    trial_dir: Path,
    task_src_dir: Path,
    benchmark: str,
) -> dict:
    """Run one question through AgentHarnessAgent and score it.

    Returns a result dict with question_id, predicted, ground_truth,
    is_correct, duration, etc.
    """
    from harbor.models.agent.context import AgentContext

    q_id = question["id"]
    trial_dir.mkdir(parents=True, exist_ok=True)

    # ── Resume: skip if already scored ──────────────────────────
    existing_result = trial_dir / "result.json"
    if existing_result.exists():
        try:
            cached = json.loads(existing_result.read_text(encoding="utf-8"))
            logger.debug("[%s] Skipping — result.json exists", q_id)
            return cached
        except (json.JSONDecodeError, OSError):
            pass  # corrupted, re-run

    # Write config.json so agent can find task.toml metadata
    atomic_write_text((trial_dir / "config.json"), 
        json.dumps({"task": {"path": str(task_src_dir)}}),
        encoding="utf-8",
    )

    # Create a fresh agent wrapper pointing to this trial
    # We can't change agent.logs_dir after init, so we set it directly
    agent.logs_dir = trial_dir

    instruction_path = task_src_dir / "instruction.md"
    instruction = instruction_path.read_text(encoding="utf-8")

    context = AgentContext()
    fake_env = MagicMock()

    t0 = time.monotonic()
    error = None
    try:
        await asyncio.wait_for(
            agent.run(instruction, fake_env, context),
            timeout=_QUESTION_TIMEOUT,
        )
    except asyncio.TimeoutError:
        error = f"timeout after {_QUESTION_TIMEOUT}s"
        logger.error("[%s] Pipeline TIMEOUT after %ds", q_id, _QUESTION_TIMEOUT)
    except Exception as exc:
        error = str(exc)
        logger.error("[%s] Pipeline error: %s", q_id, exc)
    duration = time.monotonic() - t0

    # Read answer
    answer_file = trial_dir / "agent" / "final_answer.txt"
    predicted = ""
    if answer_file.exists():
        predicted = answer_file.read_text(encoding="utf-8").strip()

    # Score: rule-based first, LLM judge if rule says wrong
    ground_truth = question["ground_truth"]
    answer_type = question["answer_type"]
    judge_config = None
    judge_details = {}
    try:
        judge_config = resolve_judge_config(benchmark, task_src_dir)
        reward, judge_method, scoring_error, rubric_score = await _score_answer(
            predicted, ground_truth, question["question"], answer_type, q_id,
            benchmark,
            judge_variant=question.get("judge_variant"), judge_config=judge_config,
            **({"judge_details": judge_details} if benchmark == "prl_bench" else {}),
        )
    except Exception as exc:  # A bad profile must not discard the completed report.
        reward, judge_method, rubric_score = None, "scoring_failed", None
        scoring_error = f"{type(exc).__name__}: {exc}"
        logger.error("[%s] Benchmark judge configuration failed: %s", q_id, scoring_error)

    result = {
        "question_id": q_id,
        "question": question["question"][:200],
        "answer_type": answer_type,
        "ground_truth": ground_truth,
        "predicted_answer": predicted,
        "is_correct": None if reward is None else reward == 1,
        "reward": reward,
        "rubric_score": rubric_score,  # None = binary judge; float = rubric value
        "judge_method": judge_method,
        "scoring_error": scoring_error,  # None = OK, string = needs manual review
        "duration_seconds": round(duration, 2),
        "error": error,
        "category": question.get("category", ""),
        "benchmark": benchmark,
        "evaluation_metadata": question.get("metadata", {}),
        "judge_variant": question.get("judge_variant") or "official",
        "judge_config": judge_config,
        "benchmark_profile": question.get("benchmark_profile"),
    }

    # Write per-question result
    if benchmark == "prl_bench":
        result["judge_details"] = judge_details
        result["judge_variant"] = "local_equal_criterion_binary_v1"
    atomic_write_text((trial_dir / "result.json"), 
        json.dumps(result, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    status = "NOT_GRADED" if reward is None else ("CORRECT" if reward == 1 else "WRONG")
    method_tag = f" [{judge_method}]" if judge_method != "rule" else ""
    logger.info(
        "[%s] %s%s | predicted=%r truth=%r (%.1fs)",
        q_id, status, method_tag, predicted[:40], ground_truth[:40], duration,
    )
    return result


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("qid", help="Question id (matches trials/<qid>/ dir)")
    p.add_argument("out_dir", help="Run output directory (contains trials/, tasks/)")
    p.add_argument("benchmark", nargs="?", default="browsecomp")
    return p.parse_args()


async def _run_async(args: argparse.Namespace) -> int:
    from dotenv import load_dotenv
    load_dotenv()
    from benchmarks.harbor_agent.agent import AgentHarnessAgent

    out_dir = Path(args.out_dir).resolve()
    trial_dir = out_dir / "trials" / args.qid
    task_src_dir = out_dir / "tasks" / args.qid

    # Resume: if already scored, no-op (master should have skipped us, but be safe).
    if (trial_dir / "result.json").exists():
        logger.info("[%s] result.json already present — nothing to do", args.qid)
        return 0

    if not task_src_dir.exists():
        logger.error("[%s] task source dir not found: %s", args.qid, task_src_dir)
        return 1

    question = load_evaluation_question(task_src_dir, args.benchmark, args.qid)
    if question is None:
        return 2

    # Bootstrap agent (one per subprocess — own :memory: DB, own kernel).
    agent = AgentHarnessAgent(logs_dir=trial_dir.parent, model_name=None)
    run_db = str(out_dir / f".worker_kernel_{args.qid}.db")
    setup_env = MagicMock(spec=[])  # no _pipeline_hint — react_base is the only workflow
    await agent.setup(setup_env, db_path=run_db)

    await run_one_question(agent, question, trial_dir, task_src_dir, args.benchmark)

    # Cleanup the per-worker kernel db file (best-effort).
    try:
        Path(run_db).unlink()
    except FileNotFoundError:
        pass

    return 0


def load_evaluation_question(task_src_dir: Path, benchmark: str, qid: str) -> dict | None:
    """Load the exact generated row; old task layouts fall back to the registry."""
    snapshot = task_src_dir / "tests" / "evaluation.json"
    if snapshot.exists():
        question = json.loads(snapshot.read_text(encoding="utf-8"))
        if str(question["id"]) != qid or question.get("benchmark") not in (None, benchmark):
            raise ValueError(f"Evaluation snapshot identity mismatch: {snapshot}")
        return question
    from benchmarks.core.registry import load_questions
    match = next((q for q in load_questions(benchmark) if q.id == qid), None)
    if match is None:
        logger.error("[%s] question id not found in %s dataset", qid, benchmark)
        return None
    return {
        "id": match.id, "question": match.question, "ground_truth": match.ground_truth,
        "answer_type": match.answer_type, "category": match.metadata.get("category", ""),
        "image_path": match.image_path, "metadata": match.metadata,
    }


def main() -> None:
    args = _parse_args()
    exit_code = asyncio.run(_run_async(args))
    # Force-exit: agent_harness kernel keeps a non-daemon aiosqlite worker
    # thread alive, which blocks threading._shutdown() forever. result.json
    # has already been flushed to disk by run_one_question; nothing else
    # in this one-shot worker process is worth waiting to clean up.
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(exit_code)


if __name__ == "__main__":
    main()
