#!/usr/bin/env python3
"""Run a benchmark via subprocess-per-question isolation.

Each question runs in its own Python subprocess — fresh kernel, in-memory
DB, asyncio loop. Lets the master SIGKILL hung workers.

Output:

  Single-run (``--runs 1``):
    <out>/{tasks,trials}/<qid>/...
    <out>/{results.json,summary.txt}

  Multi-run (``--runs N``):
    <out>/config.json
    <out>/run_<i>/{tasks,trials,results.json,summary.txt}

Inspect progress with ``python -m benchmarks.runner.check_progress``.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import sys
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-7s %(name)s — %(message)s",
)
logger = logging.getLogger("run_harbor_subprocess")

from agent_harness.core.atomic_files import atomic_write_text  # noqa: E402

from benchmarks.core.contracts import is_graded, solver_metadata  # noqa: E402
from benchmarks.judges import DEFERRED_BENCHMARKS, JUDGE_VARIANTS  # noqa: E402

_QUESTION_TIMEOUT = 30600  # 8.5h — gives inner 8h a 30min grace to score
_WORKER_SCRIPT = Path(__file__).resolve().parent / "run_one_task.py"
_DRB = "deepresearch_bench_ii"
_NATIVE_CRASH_CODES = {-6, -11}  # SIGABRT / SIGSEGV; the native root cause is unknown.


# ── Subprocess runner ────────────────────────────────────────────────

async def _tee(reader, log, mirror, prefix=""):
    while line := await reader.readline():
        s = line.decode("utf-8", errors="replace")
        log.write(s)
        log.flush()
        if mirror:
            sys.stdout.write(prefix + s if prefix else s)
            sys.stdout.flush()


def _reusable_result(result, qid: str) -> bool:
    """Keep completed solving (including pending judging), retry execution failures."""
    return (
        isinstance(result, dict) and result.get("question_id") == qid
        and not result.get("error")
        and (isinstance(result.get("is_correct"), bool)
             or bool(result.get("predicted_answer")))
    )


async def _wait_process_and_log(proc, tee):
    wait = asyncio.create_task(proc.wait())
    try:
        await asyncio.wait((wait, tee), return_when=asyncio.FIRST_EXCEPTION)
        if tee.done() and not tee.cancelled() and tee.exception() is not None:
            raise tee.exception()
        return await wait
    finally:
        if not wait.done():
            wait.cancel()
        await asyncio.gather(wait, return_exceptions=True)


async def _settle_worker(proc, tee):
    if proc is not None and proc.returncode is None:
        try:
            proc.kill()
        except ProcessLookupError:
            pass
        await proc.wait()
    if tee is not None:
        if not tee.done():
            tee.cancel()
        await asyncio.gather(tee, return_exceptions=True)


def _archive_drb_attempt(out_dir: Path, qid: str) -> Path:
    """Preserve the whole failed trial and remove its stale worker database."""
    archive = out_dir / "previous_attempts" / qid / uuid.uuid4().hex
    archive.parent.mkdir(parents=True, exist_ok=True)
    (out_dir / "trials" / qid).replace(archive)
    for suffix in ("", "-wal", "-shm"):
        database = out_dir / f".worker_kernel_{qid}.db{suffix}"
        if database.exists():
            database.replace(archive / database.name)
    return archive


async def run_task_subprocess(
    qid: str,
    out_dir: Path,
    benchmark: str,
    sem: asyncio.Semaphore,
    mirror: bool = False,
) -> dict:
    """Retry a DRB native crash once, reusing the normal resume and slot logic."""
    for attempt in range(2 if benchmark == _DRB else 1):
        result = await _run_task_subprocess_once(qid, out_dir, benchmark, sem, mirror)
        if (_reusable_result(result, qid)
                or result.get("worker_exit_code") not in _NATIVE_CRASH_CODES
                or benchmark != _DRB or attempt == 1):
            return result
        logger.warning("[%s] native crash exit=%s; retrying once through the worker semaphore",
                       qid, result["worker_exit_code"])
    raise AssertionError("unreachable")


async def _run_task_subprocess_once(
    qid: str,
    out_dir: Path,
    benchmark: str,
    sem: asyncio.Semaphore,
    mirror: bool = False,
) -> dict:
    """Spawn a worker subprocess for one question. Returns the result dict
    from result.json, or a synthetic error dict on timeout/crash."""
    trial_dir = out_dir / "trials" / qid
    result_path = trial_dir / "result.json"

    # Resume: keep completed generation even while official scoring is pending.
    if result_path.exists():
        try:
            previous = json.loads(result_path.read_text(encoding="utf-8"))
            if _reusable_result(previous, qid):
                return previous
        except json.JSONDecodeError:
            pass  # corrupted, re-run

    async with sem:
        # Keep failed/corrupt results for audit; do not read them as a fresh result
        # if the next process dies before it can publish its output.
        if result_path.exists():
            if benchmark == _DRB:
                _archive_drb_attempt(out_dir, qid)
            else:
                archive = out_dir / "previous_attempts" / qid / uuid.uuid4().hex
                archive.mkdir(parents=True, exist_ok=True)
                result_path.replace(archive / "result.json")
        trial_dir.mkdir(parents=True, exist_ok=True)
        log_path = trial_dir / "agent.log"

        log_file = open(log_path, "a", encoding="utf-8")
        tee = None
        proc = None
        try:
            t0 = time.monotonic()
            proc = await asyncio.create_subprocess_exec(
                sys.executable, str(_WORKER_SCRIPT),
                qid, str(out_dir), benchmark,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.STDOUT,
                # Emit Python stacks on native crashes into this attempt's log.
                # These are diagnostics, not a claimed native dependency fix.
                env={**os.environ, "PYTHONFAULTHANDLER": "1", "PYTHONUNBUFFERED": "1"},
            )
            tee = asyncio.create_task(_tee(proc.stdout, log_file, mirror, prefix=f"[{qid}] "))
            exit_code: int | None = None
            try:
                exit_code = await asyncio.wait_for(
                    _wait_process_and_log(proc, tee), timeout=_QUESTION_TIMEOUT,
                )
            except asyncio.TimeoutError:
                proc.kill()
                await proc.wait()
                duration = time.monotonic() - t0
                logger.error(
                    "[%s] subprocess SIGKILLed after %ds timeout",
                    qid, _QUESTION_TIMEOUT,
                )
                synth = {
                    "question_id": qid,
                    "is_correct": None if benchmark in DEFERRED_BENCHMARKS else False,
                    "reward": None if benchmark in DEFERRED_BENCHMARKS else 0,
                    "error": f"subprocess_timeout_{_QUESTION_TIMEOUT}s",
                    "duration_seconds": round(duration, 2),
                    "judge_method": ("not_graded" if benchmark in DEFERRED_BENCHMARKS
                                     else "subprocess_timeout"),
                }
                atomic_write_text(result_path, 
                    json.dumps(synth, indent=2, ensure_ascii=False),
                    encoding="utf-8",
                )
                return synth

            duration = time.monotonic() - t0
        finally:
            primary_error = sys.exc_info()[1]
            cleanup = asyncio.create_task(_settle_worker(proc, tee))
            cancellation = None
            try:
                while not cleanup.done():
                    try:
                        await asyncio.shield(cleanup)
                    except asyncio.CancelledError as exc:
                        cancellation = exc
                cleanup.result()
            finally:
                try:
                    log_file.close()
                except OSError:
                    if primary_error is None and cancellation is None:
                        raise
            if cancellation is not None and primary_error is None:
                raise cancellation

        atomic_write_text(trial_dir / "subprocess.json", json.dumps({
            "pid": proc.pid,
            "exit_code": exit_code,
            "duration_seconds": round(duration, 2),
            "native_crash": exit_code in _NATIVE_CRASH_CODES,
            "native_root_cause": "unknown" if exit_code in _NATIVE_CRASH_CODES else None,
            "faulthandler_enabled": True,
            "finished_at_utc": datetime.now(timezone.utc).isoformat(),
        }, indent=2), encoding="utf-8")

    if result_path.exists():
        try:
            result = json.loads(result_path.read_text(encoding="utf-8"))
            if not isinstance(result, dict) or result.get("question_id") != qid:
                raise ValueError("Invalid result shape or question ID")
            if exit_code in _NATIVE_CRASH_CODES and not _reusable_result(result, qid):
                result["worker_exit_code"] = exit_code
                atomic_write_text(result_path, json.dumps(result, indent=2, ensure_ascii=False))
            return result
        except (ValueError, OSError) as exc:
            logger.error("[%s] result.json unreadable: %s", qid, exc)
            if benchmark == _DRB and result_path.exists():
                result_path.replace(trial_dir / f"unreadable_result_{uuid.uuid4().hex}.json")

    logger.error(
        "[%s] worker exit=%s but no result.json (duration=%.1fs)",
        qid, exit_code, duration,
    )
    synth = {
        "question_id": qid,
        "is_correct": None if benchmark in DEFERRED_BENCHMARKS else False,
        "reward": None if benchmark in DEFERRED_BENCHMARKS else 0,
        "error": f"worker_exit_{exit_code}_no_result",
        "worker_exit_code": exit_code,
        "duration_seconds": round(duration, 2),
        "judge_method": "not_graded" if benchmark in DEFERRED_BENCHMARKS else "worker_crash",
    }
    atomic_write_text(result_path, 
        json.dumps(synth, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return synth


# ── One-run evaluator ────────────────────────────────────────────────

async def run_eval(
    args: argparse.Namespace,
    *,
    out_dir: Path,
    seed: int,
) -> dict:
    """Generate tasks + dispatch subprocesses for one run pass."""
    from benchmarks.core.registry import load_questions, get_config
    from benchmarks.core.harbor_task_generator import generate_task_dirs

    snapshot = getattr(args, "resolved_benchmark_profile", None)
    if snapshot is not None:
        out_dir.mkdir(parents=True, exist_ok=True)
        atomic_write_text(out_dir / "benchmark_profile.json", json.dumps(snapshot, indent=2))
        atomic_write_text(out_dir / "config.json", json.dumps(vars(args), indent=2, default=str))
    cfg = get_config(args.benchmark)
    pipeline = args.pipeline or cfg.default_pipeline or "react_base"
    profile = args.profile or (
        "auto" if pipeline == "dynamic_htn_informed_research"
        else "default"
    )
    if pipeline == "dynamic_htn_informed_research":
        from importlib import import_module

        driver = import_module(f"workflows.{pipeline}.nodes.dynamic_solve")
        selection = driver.prepare_profile_selection({"profile": profile})
        # Preserve built-in names; resolve user paths before passing to children.
        if Path(profile).name != profile or Path(profile).suffix in {".yaml", ".yml"}:
            profile = str(selection.path)
    judge_variant = getattr(args, "judge_variant", None)
    if judge_variant and judge_variant not in JUDGE_VARIANTS.get(args.benchmark, {}):
        raise ValueError(f"Unknown judge variant {judge_variant!r} for {args.benchmark!r}")

    questions = load_questions(
        args.benchmark,
        limit=args.limit,
        offset=args.offset,
        answer_type=args.answer_type,
        category=args.category,
        subset=getattr(args, "subset", None),
    )

    if not args.no_shuffle:
        import random
        random.Random(seed).shuffle(questions)
        logger.info("Shuffled questions with seed=%d", seed)

    if args.limit:
        questions = questions[:args.limit]

    total = len(questions)
    adoption = None
    if getattr(args, "adopt_run", None):
        if pipeline != "dynamic_htn_informed_research" or args.benchmark != "prl_bench" or getattr(args, "runs", 1) != 1:
            raise ValueError("--adopt-run currently supports a single PRL run")
        from benchmarks.runner.adopt import prepare
        adoption = prepare(args.adopt_run, questions, profile, concurrency=args.concurrency,
                           benchmark_profile=snapshot)
        if adoption.source == out_dir.resolve():
            raise ValueError("Adoption requires a new output directory")
        # Acquire slots for already-running pilot questions before new dispatch.
        questions.sort(key=lambda q: q.id not in adoption.ids)
    logger.info("Loaded %d %s questions", total, cfg.name)
    if total == 0:
        logger.error("No questions to run")
        return {}

    tasks_dir = out_dir / "tasks"
    q_dicts = [
        {
            "id": q.id,
            "question": q.question,
            "answer": q.ground_truth,
            "answer_type": q.answer_type,
            "category": q.metadata.get("category", ""),
            "image_path": q.image_path,
            "file_path": q.file_path,
            "file_name": q.file_name,
            # Inject workflow-profile so it lands in task.toml's [metadata]
            # section. Read back by AgentHarnessAgent._read_task_metadata
            # and forwarded into the pipeline state as ``state.metadata.profile``.
            "metadata": {**solver_metadata(q.metadata), "profile": profile,
                         "benchmark": args.benchmark},
            "evaluation_metadata": q.metadata,
            "benchmark_profile": getattr(args, "resolved_benchmark_profile", None),
            "benchmark": args.benchmark,
            "judge_variant": judge_variant,
        }
        for q in questions
    ]
    generate_task_dirs(q_dicts, tasks_dir, pipeline_id=pipeline)

    (out_dir / "trials").mkdir(parents=True, exist_ok=True)

    if args.benchmark == _DRB:
        from benchmarks.watch_drb import refresh_run_summary
        refresh_run_summary(out_dir, generation_report={
            "benchmark": args.benchmark, "pipeline": pipeline, "profile": profile,
            "total": total, "generation_stage": "running", "results": [],
            "config": {"concurrency": args.concurrency, "seed": seed},
        })

    sem = asyncio.Semaphore(args.concurrency)
    mirror = sys.stdout.isatty()
    t_start = time.monotonic()
    completed = 0
    correct_count = 0
    graded_count = 0

    async def wrap(q):
        nonlocal completed, correct_count, graded_count
        result = None
        if adoption is not None and q.id in adoption.ids:
            from benchmarks.runner.adopt import adopt_result
            result = await adopt_result(adoption, q.id, out_dir, sem)
        if result is None:
            result = await run_task_subprocess(q.id, out_dir, args.benchmark, sem, mirror)
        completed += 1
        if result.get("is_correct"):
            correct_count += 1
        graded_count += is_graded(result)
        if completed % 10 == 0 or completed == total:
            elapsed = time.monotonic() - t_start
            logger.info(
                "PROGRESS: %d/%d done, %d/%d graded correct, %d not graded (%.0fs elapsed)",
                completed, total, correct_count,
                graded_count, completed - graded_count, elapsed,
            )
        return result

    results = await asyncio.gather(
        *(wrap(q) for q in questions),
        return_exceptions=True,
    )

    final: list[dict] = []
    for q, r in zip(questions, results):
        if isinstance(r, Exception):
            logger.error("[%s] orchestrator error: %s", q.id, r)
            final.append({
                "question_id": q.id,
                "is_correct": None if args.benchmark in DEFERRED_BENCHMARKS else False,
                "reward": None if args.benchmark in DEFERRED_BENCHMARKS else 0,
                "error": f"orchestrator_{type(r).__name__}: {r}",
                "duration_seconds": 0,
                "judge_method": ("not_graded" if args.benchmark in DEFERRED_BENCHMARKS
                                 else "orchestrator_exception"),
            })
        else:
            final.append(r)

    total_duration = time.monotonic() - t_start

    correct = sum(1 for r in final if r.get("is_correct"))
    attempted = sum(1 for r in final if r.get("predicted_answer"))
    graded = sum(is_graded(r) for r in final)
    not_graded = total - graded
    scalar = args.benchmark == "prl_bench"
    accuracy = correct / graded if graded and not scalar else None
    accuracy_text = f"{accuracy:.2%}" if accuracy is not None else "N/A (not graded)"
    rubric_values = [r["rubric_score"] for r in final if is_graded(r) and r.get("rubric_score") is not None]
    rubric_mean = sum(rubric_values) / len(rubric_values) if rubric_values else None
    if scalar:
        accuracy_text = "N/A (continuous rubric score; no pass threshold)"

    by_method: dict[str, int] = {}
    for r in final:
        m = r.get("judge_method", "unknown")
        by_method[m] = by_method.get(m, 0) + 1

    summary = (
        f"\n{'=' * 60}\n"
        f"  {cfg.name} Results\n"
        f"{'=' * 60}\n"
        f"  Pipeline:     {pipeline}\n"
        f"  Profile:      {profile}\n"
        f"  Total:        {total}\n"
        f"  Attempted:    {attempted}\n"
        f"  Correct:      {correct}\n"
        f"  Graded:       {graded}\n"
        f"  Not graded:   {not_graded}\n"
        f"  Accuracy:     {accuracy_text}\n"
        f"  Duration:     {total_duration:.1f}s\n"
        f"  Concurrency:  {args.concurrency}\n"
        f"\n  Scoring method breakdown:\n"
    )
    for m, cnt in sorted(by_method.items()):
        summary += f"    {m}: {cnt}\n"
    if scalar:
        summary += f"  Local rubric mean (0-100): {rubric_mean}\n"
    summary += f"{'=' * 60}\n"
    if args.benchmark != _DRB:
        logger.info(summary)

    report = {
        "benchmark": args.benchmark,
        "pipeline": pipeline,
        "profile": profile,
        "total": total,
        "attempted": attempted,
        "correct": None if scalar else correct,
        "graded": graded,
        "not_graded": not_graded,
        "judge_variant": "local_equal_criterion_binary_v1" if scalar else judge_variant or "official",
        "accuracy": accuracy,
        "by_method": by_method,
        "duration_seconds": round(total_duration, 2),
        "config": {
            "limit": args.limit,
            "seed": seed,
            "concurrency": args.concurrency,
            "offset": args.offset,
            "subset": getattr(args, "subset", None),
        },
        "started_at": datetime.now(timezone.utc).isoformat(),
        "results": final,
    }
    if scalar:
        report.update(rubric_mean=rubric_mean, scoring_protocol="local_equal_criterion_binary_v1",
                      official_score_reproduction=False)
    if args.benchmark == _DRB:
        report["generation_stage"] = "completed"
        report = refresh_run_summary(out_dir, generation_report=report)
        logger.info("\n%s", (out_dir / "summary.txt").read_text(encoding="utf-8"))
        return report
    atomic_write_text((out_dir / "results.json"), 
        json.dumps(report, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    atomic_write_text((out_dir / "summary.txt"), summary, encoding="utf-8")
    logger.info("Run finished — %d/%d graded correct (%s), %d not graded",
                correct, graded, accuracy_text, not_graded)
    return report


# ── CLI ──────────────────────────────────────────────────────────────

def main() -> None:
    p = argparse.ArgumentParser(allow_abbrev=False,
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--benchmark_profile", required=True,
                   help="Benchmark profile name or YAML path (dataset and judge configuration)")
    p.add_argument("--pipeline", default=None,
                   help="AgentHarness pipeline (default: dataset's default_pipeline)")
    p.add_argument("--profile", "--meta-profile", dest="profile", default=None,
                   help="Workflow profile name or YAML path (dynamic HTN: auto meta profile). "
                        "Meta profiles map task routes to execution profiles. Written into task metadata.")
    p.add_argument("--runs", type=int, default=1,
                   help="Number of independent runs over the question set. "
                        "Each lands in <out>/run_<i>/. Default: 1.")
    p.add_argument("--concurrency", type=int, default=1,
                   help="Per-run concurrent subprocess workers (default: 1). When --runs N, all N runs launch concurrently; total in-flight = runs × concurrency.")
    p.add_argument("--seed", type=int, default=None,
                   help="Shuffle seed (default: 42). For --runs N, each "
                        "run uses seed + run_index - 1.")
    p.add_argument("--no-shuffle", action="store_true",
                   help="Disable question shuffling (run in dataset order)")
    p.add_argument("--adopt-run", default=None,
                   help="Extend a matching PRL pilot, counting its active workers toward concurrency")
    p.add_argument("--out", default=None,
                   help="Output directory (default: benchmarks/results/<benchmark>/<ts>)")
    args = p.parse_args()
    from benchmarks.core.profile import load_benchmark_profile
    try:
        selected = load_benchmark_profile(args.benchmark_profile)
    except (ValueError, OSError) as exc:
        p.error(str(exc))
    args.benchmark = selected.benchmark  # Internal dispatch identity, not a CLI option.
    args.resolved_benchmark_profile = selected.snapshot()
    for key, value in selected.configuration["dataset"].items():
        setattr(args, key, value)
    args.judge_variant = selected.configuration["judge"].get("variant")
    if args.judge_variant == "official":
        args.judge_variant = None
    if args.concurrency < 1 or args.runs < 1:
        p.error("runs and concurrency must be positive")

    if args.out is None:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.out = f"benchmarks/results/{args.benchmark}/{ts}"

    base_out = Path(args.out).resolve()
    base_out.mkdir(parents=True, exist_ok=True)
    snapshot_path = base_out / "benchmark_profile.json"
    if snapshot_path.exists():
        previous = json.loads(snapshot_path.read_text())
        if previous.get("configuration") != selected.configuration:
            p.error("Output directory belongs to a different benchmark profile; use a new --out")
    atomic_write_text(snapshot_path, json.dumps(selected.snapshot(), indent=2))
    atomic_write_text((base_out / "config.json"), 
        json.dumps(vars(args), indent=2, default=str),
        encoding="utf-8",
    )

    base_seed = args.seed if args.seed is not None else 42

    if args.runs == 1:
        asyncio.run(run_eval(args, out_dir=base_out, seed=base_seed))
    else:
        # All N runs concurrently. Each run has its own ``asyncio.Semaphore``
        # of ``args.concurrency`` — total subprocesses in-flight =
        # args.runs × args.concurrency.
        async def _run_all() -> None:
            coros = []
            for run_idx in range(1, args.runs + 1):
                run_out = base_out / f"run_{run_idx}"
                run_out.mkdir(parents=True, exist_ok=True)
                seed = base_seed + run_idx - 1
                logger.info("queue run %d / %d  (seed=%d, out=%s)",
                            run_idx, args.runs, seed, run_out)
                coros.append(run_eval(args, out_dir=run_out, seed=seed))
            await asyncio.gather(*coros)

        asyncio.run(_run_all())


if __name__ == "__main__":
    main()
