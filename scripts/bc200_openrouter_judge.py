"""Check or rejudge BC200 with GPT-5.4 using the configured OpenRouter key."""
from __future__ import annotations

import argparse
import asyncio
import json
import os
from pathlib import Path
import sys

from dotenv import dotenv_values

ROOT = Path(__file__).resolve().parents[1]


def configure():
    source = dotenv_values(ROOT / ".env")
    for field in ("OPENROUTER_API_KEY", "OPENROUTER_BASE_URL"):
        if not source.get(field):
            raise ValueError(f"Missing {field}")
    os.environ["JUDGE_API_KEY"] = source["OPENROUTER_API_KEY"]
    os.environ["JUDGE_BASE_URL"] = source["OPENROUTER_BASE_URL"]
    os.environ["BC200_JUDGE_MODEL"] = "gpt-5.4"
    os.environ["PYTHONPATH"] = str(ROOT)
    sys.path.insert(0, str(ROOT))


async def check():
    from benchmarks.judges.browsecomp import verify_browsecomp_semantic

    verdict = await asyncio.wait_for(
        verify_browsecomp_semantic("What is 1+1?", "2", "2"), timeout=90,
    )
    print(json.dumps({"provider": "openrouter", "credential": "OPENROUTER_API_KEY",
                      "model": "gpt-5.4", "verdict": verdict}))
    return 0 if verdict == "CORRECT" else 1


async def rejudge_missing(out, concurrency):
    import collections
    import fcntl
    import shutil
    from datetime import datetime, timezone
    from benchmarks.core.contracts import is_graded
    from benchmarks.runner.rejudge import _rejudge_one, _atomic_write

    lock = (out / "rejudge_openrouter.lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    paths = sorted((out / "trials").glob("*/result.json"))
    pending = [p for p in paths if not is_graded(json.loads(p.read_text()))]
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    backup = out / ("before_openrouter_rejudge_" + stamp)
    backup.mkdir()
    for p in pending:
        dest = backup / "trials" / p.parent.name / p.name
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(p, dest)
    for name in ("results.json", "summary.txt", "run_status.json"):
        if (out / name).exists():
            shutil.copy2(out / name, backup / name)
    status = {"phase": "running", "pid": os.getpid(), "total": len(pending),
              "completed": 0, "failed": 0, "provider": "openrouter", "model": "gpt-5.4",
              "credential": "OPENROUTER_API_KEY", "backup": str(backup)}

    def write_status():
        status["updated_at_utc"] = datetime.now(timezone.utc).isoformat()
        _atomic_write(out / "rejudge_status.json", status)
        print(json.dumps(status), flush=True)

    write_status()
    limit = asyncio.Semaphore(concurrency)

    async def one(p):
        async with limit:
            try:
                # Select only ungraded trials; force bypasses the legacy field-presence skip.
                outcome = await asyncio.wait_for(
                    _rejudge_one(p, "browsecomp_200", {}, asyncio.Semaphore(1), True, "semantic"),
                    timeout=240,
                )
            except Exception:
                outcome = "fail"
            status["completed"] += 1
            status["failed"] += outcome != "ok"
            write_status()

    await asyncio.gather(*(one(p) for p in pending))
    rows = [json.loads(p.read_text()) for p in paths]
    graded = sum(is_graded(r) for r in rows)
    correct = sum(r.get("is_correct") is True for r in rows)
    report = json.loads((out / "results.json").read_text())
    report.update(results=rows, correct=correct, graded=graded, not_graded=len(rows)-graded,
                  accuracy=correct/graded if graded else None,
                  by_method=dict(collections.Counter(r.get("judge_method") for r in rows)),
                  rejudge={"provider": "openrouter", "model": "gpt-5.4", "updated_trials": len(pending)})
    _atomic_write(out / "results.json", report)
    (out / "summary.txt").write_text(
        f"BC200 after OpenRouter rejudge: {correct}/{graded} graded correct; "
        f"{len(rows)-graded} not graded; total {len(rows)}.\n"
    )
    run_status = json.loads((out / "run_status.json").read_text())
    errors = sum(bool(r.get("scoring_error")) for r in rows)
    run_status.update(correct=correct, graded=graded, not_graded=len(rows)-graded,
                      scoring_errors=errors, accuracy=report["accuracy"],
                      phase="finished_with_errors" if errors or len(rows)!=graded or run_status.get("execution_errors") else "finished",
                      updated_at_utc=datetime.now(timezone.utc).isoformat())
    _atomic_write(out / "run_status.json", run_status)
    status.update(phase="finished_with_errors" if status["failed"] else "finished",
                  graded=graded, correct=correct, not_graded=len(rows)-graded)
    write_status()
    return 1 if status["failed"] else 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rejudge", type=Path, help="BC200 result directory; default only checks the judge")
    parser.add_argument("--concurrency", type=int, default=4)
    args = parser.parse_args()
    configure()
    if args.rejudge:
        config = json.loads((args.rejudge / "config.json").read_text())
        if config.get("benchmark") != "browsecomp_200":
            raise ValueError("This launcher only supports browsecomp_200")
        return asyncio.run(rejudge_missing(args.rejudge.resolve(), args.concurrency))
    return asyncio.run(check())


if __name__ == "__main__":
    sys.exit(main())
