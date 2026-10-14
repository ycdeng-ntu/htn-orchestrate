"""Backfill failed FSR grades while an existing Docker job finishes. No solving."""
import asyncio
import json
import logging
from pathlib import Path
import sys
import time
from collections import Counter

from dotenv import load_dotenv
from agent_harness.core.atomic_files import atomic_write_text
from benchmarks.runner.rejudge import _rejudge_one

load_dotenv()
logging.basicConfig(level=logging.INFO)
out, job_state = map(Path, sys.argv[1:])
status_path = out / 'backfill_status.json'
attempted = set()


def write(path, data):
    atomic_write_text(path, json.dumps(data, ensure_ascii=False, indent=2))


def refresh_summary():
    report_path = out / 'results.json'
    if not report_path.exists():
        return
    report = json.loads(report_path.read_text())
    rows = [json.loads(p.read_text()) for p in sorted((out / 'trials').glob('*/result.json'))]
    correct = sum(r.get('is_correct') is True for r in rows)
    graded = sum(isinstance(r.get('is_correct'), bool) for r in rows)
    report.update(results=rows, correct=correct, graded=graded, not_graded=len(rows)-graded,
                  accuracy=correct / graded if graded else None,
                  by_method=dict(Counter(r.get('judge_method', 'unknown') for r in rows)))
    write(report_path, report)
    atomic_write_text(out / 'summary.txt',
                      f'FSR after scoring repair\nTotal: {report["total"]}\n'
                      f'Correct: {correct}\nGraded: {graded}\nNot graded: {len(rows)-graded}\n'
                      f'Accuracy among graded: {report["accuracy"]}\n')


async def main():
    while True:
        pending = []
        for p in sorted((out / 'trials').glob('*/result.json')):
            r = json.loads(p.read_text())
            if (r.get('predicted_answer') and not r.get('error')
                    and (r.get('scoring_error') or not isinstance(r.get('is_correct'), bool))
                    and str(p) not in attempted):
                pending.append(p)
        for p in pending:
            attempted.add(str(p))
        outcomes = await asyncio.gather(*(
            _rejudge_one(p, 'frontier_science_research', {}, sem, False) for p in pending))
        if pending:
            logging.info('Backfill results: %s', list(zip([p.parent.name for p in pending], outcomes)))
        state = json.loads(job_state.read_text()).get('status')
        rows = [json.loads(p.read_text()) for p in (out / 'trials').glob('*/result.json')]
        failed = sum(bool(r.get('scoring_error')) for r in rows)
        write(status_path, {'status': 'watching' if state == 'running' else 'finished',
                            'updated_at': time.time(), 'attempted': len(attempted),
                            'remaining_scoring_errors': failed, 'runner_status': state})
        if state != 'running':
            fresh = [p for p in (out / 'trials').glob('*/result.json') if str(p) not in attempted]
            if any((r := json.loads(p.read_text())).get('predicted_answer')
                   and not r.get('error') and (r.get('scoring_error')
                   or not isinstance(r.get('is_correct'), bool)) for p in fresh):
                continue
            # One final pass sees results published just before the runner exited.
            if pending:
                continue
            refresh_summary()
            return
        await asyncio.sleep(30)


sem = asyncio.Semaphore(4)
asyncio.run(main())
