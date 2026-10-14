"""Offline integration checks; no model/API calls."""
import asyncio
import errno
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile

from dotenv import dotenv_values
from benchmarks.runner import run_subprocess as runner
from workflows.dynamic_htn_informed_research import profile  # noqa: F401

assert os.getuid() == 65534
assert not Path('/var/run/docker.sock').exists()
assert not Path('/mnt/agent-framework').exists()
assert dotenv_values('/workspace/.env')  # Validate reading without printing credentials.
for directory in ('agent_harness', 'workflows', 'data', 'benchmarks'):
    target = Path('/workspace') / directory / '.docker_write_probe'
    try:
        target.write_text('unexpected')
    except OSError as exc:
        assert exc.errno in (errno.EROFS, errno.EACCES), exc
    else:
        target.unlink()
        raise AssertionError(f'{directory} was writable')
with tempfile.TemporaryDirectory(prefix='docker-smoke-', dir='/workspace/results') as out:
    root = Path(out)
    worker = root / 'worker.py'
    worker.write_text('''import json, os, sys
from pathlib import Path
qid, out, _ = sys.argv[1:]
p = Path(out) / 'trials' / qid / 'result.json'
p.write_text(json.dumps({'question_id': qid, 'is_correct': True, 'uid': os.getuid()}))
''')
    runner._WORKER_SCRIPT = worker
    result = asyncio.run(runner.run_task_subprocess('smoke', root, 'browsecomp', asyncio.Semaphore(1)))
    assert result['uid'] == 65534 and result['is_correct'] is True
subprocess.run([sys.executable, '-m', 'benchmarks.runner.run_one_task', '--help'],
               stdout=subprocess.DEVNULL, check=True)
print(json.dumps({'nonroot': True, 'code_and_data_readonly': True,
                  'env_readable': True, 'results_writable': True,
                  'worker_subprocess': 'passed', 'worker_cli': 'passed'}))
