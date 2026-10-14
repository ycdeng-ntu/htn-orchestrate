"""Run a detached benchmark with persistent logs and exit status inside Docker."""
import fcntl
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time

from agent_harness.core.atomic_files import atomic_write_text

job, *args = sys.argv[1:]
directory = Path('/workspace/results/.jobs') / job
directory.mkdir(parents=True, exist_ok=False)
state = {'job': job, 'args': args, 'started_at': time.time(), 'status': 'running'}


def save():
    atomic_write_text(directory / 'status.json', json.dumps(state, indent=2))


save()
try:
    from scripts.benchmark_credentials import configure
    profile = args[args.index('--profile') + 1] if '--profile' in args else next(
        (a.split('=', 1)[1] for a in args if a.startswith('--profile=')), None)
    configure(profile=profile)
    output = args[args.index('--out') + 1] if '--out' in args else next(
        a.split('=', 1)[1] for a in args if a.startswith('--out='))
    lock_dir = Path('/workspace/results/.locks')
    lock_dir.mkdir(exist_ok=True)
    lock = (lock_dir / hashlib.sha256(output.encode()).hexdigest()).open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    with (directory / 'runner.log').open('w') as log:
        result = subprocess.run(
            [sys.executable, '-m', 'benchmarks.runner.run_subprocess', *args],
            stdout=log, stderr=subprocess.STDOUT,
        )
    state.update(status='completed' if result.returncode == 0 else 'failed',
                 exit_code=result.returncode)
except Exception as exc:
    state.update(status='failed', error=f'{type(exc).__name__}: {exc}')
    raise
finally:
    state['finished_at'] = time.time()
    save()
