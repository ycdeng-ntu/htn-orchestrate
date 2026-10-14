#!/usr/bin/env python3
"""Shared Docker runtime. Uses the project registry and PyYAML to snapshot benchmark profiles on the host."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
from datetime import datetime, timezone
import uuid

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
NAME = 'agentharness-benchmarks'
IMAGE = 'frontieragent-eval-local:latest'
LABEL = 'agentharness.workspace'
OUT = ROOT / 'results' / 'docker'


def docker(*args, capture=False):
    return subprocess.run(['docker', *map(str, args)], check=True,
                          capture_output=capture, text=True)


def inspect():
    p = subprocess.run(['docker', 'container', 'inspect', NAME],
                       capture_output=True, text=True)
    if p.returncode:
        if 'No such' in p.stderr:
            return None
        raise RuntimeError(p.stderr.strip())
    obj = json.loads(p.stdout)[0]
    if (obj['Config'].get('Labels') or {}).get(LABEL) != str(ROOT):
        raise RuntimeError(f'{NAME} belongs to another workspace; refusing to use it')
    return obj


def mounts():
    names = ['agent_harness', 'benchmarks', 'plugins', 'workflows', 'data',
             'scripts', 'tests', 'pyproject.toml', '.env']
    return [(ROOT / n, '/workspace/' + n, True) for n in names] + [
        (OUT, '/workspace/results', False)]


def validate(obj):
    expected = {(str(p.resolve()), dst, not ro) for p, dst, ro in mounts()}
    # DSW's Docker proxy reports bind sources under its /mainroot prefix.
    actual = {(m['Source'].removeprefix('/mainroot') if m['Source'].startswith('/mainroot/')
               else m['Source'], m['Destination'], m['RW']) for m in obj['Mounts']}
    hc = obj['HostConfig']
    if actual != expected or obj['Config']['User'] != '65534:65534':
        raise RuntimeError('Container mount/user configuration differs; stop and recreate it')
    if hc.get('Privileged') or hc.get('PidMode') == 'host' or hc.get('NetworkMode') == 'host':
        raise RuntimeError('Unexpected privileged/host namespace configuration')
    if any(hc.get(k) for k in ('Memory', 'NanoCpus', 'CpuQuota', 'CpusetCpus')):
        raise RuntimeError('Unexpected CPU/memory limit')
    if hc.get('PidsLimit') not in (None, 0, -1):
        raise RuntimeError('Unexpected process limit')


def up():
    obj = inspect()
    if obj:
        validate(obj)
        if not obj['State']['Running']:
            docker('start', NAME)
        return
    if not OUT.exists():
        OUT.mkdir(parents=True)
        os.chown(OUT, 65534, 65534)
    if OUT.stat().st_uid != 65534 or OUT.is_symlink():
        raise RuntimeError(f'{OUT} must be a dedicated directory owned by uid 65534')
    for src, _, _ in mounts():
        if not src.exists() or src.is_symlink():
            raise RuntimeError(f'Missing or symlink mount source: {src}')
    args = ['run', '-d', '--pull=never', '--name', NAME,
            '--label', f'{LABEL}={ROOT}', '--user', '65534:65534',
            '--workdir', '/workspace', '--env', 'HOME=/tmp',
            '--env', 'PYTHONPATH=/workspace', '--env', 'PYTHONDONTWRITEBYTECODE=1',
            '--env', 'PYTHONUNBUFFERED=1', '--entrypoint', 'python']
    for src, dst, ro in mounts():
        args += ['--mount', f'type=bind,src={src},dst={dst}' + (',readonly' if ro else '')]
    args += [IMAGE, '-c', 'import time; time.sleep(10**9)']
    docker(*args)
    validate(inspect())


def validate_source_generation():
    """Do not launch new work through a bind mount retaining an older source root."""
    marker = ROOT / 'workflows' / '.source_generation'
    try:
        expected = marker.read_text(encoding='utf-8').strip()
    except FileNotFoundError:
        return  # Workspaces predating source generations remain compatible.
    except (OSError, UnicodeError) as exc:
        raise RuntimeError(
            'Cannot read the host workflows source-generation marker; '
            'no new benchmark was started. Running jobs are untouched.'
        ) from exc
    if not expected:
        raise RuntimeError(
            'The host workflows source-generation marker is empty; '
            'no new benchmark was started. Running jobs are untouched.'
        )
    script = (
        'from pathlib import Path\n'
        'try:\n'
        '    value = Path("/workspace/workflows/.source_generation").read_text(encoding="utf-8")\n'
        'except FileNotFoundError:\n'
        '    value = ""\n'
        'print(value, end="")\n'
    )
    try:
        actual = docker('exec', NAME, 'python', '-c', script, capture=True).stdout.strip()
    except (OSError, UnicodeError, subprocess.CalledProcessError) as exc:
        raise RuntimeError(
            'Cannot verify the container source generation; no new benchmark was started. '
            'Running jobs are untouched. After they finish, recreate the container '
            'before starting new runs.'
        ) from exc
    if actual != expected:
        raise RuntimeError(
            'The existing container uses retained old source; no new benchmark was started. '
            'Running jobs are untouched. After they finish, recreate the container '
            'before starting new runs.'
        )


def option(args, key):
    positions = [i for i, a in enumerate(args) if a == key or a.startswith(key + '=')]
    if len(positions) > 1:
        raise ValueError(f'Duplicate {key}')
    if not positions:
        return None, None
    i = positions[0]
    if '=' in args[i]:
        return i, args[i].split('=', 1)[1]
    if i + 1 >= len(args):
        raise ValueError(f'Missing value for {key}')
    return i, args[i + 1]


def replace_option(args, key, value):
    i, _ = option(args, key)
    if i is None:
        args.extend([key, value])
    elif '=' in args[i]:
        args[i] = key + '=' + value
    else:
        args[i + 1] = value


def prepare(args, job):
    args = list(args)
    _, out = option(args, '--out')
    out = out or f'results/{job}'
    p = Path(out)
    p = p if p.is_absolute() else Path('/workspace') / p
    if ('..' in p.parts or not p.is_relative_to('/workspace/results')
            or p == Path('/workspace/results')):
        raise ValueError('--out must be results/<run> or /workspace/results/<run>')
    replace_option(args, '--out', str(p))
    _, profile_name = option(args, '--benchmark_profile')
    if not profile_name:
        raise ValueError('--benchmark_profile is required')
    from benchmarks.core.profile import load_benchmark_profile
    import yaml
    profile = load_benchmark_profile(profile_name)
    config = profile.configuration
    subset = config['dataset']['subset']
    directory = ROOT / 'data' / 'docker_inputs'
    directory.mkdir(parents=True, exist_ok=True)
    if subset:
        src = Path(subset).resolve(strict=True)
        dest = directory / (job + '.jsonl')
        shutil.copyfile(src, dest)
        dest.chmod(0o644)
        config['dataset']['subset'] = '/workspace/data/docker_inputs/' + dest.name
    dest = directory / (job + '.benchmark_profile.yaml')
    dest.write_text(yaml.safe_dump(config, sort_keys=False))
    dest.chmod(0o644)
    replace_option(args, '--benchmark_profile', '/workspace/data/docker_inputs/' + dest.name)
    return args


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('action', choices=['up', 'run', 'exec', 'status', 'stop'])
    p.add_argument('args', nargs=argparse.REMAINDER)
    ns = p.parse_args()
    args = ns.args[1:] if ns.args[:1] == ['--'] else ns.args
    if ns.action == 'up':
        up()
        print(f'Ready: {NAME}')
    elif ns.action == 'run':
        if not args:
            p.error('run requires benchmark runner arguments')
        job = datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S') + '_' + uuid.uuid4().hex[:8]
        args = prepare(args, job)
        up()
        validate_source_generation()
        docker('exec', '-d', NAME, 'python', 'scripts/docker_job.py', job, *args)
        print(f'Job: {job}\nLog/status: {OUT}/.jobs/{job}\nResults: {option(args, "--out")[1]}')
    elif ns.action == 'exec':
        if not args:
            p.error('exec requires a command')
        up()
        docker('exec', NAME, *args)
    elif ns.action == 'status':
        obj = inspect()
        print('not created' if obj is None else obj['State']['Status'])
        if obj and obj['State']['Running']:
            docker('top', NAME, '-eo', 'pid,etime,args')
    else:
        if inspect():
            docker('stop', '--time', '30', NAME)
            docker('rm', NAME)
            for path in (OUT / '.jobs').glob('*/status.json'):
                state = json.loads(path.read_text())
                if state.get('status') == 'running':
                    state.update(status='interrupted', reason='shared container stopped')
                    temp = path.with_suffix('.tmp')
                    temp.write_text(json.dumps(state, indent=2))
                    temp.replace(path)


if __name__ == '__main__':
    try:
        main()
    except (RuntimeError, ValueError, OSError, subprocess.CalledProcessError) as exc:
        print(str(exc), file=sys.stderr)
        sys.exit(1)
