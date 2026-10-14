"""Benchmark-owned dataset and scoring configuration, independent of solvers."""
from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
import hashlib
from pathlib import Path
from typing import Any

import yaml

from benchmarks.core.registry import REGISTRY

PROFILES_DIR = Path(__file__).resolve().parents[1] / 'profiles'
DATA_KEYS = frozenset({'subset', 'limit', 'offset', 'answer_type', 'category'})
JUDGE_VARIANT_NAMES = {
    'browsecomp_200': frozenset({'semantic'}),
    'frontier_science_research': frozenset({'agent_team_20260921'}),
}
JUDGE_KEYS = frozenset({'provider', 'model', 'reasoning_effort', 'variant'})


def validate_judge(config: Any) -> dict:
    if not isinstance(config, dict) or set(config) - JUDGE_KEYS:
        raise ValueError('judge must contain only provider, model, reasoning_effort and variant; credentials belong in environment variables')
    out = deepcopy(config)
    for key in ('provider', 'model'):
        if not isinstance(out.get(key), str) or not out[key].strip():
            raise ValueError(f'judge.{key} must be a nonempty string')
        out[key] = out[key].strip()
    if out['provider'] not in {'openai', 'openrouter', 'judge'}:
        raise ValueError('judge.provider must be openai, openrouter, or judge (JUDGE_* transport)')
    if out.get('reasoning_effort') not in (None, 'low', 'medium', 'high', 'xhigh'):
        raise ValueError('Invalid judge.reasoning_effort')
    if out.get('variant') is not None and not isinstance(out['variant'], str):
        raise ValueError('judge.variant must be a string or null')
    return out


def validate_config(raw: Any, *, directory: Path) -> dict:
    if not isinstance(raw, dict) or set(raw) - {'benchmark', 'dataset', 'judge', 'web_policy'}:
        raise ValueError('Benchmark profile allows only benchmark, dataset, judge, web_policy; solver and run settings belong elsewhere')
    benchmark = raw.get('benchmark')
    if not isinstance(benchmark, str) or benchmark not in REGISTRY:
        raise ValueError(f'Unknown benchmark type: {benchmark!r}')
    dataset = raw.get('dataset', {})
    if not isinstance(dataset, dict) or set(dataset) - DATA_KEYS:
        raise ValueError('Invalid dataset selection fields')
    dataset = {'subset': None, 'limit': None, 'offset': 0, 'answer_type': None, 'category': None, **dataset}
    for key in ('limit', 'offset'):
        value = dataset[key]
        if key == 'limit' and value is None:
            continue
        if type(value) is not int or value < (1 if key == 'limit' else 0):
            raise ValueError(f'dataset.{key} must be a valid nonnegative integer (limit must be positive)')
    for key in ('subset', 'answer_type', 'category'):
        if dataset[key] is not None and (not isinstance(dataset[key], str) or not dataset[key].strip()):
            raise ValueError(f'dataset.{key} must be a nonempty string or null')
    if dataset['subset'] is not None:
        path = Path(dataset['subset']).expanduser()
        dataset['subset'] = str((directory / path).resolve() if not path.is_absolute() else path.resolve())
    judge = validate_judge(raw.get('judge'))
    if benchmark in {'frontier_science_research', 'prl_bench', 'deepresearch_bench_ii'}:
        if judge['provider'] == 'judge' or not judge.get('reasoning_effort'):
            raise ValueError(f'{benchmark} requires openai/openrouter and an explicit judge reasoning_effort')
    variant = judge.get('variant')
    if variant not in (None, 'official') and variant not in JUDGE_VARIANT_NAMES.get(benchmark, ()):
        raise ValueError(f'Unknown judge variant {variant!r} for {benchmark}')
    from agent_harness.core.web_policy import validate_web_policy
    return {'benchmark': benchmark, 'dataset': dataset, 'judge': judge,
            'web_policy': validate_web_policy(raw.get('web_policy'))}


@dataclass(frozen=True)
class BenchmarkProfile:
    path: Path
    sha256: str
    configuration: dict

    @property
    def benchmark(self) -> str:
        return self.configuration['benchmark']

    def snapshot(self) -> dict:
        return {'path': str(self.path), 'sha256': self.sha256, 'configuration': deepcopy(self.configuration)}


def load_benchmark_profile(name: str | Path) -> BenchmarkProfile:
    path = Path(name).expanduser()
    if path.name == str(path) and path.suffix not in {'.yaml', '.yml'}:
        path = PROFILES_DIR / (str(path) + '.yaml')
    path = path.resolve()
    content = path.read_bytes()
    try:
        raw = yaml.safe_load(content)
    except yaml.YAMLError as exc:
        raise ValueError(f'Invalid benchmark profile YAML: {path}') from exc
    config = validate_config(raw, directory=path.parent)
    return BenchmarkProfile(path, hashlib.sha256(content).hexdigest(), config)


def snapshot_config(snapshot: dict, benchmark: str | None = None) -> dict:
    if not isinstance(snapshot, dict) or not isinstance(snapshot.get('configuration'), dict):
        raise ValueError('Missing resolved benchmark profile snapshot')
    config = validate_config(snapshot['configuration'], directory=Path(snapshot.get('path') or '.').parent)
    if benchmark is not None and config['benchmark'] != benchmark:
        raise ValueError('Benchmark profile snapshot identity mismatch')
    return config
