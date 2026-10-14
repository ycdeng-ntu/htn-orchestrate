"""Resolve scoring settings exclusively from saved evaluation inputs."""
from __future__ import annotations

import json
from pathlib import Path

from benchmarks.core.profile import snapshot_config, validate_judge


def _safe_fsr_config(config: dict) -> dict:
    # Public helper retained internally for validating explicit FSR judge calls.
    clean = validate_judge({k: v for k, v in config.items()
                            if k in {'provider', 'model', 'reasoning_effort', 'variant'}})
    if not clean.get('reasoning_effort'):
        raise ValueError('FSR judge requires an explicit reasoning_effort')
    return clean


def resolve_judge_config(benchmark: str, task_dir: Path, *, saved_config: dict | None = None) -> dict | None:
    if saved_config is not None:
        return validate_judge({k: v for k, v in saved_config.items()
                               if k in {'provider', 'model', 'reasoning_effort', 'variant'}})
    path = task_dir / 'tests' / 'evaluation.json'
    if not path.exists():
        return None
    evaluation = json.loads(path.read_text())
    snapshot = evaluation.get('benchmark_profile')
    if snapshot is None:
        return None
    return snapshot_config(snapshot, benchmark)['judge']
