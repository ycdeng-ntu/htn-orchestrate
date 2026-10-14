from __future__ import annotations

import asyncio
from importlib import import_module
import json
from pathlib import Path
import sys
import tomllib

import pytest
import yaml

from benchmarks.core.harbor_task_generator import generate_task_dirs
from benchmarks.core.profile import load_benchmark_profile, snapshot_config, validate_config
from benchmarks.judges import score_answer
from benchmarks.judges import _common
from benchmarks.runner.judge_config import resolve_judge_config

CONFIG = {"benchmark": "browsecomp_200", "dataset": {"subset": "input.jsonl", "limit": 2},
          "judge": {"provider": "openrouter", "model": "custom-judge", "reasoning_effort": "low"},
          "web_policy": {"blocked_domains": ["blocked.example"], "blocked_terms": ["answer key"]}}


def test_profile_resolution_snapshot_and_verifier_boundary(tmp_path):
    path = tmp_path / 'benchmark.yaml'
    path.write_text(yaml.safe_dump(CONFIG))
    selected = load_benchmark_profile(path)
    snapshot = selected.snapshot()
    assert selected.configuration['dataset']['subset'] == str(tmp_path / 'input.jsonl')
    path.unlink()  # Regrading must be independent of mutable source files.
    assert snapshot_config(snapshot) == selected.configuration
    questions = [{"id": "q", "question": "public question", "answer": "gold-secret",
                  "benchmark_profile": snapshot, "benchmark": "browsecomp_200",
                  "metadata": {"profile": "auto", "judge": "untrusted", "benchmark_web_policy": {}}}]
    generate_task_dirs(questions, tmp_path / 'tasks', pipeline_id='dynamic_htn_informed_research')
    task = tmp_path / 'tasks/q'
    metadata = tomllib.loads((task / 'task.toml').read_text())['metadata']
    assert json.loads(metadata['benchmark_web_policy']) == {**CONFIG['web_policy'], 'network_enabled': True}
    assert 'judge' not in metadata and 'benchmark_profile' not in metadata
    assert 'gold-secret' not in json.dumps(metadata)
    assert 'custom-judge' not in (task / 'instruction.md').read_text()
    assert resolve_judge_config('browsecomp_200', task) == CONFIG['judge']
    with pytest.raises(ValueError, match='does not support'):
        generate_task_dirs(questions, tmp_path / 'unsupported', pipeline_id='react_base')


@pytest.mark.parametrize('bad', [
    {'profile': 'auto'}, {'dataset': {'limit': True}}, {'dataset': {'offset': -1}},
    {'dataset': {'subset': []}}, {'judge': {'api_key': 'never-snapshot-this'}},
    {'judge': {'provider': 'openrouter', 'model': 'test', 'reasoning_effort': []}},
    {'web_policy': {'blocked_domains': 'not-a-list'}}, {'web_policy': {'judge': 'secret'}},
])
def test_invalid_benchmark_configuration_rejected(tmp_path, bad):
    with pytest.raises(ValueError):
        validate_config({**CONFIG, **bad}, directory=tmp_path)


@pytest.mark.parametrize('setting', [{'judge': {}}, {'evaluation': {}},
                                     {'web': {'blocked_domains': []}}])
def test_solver_profiles_cannot_own_benchmark_settings(tmp_path, setting):
    from workflows.dynamic_htn_informed_research.profile import load_profile_selection
    path = tmp_path / 'solver.yaml'
    path.write_text(yaml.safe_dump(setting))
    with pytest.raises(ValueError, match='benchmark_profile'):
        load_profile_selection({'profile': str(path)})
    with pytest.raises(ValueError, match='benchmark_profile'):
        load_profile_selection({'profile': 'auto', 'dynir_config': setting})


@pytest.mark.asyncio
async def test_concurrent_benchmark_judges_use_separate_model_effort_and_endpoint(monkeypatch):
    from benchmarks.judges import JUDGE_REGISTRY
    calls = []
    monkeypatch.setenv('OPENROUTER_API_KEY', 'router-test')
    monkeypatch.setenv('OPENROUTER_BASE_URL', 'https://router.test/v1')
    monkeypatch.setenv('OPENAI_API_KEY', 'openai-test')
    monkeypatch.setenv('OPENAI_BASE_URL', 'https://openai.test/v1')
    monkeypatch.delenv('NEW_API_API_KEY', raising=False)

    async def judge(*args):
        await asyncio.sleep(0)
        kwargs = _common._build_judge_kwargs('q', max_completion_tokens=10, default_model='wrong')
        calls.append((kwargs, _common.judge_transport()))
        return 'CORRECT'

    monkeypatch.setitem(JUDGE_REGISTRY, 'browsecomp_200', judge)
    monkeypatch.setitem(JUDGE_REGISTRY, 'frontier_science_olympiad', judge)
    configs = [CONFIG['judge'], {'provider': 'openai', 'model': 'second-judge', 'reasoning_effort': 'high'}]
    assert await asyncio.gather(*[
        score_answer(bench, 'q', 'gold', 'answer', judge_config=config)
        for bench, config in zip(['browsecomp_200', 'frontier_science_olympiad'], configs)
    ]) == [('CORRECT', None), ('CORRECT', None)]
    assert {(request['model'], request['reasoning_effort'], transport) for request, transport in calls} == {
        ('custom-judge', 'low', ('router-test', 'https://router.test/v1')),
        ('second-judge', 'high', ('openai-test', 'https://openai.test/v1'))}
    assert _common._JUDGE_CONFIG.get() is None


@pytest.mark.parametrize('package', ['dynamic_htn_informed_research'])
def test_web_policy_is_independent_of_solver_and_does_not_leak_between_tasks(package):
    config_cls = import_module(f'workflows.{package}.tools.runtime').ResearchToolConfig
    first = config_cls.from_profile({'web': {'content_cache_enabled': False}},
                                    web_policy=CONFIG['web_policy'])
    second = config_cls.from_profile({}, web_policy={'blocked_domains': ['different.example']})
    assert first.blocked_needles == ('blocked.example', 'answer key')
    assert second.blocked_needles == ('different.example',)


def test_all_builtin_profiles_load_and_semantic_variant_registry_agrees():
    from benchmarks.core.profile import PROFILES_DIR, JUDGE_VARIANT_NAMES
    from benchmarks.judges import JUDGE_VARIANTS
    for path in PROFILES_DIR.glob('*.yaml'):
        load_benchmark_profile(path)
    assert JUDGE_VARIANT_NAMES == {key: frozenset(value) for key, value in JUDGE_VARIANTS.items()}
    assert 'superchem' in load_benchmark_profile('superchem256').configuration['web_policy']['blocked_terms']
    assert 'browsecomp' not in load_benchmark_profile('deepsearchqa').configuration['web_policy']['blocked_terms']


def test_removed_benchmark_flag_is_not_an_abbreviation(monkeypatch, capsys):
    from benchmarks.runner import run_subprocess
    monkeypatch.setattr(sys, 'argv', ['run_subprocess', '--benchmark', 'browsecomp'])
    with pytest.raises(SystemExit) as error:
        run_subprocess.main()
    assert error.value.code == 2
    assert '--benchmark_profile' in capsys.readouterr().err


@pytest.mark.asyncio
@pytest.mark.parametrize('effort', [None, 'low', 'high'])
async def test_judge_effort_in_profile_is_the_actual_wire_parameter(effort):
    from types import SimpleNamespace
    calls = []

    async def create(**kwargs):
        calls.append(kwargs)
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content='CORRECT'))])

    client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
    config = {'provider': 'judge', 'model': 'gpt-5.4', 'reasoning_effort': effort}
    with _common.judge_scope(config):
        kwargs = _common._build_judge_kwargs('q', max_completion_tokens=10)
        assert await _common._judge_call_with_effort_fallback(kwargs, label='test', client=client) == 'CORRECT'
    assert calls[0].get('reasoning_effort') == effort
    if effort is None:
        assert 'reasoning_effort' not in calls[0]


def test_every_research_solver_yaml_is_valid_and_benchmark_free():
    from workflows.dynamic_htn_informed_research.profile import load_profile_selection
    directory = Path(__file__).resolve().parents[2] / 'workflows/dynamic_htn_informed_research/profiles'
    for path in directory.glob('*.yaml'):
        selected = load_profile_selection({'profile': str(path)})
        for config in [selected.initial_profile, *selected.routes.values()]:
            assert not (set(config) & {'judge', 'evaluation', 'web_policy'})
            assert not (set(config.get('web') or {}) & {'blocked_domains', 'blocked_terms'})


@pytest.mark.parametrize("profile_name", ["fso86", "frontier_science_olympiad"])
def test_fso_profiles_project_offline_permission_to_generated_tasks(tmp_path, profile_name):
    selected = load_benchmark_profile(profile_name)
    assert selected.configuration["web_policy"]["network_enabled"] is False
    generate_task_dirs([{
        "id": "offline-fso", "question": "Compute six times seven.", "answer": "42",
        "benchmark": "frontier_science_olympiad", "benchmark_profile": selected.snapshot(),
    }], tmp_path / "tasks", pipeline_id="dynamic_htn_informed_research")
    metadata = tomllib.loads((tmp_path / "tasks/offline-fso/task.toml").read_text())["metadata"]
    assert json.loads(metadata["benchmark_web_policy"])["network_enabled"] is False
