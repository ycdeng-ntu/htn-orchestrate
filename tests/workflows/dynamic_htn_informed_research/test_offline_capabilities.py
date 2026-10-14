"""Offline permissions belong to the benchmark and survive any router decision."""
from __future__ import annotations

import json
import sys
from types import SimpleNamespace

import httpx
import pytest

from agent_harness.core.web_policy import validate_web_policy
from benchmarks.core.harbor_task_generator import generate_task_dirs
from benchmarks.core.profile import load_benchmark_profile
from workflows.dynamic_htn_informed_research.runtime.worker import worker_capability_instruction
from workflows.dynamic_htn_informed_research.tools import ResearchToolConfig, ResearchToolRuntime
from workflows.dynamic_htn_informed_research.tools.runtime import ROLE_TOOLS, ResearchToolError
from workflows.dynamic_htn_informed_research.tools.sandbox import e2b_factory



@pytest.mark.parametrize('bad', [0, 1, 'false', None, [], {}])
def test_network_permission_must_be_boolean(bad):
    with pytest.raises(ValueError, match='boolean'):
        validate_web_policy({'network_enabled': bad})


def test_benchmark_permissions_override_solver_tool_overrides():
    cfg = ResearchToolConfig.from_profile(
        {}, web_policy={'network_enabled': False, 'blocked_terms': ['answer key']},
        network_enabled=True, blocked_needles=('unrelated',),
    )
    assert cfg.network_enabled is False
    assert cfg.blocked_needles == ('answer key',)
    assert ResearchToolConfig.from_profile({}).network_enabled is True


@pytest.mark.asyncio
@pytest.mark.parametrize('role', ROLE_TOOLS)
async def test_offline_tools_hide_and_reject_retrieval_for_every_role(tmp_path, role):
    async with httpx.AsyncClient(transport=httpx.MockTransport(lambda _: pytest.fail('network request'))) as client:
        runtime = ResearchToolRuntime(role, task_id='t', artifact_dir=tmp_path,
            config=ResearchToolConfig(network_enabled=False, read_roots=(tmp_path,), workspace_root=tmp_path),
            http_client=client)
        async with runtime:
            assert {t.name for t in runtime.tools} == set(ROLE_TOOLS[role]) - {'web_search', 'web_fetch', 'scholar_search'}
            assert runtime.config.read_roots == ()
            assert runtime.config.workspace_root is None
            for name in ('web_search', 'web_fetch', 'scholar_search'):
                with pytest.raises(ResearchToolError, match='disabled'):
                    await runtime._bind_tool(name)()
            with pytest.raises(ResearchToolError, match='disabled'):
                await runtime._direct('https://example.com')
            with pytest.raises(ResearchToolError, match='disabled'):
                await runtime.web_search('paper')


@pytest.mark.asyncio
async def test_offline_host_file_is_not_a_task_artifact(tmp_path):
    (tmp_path / 'answer.txt').write_text('private answer')
    def sandbox():
        raise RuntimeError('sandbox file lookup only')
    async with ResearchToolRuntime('dynir_report_researcher', task_id='t', artifact_dir=tmp_path,
            config=ResearchToolConfig(network_enabled=False, read_roots=(tmp_path,), workspace_root=tmp_path),
            sandbox_factory=sandbox) as runtime:
        # An absolute path outside allowed roots can only refer to the isolated sandbox.
        with pytest.raises(RuntimeError, match='sandbox file lookup only'):
            await runtime.read_file(str(tmp_path / 'answer.txt'))
        artifact = runtime.artifact_dir / 'computed.txt'
        artifact.write_text('computed value: 42')
        assert 'computed value: 42' in await runtime.read_file(str(artifact))




@pytest.mark.parametrize('names', [('submit_report',), ('run_python_code', 'submit_report'),
                                  ('run_python_code', 'read_file', 'submit_report')])
def test_capability_hint_tracks_actual_tools_including_final_turn(names):
    tools = [SimpleNamespace(name=n) for n in names]
    text = worker_capability_instruction(tools, network_enabled=False)
    assert ('offline Python computation' in text) == ('run_python_code' in names)
    assert ('file readback' in text) == ('read_file' in names)
    assert 'report submission' in text
    assert worker_capability_instruction(tools, network_enabled=True) == ''


class FakeSandbox:
    def __init__(self, codes, *, fail_disconnect=False):
        self.codes = iter(codes)
        self.events = []
        self.commands = SimpleNamespace(run=self.run)
        self.fail_disconnect = fail_disconnect
    def run(self, command, **kwargs):
        self.events.append(('command', command))
        return SimpleNamespace(exit_code=next(self.codes))
    def update_network(self, policy):
        enabled = policy['allow_internet_access']
        self.events.append(('network', enabled))
        if not enabled and self.fail_disconnect:
            raise RuntimeError('disconnect failed')
    def kill(self):
        self.events.append(('kill',))


@pytest.mark.parametrize('codes,fail_disconnect,expected_error', [
    ([0], False, None), ([1, 0, 0], False, None),
    ([1, 1], False, 'preparation failed'), ([1, 0, 1], False, 'requires preinstalled'),
    ([1, 0], True, 'disconnect failed'),
])
def test_offline_sandbox_prepares_then_disconnects_or_kills(monkeypatch, codes, fail_disconnect, expected_error):
    sandbox = FakeSandbox(codes, fail_disconnect=fail_disconnect)
    created = []
    def create(**kwargs):
        created.append(kwargs)
        return sandbox
    monkeypatch.setitem(sys.modules, 'e2b_code_interpreter', SimpleNamespace(Sandbox=SimpleNamespace(create=create)))
    def run():
        return e2b_factory(api_key='test', template='test', timeout=600, allow_internet_access=False)
    if expected_error:
        with pytest.raises(RuntimeError, match=expected_error):
            run()
        assert sandbox.events[-1] == ('kill',)
    else:
        assert run() is sandbox
    assert created[0]['allow_internet_access'] is False
    network = [event for event in sandbox.events if event[0] == 'network']
    assert network == ([] if codes[0] == 0 else [('network', True), ('network', False)])


def test_prl_task_uses_base_auto_and_projects_only_access_policy(tmp_path):
    import tomllib
    from benchmarks.core.registry import get_config
    cfg = get_config('prl_bench')
    assert cfg.default_pipeline == 'dynamic_htn_informed_research'
    profile = load_benchmark_profile('prl_bench')
    assert profile.configuration['web_policy']['network_enabled'] is False
    question = {'id': 'q', 'question': 'Derive a result.', 'answer': 'private rubric',
                'benchmark': 'prl_bench', 'benchmark_profile': profile.snapshot(), 'metadata': {'profile': 'auto'}}
    generate_task_dirs([question], tmp_path / 'tasks', pipeline_id=cfg.default_pipeline)
    metadata = tomllib.loads((tmp_path / 'tasks/q/task.toml').read_text())['metadata']
    assert metadata['profile'] == 'auto'
    assert json.loads(metadata['benchmark_web_policy'])['network_enabled'] is False
    assert 'private rubric' not in json.dumps(metadata)
    with pytest.raises(ValueError, match='offline execution'):
        generate_task_dirs([question], tmp_path / 'unsupported', pipeline_id='react_base')


@pytest.mark.asyncio
async def test_prl_runner_keeps_auto_profile_until_routing(tmp_path, monkeypatch):
    from benchmarks.core import harbor_task_generator, registry
    from benchmarks.runner.run_subprocess import run_eval
    profile = load_benchmark_profile('prl_bench')
    question = SimpleNamespace(id='q', question='Derive the physics.', ground_truth='private rubric',
        answer_type='report', category='physics', metadata={}, image_path=None, file_path=None, file_name=None)
    monkeypatch.setattr(registry, 'load_questions', lambda *a, **kw: [question])
    calls = []
    class TasksReady(Exception):
        pass
    def generate(questions, out_dir, *, pipeline_id):
        calls.append((questions, pipeline_id))
        raise TasksReady
    monkeypatch.setattr(harbor_task_generator, 'generate_task_dirs', generate)
    args = SimpleNamespace(benchmark='prl_bench', pipeline=None, profile=None,
        resolved_benchmark_profile=profile.snapshot(), no_shuffle=True, concurrency=1,
        **profile.configuration['dataset'])
    with pytest.raises(TasksReady):
        await run_eval(args, out_dir=tmp_path, seed=42)
    questions, pipeline = calls[0]
    assert pipeline == 'dynamic_htn_informed_research'
    assert questions[0]['metadata']['profile'] == 'auto'
    assert questions[0]['benchmark_profile']['configuration']['web_policy']['network_enabled'] is False
    assert args.profile is None  # The benchmark never selects a solver profile.


@pytest.mark.asyncio
async def test_sandbox_directory_read_is_an_explicit_tool_error(tmp_path):
    reads = []
    sandbox = SimpleNamespace(
        files=SimpleNamespace(get_info=lambda p: SimpleNamespace(type='dir'), read=lambda p: reads.append(p)),
        kill=lambda: None,
    )
    async with ResearchToolRuntime('dynir_report_researcher', task_id='t', artifact_dir=tmp_path,
            sandbox_factory=lambda: sandbox) as runtime:
        with pytest.raises(ResearchToolError, match='is a directory'):
            await runtime.read_file('/remote/reports')
    assert not reads
