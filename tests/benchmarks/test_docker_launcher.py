import importlib.util
from pathlib import Path

import pytest

spec = importlib.util.spec_from_file_location(
    'docker_agent', Path(__file__).resolve().parents[2] / 'scripts/docker_agent.py')
launcher = importlib.util.module_from_spec(spec)
spec.loader.exec_module(launcher)


@pytest.mark.parametrize('out', ['/tmp/out', '../other', 'results/../escape', 'results'])
def test_reject_output_outside_dedicated_run(out):
    with pytest.raises(ValueError):
        launcher.prepare(['--out', out], 'job')


def test_preserve_profile_and_concurrency(tmp_path, monkeypatch):
    monkeypatch.setattr(launcher, "ROOT", tmp_path)
    args = launcher.prepare(['--benchmark_profile', 'frontier_science_research', '--profile', 'fsr_deepseek_v4_1_flash_miromind',
                             '--concurrency', '16'], 'job')
    assert launcher.option(args, '--profile')[1] == 'fsr_deepseek_v4_1_flash_miromind'
    assert launcher.option(args, '--concurrency')[1] == '16'
    assert launcher.option(args, '--out')[1] == '/workspace/results/job'


def test_import_subset_without_mounting_parent(tmp_path, monkeypatch):
    root = tmp_path / 'project'
    root.mkdir()
    subset = tmp_path / 'private' / 'questions.jsonl'
    subset.parent.mkdir()
    subset.write_text('{"id": "1"}\n')
    monkeypatch.setattr(launcher, 'ROOT', root)
    import yaml
    profile = tmp_path / "evaluation.yaml"
    profile.write_text(yaml.safe_dump({"benchmark": "browsecomp", "dataset": {"subset": str(subset)},
                                       "judge": {"provider": "judge", "model": "test"}}))
    args = launcher.prepare(['--benchmark_profile=' + str(profile)], 'job')
    assert launcher.option(args, '--benchmark_profile')[1] == '/workspace/data/docker_inputs/job.benchmark_profile.yaml'
    saved = yaml.safe_load((root / 'data/docker_inputs/job.benchmark_profile.yaml').read_text())
    assert saved['dataset']['subset'] == '/workspace/data/docker_inputs/job.jsonl'
    assert (root / 'data/docker_inputs/job.jsonl').read_text() == subset.read_text()


def test_refuse_dangerous_container_reuse():
    obj = {'Mounts': [], 'Config': {'User': '0'}, 'HostConfig': {}}
    with pytest.raises(RuntimeError):
        launcher.validate(obj)


def test_reject_duplicate_options():
    with pytest.raises(ValueError):
        launcher.prepare(['--out', 'results/a', '--out=results/b'], 'job')
