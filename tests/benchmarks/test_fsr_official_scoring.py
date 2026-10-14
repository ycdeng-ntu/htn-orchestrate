"""Regressions for the publicly documented FrontierScience Research protocol."""
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from benchmarks.judges import frontier_science as fs
from benchmarks.judges import _common


@pytest.mark.parametrize('content,expected', [
    ('VERDICT: 0', 0.0), ('VERDICT: 7', 7.0), ('VERDICT: 10', 10.0),
    ('Reasoning\nVERDICT: 6.99\n', 6.99),
    ('Intermediate VERDICT: 4\nFinal reasoning\nVERDICT: 9', 9.0),
    ('VERDICT: 11', None), ('VERDICT: -1', None), ('VERDICT: nan', None),
    ('VERDICT: 7.2.3', None), ('VERDICT: 7/10', None),
    ('VERDICT: 9\nVERDICT: 11', None),
    ('VERDICT: 9\nNo final verdict.', None), ('', None),
])
def test_final_line_score_contract(content, expected):
    assert fs._parse_rubric_score(content) == expected


@pytest.mark.asyncio
@pytest.mark.parametrize('score,verdict', [(6.99, 'INCORRECT'), (7., 'CORRECT'), (10., 'CORRECT'), (None, 'NOT_ATTEMPTED')])
async def test_complete_submission_and_threshold_are_identical_for_both_entrypoints(monkeypatch, score, verdict):
    call = AsyncMock(return_value=score)
    monkeypatch.setattr(fs, '_call_rubric_judge_once', call)
    report = '\n# Derivation\nWork\n## References\nPaper\n## Conclusion\n42\n<!-- citation-map: {} -->\n'
    assert await fs.score_frontier_science('Question', 'Rubric', report) == (verdict, score)
    assert await fs.verify_frontier_science('Question', 'Rubric', report) == verdict
    assert call.await_count == 2  # One grade per submitted response, not 30 regrades.
    for args in call.await_args_list:
        prompt = args.args[0]
        assert 'The attempted answer: ' + report + '\n\n***' in prompt
        assert prompt.index('The problem: Question') < prompt.index('The rubric: Rubric')
        assert prompt.index('The attempted answer:') < prompt.index('First, think step-by-step')


@pytest.mark.asyncio
async def test_research_requests_disable_effort_fallback(monkeypatch):
    call = AsyncMock(return_value='VERDICT: 7')
    monkeypatch.setattr(fs, '_judge_call_with_effort_fallback', call)
    assert await fs._call_rubric_judge_once('prompt') == 7
    kwargs = call.await_args.args[0]
    assert kwargs['model'] == 'openai/gpt-5.4'
    assert kwargs['reasoning_effort'] == 'high'
    assert call.await_args.kwargs['allow_effort_fallback'] is False


@pytest.mark.asyncio
async def test_fixed_effort_retries_do_not_downgrade(monkeypatch):
    create = AsyncMock(return_value=SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content=''))]))
    client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
    monkeypatch.setattr(_common.asyncio, 'sleep', AsyncMock())
    kwargs = {'model': 'gpt-5', 'reasoning_effort': 'high', 'messages': []}
    assert await _common._judge_call_with_effort_fallback(
        kwargs, label='test', client=client, allow_effort_fallback=False) == ''
    assert create.await_count == 2
    assert all(c.kwargs['reasoning_effort'] == 'high' for c in create.await_args_list)
    assert kwargs['reasoning_effort'] == 'high'
