"""Offline compatibility checks for the frozen Agent Team judge protocol."""
import asyncio
import hashlib
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from benchmarks.core.profile import load_benchmark_profile
from benchmarks.judges import score_answer
from benchmarks.judges import frontier_science_team as team

CONFIG = load_benchmark_profile('fsr60_agent_team').configuration['judge']


def response(content='VERDICT: 8', finish='stop'):
    return SimpleNamespace(choices=[SimpleNamespace(
        finish_reason=finish, message=SimpleNamespace(content=content),
    )])


def mock_transport(monkeypatch, responses):
    create = AsyncMock(side_effect=responses)
    client = AsyncMock()
    client.__aenter__.return_value = client
    client.chat.completions.create = create
    providers = []

    def factory(provider):
        providers.append(provider)
        return client

    monkeypatch.setattr(team, '_profile_judge_client', factory)
    monkeypatch.setattr(team.asyncio, 'sleep', AsyncMock())
    return create, client, providers


@pytest.mark.asyncio
async def test_profile_dispatch_preserves_score_and_campaign_request(monkeypatch):
    create, client, providers = mock_transport(monkeypatch, [response()])
    result = await score_answer('frontier_science_research', 'q', 'r', 'a', judge_config=CONFIG)
    assert result == ('CORRECT', 8.0)
    assert create.await_args.kwargs == {
        'model': 'openai/gpt-5.5', 'reasoning_effort': 'high',
        'max_completion_tokens': 65536,
        'messages': [{'role': 'user', 'content': team.JUDGE_PROMPT_FS.format(
            question='q', rubric='r', answer='a')}],
    }
    assert providers == ['openrouter']
    assert client.__aexit__.await_count == 1
    assert load_benchmark_profile('fsr60_agent_team').configuration['web_policy']['network_enabled'] is False
    assert load_benchmark_profile('fsr60').configuration['judge']['model'] == 'openai/gpt-5.4'


@pytest.mark.asyncio
async def test_nonstop_empty_and_errors_consume_attempts_before_fallback(monkeypatch):
    create, client, _ = mock_transport(monkeypatch, [
        response(finish='length'), response(''), RuntimeError('test error'),
        response(finish='content_filter'), response('VERDICT: 6.5'),
    ])
    assert await score_answer('frontier_science_research', 'q', 'r', 'a', judge_config=CONFIG) == ('INCORRECT', 6.5)
    assert [c.kwargs['reasoning_effort'] for c in create.await_args_list] == [
        'high', 'high', 'medium', 'medium', 'low',
    ]
    assert client.__aexit__.await_count == 5


@pytest.mark.asyncio
async def test_exhausted_retries_remain_ungraded(monkeypatch):
    create, _, _ = mock_transport(monkeypatch, [response(finish='length')] * 6)
    assert await score_answer('frontier_science_research', 'q', 'r', 'a', judge_config=CONFIG) == ('NOT_ATTEMPTED', None)
    assert create.await_count == 6


@pytest.mark.asyncio
@pytest.mark.parametrize(('content', 'expected'), [
    ('no verdict', ('NOT_ATTEMPTED', None)),
    ('VERDICT: 7\ntrailing text', ('CORRECT', 7.0)),
    ('VERDICT: 6\nVERDICT: 9', ('INCORRECT', 6.0)),
])
async def test_campaign_parser_and_no_retry_after_complete_response(monkeypatch, content, expected):
    create, _, _ = mock_transport(monkeypatch, [response(content)])
    assert await score_answer('frontier_science_research', 'q', 'r', 'a', judge_config=CONFIG) == expected
    assert create.await_count == 1


@pytest.mark.asyncio
async def test_cancellation_propagates_and_closes_client(monkeypatch):
    create, client, _ = mock_transport(monkeypatch, [asyncio.CancelledError()])
    with pytest.raises(asyncio.CancelledError):
        await score_answer('frontier_science_research', 'q', 'r', 'a', judge_config=CONFIG)
    assert create.await_count == client.__aexit__.await_count == 1


def test_prompt_matches_frozen_campaign_sha256():
    assert hashlib.sha256(team.JUDGE_PROMPT_FS.encode()).hexdigest() == '96b5cd1246527c0d93b48f6b18f1f2edcdc2a79631e5ea77c6c7aafe41069fad'
