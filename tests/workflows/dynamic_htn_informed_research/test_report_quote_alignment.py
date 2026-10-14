"""Citation alignment accepts presentation changes without bypassing review integrity."""
import asyncio
import json
import time

import pytest

from workflows.dynamic_htn_informed_research.htn.llm_types import LLMResponse
from workflows.dynamic_htn_informed_research.task_strategies.report_research.quote_alignment import (
    ALIGNMENT_SYSTEM, QuoteAlignmentError, align_review,
)


class Aligner:
    model = 'offline-aligner'

    def __init__(self, status='matched', wrong_source=False, incomplete=False):
        self.calls = []
        self.status, self.wrong_source, self.incomplete = status, wrong_source, incomplete

    async def chat(self, messages, tools=None):
        self.calls.append((messages, tools))
        data = json.loads(messages[1]['content'])
        matches = []
        for citation in data['citations']:
            source = citation['source']
            if self.wrong_source:
                source = 'draft' if source == 'question' else 'question'
            matches.append({'quote_id': citation['quote_id'], 'status': self.status,
                            'source_ids': [data['sources'][source][0]['source_id']],
                            'reason': 'Compared the claim, values and conditions'})
        if self.incomplete:
            matches = matches[:-1]
        return LLMResponse(tool_calls=[{'function': {
            'name': 'align_review_quotes', 'arguments': json.dumps({'matches': matches}),
        }}])


def payload(quote, question_quote='Calculate the duration.'):
    return {'verdict': 'pass', 'checks': [{
        'id': 'duration', 'question_quote': question_quote, 'report_quote': quote,
        'method': 'Recompute duration', 'result': 'Computed and compared against draft',
        'status': 'supported',
    }], 'findings': [], 'finding_reviews': []}


async def check(p, draft, llm):
    return await align_review(json.dumps(p), question='Calculate the duration.', draft=draft,
                              previous_findings=[], research_node_ids=set(), llm=llm,
                              deadline=time.monotonic() + 5)


@pytest.mark.asyncio
async def test_exact_quotes_do_not_call_llm():
    model = Aligner()
    review, audit = await check(payload('7 days'), '7 days', model)
    assert not model.calls and audit['status'] == 'exact' and review['verdict'] == 'pass'


@pytest.mark.asyncio
@pytest.mark.parametrize(('quote', 'draft'), [
    ('The duration is 7 days', 'The duration is **7 days**.'),
    (r'\approx 7.22\text{ days}', r'\(\approx 7.22\) days'),
    (r'x=\sqrt{2}y', r'x = \sqrt2\, y'),
    ('It takes about a week', 'The duration is approximately seven days.'),
])
async def test_semantic_match_uses_original_passage_without_literal_copy(quote, draft):
    model = Aligner()
    review, audit = await check(payload(quote, 'Compute the time required.'), draft, model)
    assert len(model.calls) == 1  # Question and report citations share a single batch.
    assert len(audit['matches']) == 2
    assert audit['matches'][1]['source_passages'] == [draft]
    assert review['checks'][0]['report_quote'] == quote  # Original review stays auditable.
    request = json.loads(model.calls[0][0][1]['content'])
    assert set(request) == {'sources', 'citations'}  # No rubric, research or old drafts.
    assert 'NOT pursue word-for-word' in ALIGNMENT_SYSTEM


@pytest.mark.asyncio
@pytest.mark.parametrize(('quote', 'draft'), [
    ('Duration = 6 days', 'Duration = 7 days'),
    ('x = -2', 'x = +2'),
    ('The approximation always holds', 'The approximation holds only for small x'),
    ('The claim is supported', 'The claim is not supported'),
])
async def test_substantive_mismatch_cannot_pass(quote, draft):
    with pytest.raises(QuoteAlignmentError) as error:
        await check(payload(quote), draft, Aligner('mismatched'))
    assert error.value.status == 'mismatched'
    assert error.value.details[0]['source_passages'] == [draft]


@pytest.mark.asyncio
@pytest.mark.parametrize('option', ['wrong_source', 'incomplete'])
async def test_alignment_cannot_invent_sources_or_drop_citations(option):
    model = Aligner(**{option: True})
    with pytest.raises(ValueError):
        await check(payload('roughly one week'), '7 days', model)
    assert len(model.calls) == 2  # At most one alignment protocol correction.


@pytest.mark.asyncio
async def test_uncertainty_is_not_approval():
    with pytest.raises(QuoteAlignmentError) as error:
        await check(payload('roughly one week'), '7 days', Aligner('uncertain'))
    assert error.value.status == 'uncertain'


@pytest.mark.asyncio
async def test_schema_and_finding_links_are_checked_before_alignment():
    model = Aligner()
    p = payload('roughly one week')
    p['checks'][0]['status'] = 'unresolved'  # An unlinked unresolved check cannot pass.
    with pytest.raises(ValueError):
        await check(p, '7 days', model)
    assert not model.calls


@pytest.mark.asyncio
async def test_alignment_cancellation_propagates():
    class Cancel:
        async def chat(self, *args, **kwargs):
            raise asyncio.CancelledError()
    with pytest.raises(asyncio.CancelledError):
        await check(payload('roughly one week'), '7 days', Cancel())
