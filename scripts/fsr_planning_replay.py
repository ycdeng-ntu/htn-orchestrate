"""Paid planning-only replay: fresh planning and audit/revision of historical plans."""
from __future__ import annotations

import argparse
import asyncio
from copy import deepcopy
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import time

import yaml
from dotenv import load_dotenv

from agent_harness.core.atomic_files import atomic_write_text
from agent_harness.core.llm_errors import quota_error
from agent_harness.infra.openai_client import OpenAIClient
from workflows.dynamic_htn_informed_research.adapters import HighEffortTimeoutFallbackLLMClient
from workflows.dynamic_htn_informed_research.diagnostics import DiagnosticRecorder, scope_snapshot
from workflows.dynamic_htn_informed_research.htn.llm_types import LLMResponse
from workflows.dynamic_htn_informed_research.htn.planner import HTNPlanner
from workflows.dynamic_htn_informed_research.htn.task_node import HTNTaskNode
from workflows.dynamic_htn_informed_research.htn.task_tree import HTNTree
from workflows.dynamic_htn_informed_research.runtime.llm import NativeHTNClient
from workflows.dynamic_htn_informed_research.task_strategies.report_research.planning import _REPORT_DECOMPOSITION_SYSTEM
from workflows.dynamic_htn_informed_research.task_strategies.report_research.prompt_access import report_system_prompt
from workflows.dynamic_htn_informed_research.task_strategies.report_research.topology import ReportPlanningStrategy

ROOT = Path(__file__).resolve().parents[1]
WORKFLOW = ROOT / 'workflows/dynamic_htn_informed_research'
PROFILE = WORKFLOW / 'profiles/report_research.yaml'
LOW = 'fsr60_v41_offline_c10_semantic_alignment_teamjudge_20260923_183932'
HIGH = 'fsr6_v41_offline_c6_high_reasoning_20260924_052400'
ALLOWED = {'plan_report_research', 'audit_report_plan', 'audit_report_fidelity',
           'audit_report_decomposition', 'synthesize_report_plan_audit', 'revise_report_plan'}


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_text(path, json.dumps(value, ensure_ascii=False, indent=2))


def cases():
    result = []
    for q in ('28', '21', '33', '24'):
        run = HIGH if q == '28' else LOW
        paths = list((ROOT / 'results/docker' / run / 'trials' / q).glob(
            'agent/dynir/*/diagnostics/*/tree_snapshots/epoch-0-iter-0-planned-root.json'))
        assert len(paths) == 1, (q, len(paths))
        path = paths[0]
        tree = json.loads(path.read_text())
        # Use task-only fields, excluding runtime delivery, results and grading artifacts.
        plan = {'nodes': [{k: n[k] for k in ('id', 'title', 'task', 'depends_on')}
                          for key, n in tree.items()
                          if key != 'root' and n['execution_kind'] != 'report_delivery'],
                'rationale': 'Historical research plan reconstructed from its accepted tree snapshot.'}
        for mode in ('fresh', 'historical'):
            result.append({'case_id': f'q{q}-{mode}', 'question_id': q, 'mode': mode,
                           'question': tree['root']['task'], 'historical_plan': plan,
                           'source': str(path.relative_to(ROOT)),
                           'source_sha256': hashlib.sha256(path.read_bytes()).hexdigest()})
    return result


def fresh_cases_from_tasks(task_dir):
    """Read public task instructions only; never load evaluation/answer files."""
    directory = Path(task_dir).resolve(strict=True)
    paths = sorted(directory.glob('*/instruction.md'),
                   key=lambda p: (0, int(p.parent.name)) if p.parent.name.isdigit()
                   else (1, p.parent.name))
    if not paths:
        raise ValueError(f'No task instructions in {directory}')
    return [{'case_id': f'q{p.parent.name}-fresh', 'question_id': p.parent.name,
             'mode': 'fresh', 'question': p.read_text().strip(),
             'source': str(p), 'source_sha256': hashlib.sha256(p.read_bytes()).hexdigest()}
            for p in paths]


class RecordedPlanner:
    model = 'recorded-plan-no-api-call'

    def __init__(self, plan):
        self.plan = plan

    async def chat(self, messages, tools=None):
        assert [t['function']['name'] for t in tools] == ['plan_report_research']
        return LLMResponse(tool_calls=[{'function': {
            'name': 'plan_report_research', 'arguments': json.dumps(self.plan)}}])


async def main(args):
    config = yaml.safe_load(PROFILE.read_text())
    selected = fresh_cases_from_tasks(args.task_dir) if args.task_dir else cases()
    if args.dry_run:
        print(json.dumps({'cases': [c['case_id'] for c in selected],
                          'model': config['llm']['model'], 'reasoning_effort': 'high',
                          'concurrency': args.concurrency, 'allowed_tools': sorted(ALLOWED),
                          'workers_executed': 0, 'delivery_executed': False}))
        return
    load_dotenv(ROOT / '.env', override=False)
    key = os.environ.get('OPENROUTER_API_KEY')
    if not key:
        raise RuntimeError('OPENROUTER_API_KEY is missing')
    out = Path(args.out).resolve()
    out.mkdir(parents=True, exist_ok=False)
    source_files = [Path(__file__), PROFILE, *WORKFLOW.rglob('*.py')]
    hashes = {}
    for p in source_files:
        rel = p.relative_to(ROOT)
        hashes[str(rel)] = hashlib.sha256(p.read_bytes()).hexdigest()
        target = out / 'source' / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(p, target)
    save(out / 'manifest.json', {
        'started_at': datetime.now(timezone.utc).isoformat(), 'cases': selected,
        'profile': config, 'source_sha256': hashes, 'concurrency': args.concurrency,
        'case_timeout_s': args.timeout, 'network_enabled_for_research': False,
        'api_access': True, 'rebuild_context': None, 'runs_per_case': 1,
        'historical_plan_source': None if args.task_dir else 'Accepted epoch-0 research nodes; runtime delivery excluded',
        'task_dir': str(Path(args.task_dir).resolve()) if args.task_dir else None,
        'tools_allowed': sorted(ALLOWED), 'workers_executed': 0, 'delivery_executed': False,
    })
    semaphore = asyncio.Semaphore(args.concurrency)
    quota_failed = asyncio.Event()
    results, usage_rows = [], []

    def summarize():
        costs = [r['usage'].get('cost') for r in usage_rows if r.get('usage')]
        accepted = [r for r in results if r.get('accepted') and not r.get('fallback') and not r.get('error_type')]
        counts = [r['research_nodes'] for r in accepted]
        save(out / 'summary.json', {
            'status': 'completed' if len(results) == len(selected) else 'running',
            'total': len(selected), 'completed': len(results), 'results': results,
            'accepted': len(accepted),
            'initial_audit_accepted': sum(r.get('accepted_by_audit', False) for r in accepted),
            'revision_mechanically_accepted': sum(r.get('revision_mechanically_validated', False) for r in accepted),
            'fallbacks': sum(bool(r.get('fallback')) for r in results),
            'errors': sum(bool(r.get('error_type')) for r in results),
            'revised': sum(bool(r.get('revised')) for r in results),
            'accepted_research_node_distribution': dict(sorted(Counter(counts).items())),
            'accepted_mean_research_nodes': sum(counts) / len(counts) if counts else None,
            'workers_executed': 0, 'delivery_executed': False, 'judge_executed': False,
            'api_attempts': len(usage_rows),
            'successful_api_responses': sum(r.get('status') == 'completed' for r in usage_rows),
            'prompt_tokens': sum(r.get('usage', {}).get('prompt_tokens', 0) for r in usage_rows),
            'completion_tokens': sum(r.get('usage', {}).get('completion_tokens', 0) for r in usage_rows),
            'reported_cost_usd': sum(c for c in costs if isinstance(c, (int, float))),
            'responses_with_reported_cost': sum(isinstance(c, (int, float)) for c in costs),
        })

    async def run(case):
        async with semaphore:
            directory = out / case['case_id']
            diag = DiagnosticRecorder(directory)
            save(directory / 'input.json', case)
            clients = []
            started = time.monotonic()
            result = {k: case[k] for k in ('case_id', 'question_id', 'mode')}

            def client_for(effort):
                cfg = config['llm']
                extra = deepcopy(cfg.get('extra_body') or {})
                extra.update(reasoning_effort=effort, top_p=cfg['top_p'])
                client = OpenAIClient(cfg['model'], api_key=key,
                    base_url=os.environ.get('OPENROUTER_BASE_URL') or 'https://openrouter.ai/api/v1',
                    temperature=cfg['temperature'], max_completion_tokens=cfg['max_tokens'],
                    timeout=cfg['high_effort_timeout_s'] + 30, extra_body=extra)
                clients.append(client)
                api = client._client.chat.completions.with_raw_response
                original = api.create

                async def capture(**kwargs):
                    assert len(kwargs.get('tools', [])) == 1
                    tool = kwargs['tools'][0]['function']['name']
                    assert tool in ALLOWED
                    if quota_failed.is_set():
                        raise RuntimeError('Quota failure in another replay case')
                    row = {'case_id': case['case_id'], 'tool': tool, 'effort': effort,
                           'scope': scope_snapshot(), 'status': 'started'}
                    usage_rows.append(row)
                    tick = time.monotonic()
                    try:
                        raw = await original(**kwargs)
                        response = raw.parse()
                        row.update(status='completed', generation_id=response.id,
                                   response_model=response.model,
                                   usage=response.usage.model_dump(mode='json') if response.usage else {})
                        return raw
                    except BaseException as exc:
                        row.update(status=type(exc).__name__)
                        if quota_error(exc) is not None:
                            quota_failed.set()
                        raise
                    finally:
                        row['duration_s'] = round(time.monotonic() - tick, 3)
                        with (directory / 'api_usage.jsonl').open('a') as f:
                            f.write(json.dumps(row) + '\n')
                        summarize()
                api.create = capture
                return NativeHTNClient(client, diagnostics=diag, reasoning_effort=effort)

            try:
                high, low = client_for('high'), client_for('low')
                live = HighEffortTimeoutFallbackLLMClient(high, low,
                    timeout_s=config['llm']['high_effort_timeout_s'], diagnostics=diag)
                strategy = ReportPlanningStrategy(
                    planner_llm=RecordedPlanner(case['historical_plan']) if case['mode'] == 'historical' else live,
                    audit_llm=live, revision_llm=live,
                    planner_system=report_system_prompt(_REPORT_DECOMPOSITION_SYSTEM, network_enabled=False),
                    network_enabled=False, deadline=started + args.timeout, diagnostics=diag)
                root = HTNTaskNode(id='root', task=case['question'], needs_decomposition=True)
                tree = HTNTree(root_id='root', nodes={'root': root})
                planner = HTNPlanner(strategy=strategy, max_depth=config['agent']['max_depth'],
                                     max_children=config['agent']['max_children'])
                children = await planner.decompose(tree, 'root')
                violations = tree.validate()
                assert not violations, violations
                assert all(n.status.value == 'pending' for n in children)
                save(directory / 'final_tree.json', {k: n.model_dump(mode='json') for k, n in tree.nodes.items()})
                audit_paths = sorted((directory / 'report_delivery/epoch-0').glob('plan-audit-*.json'))
                audits = [json.loads(p.read_text()) for p in audit_paths]
                research = [n for n in children if n.execution_kind != 'report_delivery']
                proposed = directory / 'report_delivery/epoch-0/proposed-plan.json'
                original = json.loads(proposed.read_text())['output'] if proposed.exists() else None
                result['initial_node_count'] = len(original['nodes']) if original else None
                result['initial_research_nodes_by_audit'] = sum(n['kind'] == 'research' for n in audits[0]['nodes']) if audits else None
                result.update(audit_verdicts=[a['plan_review']['verdict'] for a in audits],
                              first_issues=audits[0]['plan_review']['issues'] if audits else [],
                              revised=(directory / 'report_delivery/epoch-0/revised-plan.json').exists(),
                              research_nodes=len(research),
                              final_plan=[{k: n.model_dump()[k] for k in ('id', 'title', 'task', 'depends_on')} for n in research],
                              tree_valid=True, workers_executed=0)
                events = [json.loads(l) for l in (directory / 'events.jsonl').read_text().splitlines()]
                acceptance_path = directory / 'report_delivery/epoch-0/plan-acceptance.json'
                result.update(json.loads(acceptance_path.read_text()))
                result['low_effort_fallbacks'] = sum(e['event'] == 'low_effort_fallback_start' for e in events)
            except Exception as exc:
                result.update(error_type=type(exc).__name__)
            finally:
                for client in clients:
                    await client._client.close()
            result['duration_s'] = round(time.monotonic() - started, 3)
            save(directory / 'result.json', result)
            results.append(result)
            summarize()
            print(json.dumps({k: result.get(k) for k in ('case_id', 'audit_verdicts', 'research_nodes', 'fallback', 'duration_s', 'error_type')}), flush=True)

    summarize()
    await asyncio.gather(*(run(case) for case in selected))
    print('RESULTS ' + str(out), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', required=True)
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--task-dir', help='Fresh planning for every public */instruction.md in this directory; no historical plans or workers')
    parser.add_argument('--concurrency', type=int, default=6)
    parser.add_argument('--timeout', type=float, default=1800)
    args = parser.parse_args()
    assert args.concurrency > 0 and args.timeout > 0
    asyncio.run(main(args))
