"""Offline native-tool contracts: fake HTTP and E2B, never real requests."""
from __future__ import annotations

import asyncio
import json
from pathlib import Path
from types import SimpleNamespace

import httpx
import pytest

from workflows.dynamic_htn_informed_research.tools import ResearchToolConfig, ResearchToolError, build_research_tools
from workflows.dynamic_htn_informed_research.tools._pure import format_scholar, sanitize_code
from workflows.dynamic_htn_informed_research.tools.summary import SummaryExtractor, raw_fallback, summary_candidates


def config(**kwargs):
    return ResearchToolConfig(serper_api_key="fake-search-key", jina_api_key="fake-jina-key",
                             summary_use_environment=False, retry_delay=0, direct_fetch_enabled=False, **kwargs)


def get_tool(runtime, name):
    return next(t for t in runtime.tools if t.name == name)


class FakeSandbox:
    def __init__(self, stdout="ok", stderr="", exit_code=0):
        self.content = {}
        self.kills = 0
        self.commands_seen = []
        self.files = SimpleNamespace(write=self.write, read=lambda path: self.content[path])
        self.commands = SimpleNamespace(run=self.run)
        self.result = SimpleNamespace(stdout=stdout, stderr=stderr, exit_code=exit_code)

    def write(self, path, content):
        self.content[path] = content

    def run(self, command, timeout):
        self.commands_seen.append((command, timeout))
        return self.result

    def kill(self):
        self.kills += 1


@pytest.mark.asyncio
async def test_original_schemas_role_allowlists_and_independent_latch(tmp_path):
    runtimes = [build_research_tools(role, task_id="t", artifact_dir=tmp_path, config=config())
                for role in ("dynir_researcher", "dynir_verifier", "dynir_report_researcher", "dynir_science_solver")]
    try:
        researcher, verifier, reporter, scientist = runtimes
        assert len(researcher.tools) == len(scientist.tools) == 6
        assert [t.name for t in verifier.tools] == ["web_search", "web_fetch", "scholar_search", "run_python_code", "submit_report"]
        assert [t.name for t in reporter.tools] == [t.name for t in researcher.tools]
        expected = {"web_search": ["q", "num_results", "gl", "hl", "tbs"], "web_fetch": ["url", "info_to_extract"],
                    "scholar_search": ["query", "num_results", "gl", "hl"], "run_python_code": ["code", "timeout"],
                    "read_file": ["path", "start_line", "end_line", "start_char", "max_chars"], "submit_report": ["content", "confidence"]}
        for tool in researcher.tools:
            spec = tool.to_openai_schema()["function"]
            assert list(spec["parameters"]["properties"]) == expected[tool.name]
            assert spec["parameters"]["required"] == expected[tool.name][:1]
        assert get_tool(researcher, "web_fetch").parameters["properties"]["url"] == {
            "anyOf": [{"type": "string"}, {"type": "array", "items": {"type": "string"}}]}
        get_tool(researcher, "web_search").parameters["properties"]["q"]["extra"] = True
        assert "extra" not in get_tool(scientist, "web_search").parameters["properties"]["q"]
        with pytest.raises(ResearchToolError, match="empty"):
            await get_tool(researcher, "submit_report").ainvoke({"content": " "})
        assert not researcher.report_latch.submitted
        result = await get_tool(researcher, "submit_report").ainvoke({"content": " report ", "confidence": 1.5})
        assert result == "Report submitted (6 chars, confidence=1.00). Task loop will exit after this turn."
        assert researcher.report_latch.report == "report" and not verifier.report_latch.submitted
        with pytest.raises(ResearchToolError, match="already"):
            await get_tool(researcher, "submit_report").ainvoke({"content": "changed"})
        with pytest.raises(TypeError):
            await get_tool(reporter, "web_search").ainvoke({"_name": "run_python_code", "code": "print(1)"})
    finally:
        await asyncio.gather(*(r.aclose() for r in runtimes))


@pytest.mark.asyncio
async def test_search_multi_query_formatter_dedup_trace_and_source_cards(tmp_path):
    requests = []
    def provider(request):
        requests.append(request)
        q = json.loads(request.content)["q"]
        return httpx.Response(200, json={"answerBox": {"title": "Answer", "answer": "42"},
            "organic": [{"title": "Paper", "link": "https://papers.example/a", "snippet": "Value 42"},
                        {"title": q, "link": f"https://papers.example/{q}", "snippet": "Novel"},
                        {"title": "Video", "link": "https://youtube.com/a", "snippet": "filtered"}]})
    async with httpx.AsyncClient(transport=httpx.MockTransport(provider)) as http:
        async with build_research_tools("dynir_researcher", task_id="task 中文", artifact_dir=tmp_path, config=config(), http_client=http) as runtime:
            result = await get_tool(runtime, "web_search").ainvoke({"q": '["first", "second"]', "num_results": 900, "tbs": "qdr:y"})
            assert result.count("URL: https://papers.example/a") == 1
            assert "## Direct Answer\n**Answer**\n42" in result
            assert "## Query: second" in result and "\n\n---\n\n" in result
            assert "youtube.com" not in result
            assert len(runtime.evidence_cards) == 3
            assert all(json.loads(r.content)["num"] == 100 for r in requests)
            assert all(json.loads(r.content)["tbs"] == "qdr:y" for r in requests)
            assert requests[0].headers["X-MiroAPI-Batch-Id"] == requests[1].headers["X-MiroAPI-Batch-Id"]
            assert requests[0].headers["X-MiroAPI-Trace-Id"] != requests[1].headers["X-MiroAPI-Trace-Id"]


@pytest.mark.asyncio
async def test_search_retries_raise_and_do_not_advertise_failed_batch_sources(tmp_path):
    attempts = 0
    def provider(request):
        nonlocal attempts
        attempts += 1
        return httpx.Response(429)
    async with httpx.AsyncClient(transport=httpx.MockTransport(provider)) as http:
        async with build_research_tools("dynir_researcher", task_id="t", artifact_dir=tmp_path, config=config(), http_client=http) as runtime:
            with pytest.raises(ResearchToolError, match="retries exhausted"):
                await get_tool(runtime, "web_search").ainvoke({"q": "test"})
            assert attempts == 3 and not runtime.evidence_cards


@pytest.mark.asyncio
@pytest.mark.parametrize("failure_status", [401, 429])
async def test_search_partial_failure_keeps_delayed_siblings_and_deduplicates(tmp_path, failure_status):
    attempts = {}
    sibling_started = asyncio.Event()
    sibling_cancelled = False

    async def provider(request):
        nonlocal sibling_cancelled
        query = json.loads(request.content)["q"]
        attempts[query] = attempts.get(query, 0) + 1
        if query == "bad":
            await sibling_started.wait()
            return httpx.Response(failure_status)
        sibling_started.set()
        try:
            await asyncio.sleep(0.01)
        except asyncio.CancelledError:
            sibling_cancelled = True
            raise
        return httpx.Response(200, json={"organic": [
            {"title": "Shared", "link": "https://source.test/shared", "snippet": "retained evidence"},
            {"title": query, "link": f"https://source.test/{query}", "snippet": "unique evidence"},
        ]})

    async with httpx.AsyncClient(transport=httpx.MockTransport(provider)) as http:
        async with build_research_tools("dynir_researcher", task_id="t", artifact_dir=tmp_path,
                                       config=config(), http_client=http) as runtime:
            output = await get_tool(runtime, "web_search").ainvoke({"q": ["bad", "first", "second"]})
            assert f"## Query: bad\n\nError: Serper HTTP {failure_status}" in output
            assert output.count("URL: https://source.test/shared") == 1
            assert "URL: https://source.test/first" in output
            assert "URL: https://source.test/second" in output
            assert {card["source"]["url"] for card in runtime.evidence_cards} == {
                "https://source.test/shared", "https://source.test/first", "https://source.test/second",
            }
            assert attempts == {"bad": 3 if failure_status == 429 else 1, "first": 1, "second": 1}
            assert not sibling_cancelled


@pytest.mark.asyncio
async def test_search_all_failed_batch_is_an_error_with_no_sources(tmp_path):
    async with httpx.AsyncClient(transport=httpx.MockTransport(lambda request: httpx.Response(401))) as http:
        async with build_research_tools("dynir_researcher", task_id="t", artifact_dir=tmp_path,
                                       config=config(), http_client=http) as runtime:
            with pytest.raises(ResearchToolError) as error:
                await get_tool(runtime, "web_search").ainvoke({"q": ["first", "second"]})
            assert "## Query: first" in str(error.value) and "## Query: second" in str(error.value)
            assert not runtime.evidence_cards


@pytest.mark.asyncio
async def test_search_empty_success_is_not_a_failed_query(tmp_path):
    def provider(request):
        return httpx.Response(401) if json.loads(request.content)["q"] == "bad" else httpx.Response(200, json={})

    async with httpx.AsyncClient(transport=httpx.MockTransport(provider)) as http:
        async with build_research_tools("dynir_researcher", task_id="t", artifact_dir=tmp_path,
                                       config=config(), http_client=http) as runtime:
            output = await get_tool(runtime, "web_search").ainvoke({"q": ["bad", "empty"]})
            assert "## Query: bad\n\nError:" in output
            assert "## Query: empty\n\nNo results (error or empty)." in output
            assert not runtime.evidence_cards


@pytest.mark.asyncio
async def test_cancelled_search_batch_settles_all_owned_requests(tmp_path):
    started, cancelled = set(), set()
    all_started = asyncio.Event()

    async def provider(request):
        query = json.loads(request.content)["q"]
        started.add(query)
        if len(started) == 2:
            all_started.set()
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            cancelled.add(query)
            raise

    async with httpx.AsyncClient(transport=httpx.MockTransport(provider)) as http:
        async with build_research_tools("dynir_researcher", task_id="t", artifact_dir=tmp_path,
                                       config=config(), http_client=http) as runtime:
            task = asyncio.create_task(get_tool(runtime, "web_search").ainvoke({"q": ["first", "second"]}))
            try:
                await asyncio.wait_for(all_started.wait(), 2)
            finally:
                task.cancel()
                with pytest.raises(asyncio.CancelledError):
                    await task
            assert cancelled == {"first", "second"}
            assert not runtime.evidence_cards


def test_scholar_preserves_citation_order_and_pdf():
    text = format_scholar({"organic": [{"title": "Low", "citedBy": 1},
        {"title": "High", "citedBy": "1,234", "year": 2020, "publicationInfo": "A, Journal",
         "link": "https://journal.example/a", "pdfUrl": "https://oa.example/a.pdf"}]})
    assert text.startswith("## Scholar Results\n[1] **High** (2020) — cited by 1,234")
    assert "PDF: https://oa.example/a.pdf" in text
    assert format_scholar({}) == "No scholarly results found."


@pytest.mark.asyncio
async def test_scholar_http_contract_and_blocked_pdf(tmp_path):
    payloads = []
    def provider(request):
        assert request.url.path == "/scholar"
        payloads.append(json.loads(request.content))
        return httpx.Response(200, json={"organic": [{"title": "Paper", "link": "https://journal.example/a", "pdfUrl": "https://dataset.example/a.pdf"}]})
    async with httpx.AsyncClient(transport=httpx.MockTransport(provider)) as http:
        async with build_research_tools("dynir_researcher", task_id="t", artifact_dir=tmp_path,
                                       config=config(blocked_needles=("dataset.example",)), http_client=http) as runtime:
            result = await get_tool(runtime, "scholar_search").ainvoke({"query": "  equations  ", "num_results": 99})
            assert payloads == [{"q": "equations", "num": 20, "gl": "us", "hl": "en"}]
            assert "PDF:" not in result and len(runtime.evidence_cards) == 1


@pytest.mark.asyncio
async def test_fetch_focus_short_content_raw_provenance_and_input_pairing(tmp_path):
    focuses, urls = [], []
    def provider(request):
        url = json.loads(request.content)["url"]
        urls.append(url)
        return httpx.Response(200, json={"data": {"content": "# Real paper\nexact raw value = 42"}})
    async def summarize(content, focus):
        focuses.append((content, focus))
        return "EXTRACTED " + focus
    async with httpx.AsyncClient(transport=httpx.MockTransport(provider)) as http:
        async with build_research_tools("dynir_researcher", task_id="t", artifact_dir=tmp_path, config=config(),
                                        http_client=http, summary_client=summarize) as runtime:
            result = await get_tool(runtime, "web_fetch").ainvoke({"url": ["arxiv.org/pdf/1234.5.pdf", "https://paper.example/b"], "info_to_extract": ["exact value"]})
            assert urls[0] == "https://arxiv.org/abs/1234.5"
            assert len(focuses) == 1  # A short focus list pads with empty strings.
            assert "[1] URL: arxiv.org/pdf/1234.5.pdf\n    Info: EXTRACTED exact value" in result
            assert "[2] URL: https://paper.example/b\n    Info: # Real paper" in result
            assert len(runtime.evidence_cards) == 2
            stored = json.loads(Path(runtime.evidence_cards[0]["raw_path"]).read_text())
            assert stored["content"].endswith("raw value = 42") and stored["focus"] == "exact value"
            assert len(stored["sha256"]) == 64


@pytest.mark.asyncio
async def test_jina_browser_retry_and_negative_cache(tmp_path):
    engines = []
    def provider(request):
        engines.append(request.headers.get("X-Engine", "default"))
        return httpx.Response(403)
    async with httpx.AsyncClient(transport=httpx.MockTransport(provider)) as http:
        async with build_research_tools("dynir_researcher", task_id="t", artifact_dir=tmp_path, config=config(), http_client=http) as runtime:
            for _ in range(2):
                with pytest.raises(ResearchToolError):
                    await get_tool(runtime, "web_fetch").ainvoke({"url": "https://paper.example/a"})
            assert engines == ["default", "browser"]
            assert not runtime.evidence_cards


@pytest.mark.asyncio
async def test_pmc_route_uses_bioc_without_jina(tmp_path):
    seen = []
    def provider(request):
        seen.append(str(request.url))
        return httpx.Response(200, json=[{"metadata": "x" * 500, "documents": [{"passages": [{"text": "Full exact result"}, {"text": "Method"}]}]}])
    async with httpx.AsyncClient(transport=httpx.MockTransport(provider)) as http:
        async with build_research_tools("dynir_researcher", task_id="t", artifact_dir=tmp_path, config=config(), http_client=http) as runtime:
            result = await get_tool(runtime, "web_fetch").ainvoke({"url": "https://pmc.ncbi.nlm.nih.gov/articles/PMC123/"})
            assert result == "Full exact result\n\nMethod"
            assert len(seen) == 1 and "BioC_json/PMC123/unicode" in seen[0]


@pytest.mark.asyncio
async def test_paywall_oa_route_keeps_requested_and_actual_provenance(tmp_path):
    seen = []
    def provider(request):
        seen.append(str(request.url))
        if request.url.host == "api.unpaywall.org":
            assert request.url.params["email"] == "research@example.org"
            return httpx.Response(200, json={"best_oa_location": {"url_for_pdf": "https://repository.example/paper.pdf"}})
        assert json.loads(request.content)["url"] == "https://repository.example/paper.pdf"
        return httpx.Response(200, json={"data": {"content": "# Open paper\n" + "methods and exact values\n" * 40}})
    async with httpx.AsyncClient(transport=httpx.MockTransport(provider)) as http:
        async with build_research_tools("dynir_researcher", task_id="t", artifact_dir=tmp_path,
                                       config=config(unpaywall_email="research@example.org"), http_client=http) as runtime:
            requested = "https://pubs.acs.org/doi/10.1234/example"
            await get_tool(runtime, "web_fetch").ainvoke({"url": requested})
            assert len(seen) == 2
            assert runtime.evidence_cards[0]["source"]["url"] == requested
            assert runtime.evidence_cards[0]["actual_url"] == "https://repository.example/paper.pdf"


@pytest.mark.asyncio
async def test_failed_fetch_batch_preserves_sibling_results_and_cards(tmp_path):
    sibling_started, sibling_cancelled = asyncio.Event(), asyncio.Event()
    async def provider(request):
        url = json.loads(request.content)["url"]
        if url.endswith("/bad"):
            await sibling_started.wait()
            return httpx.Response(401)
        sibling_started.set()
        try:
            await asyncio.sleep(0.01)
            return httpx.Response(200, json={"data": {"content": "# Useful source\nVerified value 42."}})
        except asyncio.CancelledError:
            sibling_cancelled.set()
            raise
    async with httpx.AsyncClient(transport=httpx.MockTransport(provider)) as http:
        async with build_research_tools("dynir_researcher", task_id="t", artifact_dir=tmp_path, config=config(), http_client=http) as runtime:
            result = await get_tool(runtime, "web_fetch").ainvoke({"url": ["https://paper.example/bad", "https://paper.example/pending"]})
            assert "Error:" in result and "Verified value 42" in result
            assert not sibling_cancelled.is_set()
            assert len(runtime.evidence_cards) == 1
            assert runtime.evidence_cards[0]["source"]["url"] == "https://paper.example/pending"


@pytest.mark.asyncio
async def test_summary_fallback_thinking_retry_science_prompt_and_usage():
    requests, usage = [], []
    def provider(request):
        payload = json.loads(request.content)
        requests.append(payload)
        if len(requests) == 1:
            return httpx.Response(403)
        if len(requests) == 2:
            return httpx.Response(200, json={"choices": [{"message": {"content": ""}, "finish_reason": "length"}], "usage": {"prompt_tokens": 10}})
        return httpx.Response(200, json={"choices": [{"message": {"content": "x = 1/√λ"}}]})
    async with httpx.AsyncClient(transport=httpx.MockTransport(provider)) as http:
        extractor = SummaryExtractor(http, {"base_url": "https://primary.example/v1", "model": "primary",
             "fallback": {"base_url": "https://secondary.example/v1", "model": "deepseek/science", "reasoning_effort": "low",
                          "retry_empty_or_length_with_thinking_disabled": True}},
             use_environment=False, extract_mode="science", retry_delay=0, usage_sink=lambda **kw: usage.append(kw))
        assert await extractor("raw content", "derive x") == "x = 1/√λ"
        assert "PRESERVE SPECIFICS VERBATIM" in requests[0]["messages"][0]["content"]
        assert requests[1]["reasoning_effort"] == "low"
        assert requests[2]["thinking"] == {"type": "disabled"}
        assert len(usage) == 3
        assert usage[0]["status"] == "HTTPStatusError" and not usage[0]["usage_known"]
        assert usage[1]["prompt_tokens"] == 10 and usage[1]["usage_known"]
        assert not usage[2]["usage_known"]


@pytest.mark.asyncio
async def test_summary_unconfigured_raw_fallback_and_context_retry():
    contents = []
    def provider(request):
        contents.append(json.loads(request.content)["messages"][0]["content"])
        if len(contents) == 1:
            return httpx.Response(400, text="maximum context length")
        return httpx.Response(200, json={"choices": [{"message": {"content": "extracted"}}]})
    async with httpx.AsyncClient(transport=httpx.MockTransport(provider)) as http:
        disabled = SummaryExtractor(http, use_environment=False)
        assert await disabled("x" * 25_000, "focus") == raw_fallback("x" * 25_000)
        assert not contents
        configured = SummaryExtractor(http, {"base_url": "https://summary.example/v1", "model": "gpt-5"}, use_environment=False)
        assert await configured("x" * 90_000, "focus") == "extracted"
        assert len(contents[0]) - len(contents[1]) == 40_960 - len("[...truncated]")


@pytest.mark.asyncio
async def test_summary_explicit_environment_supports_dotenv_only_and_third_fallback():
    environment = {"SUMMARY_LLM_BASE_URL": "https://env-primary.example/v1", "SUMMARY_LLM_MODEL_NAME": "env-primary",
                   "SUMMARY_LLM_FALLBACK_BASE_URL": "https://env-fallback.example/v1", "SUMMARY_LLM_FALLBACK_MODEL": "env-third"}
    assert [c["model"] for c in summary_candidates({}, environment=environment)] == ["env-primary", "env-third"]
    assert summary_candidates({}, environment={}) == []
    profile = {"base_url": "https://profile.example/v1", "model": "profile-primary",
               "fallback": {"base_url": "https://profile-fallback.example/v1", "model": "profile-second"}}
    requests = []
    def provider(request):
        model = json.loads(request.content)["model"]
        requests.append(model)
        if model != "env-third":
            return httpx.Response(403)
        return httpx.Response(200, json={"choices": [{"message": {"content": "third candidate extraction"}}]})
    async with httpx.AsyncClient(transport=httpx.MockTransport(provider)) as http:
        extractor = SummaryExtractor(http, profile, environment=environment)
        assert await extractor("source content", "focus") == "third candidate extraction"
    assert requests == ["profile-primary", "profile-second", "env-third"]


@pytest.mark.asyncio
@pytest.mark.parametrize("role", ["dynir_researcher", "dynir_report_researcher", "dynir_report_verifier"])
async def test_worker_sandbox_reuse_overflow_read_lines_and_release(tmp_path, role):
    sandbox = FakeSandbox(stdout="line\n" * 4000)
    builds = []
    def factory():
        builds.append(True)
        return sandbox
    runtime = build_research_tools(role, task_id="t", artifact_dir=tmp_path, config=config(), sandbox_factory=factory)
    try:
        result = await get_tool(runtime, "run_python_code").ainvoke({"code": "print(π)\n!pip install test", "timeout": 17})
        assert "result truncated at 10,000 chars" in result
        assert list(sandbox.content.values()) == ["print(pi)"]
        assert sandbox.commands_seen[0][0].startswith("timeout -s KILL 17s python3 /tmp/exec_")
        artifact = runtime.provenance[-1]["path"]
        recovered = await get_tool(runtime, "read_file").ainvoke({"path": artifact, "start_line": 3001, "end_line": 3002})
        assert recovered == "[lines 3001-3002 of 4000]\nline\nline\n"
        sandbox.content["/tmp/result.txt"] = "a\nb\nc"
        assert await get_tool(runtime, "read_file").ainvoke({"path": "/tmp/result.txt", "start_line": 2, "end_line": 2}) == "[lines 2-2 of 3]\nb\n"
        assert len(builds) == 1
    finally:
        await runtime.aclose()
        await runtime.aclose()
    assert sandbox.kills == 1
    with pytest.raises(ResearchToolError, match="closed"):
        await get_tool(runtime, "run_python_code").ainvoke({"code": "print(1)"})


@pytest.mark.asyncio
async def test_nonzero_python_exit_is_error_and_preserves_stderr(tmp_path):
    sandbox = FakeSandbox(stdout="partial", stderr="ValueError", exit_code=1)
    async with build_research_tools("dynir_researcher", task_id="t", artifact_dir=tmp_path, config=config(), sandbox_factory=lambda: sandbox) as runtime:
        with pytest.raises(ResearchToolError, match=r"\[exit code 1\]\npartial\n--- stderr ---\nValueError"):
            await get_tool(runtime, "run_python_code").ainvoke({"code": "bad code"})
    assert sandbox.kills == 1


@pytest.mark.asyncio
async def test_cancel_during_sandbox_creation_still_releases(tmp_path):
    started, release = asyncio.Event(), asyncio.Event()
    sandbox = FakeSandbox()
    async def factory():
        started.set()
        await release.wait()
        return sandbox
    runtime = build_research_tools("dynir_researcher", task_id="t", artifact_dir=tmp_path, config=config(), sandbox_factory=factory)
    task = asyncio.create_task(get_tool(runtime, "run_python_code").ainvoke({"code": "print(1)"}))
    await started.wait()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    cleanup = asyncio.create_task(runtime.aclose())
    await asyncio.sleep(0)
    release.set()
    await cleanup
    assert sandbox.kills == 1


@pytest.mark.asyncio
async def test_blocked_source_and_workspace_symlink_do_not_leak(tmp_path):
    sandbox = FakeSandbox()
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    outside = tmp_path / "outside.txt"
    outside.write_text("outside")
    (workspace / "escape.txt").symlink_to(outside)
    (workspace / "note.txt").write_text("inside")
    def provider(request):
        return httpx.Response(200, json={"organic": [{"title": "Blocked", "link": "https://blocked.example/a"},
                                                      {"title": "Allowed", "link": "https://allowed.example/a"}]})
    async with httpx.AsyncClient(transport=httpx.MockTransport(provider)) as http:
        async with build_research_tools("dynir_researcher", task_id="t", artifact_dir=tmp_path, config=config(blocked_needles=("blocked.example",), workspace_root=workspace), http_client=http, sandbox_factory=lambda: sandbox) as runtime:
            with pytest.raises(ResearchToolError, match="blocked"):
                await get_tool(runtime, "web_search").ainvoke({"q": "site:blocked.example"})
            text = await get_tool(runtime, "web_search").ainvoke({"q": "query"})
            assert "Blocked" not in text and len(runtime.evidence_cards) == 1
            assert await get_tool(runtime, "read_file").ainvoke({"path": "note.txt"}) == "inside"
            with pytest.raises(ResearchToolError, match="outside"):
                await get_tool(runtime, "read_file").ainvoke({"path": "escape.txt"})


def test_configuration_and_sanitizer_are_explicit():
    cfg = ResearchToolConfig.from_profile({"summary_llm": {"model": "summary"}}, web_policy={"blocked_terms": ["answer key", "dataset.example"]})
    assert cfg.blocked_needles == ("answer key", "dataset.example")
    assert cfg.summary_llm["model"] == "summary"
    assert sanitize_code("# 中文 π\nx = 2²\n!pip install x") == "# ?? pi\nx = 2**2"


def test_pinned_summary_never_switches_to_environment_models():
    pinned = {
        "base_url": "https://router.test/v1", "model": "deepseek/deepseek-v4.1-flash",
        "provider_routing": {"only": ["deepseek"], "allow_fallbacks": False},
        "allow_environment_fallback": False,
    }
    environment = {
        "SUMMARY_LLM_BASE_URL": "https://other.test/v1", "SUMMARY_LLM_MODEL": "other-primary",
        "SUMMARY_LLM_FALLBACK_BASE_URL": "https://fallback.test/v1", "SUMMARY_LLM_FALLBACK_MODEL": "other-fallback",
    }
    assert summary_candidates(pinned, environment=environment) == [pinned]
    # Missing configured endpoint must not silently replace a pinned model.
    incomplete = {key: value for key, value in pinned.items() if key != "base_url"}
    assert summary_candidates(incomplete, environment=environment) == []
