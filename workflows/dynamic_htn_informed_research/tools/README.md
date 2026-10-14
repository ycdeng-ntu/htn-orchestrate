# Research Tools

`build_research_tools(role_id, *, task_id, artifact_dir, config=None,
http_client=None, summary_client=None, sandbox_factory=None)` returns an owned
`ResearchToolRuntime`. Pass `runtime.tools` to the native loop; inspect
`runtime.report_latch` (`submitted`, `report`, `confidence`) after a turn and
always `await runtime.aclose()` in `finally`. An async context manager is also
supported. Every worker gets independent tool instances, source cards, artifact
directory and lazy E2B sandbox. Workers in one question share URL retry state
through `ResearchToolConfig.scrape_cache`, owned by `TaskRuntime`; constructing
standalone tools without this field creates an isolated retry state.

The six public JSON schemas are frozen in `schemas.json`. Role allowlists match
the four strategies: research verifiers omit `read_file`; report workers expose
only search, fetch, scholar, and submission.

## Configuration

`ResearchToolConfig.from_profile(profile, blocked_needles=(), **overrides)`
accepts a resolved profile. `summary_llm` retains its independent primary,
profile fallback, environment fallback, extraction prompt, context-length retry,
thinking-disabled retry, and 20,000-character raw fallback. The caller explicitly
sets `extract_mode="science"` when the benchmark extraction policy requests it.
`usage_sink(**event)` receives one event per actual summary HTTP request,
including failed, cancelled, and usage-less requests. A supplied
`summary_client(content, focus)` replaces that entire summary policy.
`summary_environment` may supply an explicit task-local mapping of
`SUMMARY_LLM_*` values (for example, resolved from a dotenv file). When supplied,
candidate resolution reads that mapping instead of process environment values.

## Backends

Supported backends are Serper `/search` and `/scholar`, Jina Reader with browser
escalation, direct HTML via trafilatura, PMC/PubMed/bioRxiv/medRxiv and
Unpaywall/CrossRef, and E2B for code execution. The caller can inject an
`httpx.AsyncClient` and a sync/async `sandbox_factory()` for offline tests.
The sandbox's minimal interface is `files.write/read`, `commands.run`, and
`kill`; no host subprocess fallback is installed. Caller-owned HTTP clients are
not closed. Sandbox ownership always transfers to the runtime.

## Sandbox provisioning

The production E2B factory prepares the research Python environment before its
first use. It imports all 18 scientific packages listed in `sandbox_bootstrap.py`
(including Pillow, Beautiful Soup, PyMuPDF, and networkx). Missing packages are
installed with `python3 -m pip`, then all imports are verified again. The
package/import mapping handles name differences (Pillow to PIL, Beautiful Soup
to bs4, PyMuPDF to fitz). A prepared template skips pip, and all calls in one
worker reuse the same initialized sandbox. Injected test factories supply their
own ready-to-use sandbox and do not invoke production provisioning.

Each import probe has a 60-second remote deadline and installation has a
180-second remote deadline, with the SDK wait allowing another five seconds.
This first-use setup happens before the individual Python code timeout starts.
Installation or verification failure prevents code execution and releases the
created sandbox.

## Error handling and retries

- Backend, execution, or invalid-submission failures raise exceptions that mark
  tool results as errors. A mixed `web_fetch` batch returns an `Info` or `Error`
  block per URL and commits successful evidence cards. Multi-query `web_search`
  likewise retains successful results and source cards, with an error block for
  each failed query.
- Failed URL retrievals share a question-local retry list across workers, epochs,
  and report gap recovery. The first failure starts a 60-second cooldown, the
  second 120 seconds, and the third 240 seconds. A fourth consecutive failure
  stops that URL for the rest of the question. Cooldown expiry permits a later
  request but does not schedule a retry. Successful retrieval clears the
  consecutive failure count. Other questions start fresh.
- One attempt means the complete retrieval pipeline, including browser/direct/OA
  fallbacks, not one HTTP request. Jina 403/422 can still escalate to the browser
  engine. Per-URL guards make concurrent workers recheck retry state after the
  current retrieval finishes. Unrelated URLs remain parallel.
- `web_fetch_attempt` artifacts record failed backend attempts, actual HTTP
  statuses or transport categories, and the final retrieval/skip outcome.
  `fetch_retry_state.json` saves the question's retry list on runtime cleanup.
- Raw fetched text is persisted before summarization with requested/actual URL,
  focus, digest, and task/worker IDs. Web output defaults to 140,000 characters;
  Python output retains a 10,000-character limit. `read_file` unwraps owned JSON
  artifacts and supports inclusive line ranges.
- `UNPAYWALL_EMAIL` must be configured to enable the Unpaywall route. E2B is the
  only code execution backend; template and TTL are configurable.

## Tests

Offline coverage is in `tests/workflows/dynamic_htn_informed_research/test_tools_native.py`.
No real provider or E2B request is required by these tests.
