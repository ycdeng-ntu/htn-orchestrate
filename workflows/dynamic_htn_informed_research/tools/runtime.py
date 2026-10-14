"""Native six-tool Miro research family, independent for every worker.

Selected production backends are Serper, Jina (+ direct HTML), academic OA APIs
and E2B. Credentials are resolved at construction, never by importing a source
checkout. URL retry state is shared within each question; artifacts and sandbox
ownership remain per worker. Mixed fetch batches preserve successful results.
"""
from __future__ import annotations

import asyncio
import hashlib
import json
import math
import os
import re
import shlex
import time
import uuid
from copy import deepcopy
from contextvars import ContextVar
from dataclasses import dataclass, field, replace
from functools import partial
from pathlib import Path
from typing import Any, Callable, Mapping
from urllib.parse import unquote

import httpx

from agent_harness.core.atomic_files import atomic_write_text
from agent_harness.core.tool import Tool, ToolExecutionError, ToolOutput

from ._pure import (coerce_json_list, format_search, format_scholar, _is_blacklisted,
                    _normalise_inputs, _extract_jina_body, biorxiv_to_pdf, extract_pmcid,
                    is_garbage_content, route_url, sanitize_code)
from .academic import AcademicFetcher
from .html_extract import HtmlExtractionError, extract_html, is_html_response
from .readback import paginate_text
from .sandbox import WorkerSandbox, call_adapter, e2b_factory
from .scrape_cache import _ScrapeCache, format_skip_message
from .summary import SummaryExtractor, raw_fallback

_SCHEMAS = json.loads(Path(__file__).with_name("schemas.json").read_text(encoding="utf-8"))
_ALL_TOOLS = ("web_search", "web_fetch", "scholar_search", "run_python_code", "read_file", "submit_report")
ROLE_TOOLS = {
    "dynir_researcher": _ALL_TOOLS,
    "dynir_verifier": tuple(t for t in _ALL_TOOLS if t != "read_file"),
    **{role: _ALL_TOOLS for role in ("dynir_science_solver", "dynir_science_checker", "dynir_science_mcq_solver", "dynir_science_mcq_checker")},
    **{role: _ALL_TOOLS for role in ("dynir_report_researcher", "dynir_report_verifier", "dynir_report_final_verifier")},
}


class ResearchToolError(ToolExecutionError):
    """An actual tool failure; never disguised as a successful result string."""


@dataclass
class ReportLatch:
    report: str = ""
    confidence: float = 0.7
    submitted: bool = False


@dataclass
class ResearchToolConfig:
    network_enabled: bool = True
    serper_api_key: str = field(default_factory=lambda: os.getenv("SERPER_API_KEY", ""))
    serper_base_url: str = field(default_factory=lambda: os.getenv("SERPER_BASE_URL", "https://google.serper.dev"))
    jina_api_key: str = field(default_factory=lambda: os.getenv("JINA_API_KEY", ""))
    jina_base_url: str = field(default_factory=lambda: os.getenv("JINA_BASE_URL", "https://r.jina.ai"))
    e2b_api_key: str = field(default_factory=lambda: os.getenv("E2B_API_KEY", ""))
    e2b_template: str = field(default_factory=lambda: os.getenv("E2B_TEMPLATE", "code-interpreter-v1"))
    e2b_timeout: int = 1800
    ncbi_api_key: str = field(default_factory=lambda: os.getenv("NCBI_API_KEY", ""))
    unpaywall_email: str = field(default_factory=lambda: os.getenv("UNPAYWALL_EMAIL", ""))
    blocked_needles: tuple[str, ...] = ()
    read_roots: tuple[str | Path, ...] = field(default_factory=lambda: (
        Path(__file__).resolve().parents[3] / "plugins" / "skills",
        Path(__file__).resolve().parents[3] / "data",
    ))
    workspace_root: str | Path | None = None
    summary_llm: Mapping[str, Any] = field(default_factory=dict)
    summary_environment: Mapping[str, Any] | None = None
    summary_use_environment: bool = True
    extract_mode: str = ""
    usage_sink: Callable[..., Any] | None = None
    retry_delay: float = 1.0
    direct_fetch_enabled: bool = True
    content_cache_enabled: bool = False
    scrape_cache: _ScrapeCache | None = field(default=None, repr=False, compare=False)
    # Keep comfortably under AgentHarness's 150,000-char tool-result ceiling;
    # raw text is persisted first and can be read with read_file.
    max_inline_chars: int = 140_000

    @classmethod
    def from_profile(cls, profile: Mapping[str, Any], *, blocked_needles: tuple[str, ...] = (), web_policy: dict | None = None, **overrides: Any) -> "ResearchToolConfig":
        web = profile.get("web") or {}
        data = {"summary_llm": deepcopy(profile.get("summary_llm") or {}),
                "blocked_needles": tuple(dict.fromkeys(blocked_needles)),
                "content_cache_enabled": bool(web.get("content_cache_enabled", False)),
                "extract_mode": str(web.get("extract_mode") or profile.get("extract_mode") or "")}
        data.update(overrides)
        if web_policy is not None:
            from agent_harness.core.web_policy import blocked_sources, validate_web_policy
            policy = validate_web_policy(web_policy)
            data["blocked_needles"] = blocked_sources(policy)
            data["network_enabled"] = policy["network_enabled"]
        return cls(**data)


def _safe_id(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9._:+-]", "_", str(value))[:180] or "worker"


async def _gather_owned(*coroutines: Any, return_exceptions: bool = False) -> list[Any]:
    """Own all children through completion or parent cancellation."""
    tasks = [asyncio.create_task(coro) for coro in coroutines]
    try:
        return list(await asyncio.gather(*tasks, return_exceptions=return_exceptions))
    except BaseException:
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        raise


class ResearchToolRuntime:
    def __init__(self, role_id: str, *, task_id: str, artifact_dir: str | Path,
                 config: ResearchToolConfig | None = None, http_client: httpx.AsyncClient | None = None,
                 summary_client: Callable[..., Any] | None = None, sandbox_factory: Callable[..., Any] | None = None):
        if role_id not in ROLE_TOOLS:
            raise ValueError(f"Unknown research worker role: {role_id}")
        self.config = replace(config, summary_llm=deepcopy(config.summary_llm),
                              summary_environment=deepcopy(config.summary_environment)) if config else ResearchToolConfig()
        if not isinstance(self.config.network_enabled, bool):
            raise ValueError("network_enabled must be boolean")
        if not self.config.network_enabled:
            # Host dataset/skill roots are not offline task inputs.
            self.config.read_roots = ()
            self.config.workspace_root = None
        if self.config.content_cache_enabled:
            raise ValueError("Persistent Miro content_cache_enabled is not implemented; migrated profiles disable it")
        if not 1 <= self.config.max_inline_chars <= 140_000:
            raise ValueError("max_inline_chars must be between 1 and 140000")
        self.role_id, self.task_id = role_id, task_id
        self.worker_id = f"{_safe_id(role_id)}-{uuid.uuid4().hex[:12]}"
        # Unique worker suffix prevents file/provenance clobbering even when
        # multiple same-role workers belong to the same task.
        self.artifact_dir = Path(artifact_dir).resolve() / "tools" / self.worker_id
        self.artifact_dir.mkdir(parents=True, exist_ok=True)
        self._owns_http = http_client is None
        self.http = http_client if http_client is not None else httpx.AsyncClient(follow_redirects=True)
        self.report_latch = ReportLatch()
        self.evidence_cards: list[dict[str, Any]] = []
        self.provenance: list[dict[str, Any]] = []
        self.backend_failures: list[dict[str, Any]] = []
        self._negative_cache = self.config.scrape_cache if self.config.scrape_cache is not None else _ScrapeCache()
        self._fetch_attempts: ContextVar[list[dict[str, Any]] | None] = ContextVar(
            f"fetch-attempts-{self.worker_id}", default=None,
        )
        self._pending_cards: ContextVar[list[dict[str, Any]] | None] = ContextVar(f"cards-{self.worker_id}", default=None)
        self._pending_artifacts: ContextVar[list[dict[str, Any]] | None] = ContextVar(
            f"artifacts-{self.worker_id}", default=None,
        )
        self.closed = False
        self._close_task: asyncio.Task[None] | None = None
        self._active_calls: set[asyncio.Task[Any]] = set()
        self.sandbox = WorkerSandbox(sandbox_factory or partial(e2b_factory, api_key=self.config.e2b_api_key,
                                                              template=self.config.e2b_template, timeout=self.config.e2b_timeout,
                                                              **({"allow_internet_access": False} if not self.config.network_enabled else {})))
        self.academic = AcademicFetcher(self.http, ncbi_api_key=self.config.ncbi_api_key, unpaywall_email=self.config.unpaywall_email)
        self.summary_client = summary_client or SummaryExtractor(self.http, self.config.summary_llm,
                            extract_mode=self.config.extract_mode, use_environment=self.config.summary_use_environment,
                            usage_sink=self.config.usage_sink, retry_delay=self.config.retry_delay,
                            environment=self.config.summary_environment)
        self.tools: list[Tool] = []
        for name in ROLE_TOOLS[role_id]:
            if not self.config.network_enabled and name in {"web_search", "web_fetch", "scholar_search"}:
                continue
            schema = deepcopy(_SCHEMAS[name])
            spec = schema["function"]
            self.tools.append(Tool(name=name, description=spec["description"], parameters=spec["parameters"], func=self._bind_tool(name),
                                   metadata={"openai_schema": schema, "role_id": role_id, "worker_id": self.worker_id}))

    def _bind_tool(self, name: str) -> Callable[..., Any]:
        # Bind through the closure, never a hidden default argument that model
        # kwargs could override to reach tools forbidden for this role.
        async def invoke(**kwargs: Any) -> str:
            if name in {"web_search", "web_fetch", "scholar_search"}:
                self._require_network()
            if self.closed:
                raise ResearchToolError("Worker tools have been closed")
            task = asyncio.current_task()
            if task is not None:
                self._active_calls.add(task)
            pending: list[dict[str, Any]] = []
            token = self._pending_cards.set(pending)
            artifacts: list[dict[str, Any]] = []
            artifact_token = self._pending_artifacts.set(artifacts)
            try:
                result = await getattr(self, name)(**kwargs)
                self.evidence_cards.extend(pending)
                metadata = deepcopy(result.metadata) if isinstance(result, ToolOutput) else {}
                metadata["artifact_refs"] = [*artifacts, *metadata.get("artifact_refs", [])]
                return ToolOutput(str(result), metadata=metadata)
            except ToolExecutionError as exc:
                exc.metadata = deepcopy(exc.metadata)
                exc.metadata["artifact_refs"] = [*artifacts, *exc.metadata.get("artifact_refs", [])]
                raise
            finally:
                self._pending_artifacts.reset(artifact_token)
                self._pending_cards.reset(token)
                if task is not None:
                    self._active_calls.discard(task)
        return invoke

    def _require_network(self) -> None:
        if not self.config.network_enabled:
            raise ResearchToolError("External retrieval is disabled by the benchmark web policy")

    def _check_allowed(self, text: str) -> None:
        self._require_network()
        normalized = unquote(str(text)).lower()
        if any(str(needle).lower() in normalized for needle in self.config.blocked_needles if needle):
            raise ResearchToolError("Request blocked by the task's source policy")

    def _allowed(self, text: str) -> bool:
        try:
            self._check_allowed(text)
            return True
        except ResearchToolError:
            return False

    def _persist(self, tool_name: str, content: str, *, artifact_kind: str = "trace", **metadata: Any) -> Path:
        call_id = uuid.uuid4().hex
        path = self.artifact_dir / f"{call_id}.json"
        digest = hashlib.sha256(content.encode()).hexdigest()
        record = {"tool_name": tool_name, "task_id": self.task_id, "worker_id": self.worker_id,
                  "call_id": call_id, "timestamp": time.time(), "result_chars": len(content),
                  "content": content, "artifact_kind": artifact_kind, "sha256": digest, **metadata}
        atomic_write_text(path, json.dumps(record, ensure_ascii=False))
        self.provenance.append({k: v for k, v in record.items() if k != "content"} | {"path": str(path)})
        pending = self._pending_artifacts.get()
        if pending is not None:
            pending.append({"path": str(path), "kind": artifact_kind,
                            "char_count": len(content), "sha256": digest})
        return path

    def _inline(self, tool_name: str, content: str, limit: int | None = None) -> str:
        limit = limit or self.config.max_inline_chars
        if len(content) <= limit:
            return content
        digest = hashlib.sha256(content.encode()).hexdigest()
        existing = next((ref for ref in self._pending_artifacts.get() or []
                         if ref["sha256"] == digest and ref["kind"] in {"source", "output_part"}), None)
        path = Path(existing["path"]) if existing else self._persist(
            tool_name, content, artifact_kind="output_part",
        )
        truncated = content[:limit]
        last = truncated.rfind("\n", max(0, limit - 500))
        if last > limit // 2:
            truncated = truncated[:last]
        rendered = truncated + (f"\n\n[... result truncated at {limit:,} chars. Full content ({len(content):,} chars) "
                                f"saved to: {path}. Use read_file to access if needed.]")
        return ToolOutput(rendered, metadata={"artifact_refs": [
            {"path": str(path), "kind": "tool_output", "char_count": len(content), "sha256": digest},
        ]})

    def _record_source(self, name: str, url: str, title: str = "", **kwargs: Any) -> None:
        if url.startswith(("http://", "https://")):
            pending = self._pending_cards.get()
            target = pending if pending is not None else self.evidence_cards
            target.append({"tool": name, "source": {"url": url, "title": title}, **kwargs})

    async def _serper(self, query: str, *, scholar: bool, num_results: int, gl: str, hl: str, tbs: str = "") -> dict[str, Any]:
        self._check_allowed(query)
        if not self.config.serper_api_key:
            raise ResearchToolError("SERPER_API_KEY is required for search")
        batch = _safe_id(f"dynir-{self.task_id}")
        headers = {"X-API-KEY": self.config.serper_api_key, "Content-Type": "application/json",
                   "X-MiroAPI-Batch-Id": batch, "X-MiroAPI-Trace-Id": f"{batch}:{_safe_id(self.worker_id)}:{uuid.uuid4().hex}"}
        payload: dict[str, Any] = {"q": query, "num": max(1, min(20 if scholar else 100, num_results)), "gl": gl, "hl": hl}
        if tbs:
            payload["tbs"] = tbs
        endpoint = self.config.serper_base_url.rstrip("/") + ("/scholar" if scholar else "/search")
        for attempt in range(3):
            try:
                response = await self.http.post(endpoint, headers=headers, json=payload, timeout=60)
                response.raise_for_status()
                data = response.json()
                if not isinstance(data, dict):
                    raise ResearchToolError("Serper returned a non-object response")
                if not data:
                    return data
                # Filtering occurs before both presentation and evidence indexing.
                data["organic"] = [item for item in (data.get("organic") or []) if isinstance(item, dict)
                                   and self._allowed(str(item.get("link") or item.get("url") or ""))]
                for item in data["organic"]:
                    if item.get("pdfUrl") and not self._allowed(str(item["pdfUrl"])):
                        item.pop("pdfUrl")
                for section in ("answerBox", "knowledgeGraph"):
                    item = data.get(section)
                    if isinstance(item, dict) and not self._allowed(str(item.get("link") or item.get("website") or "")):
                        data.pop(section)
                return data
            except httpx.HTTPStatusError as exc:
                if exc.response.status_code != 429 and exc.response.status_code < 500:
                    raise ResearchToolError(f"Serper HTTP {exc.response.status_code}") from exc
                if attempt == 2:
                    raise ResearchToolError(f"Serper HTTP {exc.response.status_code}; retries exhausted") from exc
            except httpx.TransportError as exc:
                if attempt == 2:
                    raise ResearchToolError("Serper transport failed; retries exhausted") from exc
            except ValueError as exc:
                raise ResearchToolError("Serper returned invalid JSON") from exc
            await asyncio.sleep(self.config.retry_delay * 2 ** attempt)
        raise AssertionError("unreachable")

    def _search_cards(self, data: dict[str, Any], *, scholar: bool) -> None:
        name = "scholar_search" if scholar else "web_search"
        for item in data.get("organic") or []:
            url = str(item.get("link") or "")
            if not scholar and _is_blacklisted(url):
                continue
            self._record_source(name, url, str(item.get("title") or ""))
            if scholar and item.get("pdfUrl"):
                self._record_source(name, str(item["pdfUrl"]), str(item.get("title") or ""))

    async def web_search(self, q: str | list[str], num_results: int = 10, gl: str = "us", hl: str = "en", tbs: str = "") -> str:
        queries = coerce_json_list(q)
        queries = [queries] if isinstance(queries, str) else queries
        queries = [s.strip() for s in queries if isinstance(s, str) and s.strip()] if isinstance(queries, list) else []
        if not queries:
            raise ResearchToolError("Search query is required")
        results = await _gather_owned(
            *(self._serper(query, scholar=False, num_results=num_results, gl=gl, hl=hl, tbs=tbs) for query in queries),
            return_exceptions=True,
        )
        for result in results:
            if isinstance(result, BaseException) and not isinstance(result, Exception):
                raise result
        if len(results) == 1 and isinstance(results[0], Exception):
            raise results[0]
        seen: set[str] = set()
        parts = []
        for query, data in zip(queries, results):
            if isinstance(data, Exception):
                detail = str(data) if isinstance(data, ResearchToolError) else type(data).__name__
                parts.append(f"## Query: {query}\n\nError: {detail}")
                continue
            if not data:
                parts.append(f"## Query: {query}\n\nNo results (error or empty)." if len(queries) > 1 else f"No results found for: {query}")
                continue
            unique = []
            for item in data.get("organic") or []:
                link = str(item.get("link") or item.get("url") or "")
                if link and link in seen:
                    continue
                if link:
                    seen.add(link)
                unique.append(item)
            data["organic"] = unique
            self._search_cards(data, scholar=False)
            formatted = format_search(data)
            parts.append(f"## Query: {query}\n\n{formatted}" if len(queries) > 1 else formatted)
        rendered = self._inline("web_search", "\n\n---\n\n".join(parts))
        if all(isinstance(result, Exception) for result in results):
            raise ResearchToolError(rendered)
        return rendered

    async def scholar_search(self, query: str, num_results: int = 10, gl: str = "us", hl: str = "en") -> str:
        if not query.strip():
            raise ResearchToolError("Scholar query is required")
        data = await self._serper(query.strip(), scholar=True, num_results=num_results, gl=gl, hl=hl)
        if not data:
            return f"No scholarly results found for: {query.strip()}"
        self._search_cards(data, scholar=True)
        return self._inline("scholar_search", format_scholar(data))

    def _fetch_failure(self, backend: str, url: str, *, reason: str,
                       status: int | None = None, error: str | None = None) -> None:
        record = {"backend": backend, "url": url, "reason": reason, "status": status}
        if error:
            record["error"] = error
        self.backend_failures.append(record)
        attempts = self._fetch_attempts.get()
        if attempts is not None:
            attempts.append(record)

    async def _jina(self, url: str) -> str:
        self._check_allowed(url)
        if not self.config.jina_api_key:
            self._fetch_failure("jina", url, reason="not_configured")
            return ""
        browser = False
        # A browser escalation is an alternate extraction mode. Repeating an
        # unsuccessful pipeline is owned by the question's delayed retry policy.
        for _ in range(2):
            headers = {"Authorization": f"Bearer {self.config.jina_api_key}", "Content-Type": "application/json",
                       "Accept": "application/json", "X-Timeout": "30" if browser else "20"}
            if browser:
                headers["X-Engine"] = "browser"
            try:
                response = await self.http.post(self.config.jina_base_url, headers=headers, json={"url": url}, timeout=60)
                status = response.status_code
                if status == 200:
                    content = _extract_jina_body(response.text)
                    if not content:
                        self._fetch_failure("jina", url, reason="empty_content", status=status)
                    return content
                self._fetch_failure("jina", url, reason="http_error", status=status)
                if status in (403, 422):
                    if not browser:
                        browser = True
                        continue
                return ""
            except httpx.TransportError as exc:
                self._fetch_failure(
                    "jina", url, reason="timeout" if isinstance(exc, httpx.TimeoutException) else "transport_error",
                    error=type(exc).__name__,
                )
                return ""
        return ""

    async def _direct(self, url: str) -> str:
        self._check_allowed(url)
        if not self.config.direct_fetch_enabled:
            return ""
        try:
            response = await self.http.get(url, headers={"User-Agent": "AgentHarness/1.0"}, timeout=30, follow_redirects=True)
            response.raise_for_status()
            self._check_allowed(str(response.url))
            if not is_html_response(response.content, response.headers.get("content-type", "")):
                self._fetch_failure("direct", url, reason="unsupported_content_type", status=response.status_code)
                return ""
            content = await extract_html(response.text)
            if not content:
                self._fetch_failure("direct", url, reason="empty_content", status=response.status_code)
            return content
        except HtmlExtractionError as exc:
            if exc.diagnostic:
                self._persist("html_extraction_error", exc.diagnostic, requested_url=url,
                              error=str(exc), worker_exit_code=exc.returncode)
            self._fetch_failure(
                "direct", url, reason="timeout" if exc.reason == "timeout" else "extraction_error",
                error=str(exc),
            )
            return ""
        except (httpx.HTTPError, TimeoutError, ValueError) as exc:
            self._fetch_failure(
                "direct", url,
                reason=("http_error" if isinstance(exc, httpx.HTTPStatusError)
                        else "timeout" if isinstance(exc, (httpx.TimeoutException, TimeoutError))
                        else "transport_error" if isinstance(exc, httpx.TransportError)
                        else "extraction_error"),
                status=exc.response.status_code if isinstance(exc, httpx.HTTPStatusError) else None,
                error=type(exc).__name__,
            )
            return ""

    async def _fetch_content(self, url: str) -> tuple[str, str]:
        self._check_allowed(url)
        route = route_url(url)
        content, actual_url = "", url
        if route == "pmc":
            content = await self.academic.pmc(extract_pmcid(url))
        elif route == "pubmed":
            pmcid = await self.academic.pubmed_to_pmc(url)
            content = await self.academic.pmc(pmcid)
            if content:
                actual_url = f"https://pmc.ncbi.nlm.nih.gov/articles/{pmcid}/"
        elif route in ("biorxiv", "paywall"):
            alternative = biorxiv_to_pdf(url) if route == "biorxiv" else await self.academic.oa_url(url)
            if alternative:
                content = await self._jina(alternative)
                if content:
                    actual_url = alternative
            if not content or len(content) < 500:
                fallback = await self._jina(url)
                if fallback and not is_garbage_content(fallback) and (
                    is_garbage_content(content) or len(fallback) > len(content)
                ):
                    content, actual_url = fallback, url
        if not content and route not in ("biorxiv", "paywall"):
            content, actual_url = await self._jina(url), url
        if (not content or is_garbage_content(content)) and route == "jina":
            content = await self._direct(url)
        if route not in ("pmc", "pubmed") and (is_garbage_content(content) or (route == "paywall" and len(content) < 500)):
            alternative = await self.academic.oa_url(url)
            if alternative:
                recovered = await self._jina(alternative)
                if recovered and not is_garbage_content(recovered) and (
                    is_garbage_content(content) or len(recovered) > len(content)
                ):
                    content, actual_url = recovered, alternative
        return content, actual_url

    async def _fetch_one(self, url: str, focus: str) -> str:
        url = url.strip()
        if not url.startswith(("http://", "https://")):
            url = "https://" + url
        if "arxiv.org/pdf/" in url:
            url = url.replace("/pdf/", "/abs/").split(".pdf")[0]
        elif "arxiv.org/pdf" in url:
            url = url.replace("/pdf", "/abs")
        self._check_allowed(url)
        # Hold only through retrieval, not per-worker persistence or summary.
        # A failed owner populates shared state before any waiter may retry.
        async with self._negative_cache.guard(url):
            cached = self._negative_cache.check(url)
            if cached is not None:
                message = format_skip_message(url, cached)
                self._persist("web_fetch_attempt", message, requested_url=url, outcome="skipped")
                raise ResearchToolError(message)
            attempts: list[dict[str, Any]] = []
            token = self._fetch_attempts.set(attempts)
            try:
                try:
                    content, actual_url = await self._fetch_content(url)
                except ResearchToolError:
                    raise  # Source-policy rejection is not a retrieval failure.
                except Exception as exc:
                    self._fetch_failure("fetch", url, reason="fetch_error", error=type(exc).__name__)
                    content, actual_url = "", url
                if not content or is_garbage_content(content):
                    if content:
                        self._fetch_failure("content_validation", url, reason="unusable_content")
                    failures = [item for item in attempts if item["url"] == url]
                    failure = (failures or attempts or [{"reason": "empty_content", "backend": "fetch", "status": None}])[-1]
                    entry = self._negative_cache.record_failure(
                        url, failure["status"], reason=failure["reason"], backend=failure["backend"],
                    )
                    message = f"Could not extract content from {url}. " + format_skip_message(url, entry)
                    self._persist("web_fetch_attempt", message, requested_url=url,
                                  outcome="failed", backend_attempts=attempts,
                                  retry_state=self._negative_cache.snapshot().get(url))
                    raise ResearchToolError(message)
                self._negative_cache.record_success(url)
                self._persist("web_fetch_attempt", "", requested_url=url, actual_url=actual_url,
                              outcome="succeeded", backend_attempts=attempts)
            except asyncio.CancelledError:
                # A cancelled worker is not evidence that the URL is broken.
                raise
            finally:
                self._fetch_attempts.reset(token)
        raw_path = self._persist("web_fetch", content, artifact_kind="source", requested_url=url, actual_url=actual_url, focus=focus,
                                 sha256=hashlib.sha256(content.encode()).hexdigest())
        title = next((line.lstrip("# ").strip() for line in content.splitlines() if line.startswith("# ")), "")
        self._record_source("web_fetch", url, title, raw_path=str(raw_path), actual_url=actual_url)
        if focus and focus.strip():
            try:
                extracted = await call_adapter(self.summary_client, content, focus)
            except Exception as exc:
                # Retrieval succeeded: preserve the raw evidence even if its
                # optional extraction stage fails, without banning the URL.
                self._persist("web_fetch_summary", "", requested_url=url,
                              outcome="raw_fallback", error=type(exc).__name__)
                extracted = raw_fallback(content)
            return self._inline("web_fetch", str(extracted) if extracted else raw_fallback(content))
        return self._inline("web_fetch", content)

    async def web_fetch(self, url: str | list[str], info_to_extract: str | list[str] = "") -> str:
        urls, focuses = _normalise_inputs(url, info_to_extract)
        if not urls:
            raise ResearchToolError("URL is required")
        results = await _gather_owned(
            *(self._fetch_one(u, focus) for u, focus in zip(urls, focuses)), return_exceptions=True,
        )
        for result in results:
            if isinstance(result, BaseException) and not isinstance(result, Exception):
                raise result
        if len(results) == 1:
            if isinstance(results[0], Exception):
                raise results[0]
            return results[0]
        parts = []
        for i, (u, result) in enumerate(zip(urls, results), 1):
            if isinstance(result, Exception):
                detail = str(result) if isinstance(result, ResearchToolError) else type(result).__name__
                parts.append(f"[{i}] URL: {u}\n    Error: {detail}")
            else:
                parts.append(f"[{i}] URL: {u}\n    Info: {result}")
        rendered = self._inline("web_fetch", "\n\n".join(parts))
        if all(isinstance(result, Exception) for result in results):
            raise ResearchToolError(rendered)
        return rendered

    async def run_python_code(self, code: str, timeout: int = 60) -> str:
        if not code.strip():
            raise ResearchToolError("Error: empty code provided.")
        if isinstance(timeout, bool) or not isinstance(timeout, int) or timeout <= 0:
            raise ResearchToolError("timeout must be a positive integer")
        async with self.sandbox.command_lock:
            sandbox = await self.sandbox.get()
            path = f"/tmp/exec_{uuid.uuid4().hex[:8]}.py"
            await call_adapter(sandbox.files.write, path, sanitize_code(code))
            try:
                # The process deadline also stops remote Python when its caller's
                # SDK wait times out; worker cleanup kills the entire sandbox.
                result = await call_adapter(sandbox.commands.run,
                    f"timeout -s KILL {timeout}s python3 {shlex.quote(path)}", timeout=timeout + 5)
            except Exception as exc:
                if type(exc).__name__ == "CommandExitException" and hasattr(exc, "exit_code"):
                    result = exc
                elif isinstance(exc, TimeoutError) or "Timeout" in type(exc).__name__:
                    raise ResearchToolError(f"Error: execution timed out after {timeout}s.") from exc
                else:
                    raise ResearchToolError(f"Python sandbox failed: {type(exc).__name__}") from exc
            output = str(getattr(result, "stdout", "") or "")
            if getattr(result, "stderr", ""):
                output += "\n--- stderr ---\n" + str(result.stderr)
            if result.exit_code != 0:
                # Preserve the useful diagnostic, but classify execution failure
                # correctly in the host loop's ToolResult.
                raise ResearchToolError(self._inline("run_python_code", f"[exit code {result.exit_code}]\n{output}", 10_000))
            return self._inline("run_python_code", output, 10_000)

    async def read_file(
        self, path: str, start_line: int = 0, end_line: int = 0,
        start_char: int | None = None, max_chars: int = 15_000,
    ) -> str:
        if not path.strip():
            raise ResearchToolError("Error: file path is required.")
        read_args = {"start_line": start_line, "end_line": end_line,
                     "start_char": start_char, "max_chars": max_chars}
        try:
            # Validate before an invalid request can start a remote sandbox.
            if start_char is not None and (
                isinstance(start_char, bool) or not isinstance(start_char, int) or start_char < 0
            ):
                raise ValueError("start_char must be a non-negative integer")
            paginate_text("", **{**read_args, "start_char": 0 if start_char is not None else None})
        except ValueError as exc:
            raise ResearchToolError(str(exc)) from exc
        # Host reads are limited to explicitly supplied roots and this worker's
        # artifacts, using resolved paths (including symlink containment).
        raw = Path(path).expanduser()
        roots = [self.artifact_dir, *(Path(p).resolve() for p in self.config.read_roots)]
        if self.config.workspace_root:
            roots.append(Path(self.config.workspace_root).resolve())
        candidates = [raw.resolve()] if raw.is_absolute() else [raw.resolve(), *((r / raw).resolve() for r in roots)]
        local = next((p for p in candidates if any(p.is_relative_to(root) for root in roots) and p.is_file()), None)
        if local:
            if any(needle in local.name.lower() for needle in (".env", "credentials", "secret", "password", "token", ".key", ".pem", ".cert")):
                raise ResearchToolError("Access to credential files is blocked")
            def read_local() -> str:
                # Character cursors refer to the actual text, including CRLF.
                with local.open(encoding="utf-8", newline="") as stream:
                    return stream.read()

            content = await asyncio.to_thread(read_local)
            # Own overflow/provenance JSON envelopes expose the original text,
            # allowing line ranges to recover content after the inline prefix.
            if local.parent == self.artifact_dir and local.suffix == ".json":
                stored = json.loads(content)
                if isinstance(stored, dict) and isinstance(stored.get("content"), str):
                    content = stored["content"]
        elif raw.is_absolute():
            sandbox = await self.sandbox.get()
            # E2B's /files download endpoint rejects directories with HTTP 400.
            # Stat uses the filesystem RPC and preserves the file-only contract
            # without executing code or treating a listing as file contents.
            get_info = getattr(sandbox.files, "get_info", None)
            if callable(get_info):
                info = await call_adapter(get_info, str(raw))
                entry_type = getattr(info, "type", None)
                if getattr(entry_type, "value", entry_type) == "dir":
                    raise ResearchToolError(
                        f"read_file requires a file path; {str(raw)!r} is a directory. "
                        "To list this sandbox directory, use run_python_code with "
                        f"`import os; print(os.listdir({str(raw)!r}))`, then read a specific file."
                    )
            content = await call_adapter(sandbox.files.read, str(raw))
            if isinstance(content, bytes):
                content = content.decode("utf-8")
        else:
            raise ResearchToolError("File is outside configured read roots or does not exist")
        content = str(content)
        try:
            page = paginate_text(content, **read_args)
        except ValueError as exc:
            raise ResearchToolError(str(exc)) from exc
        metadata = {}
        if local:
            metadata["read_ref"] = {
                "path": str(local), "read_args": read_args, "total_chars": len(content),
            }
        return ToolOutput(page, metadata=metadata)

    async def submit_report(self, content: str, confidence: float = 0.7) -> str:
        text = (content or "").strip()
        if not text:
            raise ResearchToolError("submit_report rejected: `content` was empty. Pass the complete Scope/Finding/Evidence report in `content`.")
        if self.report_latch.submitted:
            raise ResearchToolError("submit_report was already called for this worker")
        try:
            conf = float(confidence)
            conf = max(0.0, min(1.0, conf)) if math.isfinite(conf) else 0.7
        except (TypeError, ValueError):
            conf = 0.7
        self.report_latch.report, self.report_latch.confidence, self.report_latch.submitted = text, conf, True
        return f"Report submitted ({len(text)} chars, confidence={conf:.2f}). Task loop will exit after this turn."

    async def _release(self) -> None:
        active = tuple(self._active_calls)
        for task in active:
            task.cancel()
        if active:
            await asyncio.gather(*active, return_exceptions=True)
        try:
            await self.sandbox.aclose()
        finally:
            if self._owns_http:
                await self.http.aclose()

    async def aclose(self) -> None:
        self.closed = True
        if self._close_task is None:
            self._close_task = asyncio.create_task(self._release())
        try:
            await asyncio.shield(self._close_task)
        except asyncio.CancelledError:
            await self._close_task
            raise

    async def __aenter__(self) -> "ResearchToolRuntime":
        return self

    async def __aexit__(self, *_: Any) -> None:
        await self.aclose()


def build_research_tools(role_id: str, *, task_id: str, artifact_dir: str | Path,
                         config: ResearchToolConfig | None = None, http_client: httpx.AsyncClient | None = None,
                         summary_client: Callable[..., Any] | None = None, sandbox_factory: Callable[..., Any] | None = None) -> ResearchToolRuntime:
    return ResearchToolRuntime(role_id, task_id=task_id, artifact_dir=artifact_dir, config=config,
                               http_client=http_client, summary_client=summary_client, sandbox_factory=sandbox_factory)
