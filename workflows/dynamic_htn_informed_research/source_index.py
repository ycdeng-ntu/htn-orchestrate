"""Deterministic source-title enrichment for submitted worker reports.

Workers already accumulate evidence cards from real ``web_search`` and
``web_fetch`` tool results.  This module uses URLs cited in the final report as
join keys into those observed cards.  It never invents a title and never treats
an unmatched or ambiguous URL as observed.
"""

from __future__ import annotations

import re
import unicodedata
from bisect import bisect_left
from dataclasses import dataclass
from typing import Any, Iterable
from urllib.parse import parse_qsl, unquote, urlencode, urlparse, urlunparse


_URL_RE = re.compile(r"https?://[^\s<>\"']+", re.IGNORECASE)
_TRAILING_URL_PUNCTUATION = ").,;:!?]}"
_TRACKING_QUERY_KEYS = {
    "fbclid",
    "gclid",
    "dclid",
    "mc_cid",
    "mc_eid",
    "msclkid",
}
_FUZZY_LCS_RATIO = 0.85
_FUZZY_TOKEN_OVERLAP = 0.60
_FUZZY_WINNER_MARGIN = 0.10
_MAX_FUZZY_TEXT_CHARS = 2_048
_MAX_INDEXED_REPORT_URLS = 16
_RENDERED_OBSERVED_SOURCE = re.compile(
    r'^- \[(?:exact_observed|canonical_observed|fuzzy_observed(?: score=[0-9.]+)?)\] '
    r'"(?P<title>.*?)" — (?:observed: )?(?P<url>https?://\S+)',
    re.MULTILINE,
)


@dataclass(frozen=True)
class _ObservedSource:
    url: str
    title: str


@dataclass(frozen=True)
class _IndexedSource:
    url: str
    title: str
    canonical_url: str
    host: str
    path: str
    query: str
    comparison_text: str
    tokens: frozenset[str]


@dataclass(frozen=True)
class SourceMatch:
    reported_url: str
    status: str
    observed_url: str | None = None
    title: str | None = None
    score: float | None = None


def _clean_url(url: str) -> str:
    return str(url or "").strip().rstrip(_TRAILING_URL_PUNCTUATION)


def extract_report_urls(report: str) -> list[str]:
    """Return unique HTTP(S) URLs in report order."""

    seen: set[str] = set()
    urls: list[str] = []
    for match in _URL_RE.finditer(report or ""):
        url = _clean_url(match.group(0))
        if url and url not in seen:
            seen.add(url)
            urls.append(url)
    return urls


def _unwrap_known_wrapper(url: str) -> str:
    """Unwrap a small set of deterministic proxy/archive URL shapes."""

    current = _clean_url(url)
    for _ in range(2):
        parsed = urlparse(current)
        host = (parsed.hostname or "").lower()
        path = unquote(parsed.path or "")
        replacement: str | None = None

        if host == "r.jina.ai" and path.lstrip("/").startswith(("http://", "https://")):
            replacement = path.lstrip("/")
        elif host.endswith("web.archive.org"):
            archive_match = re.match(r"^/web/[^/]+/(https?://.+)$", path)
            if archive_match:
                replacement = archive_match.group(1)
        elif host.endswith("google.com") and path == "/url":
            params = dict(parse_qsl(parsed.query, keep_blank_values=True))
            candidate = params.get("q") or params.get("url")
            if candidate and candidate.startswith(("http://", "https://")):
                replacement = candidate

        if not replacement or replacement == current:
            break
        current = replacement
    return current


def _normalised_parts(url: str) -> tuple[str, str, str]:
    parsed = urlparse(_unwrap_known_wrapper(url))
    host = (parsed.hostname or "").lower()
    if host.startswith("www."):
        host = host[4:]
    try:
        port = parsed.port
    except ValueError:
        port = None
    if port and not (
        parsed.scheme.lower() == "http" and port == 80
        or parsed.scheme.lower() == "https" and port == 443
    ):
        host = f"{host}:{port}"

    path = re.sub(r"/{2,}", "/", unquote(parsed.path or "/"))
    if path != "/":
        path = path.rstrip("/")

    query_items = []
    for key, value in parse_qsl(parsed.query, keep_blank_values=True):
        folded = key.lower()
        if folded.startswith("utm_") or folded in _TRACKING_QUERY_KEYS:
            continue
        query_items.append((key, value))
    query_items.sort()
    return host, path, urlencode(query_items, doseq=True)


def canonicalise_url(url: str) -> str:
    """Canonical comparison key; scheme and tracking fragments are irrelevant."""

    host, path, query = _normalised_parts(url)
    return urlunparse(("", host, path, "", query, ""))


def _normalise_title(title: str) -> str:
    """Comparison key for exact title equality across hosts."""

    folded = unicodedata.normalize("NFKC", title).casefold()
    return " ".join(re.sub(r"[^\w]+", " ", folded).split())


def _is_generic_extraction_title(title_key: str) -> bool:
    """Reject summary-extractor headings that are not real page titles."""

    return title_key == "extracted information" or title_key.startswith(
        "extracted information about "
    )


def build_cross_host_title_collision_warning(
    source_indexes: Iterable[str],
) -> str | None:
    """Flag exact normalized-title reuse across distinct observed hosts.

    This is deliberately a narrow signal: it does not fuzzy-match titles or
    infer a common publisher/source family. The coordinator keeps the raw
    source metadata and decides whether separate original provenance exists.
    """

    by_title: dict[str, dict[str, Any]] = {}
    for source_index in source_indexes:
        for match in _RENDERED_OBSERVED_SOURCE.finditer(source_index or ""):
            title = match.group("title").strip()
            if not title or title == "unknown":
                continue
            host, _, _ = _normalised_parts(match.group("url"))
            title_key = _normalise_title(title)
            if not host or not title_key or _is_generic_extraction_title(title_key):
                continue
            group = by_title.setdefault(title_key, {"title": title, "hosts": set()})
            group["hosts"].add(host)

    collisions = [
        group for group in by_title.values() if len(group["hosts"]) >= 2
    ]
    if not collisions:
        return None
    collisions.sort(key=lambda group: (str(group["title"]).casefold(), group["title"]))
    lines = [
        "Deterministic cross-host title collision warning (exact normalized-title "
        "equality only; not a source-family inference):"
    ]
    for group in collisions:
        hosts = ", ".join(sorted(group["hosts"]))
        lines.append(f'- "{group["title"]}" appears on distinct hosts: {hosts}')
    lines.append(
        "Treat each collision group as one evidence lead unless the reports establish "
        "distinct original provenance; title equality alone is not a reason to discard it."
    )
    return "\n".join(lines)


def _same_site(left_host: str, right_host: str) -> bool:
    """Conservative same-site gate without a public-suffix dependency."""

    return bool(
        left_host
        and right_host
        and (
            left_host == right_host
            or left_host.endswith(f".{right_host}")
            or right_host.endswith(f".{left_host}")
        )
    )


def _lcs_ratio(left: str, right: str) -> float:
    """Symmetric LCS similarity without an absolute-length requirement."""

    if not left or not right:
        return 0.0
    if left == right:
        return 1.0
    positions: dict[str, list[int]] = {}
    for index, character in enumerate(right):
        positions.setdefault(character, []).append(index)
    tails: list[int] = []
    for character in left:
        for position in reversed(positions.get(character, ())):
            insertion = bisect_left(tails, position)
            if insertion == len(tails):
                tails.append(position)
            else:
                tails[insertion] = position
    return (2.0 * len(tails)) / (len(left) + len(right))


def _url_tokens(path: str, query: str) -> frozenset[str]:
    return frozenset(
        token
        for token in re.split(r"[^a-z0-9]+", f"{path}?{query}".lower())
        if token
    )


def _token_overlap(left: frozenset[str], right: frozenset[str]) -> float:
    if not left or not right:
        return 0.0
    return len(left & right) / len(left | right)


def _observed_sources(evidence_cards: Any) -> list[_ObservedSource]:
    if not isinstance(evidence_cards, list):
        return []
    by_url: dict[str, _ObservedSource] = {}
    for card in evidence_cards:
        if not isinstance(card, dict):
            continue
        source = card.get("source")
        if not isinstance(source, dict):
            continue
        url = _clean_url(source.get("url") or "")
        if not url.startswith(("http://", "https://")):
            continue
        title = re.sub(r"\s+", " ", str(source.get("title") or "")).strip()
        if _clean_url(title) == url:
            title = ""
        candidate = _ObservedSource(url=url, title=title or "unknown")
        current = by_url.get(url)
        if current is None or current.title == "unknown":
            by_url[url] = candidate
    return list(by_url.values())


class _SourceMatcher:
    """Pre-index one worker's observed sources for repeated report-URL lookup."""

    def __init__(self, observed_sources: Iterable[_ObservedSource]) -> None:
        self._exact: dict[str, _IndexedSource] = {}
        self._canonical: dict[str, _IndexedSource] = {}
        self._by_host: dict[str, list[_IndexedSource]] = {}
        self._compatible_host_cache: dict[str, tuple[_IndexedSource, ...]] = {}

        for source in observed_sources:
            host, path, query = _normalised_parts(source.url)
            canonical_url = canonicalise_url(source.url)
            indexed = _IndexedSource(
                url=source.url,
                title=source.title,
                canonical_url=canonical_url,
                host=host,
                path=path,
                query=query,
                comparison_text=f"{path}?{query}".rstrip("?"),
                tokens=_url_tokens(path, query),
            )
            current_exact = self._exact.get(source.url)
            if current_exact is None or current_exact.title == "unknown":
                self._exact[source.url] = indexed

            current_canonical = self._canonical.get(canonical_url)
            if current_canonical is None or current_canonical.title == "unknown":
                self._canonical[canonical_url] = indexed

        # Fuzzy comparison needs only one representative for each canonical URL.
        for indexed in self._canonical.values():
            self._by_host.setdefault(indexed.host, []).append(indexed)

    def _compatible_sources(self, host: str) -> tuple[_IndexedSource, ...]:
        cached = self._compatible_host_cache.get(host)
        if cached is not None:
            return cached
        compatible = tuple(
            source
            for candidate_host, sources in self._by_host.items()
            if _same_site(host, candidate_host)
            for source in sources
        )
        self._compatible_host_cache[host] = compatible
        return compatible

    def match(self, reported_url: str) -> SourceMatch:
        reported_url = _clean_url(reported_url)
        exact = self._exact.get(reported_url)
        if exact is not None:
            return SourceMatch(
                reported_url=reported_url,
                status="exact_observed",
                observed_url=exact.url,
                title=exact.title,
                score=1.0,
            )

        reported_key = canonicalise_url(reported_url)
        canonical = self._canonical.get(reported_key)
        if canonical is not None:
            return SourceMatch(
                reported_url=reported_url,
                status="canonical_observed",
                observed_url=canonical.url,
                title=canonical.title,
                score=1.0,
            )

        reported_host, reported_path, reported_query = _normalised_parts(reported_url)
        reported_text = f"{reported_path}?{reported_query}".rstrip("?")
        reported_tokens = _url_tokens(reported_path, reported_query)
        scored: list[tuple[float, _IndexedSource]] = []
        for source in self._compatible_sources(reported_host):
            if (
                len(reported_text) > _MAX_FUZZY_TEXT_CHARS
                or len(source.comparison_text) > _MAX_FUZZY_TEXT_CHARS
            ):
                continue
            token_overlap = _token_overlap(reported_tokens, source.tokens)
            if token_overlap < _FUZZY_TOKEN_OVERLAP:
                continue
            lcs_ratio = _lcs_ratio(reported_text, source.comparison_text)
            if lcs_ratio < _FUZZY_LCS_RATIO:
                continue
            score = 0.7 * lcs_ratio + 0.3 * token_overlap
            scored.append((score, source))

        scored.sort(key=lambda item: item[0], reverse=True)
        if not scored:
            return SourceMatch(reported_url=reported_url, status="unobserved")
        if len(scored) > 1 and scored[0][0] - scored[1][0] < _FUZZY_WINNER_MARGIN:
            return SourceMatch(reported_url=reported_url, status="ambiguous")
        score, source = scored[0]
        return SourceMatch(
            reported_url=reported_url,
            status="fuzzy_observed",
            observed_url=source.url,
            title=source.title,
            score=round(score, 3),
        )


def match_report_url(
    reported_url: str,
    observed_sources: Iterable[_ObservedSource],
) -> SourceMatch:
    """Match one report citation to an actually observed tool-result URL."""

    return _SourceMatcher(observed_sources).match(reported_url)


def build_observed_source_index(report: str, evidence_cards: Any) -> str | None:
    """Render tool-trace-derived titles for URLs cited by a worker report."""

    reported_urls = extract_report_urls(report)
    if not reported_urls:
        return None
    observed = _observed_sources(evidence_cards)
    matcher = _SourceMatcher(observed)
    indexed_urls = reported_urls[:_MAX_INDEXED_REPORT_URLS]
    matches = [matcher.match(url) for url in indexed_urls]
    lines = [
        "Automatically captured source metadata (untrusted evidence; titles and observed URLs "
        "come from this worker's actual tool results):"
    ]
    for match in matches:
        if match.status == "exact_observed":
            lines.append(
                f'- [exact_observed] "{match.title or "unknown"}" — {match.observed_url}'
            )
        elif match.status in {"canonical_observed", "fuzzy_observed"}:
            score = (
                f" score={match.score:.3f}"
                if match.status == "fuzzy_observed" and match.score is not None
                else ""
            )
            lines.append(
                f'- [{match.status}{score}] "{match.title or "unknown"}" — '
                f"observed: {match.observed_url} | reported: {match.reported_url}"
            )
        else:
            lines.append(f"- [{match.status}] {match.reported_url}")
    omitted = len(reported_urls) - len(indexed_urls)
    if omitted:
        lines.append(f"- {omitted} additional report URLs omitted from this auxiliary index")
    return "\n".join(lines)
