"""Pure formatting and normalization ported from MiroHarness native tools.

No runtime dependency on the source checkout.
"""

from __future__ import annotations

import json

import logging

import re

from urllib.parse import urlparse

logger = logging.getLogger(__name__)

_DOMAIN_BLACKLIST: set[str] = {
    "youtube.com", "youtu.be",
    "tiktok.com",
    "facebook.com", "fb.com",
    "instagram.com",
    "pinterest.com",
    "twitter.com", "x.com",
    "linkedin.com",
    "quora.com",
    "amazon.com",
    "ebay.com",
    "walmart.com",
    "etsy.com",
    "aliexpress.com",
}

def _is_blacklisted(url: str) -> bool:
    """Check if a URL belongs to a blacklisted domain."""
    url_lower = url.lower()
    for domain in _DOMAIN_BLACKLIST:
        if domain in url_lower:
            return True
    return False

def format_search(data: dict) -> str:
    """Format Serper response into readable text with all result types."""
    parts = []

    # Answer Box (direct answer — highest priority)
    answer_box = data.get("answerBox")
    if answer_box:
        answer = answer_box.get("answer") or answer_box.get("snippet", "")
        title = answer_box.get("title", "")
        if answer:
            parts.append(f"## Direct Answer\n**{title}**\n{answer}")

    # Knowledge Graph (entity info)
    kg = data.get("knowledgeGraph")
    if kg:
        kg_title = kg.get("title", "")
        kg_type = kg.get("type", "")
        kg_desc = kg.get("description", "")
        kg_parts = [f"## Knowledge Graph: {kg_title}"]
        if kg_type:
            kg_parts.append(f"Type: {kg_type}")
        if kg_desc:
            kg_parts.append(kg_desc)
        for key in ("founded", "headquarters", "ceo", "revenue", "employees",
                     "website", "born", "died", "nationality"):
            if key in kg:
                kg_parts.append(f"{key.title()}: {kg[key]}")
        parts.append("\n".join(kg_parts))

    # Organic Results (filtered: skip low-value domains)
    organic = data.get("organic", [])
    if organic:
        results = []
        filtered_count = 0
        idx = 0
        for r in organic:
            link = r.get("link", "")
            if _is_blacklisted(link):
                filtered_count += 1
                continue
            idx += 1
            title = r.get("title", "")
            snippet = r.get("snippet", "")
            results.append(f"[{idx}] **{title}**\n{snippet}\nURL: {link}")
        if results:
            header = "## Search Results"
            if filtered_count:
                header += f"\n({filtered_count} low-value results filtered)"
            parts.append(header + "\n" + "\n\n".join(results))
        elif filtered_count:
            parts.append(
                f"## Search Results\nAll {filtered_count} results were from "
                f"low-value domains (video/social). Try a more specific query."
            )

    return "\n\n---\n\n".join(parts) if parts else "No results found."

def format_scholar(data: dict) -> str:
    """Format Scholar response into readable Markdown with citation counts."""
    organic = data.get("organic") or []
    if not organic:
        return "No scholarly results found."

    # Sort by citation count descending so high-impact papers surface first.
    def _cited(item: dict) -> int:
        c = item.get("citedBy")
        if isinstance(c, int):
            return c
        if isinstance(c, str):
            try:
                return int(c.replace(",", "").strip())
            except ValueError:
                return 0
        return 0

    sorted_hits = sorted(organic, key=_cited, reverse=True)

    lines = ["## Scholar Results"]
    for idx, item in enumerate(sorted_hits, start=1):
        title = (item.get("title") or "").strip() or "(untitled)"
        link = (item.get("link") or "").strip()
        snippet = (item.get("snippet") or "").strip()
        pub = (item.get("publicationInfo") or "").strip()
        year = item.get("year")
        cited_by = _cited(item)
        pdf = (item.get("pdfUrl") or "").strip()

        header_bits = [f"[{idx}] **{title}**"]
        if year:
            header_bits.append(f"({year})")
        if cited_by:
            header_bits.append(f"— cited by {cited_by:,}")
        lines.append(" ".join(header_bits))

        if pub:
            lines.append(pub)
        if snippet:
            lines.append(snippet)
        if link:
            lines.append(f"URL: {link}")
        if pdf and pdf != link:
            lines.append(f"PDF: {pdf}")
        lines.append("")  # blank line between hits

    return "\n".join(lines).rstrip()

def coerce_json_list(value: object) -> object:
    """Return ``value`` unchanged, or parse it if it's a JSON-list string.

    ``value`` is returned as-is when it's already a list, isn't a string,
    doesn't look like JSON, or fails to parse. A bare JSON object is
    wrapped in a single-element list so the caller sees a uniform shape.
    """
    if not isinstance(value, str):
        return value
    s = value.strip()
    if not (s.startswith("[") or s.startswith("{")):
        return value
    try:
        parsed = json.loads(s)
    except (json.JSONDecodeError, ValueError):
        return value
    if isinstance(parsed, dict):
        return [parsed]
    return parsed

PAYWALL_DOMAINS: frozenset[str] = frozenset({
    "www.sciencedirect.com", "linkinghub.elsevier.com",
    "onlinelibrary.wiley.com", "chemistry-europe.onlinelibrary.wiley.com",
    "advanced.onlinelibrary.wiley.com", "analyticalsciencejournals.onlinelibrary.wiley.com",
    "nph.onlinelibrary.wiley.com", "faseb.onlinelibrary.wiley.com",
    "link.aps.org", "journals.aps.org",
    "www.cell.com",
    "academic.oup.com",
    "www.tandfonline.com",
    "pubs.rsc.org", "www.rsc.org",
    "www.science.org",
    "www.jneurosci.org",
    "ashpublications.org",
    "pubs.aip.org",
    "journals.sagepub.com",
    "pubs.acs.org",
})

_GARBAGE_SIGNALS: tuple[str, ...] = (
    "access denied", "security verification required", "browser check",
    "checking your browser", "checking if the site connection is secure",
    "verify you are human", "verify that you are human", "verify you are a human",
    "please verify you are human", "please verify that you are human",
    "enable javascript and cookies to continue", "enable javascript to continue",
    "please enable javascript to continue", "please enable cookies to continue",
    "cookies are required to continue", "sign in to access", "log in to access",
    "subscribe to read", "purchase this article",
)

def extract_pmcid(url: str) -> str:
    """Extract PMC id from URL. e.g. ``pmc.ncbi.nlm.nih.gov/articles/PMC1234567/``."""
    match = re.search(r"(PMC\d+)", url)
    return match.group(1) if match else ""

def extract_doi(url: str) -> str:
    """Try to extract DOI from URL. Returns empty string if not found."""
    match = re.search(r"doi\.org/(10\.\d{4,}/[^\s]+)", url)
    if match:
        return match.group(1).rstrip("/")
    match = re.search(r"/(10\.\d{4,}/[^\s?#]+)", url)
    if match:
        return match.group(1).rstrip("/")
    return ""

def route_url(url: str) -> str:
    """Classify URL for backend selection. Pure — no I/O."""
    domain = urlparse(url).netloc
    if "pmc.ncbi.nlm.nih.gov" in domain:
        return "pmc"
    if "pubmed.ncbi.nlm.nih.gov" in domain:
        return "pubmed"
    if "biorxiv.org" in domain or "medrxiv.org" in domain:
        return "biorxiv"
    if domain in PAYWALL_DOMAINS:
        return "paywall"
    return "jina"

def is_garbage_content(text: str) -> bool:
    """Conservatively recognize empty content and short, explicit block pages.

    Product names and subjects such as robots or CAPTCHAs are also ordinary
    research content. Require a page-opening instruction/status instead of
    matching those words anywhere in an otherwise useful article.
    """
    content = text.strip()
    if not content:
        return True
    if len(content) > 2000:
        return False
    for line in content[:400].lower().splitlines():
        line = re.sub(r"^\s*(?:#{1,6}\s+|title:\s*)", "", line).strip(" *_")
        if len(line) > 180:
            continue
        for signal in _GARBAGE_SIGNALS:
            if not line.startswith(signal):
                continue
            remainder = line[len(signal):]
            # Whole status lines, explicit sentences, or a short instruction
            # about continuing/accessing the page. A body sentence such as
            # "browser check technologies have improved" is not a challenge.
            if not remainder or re.match(
                r"^[.!?:](?:\s|$)|^\s+(?:to |before |in order to |this |the |full )",
                remainder,
            ):
                return True
    return False

def biorxiv_to_pdf(url: str) -> str:
    """Convert bioRxiv/medRxiv URL to full PDF URL. Returns empty if not applicable."""
    if not ("biorxiv.org" in url or "medrxiv.org" in url):
        return ""
    clean = url.split("?")[0].split("#")[0].rstrip("/")
    if clean.endswith(".pdf"):
        return url
    if "/content/" in clean:
        return clean + ".full.pdf"
    return ""

def _normalise_inputs(
    url: str | list[str], info_to_extract: str | list[str],
) -> tuple[list[str], list[str]]:
    """Coerce the LangChain payload into paired URL + focus lists."""
    urls = coerce_json_list(url) if isinstance(url, str) else url
    if isinstance(urls, str):
        urls = [urls]
    elif not isinstance(urls, list):
        urls = []
    urls = [u.strip() for u in urls if isinstance(u, str) and u.strip()]

    focuses = (
        coerce_json_list(info_to_extract) if isinstance(info_to_extract, str)
        else info_to_extract
    )
    if isinstance(focuses, str):
        focuses = [focuses] * len(urls)
    elif isinstance(focuses, list):
        focuses = [str(f) for f in focuses]
        if len(focuses) < len(urls):
            focuses = focuses + [""] * (len(urls) - len(focuses))
    else:
        focuses = [""] * len(urls)
    return urls, focuses

def _extract_jina_body(raw: str) -> str:
    """Return the usable content from a Jina response body.

    Jina's POST endpoint returns JSON wrapped as ``{"code":200,"status":...,
    "data":{"title":...,"content":"markdown..."}}`` when ``Accept: application/json``
    is set. Some proxies may return raw markdown instead. Handle both shapes
    and also detect balance errors.
    """
    if not raw:
        return ""
    stripped = raw.lstrip()
    if not stripped.startswith("{"):
        return raw
    try:
        import json as _json
        data = _json.loads(raw)
    except (ValueError, TypeError):
        return raw

    if isinstance(data, dict):
        # Balance-error sentinel (from MiroFlow).
        if data.get("name") == "InsufficientBalanceError":
            logger.error("Jina API: Insufficient balance")
            return ""
        # Wrapped success payload.
        inner = data.get("data")
        if isinstance(inner, dict) and isinstance(inner.get("content"), str):
            return inner["content"]
    return raw

_UNICODE_REPLACEMENTS: dict[str, str] = {
    "\u222B": "integral",   # Integral sign
    "\u2211": "sum",        # Summation
    "\u221E": "inf",        # Infinity
    "\u00B2": "**2",        # Superscript 2
    "\u00B3": "**3",        # Superscript 3
    "\u2074": "**4",        # Superscript 4
    "\u207B": "-",          # Superscript minus
    "\u00B9": "**1",        # Superscript 1
    "\u27E8": "<",          # Left angle bracket
    "\u27E9": ">",          # Right angle bracket
    "\u2329": "<",          # Left-pointing angle bracket
    "\u232A": ">",          # Right-pointing angle bracket
    "\u2014": "-",          # Em dash
    "\u2013": "-",          # En dash
    "\u2018": "'",          # Left single quote
    "\u2019": "'",          # Right single quote
    "\u201C": '"',          # Left double quote
    "\u201D": '"',          # Right double quote
    "\u2264": "<=",         # Less than or equal
    "\u2265": ">=",         # Greater than or equal
    "\u2260": "!=",         # Not equal
    "\u00D7": "*",          # Multiplication sign
    "\u00F7": "/",          # Division sign
    "\u2248": "==",         # Almost equal
    "\u2245": "==",         # Approximately equal
    "\u2261": "==",         # Identical to
    "\u2192": "->",         # Right arrow
    "\u2190": "<-",         # Left arrow
    "\u221A": "sqrt",       # Square root
    "\u03C0": "pi",         # Pi
}

def sanitize_code(code: str) -> str:
    """Replace Unicode math symbols with ASCII equivalents and strip non-ASCII
    bytes from comments.

    Also removes lines that start with a shell/jupyter escape (``!pip install…``)
    — those get fed to the Python interpreter, which rejects them with a
    ``SyntaxError``.

    Code-line behaviour (non-comment):
      * any char present in the replacement table is mapped to its ASCII form
      * non-ASCII chars that have no replacement are left alone — they may
        be legitimate (e.g. a unicode string literal)

    Comment-line behaviour:
      * symbol replacement runs first
      * any remaining non-ASCII bytes are replaced by ``?`` (lossy) — comments
        never affect execution, so we drop noise eagerly
    """
    if not code:
        return code

    out_lines: list[str] = []
    for line in code.split("\n"):
        # Strip jupyter/shell escapes.
        if line.lstrip().startswith("!"):
            continue

        if any(ord(ch) > 127 for ch in line):
            stripped = line.lstrip()
            if stripped.startswith("#"):
                for sym, repl in _UNICODE_REPLACEMENTS.items():
                    line = line.replace(sym, repl)
                # Lossy: comments may be in any locale, we don't need them
                # round-tripped.
                line = line.encode("ascii", "replace").decode("ascii")
            else:
                for sym, repl in _UNICODE_REPLACEMENTS.items():
                    line = line.replace(sym, repl)
        out_lines.append(line)

    return "\n".join(out_lines)
