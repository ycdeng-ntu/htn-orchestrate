"""Miro's OA-first academic routes, using the worker's injectable transport."""
from __future__ import annotations

import re
from typing import Any

import httpx

from ._pure import extract_doi


class AcademicFetcher:
    def __init__(self, client: httpx.AsyncClient, *, ncbi_api_key: str = "", unpaywall_email: str = ""):
        self.client = client
        self.ncbi_api_key = ncbi_api_key
        # Do not send MiroMind's private contact identity from the source code.
        self.unpaywall_email = unpaywall_email
        self.failures: list[dict[str, Any]] = []

    async def _get(self, url: str, *, timeout: int = 15, min_chars: int = 0, **kwargs: Any) -> Any:
        try:
            response = await self.client.get(url, timeout=timeout, **kwargs)
            response.raise_for_status()
            if len(response.text) < min_chars:
                return None
            return response.json()
        except (httpx.HTTPError, ValueError) as exc:
            self.failures.append({"url": url, "error": type(exc).__name__})
            return None  # Recoverable route failure; the caller still attempts Jina.

    def _ncbi_params(self, **params: str) -> dict[str, str]:
        if self.ncbi_api_key:
            params["api_key"] = self.ncbi_api_key
        return params

    async def pmc(self, pmcid: str) -> str:
        if not pmcid:
            return ""
        data = await self._get("https://www.ncbi.nlm.nih.gov/research/bionlp/RESTful/pmcoa.cgi/BioC_json/" + pmcid + "/unicode",
                               timeout=30, min_chars=500, params=self._ncbi_params())
        items = data if isinstance(data, list) else [data]
        return "\n\n".join(str(p["text"]) for item in items if isinstance(item, dict)
                           for doc in item.get("documents", []) if isinstance(doc, dict)
                           for p in doc.get("passages", []) if isinstance(p, dict) and p.get("text"))

    async def pubmed_to_pmc(self, url: str) -> str:
        match = re.search(r"pubmed\.ncbi\.nlm\.nih\.gov/(\d+)", url)
        if not match:
            return ""
        data = await self._get("https://eutils.ncbi.nlm.nih.gov/entrez/eutils/elink.fcgi",
                               params=self._ncbi_params(dbfrom="pubmed", db="pmc", id=match[1], retmode="json"))
        if not isinstance(data, dict):
            return ""
        for linkset in data.get("linksets", []):
            for db in linkset.get("linksetdbs", []):
                if db.get("dbto") == "pmc" and db.get("links"):
                    return f"PMC{db['links'][0]}"
        return ""

    async def resolve_doi(self, url: str) -> str:
        doi = extract_doi(url)
        if doi:
            return doi
        data = await self._get("https://api.crossref.org/works", timeout=10,
                               params={"query": url, "rows": 1}, headers={"User-Agent": "AgentHarness/1.0"})
        if not isinstance(data, dict):
            return ""
        items = (data.get("message") or {}).get("items") or []
        return str(items[0].get("DOI") or "") if items else ""

    async def oa_url(self, url: str) -> str:
        if not self.unpaywall_email:
            return ""
        doi = await self.resolve_doi(url)
        if not doi:
            return ""
        data = await self._get("https://api.unpaywall.org/v2/" + doi, params={"email": self.unpaywall_email})
        if not isinstance(data, dict):
            return ""
        for location in [data.get("best_oa_location") or {}, *(data.get("oa_locations") or [])]:
            target = location.get("url_for_pdf") or location.get("url")
            if target:
                return str(target)
        return ""
