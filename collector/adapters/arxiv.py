"""arXiv adapter. Preprints via Atom API.

Corpus totals: arXiv search API cannot count the whole corpus. Totals are taken from
OpenAlex filtered by the arXiv source id (S4306402567). If that call fails, count_total
returns None and arXiv is excluded from growth (pipeline.md).
"""

from __future__ import annotations

import xml.etree.ElementTree as ET
from datetime import date
from typing import Any

import httpx

from collector.adapters.base import default_trust
from collector.constants import inclusive_end
from collector.exceptions import AdapterError
from collector.http import HttpTransport, HttpxTransport, RateLimiter
from collector.models import Document, parse_utc_date, require_source_type
from collector.settings import Settings

ARXIV_API = "http://export.arxiv.org/api/query"
OPENALEX_WORKS = "https://api.openalex.org/works"
ARXIV_OPENALEX_SOURCE = "S4306402567"
ATOM = "{http://www.w3.org/2005/Atom}"
ARXIV_NS = "{http://arxiv.org/schemas/atom}"


class ArxivAdapter:
    source = "arxiv"
    source_type = "preprint"

    def __init__(
        self,
        transport: HttpTransport | None = None,
        settings: Settings | None = None,
        mailto: str | None = None,
    ) -> None:
        self._settings = settings or Settings()
        self._transport = transport or HttpxTransport(self._settings)
        self._mailto = mailto or self._settings.openalex_mailto
        self._limiter = RateLimiter(0.0 if transport is not None else 3.1)

    def search(
        self,
        query: str,
        date_from: date,
        date_to_exclusive: date,
        *,
        limit: int,
    ) -> list[Document]:
        collected: list[Document] = []
        start = 0
        page = min(50, max(1, limit))
        while len(collected) < limit:
            raw = self._query_atom(query, date_from, date_to_exclusive, start=start, max_results=page)
            entries = _parse_entries(raw)
            if not entries:
                break
            for entry in entries:
                doc = self._to_document(entry)
                if doc is None:
                    continue
                if not (date_from <= doc.published_at < date_to_exclusive):
                    continue
                collected.append(doc)
                if len(collected) >= limit:
                    break
            start += len(entries)
            if len(entries) < page:
                break
        return collected

    def count_total(self, date_from: date, date_to_exclusive: date) -> int | None:
        """Whole-arXiv count via OpenAlex. None if the helper source is unavailable."""
        date_to = inclusive_end(date_to_exclusive)
        params: dict[str, Any] = {
            "filter": (
                f"primary_location.source.id:{ARXIV_OPENALEX_SOURCE},"
                f"from_publication_date:{date_from.isoformat()},"
                f"to_publication_date:{date_to.isoformat()}"
            ),
            "per_page": 1,
        }
        if self._mailto:
            params["mailto"] = self._mailto
        try:
            response = self._transport.get(OPENALEX_WORKS, params=params)
            count = (response.json().get("meta") or {}).get("count")
            return int(count) if count is not None else None
        except (httpx.HTTPError, ValueError, TypeError):
            return None

    def _query_atom(
        self,
        query: str,
        date_from: date,
        date_to_exclusive: date,
        *,
        start: int,
        max_results: int,
    ) -> str:
        date_to = inclusive_end(date_to_exclusive)
        submitted = (
            f"submittedDate:[{date_from.strftime('%Y%m%d')}0000 TO {date_to.strftime('%Y%m%d')}2359]"
        )
        search = f"all:{_quote_term(query)} AND {submitted}"
        params = {
            "search_query": search,
            "start": start,
            "max_results": max_results,
            "sortBy": "submittedDate",
            "sortOrder": "descending",
        }
        self._limiter.wait()
        try:
            response = self._transport.get(ARXIV_API, params=params)
            return response.text
        except httpx.HTTPError as exc:
            raise AdapterError(self.source, str(exc)) from exc

    def _to_document(self, entry: dict[str, str]) -> Document | None:
        published = entry.get("published")
        if not published:
            return None
        title = entry.get("title", "").strip()
        url = entry.get("id", "").strip()
        if not title or not url:
            return None
        require_source_type(self.source_type, self.source)
        orgs = [org for org in entry.get("affiliations", "").split("|") if org]
        return Document(
            published_at=parse_utc_date(published),
            source=self.source,
            source_type=self.source_type,
            title=" ".join(title.split()),
            url=url,
            language="en",
            trust_level=default_trust(self.source_type),
            organizations=orgs,
            text=entry.get("summary", "").strip(),
        )


def _quote_term(query: str) -> str:
    cleaned = " ".join(query.split())
    if " " in cleaned:
        return f'"{cleaned}"'
    return cleaned


def _parse_entries(xml_text: str) -> list[dict[str, str]]:
    try:
        root = ET.fromstring(xml_text)
    except ET.ParseError as exc:
        raise AdapterError("arxiv", f"invalid atom xml: {exc}") from exc
    entries: list[dict[str, str]] = []
    for node in root.findall(f"{ATOM}entry"):
        affiliations = [
            (aff.text or "").strip()
            for aff in node.findall(f"{ARXIV_NS}affiliation")
            if (aff.text or "").strip()
        ]
        published = (node.findtext(f"{ATOM}published") or node.findtext(f"{ATOM}updated") or "").strip()
        entries.append(
            {
                "id": (node.findtext(f"{ATOM}id") or "").strip(),
                "title": (node.findtext(f"{ATOM}title") or "").strip(),
                "summary": (node.findtext(f"{ATOM}summary") or "").strip(),
                "published": published,
                "affiliations": "|".join(affiliations),
            }
        )
    return entries
