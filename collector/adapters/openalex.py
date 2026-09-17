"""OpenAlex works adapter. Papers; corpus totals via meta.count."""

from __future__ import annotations

from datetime import date
from typing import Any

import httpx

from collector.adapters.base import default_trust
from collector.constants import inclusive_end
from collector.exceptions import AdapterError
from collector.http import HttpTransport, HttpxTransport, RateLimiter
from collector.models import Document, require_source_type
from collector.settings import Settings

OPENALEX_WORKS = "https://api.openalex.org/works"

_TYPE_MAP = {
    "article": "paper",
    "journal-article": "paper",
    "review": "paper",
    "letter": "paper",
    "preprint": "preprint",
    "posted-content": "preprint",
    "report": "report",
    "book": "report",
    "book-chapter": "report",
    "standard": "standard",
    "dataset": "report",
    "dissertation": "paper",
}


def reconstruct_abstract(inverted: dict[str, list[int]] | None) -> str:
    if not inverted:
        return ""
    positions: list[tuple[int, str]] = []
    for word, idxs in inverted.items():
        for idx in idxs:
            positions.append((idx, word))
    if not positions:
        return ""
    positions.sort()
    return " ".join(word for _, word in positions)


class OpenAlexAdapter:
    source = "openalex"
    source_type = "paper"

    def __init__(
        self,
        transport: HttpTransport | None = None,
        settings: Settings | None = None,
        mailto: str | None = None,
    ) -> None:
        self._settings = settings or Settings()
        self._transport = transport or HttpxTransport(self._settings)
        self._mailto = mailto or self._settings.openalex_mailto
        self._limiter = RateLimiter(0.0 if transport is not None else 0.12)

    def search(
        self,
        query: str,
        date_from: date,
        date_to_exclusive: date,
        *,
        limit: int,
    ) -> list[Document]:
        collected: list[Document] = []
        cursor = "*"
        per_page = min(50, max(1, limit))
        while len(collected) < limit and cursor:
            payload = self._get_works(
                search=query,
                date_from=date_from,
                date_to_exclusive=date_to_exclusive,
                per_page=min(per_page, limit - len(collected)),
                cursor=cursor,
            )
            results = payload.get("results") or []
            for work in results:
                doc = self._to_document(work)
                if doc is not None:
                    collected.append(doc)
                if len(collected) >= limit:
                    break
            cursor = (payload.get("meta") or {}).get("next_cursor") or ""
            if not results:
                break
        return collected

    def count_total(self, date_from: date, date_to_exclusive: date) -> int | None:
        payload = self._get_works(
            search=None,
            date_from=date_from,
            date_to_exclusive=date_to_exclusive,
            per_page=1,
            cursor=None,
        )
        count = (payload.get("meta") or {}).get("count")
        return int(count) if count is not None else None

    def _get_works(
        self,
        *,
        search: str | None,
        date_from: date,
        date_to_exclusive: date,
        per_page: int,
        cursor: str | None,
    ) -> dict[str, Any]:
        date_to = inclusive_end(date_to_exclusive)
        filters = [
            f"from_publication_date:{date_from.isoformat()}",
            f"to_publication_date:{date_to.isoformat()}",
        ]
        params: dict[str, Any] = {
            "filter": ",".join(filters),
            "per_page": per_page,
            "select": (
                "id,doi,title,display_name,publication_date,type,primary_location,"
                "abstract_inverted_index,language,authorships"
            ),
        }
        if search:
            params["search"] = search
        if cursor:
            params["cursor"] = cursor
        if self._mailto:
            params["mailto"] = self._mailto
        self._limiter.wait()
        try:
            response = self._transport.get(OPENALEX_WORKS, params=params)
            return response.json()
        except (httpx.HTTPError, ValueError) as exc:
            raise AdapterError(self.source, str(exc)) from exc

    def _to_document(self, work: dict[str, Any]) -> Document | None:
        published = work.get("publication_date")
        if not published:
            return None
        title = (work.get("display_name") or work.get("title") or "").strip()
        if not title:
            return None
        try:
            url = _work_url(work)
        except ValueError:
            return None
        raw_type = str(work.get("type") or "article")
        source_type = _TYPE_MAP.get(raw_type, "paper")
        require_source_type(source_type, self.source)
        orgs: list[str] = []
        for authorship in work.get("authorships") or []:
            for inst in authorship.get("institutions") or []:
                name = (inst.get("display_name") or "").strip()
                if name and name not in orgs:
                    orgs.append(name)
        language = (work.get("language") or "en").split("-")[0] or "en"
        return Document(
            published_at=published,
            source=self.source,
            source_type=source_type,
            title=title,
            url=url,
            language=language,
            trust_level=default_trust(source_type),
            organizations=orgs,
            text=reconstruct_abstract(work.get("abstract_inverted_index")),
        )


def _work_url(work: dict[str, Any]) -> str:
    location = work.get("primary_location") or {}
    landing = (location.get("landing_page_url") or "").strip()
    if landing:
        return landing
    doi = (work.get("doi") or "").strip()
    if doi:
        return doi if doi.startswith("http") else f"https://doi.org/{doi.removeprefix('https://doi.org/')}"
    openalex_id = str(work.get("id") or "").strip()
    if openalex_id:
        return openalex_id
    raise ValueError("OpenAlex work has no url")
