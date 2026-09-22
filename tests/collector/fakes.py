"""Test doubles. Source names in collector tests are s1/s2 (pipeline.md)."""

from __future__ import annotations

from datetime import date

from collector.models import Document, SearchTerms, require_source_type


class FakeAdapter:
    def __init__(
        self,
        source: str,
        source_type: str,
        documents: list[Document] | None = None,
        totals: dict[tuple[date, date], int | None] | None = None,
        fail_source_type: str | None = None,
        counts: dict[tuple[date, date], int | None] | None = None,
        type_filter: str | None = None,
        query_variant: str = "phrase|all",
        totals_endpoint: str = "https://fake/totals",
    ) -> None:
        self.source = source
        self.source_type = source_type
        # Часть протокола SourceAdapter: под каким фильтром типа считается источник.
        self.type_filter = type_filter
        # Семантика запроса: входит в ключ строки счётчика.
        self.query_variant = query_variant
        self.totals_endpoint = totals_endpoint
        self.documents = documents or []
        self.totals = totals or {}
        self.fail_source_type = fail_source_type
        self.counts = counts or {}
        self.search_calls: list[tuple[str, date, date, int]] = []
        self.count_calls: list[tuple[date, date]] = []
        self.match_calls: list[tuple[str, date, date]] = []

    def search(
        self,
        query: str,
        date_from: date,
        date_to_exclusive: date,
        *,
        limit: int,
    ) -> list[Document]:
        self.search_calls.append((query, date_from, date_to_exclusive, limit))
        if self.fail_source_type is not None:
            require_source_type(self.fail_source_type, self.source)
        matched = [
            item
            for item in self.documents
            if date_from <= item.published_at < date_to_exclusive
        ]
        return matched[:limit]

    def count_matching(
        self,
        search: SearchTerms,
        date_from: date,
        date_to_exclusive: date,
    ) -> int | None:
        self.match_calls.append((search.query, date_from, date_to_exclusive))
        return self.counts.get((date_from, date_to_exclusive))

    def totals_request(self, date_from: date, date_to_exclusive: date) -> tuple[str, dict]:
        return self.totals_endpoint, {
            "from": date_from.isoformat(),
            "to": date_to_exclusive.isoformat(),
            "type": self.type_filter,
        }

    def count_total(self, date_from: date, date_to_exclusive: date) -> int | None:
        self.count_calls.append((date_from, date_to_exclusive))
        if (date_from, date_to_exclusive) in self.totals:
            return self.totals[(date_from, date_to_exclusive)]
        return None


def doc(
    *,
    source: str,
    source_type: str,
    published_at: str,
    url: str,
    title: str = "Sample",
    trust_level: str = "high",
    text: str = "body",
    organizations: list[str] | None = None,
    language: str = "en",
) -> Document:
    return Document(
        published_at=published_at,
        source=source,
        source_type=source_type,
        title=title,
        url=url,
        language=language,
        trust_level=trust_level,
        organizations=organizations or [],
        text=text,
    )
