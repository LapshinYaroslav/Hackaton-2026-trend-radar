"""Test doubles. Source names in collector tests are s1/s2 (pipeline.md)."""

from __future__ import annotations

from datetime import date

from collector.models import Document, require_source_type


class FakeAdapter:
    def __init__(
        self,
        source: str,
        source_type: str,
        documents: list[Document] | None = None,
        totals: dict[tuple[date, date], int | None] | None = None,
        fail_source_type: str | None = None,
    ) -> None:
        self.source = source
        self.source_type = source_type
        self.documents = documents or []
        self.totals = totals or {}
        self.fail_source_type = fail_source_type
        self.search_calls: list[tuple[str, date, date, int]] = []
        self.count_calls: list[tuple[date, date]] = []

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
