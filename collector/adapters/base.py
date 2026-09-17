"""Source adapter contract.

Every source is probed for corpus totals on both growth windows *before* collection.
If count_total returns None, the source is excluded from growth and kept for other features.
"""

from __future__ import annotations

from datetime import date
from typing import Protocol, runtime_checkable

from collector.models import Document

HIGH_TRUST_TYPES = {"paper", "patent", "standard"}
MEDIUM_TRUST_TYPES = {"preprint", "report", "news", "product"}


@runtime_checkable
class SourceAdapter(Protocol):
    source: str
    source_type: str

    def search(
        self,
        query: str,
        date_from: date,
        date_to_exclusive: date,
        *,
        limit: int,
    ) -> list[Document]:
        """Documents matching `query` with date_from <= published_at < date_to_exclusive."""
        ...

    def count_total(self, date_from: date, date_to_exclusive: date) -> int | None:
        """Corpus size of the whole source in the window, independent of technology.

        Return None if the source cannot answer this. Callers must probe both windows
        before collection starts (pipeline.md).
        """
        ...


def default_trust(source_type: str) -> str:
    if source_type in HIGH_TRUST_TYPES:
        return "high"
    if source_type in MEDIUM_TRUST_TYPES:
        return "medium"
    return "low"
