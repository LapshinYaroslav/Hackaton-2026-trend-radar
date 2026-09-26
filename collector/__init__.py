"""Public API of the document collector.

Поиск №1 (search_recent) и счётчики поиска №2 (count_history) — методы DocumentCollector.
"""

from collector.api import DocumentCollector, default_adapters
from collector.constants import COLLECTION_START, CUTOFF_DATE, WINDOWS
from collector.db import DocumentCache, MemoryCache, PostgresCache, build_cache
from collector.exceptions import CollectorError, InvalidSourceTypeError
from collector.models import Candidate, Document, RecentSearchResult, SourceTotal
from collector.settings import Settings

__all__ = [
    "COLLECTION_START",
    "CUTOFF_DATE",
    "WINDOWS",
    "Candidate",
    "CollectorError",
    "Document",
    "DocumentCache",
    "DocumentCollector",
    "InvalidSourceTypeError",
    "MemoryCache",
    "PostgresCache",
    "RecentSearchResult",
    "Settings",
    "SourceTotal",
    "build_cache",
    "default_adapters",
]
