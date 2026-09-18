"""Public API of the document collector.

Both modes return the same CollectionResult so Yaroslav's compute_features
can run unchanged on training and on a live query.
"""

from collector.api import (
    DocumentCollector,
    build_collector,
    collect_histories,
    collect_history,
    collect_training,
    default_adapters,
    search_recent,
)
from collector.constants import COLLECTION_START, CUTOFF_DATE, WINDOWS
from collector.db import DocumentCache, MemoryCache, PostgresCache, build_cache
from collector.exceptions import CollectorError, InvalidSourceTypeError
from collector.models import Candidate, CollectionResult, Document, RecentSearchResult, SourceTotal, Technology
from collector.settings import Settings

__all__ = [
    "COLLECTION_START",
    "CUTOFF_DATE",
    "WINDOWS",
    "Candidate",
    "CollectionResult",
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
    "Technology",
    "build_cache",
    "build_collector",
    "collect_histories",
    "collect_history",
    "collect_training",
    "default_adapters",
    "search_recent",
]
