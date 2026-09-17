"""Canonical dates and windows. UTC calendar days, YYYY-MM-DD.

Inclusive start, exclusive end: 2020-09-01 <= published_at < 2026-09-01.
Windows from pipeline.md:
  before = 01.09.2023–31.08.2024  →  [2023-09-01, 2024-09-01)
  now    = 01.09.2025–31.08.2026  →  [2025-09-01, 2026-09-01)
"""

from __future__ import annotations

from datetime import date, timedelta
from typing import Final

COLLECTION_START: Final[date] = date(2020, 9, 1)
CUTOFF_DATE: Final[date] = date(2026, 9, 1)

WINDOW_BEFORE_START: Final[date] = date(2023, 9, 1)
WINDOW_BEFORE_END: Final[date] = date(2024, 9, 1)
WINDOW_NOW_START: Final[date] = date(2025, 9, 1)
WINDOW_NOW_END: Final[date] = date(2026, 9, 1)

WINDOWS: Final[dict[str, tuple[date, date]]] = {
    "before": (WINDOW_BEFORE_START, WINDOW_BEFORE_END),
    "now": (WINDOW_NOW_START, WINDOW_NOW_END),
}

ALLOWED_SOURCE_TYPES: Final[frozenset[str]] = frozenset(
    {
        "paper",
        "preprint",
        "patent",
        "news",
        "press_release",
        "product",
        "report",
        "standard",
        "blog",
    }
)

ALLOWED_TRUST_LEVELS: Final[frozenset[str]] = frozenset({"high", "medium", "low"})
ALLOWED_WINDOWS: Final[frozenset[str]] = frozenset({"before", "now"})

# Weak-only types cannot be the sole basis for including a technology (pipeline.md).
SOLE_SOURCE_WEAK_TYPES: Final[frozenset[str]] = frozenset({"blog", "press_release"})

DEFAULT_RECENT_DAYS: Final[int] = 180
DEFAULT_MAX_CANDIDATES: Final[int] = 30
DEFAULT_MAX_WORKERS: Final[int] = 8
DEFAULT_MAX_DOCS_PER_SOURCE: Final[int] = 200


def inclusive_end(exclusive_end: date) -> date:
    """Last UTC calendar day included in [start, exclusive_end)."""
    return exclusive_end - timedelta(days=1)
