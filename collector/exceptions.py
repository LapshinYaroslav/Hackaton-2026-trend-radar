"""Collector errors. Invalid source_type must fail fast — features must not swallow it."""

from __future__ import annotations


class CollectorError(Exception):
    """Base error of the document collector."""


class InvalidSourceTypeError(CollectorError):
    """Adapter returned a source_type outside the allowed enum."""

    def __init__(self, source_type: str, source: str | None = None) -> None:
        self.source_type = source_type
        self.source = source
        where = f" (source={source})" if source else ""
        super().__init__(
            f"Invalid source_type={source_type!r}{where}. "
            "Allowed: paper, preprint, patent, news, press_release, "
            "product, report, standard, blog."
        )


class AdapterError(CollectorError):
    """HTTP / parse failure inside a source adapter."""

    def __init__(self, source: str, message: str) -> None:
        self.source = source
        super().__init__(f"{source}: {message}")
