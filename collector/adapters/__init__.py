from collector.adapters.arxiv import ArxivAdapter
from collector.adapters.base import SourceAdapter, default_trust
from collector.adapters.openalex import OpenAlexAdapter
from collector.adapters.techcrunch import TechCrunchAdapter

__all__ = [
    "ArxivAdapter",
    "OpenAlexAdapter",
    "SourceAdapter",
    "TechCrunchAdapter",
    "default_trust",
]
