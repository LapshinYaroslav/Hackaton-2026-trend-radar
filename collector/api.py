"""Public collector API: training mode and query-time Search #1 / Search #2.

Search #1 (search_recent) never returns source_totals and must not be fed to
compute_features — otherwise every candidate looks "fresh" (pipeline.md).

Search #2 (collect_history) is the only path that yields CollectionResult for features.
"""

from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date, datetime, timedelta, timezone
from typing import Callable, Iterable, Sequence

from collector.adapters import ArxivAdapter, OpenAlexAdapter, TechCrunchAdapter
from collector.adapters.base import SourceAdapter
from collector.constants import (
    COLLECTION_START,
    CUTOFF_DATE,
    DEFAULT_MAX_CANDIDATES,
    DEFAULT_RECENT_DAYS,
    SOLE_SOURCE_WEAK_TYPES,
    WINDOWS,
)
from collector.db import DocumentCache, MemoryCache, build_cache
from collector.exceptions import AdapterError
from collector.http import HttpTransport, HttpxTransport
from collector.models import (
    Candidate,
    CollectionResult,
    Document,
    RecentSearchResult,
    SourceTotal,
    Technology,
    copy_document,
    require_source_type,
)
from collector.settings import Settings

logger = logging.getLogger(__name__)

MainstreamPredicate = Callable[[Candidate], bool]


def search_terms(name_en: str, aliases: list[str] | None = None) -> list[str]:
    """English name + aliases, no company names (pipeline.md: Danya supplies them that way)."""
    seen: set[str] = set()
    terms: list[str] = []
    for raw in [name_en, *(aliases or [])]:
        term = " ".join(raw.split()).strip()
        key = term.casefold()
        if not term or key in seen:
            continue
        seen.add(key)
        terms.append(term)
    return terms


def tech_key(name_en: str) -> str:
    """Stable identity across query_ids. Aliases expand search terms, not the cache key."""
    return " ".join(name_en.split()).casefold()


def filter_for_features(documents: list[Document]) -> list[Document]:
    """Search #2 accounting: dated documents inside the collection window only."""
    kept: list[Document] = []
    seen_urls: set[str] = set()
    for doc in documents:
        if not doc.in_collection_window():
            continue
        url = doc.url.strip()
        if url in seen_urls:
            continue
        seen_urls.add(url)
        kept.append(doc)
    return kept


def has_independent_confirmation(documents: list[Document]) -> bool:
    """Blogs/press releases cannot be the sole basis for including a technology."""
    if not documents:
        return True
    return any(doc.source_type not in SOLE_SOURCE_WEAK_TYPES for doc in documents)


class DocumentCollector:
    def __init__(
        self,
        adapters: Sequence[SourceAdapter],
        cache: DocumentCache | None = None,
        settings: Settings | None = None,
        is_mainstream: MainstreamPredicate | None = None,
    ) -> None:
        if not adapters:
            raise ValueError("DocumentCollector requires at least one source adapter")
        self.adapters = list(adapters)
        self.cache = cache or MemoryCache()
        self.settings = settings or Settings()
        self.is_mainstream = is_mainstream

    def probe_source_totals(self, *, force: bool = False) -> list[SourceTotal]:
        """Ask every source for corpus size on both windows before any tech search."""
        cached = None if force else self.cache.get_source_totals()
        if cached is not None:
            return cached
        totals: list[SourceTotal] = []
        for adapter in self.adapters:
            totals.extend(self._probe_adapter(adapter))
        self.cache.put_source_totals(totals)
        return totals

    def _probe_adapter(self, adapter: SourceAdapter) -> list[SourceTotal]:
        rows: list[SourceTotal] = []
        for window, (start, end) in WINDOWS.items():
            try:
                count = adapter.count_total(start, end)
            except AdapterError as exc:
                logger.warning("source_totals failed for %s/%s: %s", adapter.source, window, exc)
                count = None
            available = count is not None
            rows.append(
                SourceTotal(
                    source=adapter.source,
                    window=window,
                    n_total=int(count) if available else 0,
                    available=available,
                )
            )
        if not all(row.available for row in rows):
            logger.warning(
                "source %s has no complete totals for both windows; excluded from growth",
                adapter.source,
            )
            rows = [
                SourceTotal(source=row.source, window=row.window, n_total=row.n_total, available=False)
                for row in rows
            ]
        return rows

    def search_recent(
        self,
        subqueries: Sequence[str],
        *,
        date_from: date | None = None,
        date_to_exclusive: date | None = None,
    ) -> RecentSearchResult:
        """Fresh documents by topic subqueries. Result = candidate pool, not features."""
        start, end = _recent_bounds(date_from, date_to_exclusive)
        queries = [q.strip() for q in subqueries if q and q.strip()]
        documents = self._search_queries(queries, start, end, apply_feature_window=False)
        return RecentSearchResult(
            documents=documents,
            subqueries=queries,
            date_from=start,
            date_to=end,
        )

    def collect_history(self, candidate: Candidate, *, use_cache: bool = True) -> CollectionResult:
        """History of one candidate for 2020-09-01 .. 2026-09-01. Input to compute_features."""
        totals = self.probe_source_totals()
        key = tech_key(candidate.name_en)
        if use_cache:
            cached_docs = self.cache.get_documents(key)
            if cached_docs is not None:
                return self._make_result(candidate, cached_docs, totals, cache_hit=True)
            if self.cache.has_features(key):
                logger.info(
                    "features cache hit for %s without documents; re-collecting history",
                    candidate.candidate_id,
                )
        terms = search_terms(candidate.name_en, candidate.aliases)
        documents = self._search_queries(
            terms,
            COLLECTION_START,
            CUTOFF_DATE,
            apply_feature_window=True,
        )
        self.cache.put_documents(
            key,
            documents,
            candidate_id=candidate.candidate_id,
            query_id=candidate.query_id,
        )
        return self._make_result(candidate, documents, totals, cache_hit=False)

    def collect_histories(
        self,
        candidates: Sequence[Candidate],
        *,
        max_candidates: int | None = None,
        use_cache: bool = True,
        is_mainstream: MainstreamPredicate | None = None,
    ) -> list[CollectionResult]:
        """Search #2 for many candidates: mainstream cut, cap, then parallel fetch."""
        self.probe_source_totals()
        cap = max_candidates if max_candidates is not None else self.settings.max_candidates
        cap = cap or DEFAULT_MAX_CANDIDATES
        predicate = is_mainstream or self.is_mainstream
        selected, skipped = _select_candidates(candidates, cap, predicate)

        results_by_id: dict[str, CollectionResult] = {}
        workers = max(1, min(self.settings.max_workers, len(selected) or 1))

        def _one(item: Candidate) -> CollectionResult:
            return self.collect_history(item, use_cache=use_cache)

        if selected:
            with ThreadPoolExecutor(max_workers=workers) as pool:
                futures = {pool.submit(_one, item): item for item in selected}
                for future in as_completed(futures):
                    item = futures[future]
                    try:
                        results_by_id[item.candidate_id] = future.result()
                    except Exception:
                        logger.exception("collect_history failed for %s", item.candidate_id)
                        raise

        ordered: list[CollectionResult] = []
        skipped_ids = {item.candidate_id for item in skipped}
        for item in candidates:
            if item.candidate_id in skipped_ids:
                ordered.append(
                    CollectionResult(
                        candidate_id=item.candidate_id,
                        query_id=item.query_id,
                        documents=[],
                        source_totals=self.probe_source_totals(),
                        skipped_as_mainstream=True,
                    )
                )
            elif item.candidate_id in results_by_id:
                ordered.append(results_by_id[item.candidate_id])
        return ordered

    def collect_training(
        self,
        technologies: Sequence[Technology],
        *,
        use_cache: bool = True,
    ) -> list[CollectionResult]:
        """Offline pass over the labelled list. Same CollectionResult as query Search #2."""
        self.probe_source_totals()
        candidates = [tech.as_candidate() for tech in technologies]
        workers = max(1, min(self.settings.max_workers, len(candidates) or 1))
        results_by_id: dict[str, CollectionResult] = {}

        def _one(item: Candidate) -> CollectionResult:
            return self.collect_history(item, use_cache=use_cache)

        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = {pool.submit(_one, item): item for item in candidates}
            for future in as_completed(futures):
                item = futures[future]
                results_by_id[item.candidate_id] = future.result()
        return [results_by_id[item.candidate_id] for item in candidates]

    def _search_queries(
        self,
        queries: Sequence[str],
        date_from: date,
        date_to_exclusive: date,
        *,
        apply_feature_window: bool,
    ) -> list[Document]:
        collected: list[Document] = []
        limit = self.settings.max_docs_per_source
        for query in queries:
            for adapter in self.adapters:
                try:
                    docs = adapter.search(query, date_from, date_to_exclusive, limit=limit)
                except AdapterError as exc:
                    logger.warning("search failed for %s / %r: %s", adapter.source, query, exc)
                    continue
                for doc in docs:
                    collected.append(self._normalize_document(doc, adapter.source))
        if apply_feature_window:
            return filter_for_features(collected)
        return _dedupe(collected)

    def _normalize_document(self, doc: Document, source: str) -> Document:
        require_source_type(doc.source_type, source or doc.source)
        if doc.source != source:
            return copy_document(doc, source=source)
        return doc

    def _make_result(
        self,
        candidate: Candidate,
        documents: list[Document],
        totals: list[SourceTotal],
        *,
        cache_hit: bool,
    ) -> CollectionResult:
        independent = has_independent_confirmation(documents)
        docs = documents
        if documents and not independent:
            docs = [
                copy_document(doc, trust_level="low") if doc.trust_level != "low" else doc
                for doc in documents
            ]
        return CollectionResult(
            candidate_id=candidate.candidate_id,
            query_id=candidate.query_id,
            documents=docs,
            source_totals=totals,
            independent_confirmation=independent,
            cache_hit=cache_hit,
        )


def default_adapters(
    settings: Settings | None = None,
    transport: HttpTransport | None = None,
) -> list[SourceAdapter]:
    settings = settings or Settings()
    transport = transport or HttpxTransport(settings)
    return [
        OpenAlexAdapter(transport=transport, settings=settings),
        ArxivAdapter(transport=transport, settings=settings),
        TechCrunchAdapter(transport=transport, settings=settings),
    ]


def build_collector(
    *,
    settings: Settings | None = None,
    cache: DocumentCache | None = None,
    adapters: list[SourceAdapter] | None = None,
) -> DocumentCollector:
    settings = settings or Settings.from_env()
    return DocumentCollector(
        adapters=adapters or default_adapters(settings),
        cache=cache or build_cache(settings.database_url),
        settings=settings,
    )


def collect_training(
    technologies: Sequence[Technology | dict],
    *,
    collector: DocumentCollector | None = None,
) -> list[dict]:
    """Training mode (pipeline.md step 3): history for a labelled technology list."""
    engine = collector or build_collector()
    items = [
        item if isinstance(item, Technology) else Technology.from_dict(item) for item in technologies
    ]
    return [result.to_contract_dict() for result in engine.collect_training(items)]


def search_recent(
    subqueries: Sequence[str],
    *,
    collector: DocumentCollector | None = None,
    date_from: date | None = None,
    date_to_exclusive: date | None = None,
) -> dict:
    """Query mode, Search #1 (pipeline.md step 3): fresh documents by topic subqueries."""
    engine = collector or build_collector()
    return engine.search_recent(
        subqueries, date_from=date_from, date_to_exclusive=date_to_exclusive
    ).to_dict()


def collect_history(
    candidate: Candidate | dict,
    *,
    collector: DocumentCollector | None = None,
) -> dict:
    """Query mode, Search #2 (pipeline.md step 5): 6-year history of one candidate."""
    engine = collector or build_collector()
    item = candidate if isinstance(candidate, Candidate) else Candidate.from_dict(candidate)
    return engine.collect_history(item).to_contract_dict()


def collect_histories(
    candidates: Sequence[Candidate | dict],
    *,
    collector: DocumentCollector | None = None,
    max_candidates: int | None = None,
    is_mainstream: MainstreamPredicate | None = None,
) -> list[dict]:
    """Query mode, Search #2 for many candidates. Mainstream cut happens before fetch."""
    engine = collector or build_collector()
    items = [
        item if isinstance(item, Candidate) else Candidate.from_dict(item) for item in candidates
    ]
    results = engine.collect_histories(
        items, max_candidates=max_candidates, is_mainstream=is_mainstream
    )
    return [item.to_contract_dict() for item in results if not item.skipped_as_mainstream]


def _recent_bounds(date_from: date | None, date_to_exclusive: date | None) -> tuple[date, date]:
    today = datetime.now(timezone.utc).date()
    end = date_to_exclusive or (today + timedelta(days=1))
    start = date_from or (end - timedelta(days=DEFAULT_RECENT_DAYS))
    return start, end


def _dedupe(documents: Iterable[Document]) -> list[Document]:
    kept: list[Document] = []
    seen: set[str] = set()
    for doc in documents:
        if not doc.url or doc.url in seen:
            continue
        seen.add(doc.url)
        kept.append(doc)
    return kept


def _select_candidates(
    candidates: Sequence[Candidate],
    cap: int,
    is_mainstream: MainstreamPredicate | None,
) -> tuple[list[Candidate], list[Candidate]]:
    selected: list[Candidate] = []
    skipped: list[Candidate] = []
    seen: set[str] = set()
    for item in candidates:
        if item.candidate_id in seen:
            continue
        seen.add(item.candidate_id)
        if is_mainstream and is_mainstream(item):
            skipped.append(item)
            continue
        if len(selected) >= cap:
            skipped.append(item)
            continue
        selected.append(item)
    return selected, skipped
