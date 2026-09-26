"""Public collector API: query-time Search #1 (search_recent) and Search #2 counters (count_history).

Search #1 never returns source_totals and must not be fed to features — otherwise every
candidate looks "fresh" (pipeline.md). Выгрузка документов поиска №2 и режим обучения удалены
в задаче Л: их вызывал только CLI сборщика.
"""

from __future__ import annotations

import hashlib
import json
import logging
import threading
from datetime import date, datetime, timedelta, timezone
from typing import Iterable, Sequence

from collector.adapters import ArxivAdapter, OpenAlexAdapter, TechCrunchAdapter
from collector.adapters.base import SourceAdapter, totals_signature
from collector.constants import (
    COUNTER_WINDOWS,
    YEAR_WINDOWS,
    DEFAULT_RECENT_DAYS,
    DEFAULT_RECENT_DOCS_PER_SUBQUERY,
    SOURCE_TOTALS_TTL_HOURS,
    WINDOWS,
)
from collector.db import DocumentCache, MemoryCache
from collector.exceptions import AdapterError
from collector.http import HttpTransport, HttpxTransport
from collector.models import (
    Candidate,
    Counter,
    CounterResult,
    Document,
    RecentSearchResult,
    SearchTerms,
    SourceTotal,
    build_search_terms,
    copy_document,
    require_source_type,
)
from collector.settings import Settings

logger = logging.getLogger(__name__)

# Поиск №1: язык подзапроса -> источник -> параметры его search(). Источника нет в
# списке языка — подзапрос туда не идёт. ru уходит только в OpenAlex с фильтром языка:
# arXiv и TechCrunch англоязычные. arXiv ищет слова по отдельности (замер А3).
RECENT_ROUTES: dict[str, dict[str, dict]] = {
    "en": {"openalex": {}, "arxiv": {"words": True}, "techcrunch": {}},
    "ru": {"openalex": {"language": "ru"}},
}


def terms_hash(search: SearchTerms) -> str:
    """Отпечаток поискового запроса: часть ключа кэша счётчиков.

    Отвечает ровно на один вопрос: тот же ли это набор терминов. Уточнили термины —
    отпечаток другой — старые счётчики из кэша не вернутся. Регистр и порядок
    терминов не влияют.

    Фильтра типа здесь больше нет. Это константа OpenAlex, и в общем отпечатке она
    сидела в ключе всех источников сразу: смена фильтра у одного выбрасывала бы кэш
    GitHub и Википедии, которым до типов записи OpenAlex дела нет. Семантика запроса
    переехала в query_variant — отдельное поле ключа строки, своё у каждого источника.

    Набора окон здесь нет намеренно. Окно закодировано в ключе строки отдельной
    колонкой period, и вторая копия внутри хэша делала вредное: добавление одного
    нового окна меняло отпечаток и выбрасывало все ранее собранные строки по всем
    окнам. Названия технологии здесь тоже нет — она уже в ключе как tech_key.
    """
    payload = json.dumps(
        {
            "terms": sorted(item.casefold() for item in search.terms),
            "context_terms": sorted(item.casefold() for item in search.context_terms),
        },
        ensure_ascii=False,
        sort_keys=True,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:32]


def tech_key(name_en: str) -> str:
    """Stable identity across query_ids. Aliases expand search terms, not the cache key."""
    return " ".join(name_en.split()).casefold()


class DocumentCollector:
    def __init__(
        self,
        adapters: Sequence[SourceAdapter],
        cache: DocumentCache | None = None,
        settings: Settings | None = None,
    ) -> None:
        if not adapters:
            raise ValueError("DocumentCollector requires at least one source adapter")
        self.adapters = list(adapters)
        self.cache = cache or MemoryCache()
        self.settings = settings or Settings()
        self._totals_lock = threading.Lock()
        # Источник -> окна, уже спрошенные в этом процессе. Именно окна, а не просто
        # факт пробы: спросили пять окон, потом добавили шестое — шестое надо спросить.
        self._probed_sources: dict[str, set[str]] = {}

    def probe_source_totals(
        self,
        *,
        force: bool = False,
        windows: dict[str, tuple[date, date]] | None = None,
    ) -> list[SourceTotal]:
        """Ask every source for corpus size on both windows before any tech search.

        Кэш годен сутки. Просроченные строки и источники, чей итог неизвестен,
        пробуются заново — иначе одна неудачная проба навсегда выключила бы источник
        из growth. Внутри одного процесса каждый источник пробуется не чаще раза.

        Возвращаются строки только по адаптерам этого коллектора и только по
        запрошенным окнам. В кэш при этом уходит слияние со всем, что там лежало:
        прогон по одному источнику не должен стирать итоги остальных, иначе growth
        перестанет считаться у всех сразу — поправку на фон брать будет неоткуда.
        """
        asked = windows or WINDOWS
        with self._totals_lock:
            if force:
                self._probed_sources.clear()
            stored = self.cache.get_source_totals() or []
            cached = [] if force else stored
            expected = {adapter.source: totals_signature(adapter) for adapter in self.adapters}
            reusable = _reusable_sources(cached, self._probed_sources, expected, set(asked))

            totals: list[SourceTotal] = []
            changed = False
            for adapter in self.adapters:
                if adapter.source in reusable:
                    totals.extend(row for row in cached if row.source == adapter.source)
                    continue
                totals.extend(self._probe_adapter(adapter, asked))
                self._probed_sources[adapter.source] = set(asked)
                changed = True
            if changed:
                self.cache.put_source_totals(_merge_totals(stored, totals))
            return totals

    def _probe_adapter(
        self,
        adapter: SourceAdapter,
        windows: dict[str, tuple[date, date]] | None = None,
    ) -> list[SourceTotal]:
        rows: list[SourceTotal] = []
        signature = totals_signature(adapter)
        for window, (start, end) in (windows or WINDOWS).items():
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
                    type_filter=adapter.type_filter,
                    totals_signature=signature,
                )
            )
        if not all(row.available for row in rows):
            logger.warning(
                "source %s has no complete totals for the asked windows; excluded from growth",
                adapter.source,
            )
            rows = [
                SourceTotal(
                    source=row.source,
                    window=row.window,
                    n_total=row.n_total,
                    available=False,
                    type_filter=row.type_filter,
                    totals_signature=row.totals_signature,
                )
                for row in rows
            ]
        return rows

    def search_recent(
        self,
        subqueries: Sequence[str | dict],
        *,
        date_from: date | None = None,
        date_to_exclusive: date | None = None,
        limit: int = DEFAULT_RECENT_DOCS_PER_SUBQUERY,
    ) -> RecentSearchResult:
        """Fresh documents by topic subqueries. Result = candidate pool, not features.

        Вход — подзапросы шага 2.2 ({subquery_id, language, text}) или строки (CLI):
        строка считается английским подзапросом. Источники выбираются по языку
        (RECENT_ROUTES), у каждого документа — subquery_id, по которым он найден.
        """
        start, end = _recent_bounds(date_from, date_to_exclusive)
        items = _recent_items(subqueries)
        found: list[tuple[Document, str]] = []
        for item in items:
            for adapter in self.adapters:
                options = _route(adapter.source, item["language"])
                if options is None:
                    continue
                try:
                    docs = adapter.search(item["text"], start, end, limit=limit, **options)
                except AdapterError as exc:
                    logger.warning("search failed for %s / %r: %s", adapter.source, item["text"], exc)
                    continue
                found += [(self._normalize_document(doc, adapter.source), item["subquery_id"])
                          for doc in docs]
        documents, ids = _dedupe_with_ids(found)
        return RecentSearchResult(documents=documents, subqueries=items, date_from=start,
                                  date_to=end, subquery_ids=ids)

    def count_history(self, candidate: Candidate, *, use_cache: bool = True) -> CounterResult:
        """Поиск №2 версии 0.2: счётчики по технологии, без выгрузки документов.

        Признаки считаются по этим числам. Потолок выдачи на них не влияет: источник
        отвечает, сколько записей подходит под фильтр, независимо от того, сколько
        их можно скачать (pipeline.md 0.2, «Почему счётчики, а не документы»).
        """
        search = build_search_terms(candidate.terms, candidate.context_terms)
        if not search.usable:
            raise ValueError(
                f"{candidate.candidate_id}: запрос не собрался из терминов, "
                f"нарушения: {[item['rule'] for item in search.violations]}"
            )
        totals = self.probe_source_totals(windows=YEAR_WINDOWS)
        key = tech_key(candidate.name_en)
        digest = terms_hash(search)
        # Кэш спрашивается по каждому источнику отдельно: файл и строка в базе
        # заведены на пару (технология, источник), чтобы сборщики не мешали друг другу.
        known: list[Counter] = []
        hit = False
        for adapter in self.adapters:
            rows = self.cache.get_counters(key, digest, adapter.source) if use_cache else None
            if rows is not None:
                hit = True
                known.extend(rows)
        have = {(row.source, row.window, row.query_variant) for row in known}
        wanted = {(adapter.source, window, adapter.query_variant)
                  for adapter in self.adapters for window in COUNTER_WINDOWS}
        missing = {window for _, window, _ in wanted - have}
        if hit and not missing:
            return self._make_counter_result(candidate, key, digest, known, totals, True)
        # Досчитываются пары (источник, окно), которых нет в кэше: новое окно у всех
        # источников или один источник, промолчавший по своим окнам. Пары, уже лежащие
        # в кэше, не перезапрашиваются, поэтому добор arXiv не стоит вызовов OpenAlex.
        #
        # Обратная сторона: источник, который молчит всегда, будет опрашиваться каждый
        # прогон. Для arXiv и TechCrunch это бесплатно, для OpenAlex — вызов за $0.001,
        # и это осознанный размен: молчание источника и его ноль — разные вещи, и
        # замораживать молчание в кэше значит считать признаки по неполным данным.
        counters = known + self._count_windows(
            key, search, {name: COUNTER_WINDOWS[name] for name in sorted(missing)}, have)
        by_source: dict[str, list[Counter]] = {}
        for row in counters:
            by_source.setdefault(row.source, []).append(row)
        for source, rows in by_source.items():
            self.cache.put_counters(
                key,
                digest,
                source,
                rows,
                candidate_id=candidate.candidate_id,
                query_id=candidate.query_id,
            )
        return self._make_counter_result(candidate, key, digest, counters, totals, False)

    def _count_windows(
        self,
        key: str,
        search: SearchTerms,
        windows: dict[str, tuple[date, date]] | None = None,
        have: set[tuple[str, str]] | None = None,
    ) -> list[Counter]:
        """По одному счётчику на каждую пару (источник, окно). Молчащий источник строки не даёт."""
        counters: list[Counter] = []
        for window, (start, end) in (windows or COUNTER_WINDOWS).items():
            for adapter in self.adapters:
                if have and (adapter.source, window, adapter.query_variant) in have:
                    continue
                try:
                    number = adapter.count_matching(search, start, end)
                except AdapterError as exc:
                    logger.warning("счётчик не получен: %s/%s: %s", adapter.source, window, exc)
                    number = None
                if number is None:
                    continue
                counters.append(
                    Counter(
                        tech_key=key,
                        source=adapter.source,
                        window=window,
                        date_from=start,
                        date_to=end,
                        n=number,
                        type_filter=adapter.type_filter,
                        query_variant=adapter.query_variant,
                    )
                )
        return counters

    def _make_counter_result(
        self,
        candidate: Candidate,
        key: str,
        digest: str,
        counters: list[Counter],
        totals: list[SourceTotal],
        cache_hit: bool,
    ) -> CounterResult:
        return CounterResult(
            candidate_id=candidate.candidate_id,
            tech_key=key,
            terms_hash=digest,
            counters=counters,
            source_totals=totals,
            query_id=candidate.query_id,
            cache_hit=cache_hit,
        )

    def _normalize_document(self, doc: Document, source: str) -> Document:
        require_source_type(doc.source_type, source or doc.source)
        if doc.source != source:
            return copy_document(doc, source=source)
        return doc

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


def _recent_items(subqueries: Sequence[str | dict]) -> list[dict[str, str]]:
    """Подзапросы к одному виду {subquery_id, language, text}. Строка — английский подзапрос."""
    items: list[dict[str, str]] = []
    for number, raw in enumerate(subqueries, start=1):
        item = {"language": "en", "text": raw} if isinstance(raw, str) else dict(raw)
        text = str(item.get("text") or "").strip()
        if not text:
            continue
        language = str(item.get("language") or "").strip()
        if language not in RECENT_ROUTES:
            raise ValueError(f"подзапрос {text!r}: язык {language!r} не поддержан, "
                             f"допустимы {sorted(RECENT_ROUTES)}")
        items.append({"subquery_id": str(item.get("subquery_id") or f"s{number}"),
                      "language": language, "text": text})
    return items


def _route(source: str, language: str) -> dict | None:
    """Параметры search() источника для языка или None, если подзапрос туда не идёт.

    Источник не из списка (например, заглушка в тестах) получает английские подзапросы
    без параметров, как до маршрутизации, и не получает русские.
    """
    routes = RECENT_ROUTES[language]
    return routes.get(source, {} if language == "en" else None)


def _dedupe_with_ids(found: Iterable[tuple[Document, str]]) -> tuple[list[Document], dict[str, list[str]]]:
    """Один документ на url; все subquery_id, по которым он нашёлся, в порядке находок."""
    kept: list[Document] = []
    ids: dict[str, list[str]] = {}
    for doc, subquery_id in found:
        if not doc.url:
            continue
        if doc.url not in ids:
            ids[doc.url] = []
            kept.append(doc)
        if subquery_id not in ids[doc.url]:
            ids[doc.url].append(subquery_id)
    return kept, ids


def _merge_totals(
    stored: Sequence[SourceTotal],
    fresh: Sequence[SourceTotal],
) -> list[SourceTotal]:
    """Свежие итоги поверх сохранённых. Ключ — пара (источник, окно).

    Пара, а не источник: прогон может спросить корпуса за два окна роста, а в кэше
    лежат шесть годовых. Слияние по источнику затёрло бы четыре из них и сделало бы
    growth несчитаемым, хотя данные были.

    Порядок фиксированный, чтобы файл кэша не переписывался от перестановки строк.
    """
    merged = {(row.source, row.window): row for row in stored}
    merged.update({(row.source, row.window): row for row in fresh})
    return [merged[key] for key in sorted(merged)]


def _reusable_sources(
    cached: Sequence[SourceTotal],
    probed: dict[str, set[str]],
    expected: dict[str, str],
    asked_windows: set[str],
) -> set[str]:
    """Источники, чьи итоги можно взять из кэша.

    Годен итог свежий, доступный, по обоим окнам и посчитанный тем же запросом, что
    адаптер отправил бы сейчас. Последнее обязательно: смена фильтра, эндпоинта или
    любого параметра меняет определение знаменателя growth, и старое число молча
    считалось бы вместе с новым числителем. Подпись выводится из самого запроса,
    помнить про неё не нужно.
    """
    deadline = datetime.now(timezone.utc) - timedelta(hours=SOURCE_TOTALS_TTL_HOURS)
    by_source: dict[str, list[SourceTotal]] = {}
    for row in cached:
        by_source.setdefault(row.source, []).append(row)
    reusable = set()
    for source, rows in by_source.items():
        if asked_windows <= probed.get(source, set()):
            reusable.add(source)
            continue
        windows = {row.window for row in rows}
        fresh = all(row.collected_at is not None and row.collected_at > deadline for row in rows)
        same_method = all(row.totals_signature == expected.get(source) for row in rows)
        complete = asked_windows <= windows
        if complete and fresh and same_method and all(row.available for row in rows):
            reusable.add(source)
    return reusable


def _recent_bounds(date_from: date | None, date_to_exclusive: date | None) -> tuple[date, date]:
    today = datetime.now(timezone.utc).date()
    end = date_to_exclusive or (today + timedelta(days=1))
    start = date_from or (end - timedelta(days=DEFAULT_RECENT_DAYS))
    return start, end
