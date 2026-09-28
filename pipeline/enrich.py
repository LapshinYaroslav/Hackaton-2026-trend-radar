"""Догрузка документов по ТОП-15: последний этап run_query перед done, в пределах бюджета времени."""

from __future__ import annotations

import logging
import threading
from concurrent.futures import ThreadPoolExecutor
from typing import Callable, Mapping

from collector.api import DocumentCollector, default_adapters
from collector.db import build_cache
from collector.settings import Settings
from pipeline.persist import persist_search_documents

logger = logging.getLogger(__name__)

MAX_EXTRA = 8
SOURCE_FIELDS = (
    "title",
    "url",
    "published_at",
    "source",
    "source_type",
    "language",
    "trust_level",
    "text",
)


def _as_source(doc: Mapping) -> dict:
    return {name: doc.get(name) for name in SOURCE_FIELDS if doc.get(name) is not None}


def merge_sources(existing: list[dict], extra: list[dict], limit: int = MAX_EXTRA) -> list[dict]:
    """Добавляет новые URL, не дублируя уже показанные источники."""
    seen = {item.get("url") for item in existing if item.get("url")}
    merged = list(existing)
    for item in extra:
        url = item.get("url")
        if not url or url in seen:
            continue
        seen.add(url)
        merged.append(item)
        if len(merged) >= limit:
            break
    return merged


def _source_queue(collector: DocumentCollector, names: list[tuple[int, str]],
                  time_left: Callable[[], float] | None, tick: Callable[[], None]) -> tuple[dict[int, list], bool]:
    """Один источник проходит пункты ТОП подряд: {индекс пункта: документы} и флаг остановки по бюджету."""
    found: dict[int, list] = {}
    for index, name in names:
        if time_left is not None and time_left() <= 0:
            return found, True
        try:
            result = collector.search_recent([{"subquery_id": f"enrich-{index + 1}", "language": "en", "text": name}],
                                             limit=MAX_EXTRA)
            found[index] = result.to_dict()["documents"]
        except Exception as exc:  # сбой источника на одном пункте не останавливает очередь
            logger.warning("enrich %s / %s: %s", collector.adapters[0].source, name, exc)
        tick()
    return found, False


def enrich_top(result: dict, collector: DocumentCollector | None = None, *,
               time_left: Callable[[], float] | None = None,
               progress: Callable[[str, int, int], None] | None = None, source_limit: int = MAX_EXTRA) -> dict:
    """Ищет дополнительные документы по английскому имени каждого пункта ТОП.

    Как счётчики (pipeline.fetch.parallel_fetch): очередь на источник, источники параллельно, внутри очереди
    пункты подряд. time_left() <= 0 перед очередным пунктом — очередь останавливается, enrichment = "partial".
    progress("enrich", готово, пунктов × источников). В sources не больше source_limit; все найденные документы —
    в _documents (из них строится инсайт). n_docs и n_sources не меняются: это след поиска №1.
    """
    settings = Settings.from_env()
    own = collector is None
    collector = collector or DocumentCollector(adapters=default_adapters(settings),
                                               cache=build_cache(settings.database_url), settings=settings)
    top = list(result.get("top") or [])
    names = [(index, item["name_en"]) for index, item in enumerate(top) if item.get("name_en")]
    queues = [DocumentCollector(adapters=[adapter], cache=collector.cache, settings=collector.settings)
              for adapter in collector.adapters]
    total, done, lock = len(names) * len(queues), [0], threading.Lock()

    def tick() -> None:
        with lock:
            done[0] += 1
            if progress:
                progress("enrich", done[0], total)

    if progress:
        progress("enrich", 0, total)
    try:
        with ThreadPoolExecutor(max_workers=max(len(queues), 1)) as pool:
            outcomes = list(pool.map(lambda queue: _source_queue(queue, names, time_left, tick), queues))
    finally:
        if own:
            close = getattr(collector, "close", None)
            if callable(close):
                close()
    extras: list[dict] = []
    for index, item in enumerate(top):
        docs = [doc for found, _ in outcomes for doc in found.get(index, [])]
        extras.extend(docs)
        item["sources"] = merge_sources(list(item.get("sources") or []), [_as_source(doc) for doc in docs],
                                        limit=source_limit)
    query_id = result.get("query_id")
    if query_id and extras:
        persist_search_documents(query_id, extras, top)
    documents = list(result.get("_documents") or [])
    seen = {doc.get("url") for doc in documents}
    for doc in extras:
        if doc.get("url") and doc["url"] not in seen:
            documents.append(doc)
            seen.add(doc["url"])
    result["_documents"] = documents
    result["enrichment"] = "partial" if any(stopped for _, stopped in outcomes) else "done"
    return result
