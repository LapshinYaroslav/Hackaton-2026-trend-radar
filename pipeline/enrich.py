"""Догрузка документов по ТОП-15: последний этап run_query перед done, в пределах бюджета времени."""

from __future__ import annotations

import logging
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


def enrich_top(result: dict, collector: DocumentCollector | None = None, *,
               time_left: Callable[[], float] | None = None,
               progress: Callable[[str, int, int], None] | None = None, source_limit: int = MAX_EXTRA) -> dict:
    """Ищет дополнительные документы по английскому имени каждого пункта ТОП.

    time_left() <= 0 перед очередным пунктом — остановка, enrichment = "partial"; progress("enrich", i, n).
    В sources не больше source_limit; все найденные документы — в _documents (из них строится инсайт).
    n_docs и n_sources не меняются: это след поиска №1, по нему уже решён отсев слабых источников.
    """
    settings = Settings.from_env()
    own = collector is None
    collector = collector or DocumentCollector(
        adapters=default_adapters(settings),
        cache=build_cache(settings.database_url),
        settings=settings,
    )
    extras: list[dict] = []
    top, stopped = list(result.get("top") or []), False
    try:
        for index, item in enumerate(top):
            if time_left is not None and time_left() <= 0:
                stopped = True
                break
            if progress:
                progress("enrich", index, len(top))
            name = item.get("name_en")
            if not name:
                continue
            try:
                found = collector.search_recent(
                    [
                        {
                            "subquery_id": f"enrich-{item.get('rank')}",
                            "language": "en",
                            "text": name,
                        }
                    ],
                    limit=MAX_EXTRA,
                )
            except Exception as exc:
                logger.warning("enrich %s: %s", name, exc)
                continue
            docs = found.to_dict()["documents"]
            extras.extend(docs)
            item["sources"] = merge_sources(list(item.get("sources") or []), [_as_source(doc) for doc in docs],
                                            limit=source_limit)
    finally:
        if own:
            close = getattr(collector, "close", None)
            if callable(close):
                close()
    query_id = result.get("query_id")
    if query_id and extras:
        persist_search_documents(query_id, extras, result.get("top") or [])
    documents = list(result.get("_documents") or [])
    seen = {doc.get("url") for doc in documents}
    for doc in extras:
        if doc.get("url") and doc["url"] not in seen:
            documents.append(doc)
            seen.add(doc["url"])
    result["_documents"] = documents
    result["enrichment"] = "partial" if stopped else "done"
    if progress:
        progress("enrich", len(top), len(top))
    return result
