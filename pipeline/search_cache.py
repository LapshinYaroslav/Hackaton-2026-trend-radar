"""Файловый кэш поиска №1: повторный прогон той же темы получает те же документы без сети.

Обёртка над adapter.search, сборщик не меняется. Ключ — (текст подзапроса, язык, источник,
окно дат, потолок, параметры поиска: words у arXiv, language у OpenAlex). Счётчики признаков
идут мимо кэша: у них свой путь и свой кэш.
"""
from __future__ import annotations

import hashlib
import json
from datetime import date
from pathlib import Path

from collector.models import Document

ROOT = Path(__file__).resolve().parents[1]
CACHE_DIR = ROOT / "data" / "interim" / "cache" / "search_recent"


def cache_key(query: str, source: str, date_from: date, date_to_exclusive: date, limit: int,
              options: dict) -> str:
    """sha256 от ключевых полей запроса. Язык берётся из параметров: ru только у OpenAlex."""
    raw = json.dumps({"query": query, "language": options.get("language") or "en", "source": source,
                      "date_from": date_from.isoformat(), "date_to": date_to_exclusive.isoformat(),
                      "limit": limit, "options": options}, ensure_ascii=False, sort_keys=True)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


class CachedSearch:
    """Адаптер-обёртка: search() через файловый кэш, остальное — как у исходного адаптера."""

    def __init__(self, adapter, use_cache: bool = True, cache_dir: Path | None = None) -> None:
        self._adapter, self._use_cache, self._dir = adapter, use_cache, cache_dir or CACHE_DIR
        self.hits = 0

    def __getattr__(self, name):
        return getattr(self._adapter, name)

    def search(self, query: str, date_from: date, date_to_exclusive: date, *, limit: int, **options):
        """Документы из кэша, иначе из источника; успешный ответ кладётся в кэш."""
        path = self._dir / f"{cache_key(query, self._adapter.source, date_from, date_to_exclusive, limit, options)}.json"
        if self._use_cache and path.exists():
            self.hits += 1
            return [Document(**item) for item in json.loads(path.read_text(encoding="utf-8"))["documents"]]
        docs = self._adapter.search(query, date_from, date_to_exclusive, limit=limit, **options)
        self._dir.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"query": query, "source": self._adapter.source, "options": options,
                                    "documents": [doc.to_dict() for doc in docs]}, ensure_ascii=False),
                        encoding="utf-8")
        return docs
