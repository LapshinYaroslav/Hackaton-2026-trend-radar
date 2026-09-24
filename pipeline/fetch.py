"""Счётчики кандидата для model.candidate.candidate_features: живой сборщик, источники параллельно.

Путь счёта не новый: DocumentCollector.count_history, тот же, что дал счётчики обучения.
Меняется только расписание: у каждого источника свой коллектор, и три источника
опрашиваются одновременно — arXiv со своей паузой не ждёт OpenAlex и TechCrunch.
"""
from __future__ import annotations

import hashlib
import json
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import date
from pathlib import Path
from typing import Callable, Sequence

import pandas as pd

from collector.adapters.arxiv import ARXIV_ONE_CALL_COUNTS
from collector.adapters.techcrunch import TECHCRUNCH_ONE_CALL_COUNTS
from collector.adapters.base import SourceAdapter
from collector import rospatent as rospatent_source
from collector.api import DocumentCollector, tech_key
from collector.constants import COUNTER_WINDOWS
from collector.db import build_cache
from collector.models import Candidate, Counter, SearchTerms, build_search_terms
from collector.settings import Settings

COLUMNS = ["source", "window", "n"]
ROOT = Path(__file__).resolve().parents[1]
# Файловый кэш счётчиков оркестратора: общий для прогонов, одинаковый tech_key не пересчитывается.
# Ключ — фраза, источник и способ опроса arXiv (одним вызовом или семью): числа разных способов
# могут не совпадать (сверка В: 6 пар из 1029).
COUNTERS_CACHE_DIR = ROOT / "data" / "interim" / "cache" / "counters_pipeline"


def missing_pairs(frame: pd.DataFrame, sources: set[str]) -> list[tuple[str, str]]:
    """Пары (источник, окно), по которым счётчика нет: признаки по ним не считаются."""
    have = set(zip(frame["source"], frame["window"]))
    return sorted((source, window) for source in sources for window in COUNTER_WINDOWS
                  if (source, window) not in have)


class _Counted:
    """Минимальный заменитель CounterResult: fetch читает только поле counters."""

    def __init__(self, counters: list[Counter]):
        self.counters = counters


def one_call(source: str) -> bool:
    """Считается ли источник одним вызовом с раскладкой по окнам (флаги задач В и Л)."""
    return (source == "arxiv" and ARXIV_ONE_CALL_COUNTS) or (source == "techcrunch" and TECHCRUNCH_ONE_CALL_COUNTS)


def count_source(collector: DocumentCollector, candidate: Candidate):
    """Счётчики одного источника. arXiv и TechCrunch при включённых флагах — одним вызовом с откатом на семь.

    Откат (None от count_windows_one_call): больше 2000 записей, 429, неполный ответ, —
    тогда прежний путь count_history, семь вызовов.
    """
    adapter = collector.adapters[0]
    if one_call(adapter.source):
        search = build_search_terms(candidate.terms, candidate.context_terms)
        counts = adapter.count_windows_one_call(search, COUNTER_WINDOWS)
        if counts is not None:
            key = tech_key(candidate.name_en)
            return _Counted([Counter(tech_key=key, source=adapter.source, window=window,
                                     date_from=COUNTER_WINDOWS[window][0], date_to=COUNTER_WINDOWS[window][1],
                                     n=n, type_filter=adapter.type_filter, query_variant=adapter.query_variant)
                             for window, n in counts.items()])
    return collector.count_history(candidate)


def counters_cache_path(key: str, source: str) -> Path:
    """Файл кэша для (фраза, источник, способ опроса arXiv)."""
    mode = "one_call" if one_call(source) else "per_window"
    raw = json.dumps([key, source, mode], ensure_ascii=False)
    return COUNTERS_CACHE_DIR / f"{hashlib.sha256(raw.encode('utf-8')).hexdigest()}.json"


def cached_count_source(collector: DocumentCollector, candidate: Candidate, use_cache: bool = True):
    """count_source через файловый кэш; в кэш кладутся только полные ответы по всем окнам."""
    source = collector.adapters[0].source
    path = counters_cache_path(tech_key(candidate.name_en), source)
    if use_cache and path.exists():
        rows = json.loads(path.read_text(encoding="utf-8"))
        return _Counted([Counter(**{**row, "date_from": date.fromisoformat(row["date_from"]),
                                    "date_to": date.fromisoformat(row["date_to"])}) for row in rows])
    result = count_source(collector, candidate)
    if {row.window for row in result.counters} >= set(COUNTER_WINDOWS):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps([row.to_dict() for row in result.counters], ensure_ascii=False), encoding="utf-8")
    return result


def complete(result) -> bool:
    """Пришли ли счётчики по всем окнам: ошибка или неполный ответ — сбой очереди."""
    return not isinstance(result, Exception) and         {row.window for row in result.counters} >= set(COUNTER_WINDOWS)


def relative(stats: dict, origin: float) -> dict:
    """Начало и конец очереди в секундах от старта этапа счётчиков."""
    started, finished = stats.pop("started"), stats.pop("finished")
    return {**stats, "started_s": round(started - origin, 2), "finished_s": round(finished - origin, 2),
            "duration_s": round(finished - started, 2)}


def parallel_fetch(adapters: Sequence[SourceAdapter], settings: Settings | None = None,
                   warnings: list[str] | None = None,
                   on_done: Callable[[], None] | None = None,
                   use_cache: bool = True,
                   rospatent: dict | None = None) -> Callable[[SearchTerms], pd.DataFrame]:
    """Fetch для candidate_features. Неполное покрытие -> пустая таблица и запись в warnings.

    Пустая таблица значит «кандидат не оценён» (model.ranking.NO_COUNTERS), а не падение
    всего запроса: compute_features отказывается считать по неполным данным, и это верно.

    fetch.prefetch(phrases) — очереди по источникам (задача Л5.2): каждый источник в своём
    потоке проходит всех кандидатов подряд, не дожидаясь остальных источников; fetch потом
    отдаёт собранное. Без prefetch — прежний путь: три источника параллельно на одного
    кандидата. Числа в обоих режимах одни и те же: меняется только расписание вызовов.

    rospatent — параметры collector.rospatent.count_all (datasets, token, parallel) или None.
    Если задан, prefetch запускает ещё одну очередь — n_pat по фразам — параллельно остальным;
    результаты в fetch.patents. Сводка каждой очереди (начало и конец от старта этапа, запросы,
    кэш, повторы, сбои) — в fetch.queues. Повторы внутри сборщика снаружи не видны: у трёх
    основных источников retries и n429 — None.
    """
    settings = settings or Settings()
    collectors = [DocumentCollector(adapters=[adapter], cache=build_cache(settings.database_url),
                                    settings=settings) for adapter in adapters]
    sources = {adapter.source for adapter in adapters}
    ready: dict[tuple[str, str], object] = {}
    queues: list[dict] = []
    patents: dict[str, dict] = {}

    def one(collector: DocumentCollector, phrase: str, terms: list[str], context: list[str]):
        candidate = Candidate(candidate_id=phrase, name_en=phrase, terms=terms, context_terms=context)
        try:
            return cached_count_source(collector, candidate, use_cache)
        except ValueError as exc:  # запрос не собрался из названия
            return exc

    def prefetch(phrases: Sequence[str]) -> None:
        """Очередь на источник: кандидат готов, когда пришли все источники."""
        left, lock = {phrase: len(collectors) for phrase in phrases}, threading.Lock()
        origin = time.time()

        def run(collector: DocumentCollector) -> dict:
            source = collector.adapters[0].source
            stats = {"source": source, "started": time.time(), "candidates": len(phrases),
                     "requests": 0, "cache_hits": 0, "retries": None, "n429": None, "failures": 0}
            for phrase in phrases:
                hit = use_cache and counters_cache_path(tech_key(phrase), source).exists()
                result = one(collector, phrase, [phrase], [])
                stats["cache_hits" if hit else "requests"] += 1
                stats["failures"] += not complete(result)
                with lock:
                    ready[(phrase, source)] = result
                    left[phrase] -= 1
                    if left[phrase] == 0 and on_done:
                        on_done()
            return {**stats, "finished": time.time()}

        def run_patents() -> dict:
            results, stats = rospatent_source.count_all(phrases, use_cache=use_cache, **rospatent)
            patents.update(results)
            return stats

        with ThreadPoolExecutor(max_workers=len(collectors) + 1) as pool:
            futures = [pool.submit(run, collector) for collector in collectors]
            if rospatent is not None:
                futures.append(pool.submit(run_patents))
            queues[:] = [relative(future.result(), origin) for future in futures]

    def fetch(search: SearchTerms) -> pd.DataFrame:
        phrase = search.terms[0]
        keys = [(phrase, collector.adapters[0].source) for collector in collectors]
        if all(key in ready for key in keys) and not search.context_terms and len(search.terms) == 1:
            results = [ready[key] for key in keys]
        else:
            with ThreadPoolExecutor(max_workers=len(collectors)) as pool:
                results = list(pool.map(lambda c: one(c, phrase, list(search.terms), list(search.context_terms)),
                                        collectors))
            if on_done:
                on_done()
        errors = [r for r in results if isinstance(r, Exception)]
        if errors:
            if warnings is not None:
                warnings.append(f"{phrase}: {errors[0]}")
            results = []
        frame = pd.DataFrame([{"source": row.source, "window": row.window, "n": row.n}
                              for result in results for row in result.counters], columns=COLUMNS)
        gaps = missing_pairs(frame, sources)
        if gaps:
            if warnings is not None:
                warnings.append(f"{phrase}: нет счётчиков {gaps}, кандидат не оценён")
            return pd.DataFrame(columns=COLUMNS)
        return frame

    fetch.prefetch = prefetch
    fetch.queues = queues
    fetch.patents = patents
    return fetch
