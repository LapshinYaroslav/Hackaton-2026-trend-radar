"""Признаки одного кандидата: название технологии → шесть чисел для модели.

Один путь расчёта с обучающей таблицей. Те же функции: aggregate_windows складывает
годовые счётчики в рабочие окна, compute_features считает объём, свежесть и доли,
growth_by_sources — рост научного следа, age_from_counters — возраст первого
препринта. Здесь нет ни одной своей формулы, только склейка.

Откуда берутся счётчики. Функция принимает fetch — как получить счётчики по набору
терминов. По умолчанию это файловый кэш: без сети и ровно те числа, на которых
обучалась модель. В режиме запроса сюда передаётся сборщик, ходящий в источники.

Порядок признаков — набор версии модели (model.config.VERSIONS), тот же, в котором лежат
коэффициенты. share_patent (s2a2-v1) считается по n_pat Роспатента, который передаётся
параметром: счётчики источников его не содержат. Нет n_pat (сбой, Роспатент выключен) —
признак NaN, и модель подставит медиану обучения.

Запуск: python -m model.candidate "edge model compression"
"""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Callable

import pandas as pd

from collector.api import terms_hash
from collector.constants import COUNTER_WINDOWS
from collector.models import SearchTerms
from collector.query import build_query
from model.config import DEFAULT_VERSION, NEWS_SHARE, VERSIONS
from model.corpus import EXPECTED_SOURCES, load_training_totals
from model.counters import aggregate_windows
from model.features import (RESEARCH_SOURCES, compute_features, growth_by_sources,
                            share_patent)
from model.first_mention import age_from_counters

ROOT = Path(__file__).resolve().parents[1]
COUNTERS_DIR = ROOT / "data" / "interim" / "collector" / "cache" / "counters"
Fetch = Callable[[SearchTerms], pd.DataFrame]


def features_from_counters(counters: pd.DataFrame,
                           totals: pd.DataFrame | None = None, *, n_pat: int | None = None,
                           version: str = DEFAULT_VERSION) -> dict[str, float]:
    """Шесть признаков версии модели по годовым счётчикам одной технологии.

    Вход — счётчики до агрегации: по рабочим окнам возраст первого упоминания
    не определить, они перекрываются. n_pat нужен только s2a2-v1.
    """
    corpus = aggregate_windows(totals if totals is not None else load_training_totals(),
                               value_column="n_total")
    windows = aggregate_windows(counters)
    base = compute_features(windows, corpus, expected_sources=set(EXPECTED_SOURCES))
    research = int(windows.loc[(windows["window"] == "all")
                               & windows["source"].isin(RESEARCH_SOURCES), "n"].sum())
    values = {
        "volume": base["volume"],
        "growth_research": growth_by_sources(windows, corpus, RESEARCH_SOURCES),
        "recency": base["recency"],
        NEWS_SHARE: base["share_market"],
        "share_prev6": base["share_prev6"],
        "age_first_arxiv": age_from_counters(counters, "arxiv"),
        "share_patent": math.nan if n_pat is None else share_patent(int(n_pat), research),
    }
    return {name: values[name] for name in VERSIONS[version]}


def counters_by_window(counters: pd.DataFrame) -> dict[str, dict[str, int]]:
    """Счётчики как есть: окно → источник → число документов. Для показа в выдаче."""
    grouped = counters.groupby(["window", "source"])["n"].sum()
    result: dict[str, dict[str, int]] = {}
    for (window, source), value in grouped.items():
        result.setdefault(str(window), {})[str(source)] = int(value)
    return {window: result[window] for window in COUNTER_WINDOWS if window in result}


def cache_fetch(search: SearchTerms) -> pd.DataFrame:
    """Счётчики из файлового кэша по отпечатку терминов. Без сети.

    Файл заведён на пару (технология, источник), поэтому строки складываются.
    """
    digest = terms_hash(search)
    rows: list[dict[str, object]] = []
    for path in sorted(COUNTERS_DIR.glob("*.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("terms_hash") != digest:
            continue
        rows.extend({"source": item["source"], "window": item["window"], "n": item["n"]}
                    for item in payload["counters"])
    return pd.DataFrame(rows, columns=["source", "window", "n"])


def candidate_features(name: str, *, terms: list[str] | None = None,
                       context_terms: list[str] | None = None,
                       area: str = "", fetch: Fetch | None = None,
                       n_pat: int | None = None, version: str = DEFAULT_VERSION) -> dict:
    """Название технологии → шесть чисел плюс счётчики и источники под ними.

    Возвращает словарь:
      name, area   как передали;
      query        собранный булев запрос (пустой, если терминов меньше двух);
      terms        термины, ушедшие в источники;
      features     шесть чисел в порядке набора версии (model.config.VERSIONS);
      counters     окно → источник → число документов;
      sources      источники, ответившие хоть по одному окну;
      complete     True, если признаки посчитаны; False — если счётчиков не нашлось.

    Пропуск age_first_arxiv — обычное дело: у технологии может не быть ни одного
    препринта arXiv, и тогда возраст неизвестен, а не равен нулю. В словаре он
    останется NaN; модель подставит медиану обучающей выборки.
    """
    search = SearchTerms(terms=terms or [name], context_terms=context_terms or [],
                         query="")
    counters = (fetch or cache_fetch)(search)
    query, _ = build_query(search.terms, search.context_terms)
    answer = {"name": name, "area": area, "query": query, "terms": list(search.terms),
              "counters": counters_by_window(counters) if not counters.empty else {},
              "sources": sorted(counters["source"].unique()) if not counters.empty else [],
              "features": {name: math.nan for name in VERSIONS[version]}, "complete": False}
    if counters.empty:
        return answer
    answer["features"] = features_from_counters(counters, n_pat=n_pat, version=version)
    answer["complete"] = True
    return answer


def main() -> None:
    """Печатает признаки кандидата по названию из кэша."""
    import sys
    name = sys.argv[1] if len(sys.argv) > 1 else "edge model compression"
    answer = candidate_features(name)
    print(f"{answer['name']}  запрос: {answer['query'] or '(не собран)'}")
    print(f"источники: {', '.join(answer['sources']) or 'нет'}  "
          f"посчитано: {answer['complete']}")
    for key, value in answer["features"].items():
        print(f"  {key:22s} {value:.6f}" if value == value else f"  {key:22s} пропуск")
    print("счётчики по окнам:")
    for window, by_source in answer["counters"].items():
        print(f"  {window:6s} " + ", ".join(f"{s}={n}" for s, n in sorted(by_source.items())))


if __name__ == "__main__":
    main()
