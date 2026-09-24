"""Причины исключения и объяснения ТОП-15: шаблоны по вкладам модели, без LLM.

Вклад — слагаемое логита, как его отдаёт model.ranking. Для кандидата ниже порога
берётся признак с самым отрицательным вкладом, для кандидата в ТОП-15 — два с самыми
положительными. Признак без значения (пропуск, модель подставила медиану) в текст
не идёт: число в шаблоне было бы выдуманным.
"""
from __future__ import annotations

import math
from typing import Mapping

GENERIC_BELOW = "Совокупность признаков ниже порога модели"
GENERIC_ABOVE = "Совокупность признаков выше порога модели"

NEGATIVE = {
    "share_news_wordmatch": "Большой объём научной литературы: {n_research} публикаций — "
                            "технология уже хорошо изучена",
    "recency": "Интерес не нарастает: за последние два года {recency:.0%} всех упоминаний",
    "age_first_arxiv": "Термин давно в науке: первый препринт {age:.0f} лет назад",
    "share_prev6": "Значительная часть публикаций вышла до 2020 года ({share_prev6:.0%})",
    "volume": GENERIC_BELOW,
    "growth_research": GENERIC_BELOW,
}
POSITIVE = {
    "share_news_wordmatch": "Про технологию пока мало научных публикаций: {n_research}",
    "recency": "Упоминания свежие: {recency:.0%} за последние два года",
    "age_first_arxiv": "Молодой термин: первый препринт {age:.0f} лет назад",
    "share_prev6": "Почти нет публикаций до 2020 года",
    "volume": GENERIC_ABOVE,
    "growth_research": GENERIC_ABOVE,
}
SPECIAL = {
    "no_trace": "Нет следа в научных источниках: ни один из трёх вариантов названия не встречается "
                "в OpenAlex за 2020–2026",
    "trace_unknown": "Не оценён: OpenAlex не ответил при выборе названия",
    "cap": "Не оценён: превышен лимит числа кандидатов",
    "bad_name": "Название не является устойчивым термином",
    "no_counters": "Не оценён: счётчики источников получены не полностью",
    "beyond_top": "Выше порога модели, но не вошёл в ТОП-15",
}
RESEARCH_SOURCES = ("openalex", "arxiv")
PERIOD_WINDOWS = ("2020", "2021", "2022", "2023", "2024", "2025")


def n_research(counters: Mapping[str, Mapping[str, int]]) -> int:
    """Научные публикации (OpenAlex и arXiv) за период 2020-09 … 2026-08 по счётчикам кандидата."""
    return sum(int(counters.get(window, {}).get(source, 0))
               for window in PERIOD_WINDOWS for source in RESEARCH_SOURCES)


def _known(value) -> bool:
    return value is not None and not (isinstance(value, float) and math.isnan(value))


def _fill(template: str, features: Mapping, counters: Mapping) -> str:
    """Подставляет числа в шаблон."""
    return template.format(n_research=n_research(counters), recency=features.get("recency"),
                           age=features.get("age_first_arxiv"),
                           share_prev6=features.get("share_prev6"))


def reason_below(features: Mapping, contributions: Mapping[str, float], counters: Mapping) -> str:
    """Причина исключения: признак с самым отрицательным вкладом среди известных."""
    negative = sorted((value, name) for name, value in contributions.items()
                      if value < 0 and _known(features.get(name)))
    if not negative:
        return GENERIC_BELOW
    return _fill(NEGATIVE[negative[0][1]], features, counters)


def explanation_top(features: Mapping, contributions: Mapping[str, float], counters: Mapping,
                    count: int = 2) -> list[str]:
    """Объяснение для ТОП-15: до двух признаков с самыми положительными вкладами."""
    positive = sorted(((value, name) for name, value in contributions.items()
                       if value > 0 and _known(features.get(name))), reverse=True)
    texts: list[str] = []
    for _, name in positive:
        text = _fill(POSITIVE[name], features, counters)
        if text not in texts:
            texts.append(text)
        if len(texts) == count:
            break
    return texts or [GENERIC_ABOVE]
