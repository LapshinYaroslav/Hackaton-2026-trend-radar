"""Причины исключения и объяснения ТОП-15: шаблоны по вкладам модели, без LLM.

Вклад — слагаемое логита, как его отдаёт model.ranking. Для кандидата ниже порога
берётся признак с самым отрицательным вкладом, для кандидата в ТОП-15 — два с самыми
положительными. Признак без значения (пропуск, модель подставила медиану) в текст
не идёт: число в шаблоне было бы выдуманным.

volume в боевой модели s2a2-v1 нет; его общая строка оставлена для воспроизведения
отчётов s2a1-v1. share_patent (s2a2-v1) пишется числами патентов и публикаций:
n_pat приходит от Роспатента отдельно, n_research — из счётчиков кандидата.
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
    "share_patent": "Технология активно патентуется: {patents} на {publications}",
    "volume": GENERIC_BELOW,
    "growth_research": GENERIC_BELOW,
}
POSITIVE = {
    "share_news_wordmatch": "Про технологию пока мало научных публикаций: {n_research}",
    "recency": "Упоминания свежие: {recency:.0%} за последние два года",
    "age_first_arxiv": "Молодой термин: первый препринт {age:.0f} лет назад",
    "share_prev6": "Почти нет публикаций до 2020 года",
    "share_patent": "Патентов мало относительно научных работ: {patents} на {publications}",
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
    "duplicate_of": "Дубль: тот же класс технологий уже есть в выдаче",
}
RESEARCH_SOURCES = ("openalex", "arxiv")
PERIOD_WINDOWS = ("2020", "2021", "2022", "2023", "2024", "2025")


def n_research(counters: Mapping[str, Mapping[str, int]]) -> int:
    """Научные публикации (OpenAlex и arXiv) за период 2020-09 … 2026-08 по счётчикам кандидата."""
    return sum(int(counters.get(window, {}).get(source, 0))
               for window in PERIOD_WINDOWS for source in RESEARCH_SOURCES)


def _known(value) -> bool:
    return value is not None and not (isinstance(value, float) and math.isnan(value))


def plural(number: int, one: str, few: str, many: str) -> str:
    """Число с существительным в нужной форме и пробелом между тысячами: 1 патент, 1 200 публикаций."""
    tail, tens = number % 10, number % 100
    word = one if tail == 1 and tens != 11 else few if 2 <= tail <= 4 and not 12 <= tens <= 14 else many
    return f"{number:,}".replace(",", " ") + f" {word}"


def _fill(template: str, features: Mapping, counters: Mapping, n_pat: int | None = None) -> str:
    """Подставляет числа в шаблон."""
    research = n_research(counters)
    return template.format(n_research=research, recency=features.get("recency"),
                           age=features.get("age_first_arxiv"),
                           share_prev6=features.get("share_prev6"),
                           patents=plural(n_pat or 0, "патент", "патента", "патентов"),
                           publications=plural(research, "публикацию", "публикации", "публикаций"))


def _usable(name: str, features: Mapping, n_pat: int | None) -> bool:
    """Признак идёт в текст, если у него есть значение; патентный — ещё и число патентов."""
    return _known(features.get(name)) and (name != "share_patent" or n_pat is not None)


def reason_below(features: Mapping, contributions: Mapping[str, float], counters: Mapping,
                 n_pat: int | None = None) -> str:
    """Причина исключения: признак с самым отрицательным вкладом среди известных."""
    negative = sorted((value, name) for name, value in contributions.items()
                      if value < 0 and _usable(name, features, n_pat))
    if not negative:
        return GENERIC_BELOW
    return _fill(NEGATIVE[negative[0][1]], features, counters, n_pat)


def explanation_top(features: Mapping, contributions: Mapping[str, float], counters: Mapping,
                    count: int = 2, n_pat: int | None = None) -> list[str]:
    """Объяснение для ТОП-15: до двух признаков с самыми положительными вкладами."""
    positive = sorted(((value, name) for name, value in contributions.items()
                       if value > 0 and _usable(name, features, n_pat)), reverse=True)
    texts: list[str] = []
    for _, name in positive:
        text = _fill(POSITIVE[name], features, counters, n_pat)
        if text not in texts:
            texts.append(text)
        if len(texts) == count:
            break
    return texts or [GENERIC_ABOVE]
