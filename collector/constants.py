"""Canonical dates and windows. UTC calendar days, YYYY-MM-DD.

Inclusive start, exclusive end: 2020-09-01 <= published_at < 2026-09-01.
Windows from pipeline.md:
  before = 01.09.2023–31.08.2024  →  [2023-09-01, 2024-09-01)
  now    = 01.09.2025–31.08.2026  →  [2025-09-01, 2026-09-01)
"""

from __future__ import annotations

import hashlib
import json
from datetime import date, timedelta
from typing import Final

COLLECTION_START: Final[date] = date(2020, 9, 1)
CUTOFF_DATE: Final[date] = date(2026, 9, 1)

WINDOW_BEFORE_START: Final[date] = date(2023, 9, 1)
WINDOW_BEFORE_END: Final[date] = date(2024, 9, 1)
WINDOW_NOW_START: Final[date] = date(2025, 9, 1)
WINDOW_NOW_END: Final[date] = date(2026, 9, 1)

WINDOWS: Final[dict[str, tuple[date, date]]] = {
    "before": (WINDOW_BEFORE_START, WINDOW_BEFORE_END),
    "now": (WINDOW_NOW_START, WINDOW_NOW_END),
}

ALLOWED_SOURCE_TYPES: Final[frozenset[str]] = frozenset(
    {
        "paper",
        "preprint",
        "patent",
        "news",
        "press_release",
        "product",
        "report",
        "standard",
        "blog",
    }
)

ALLOWED_TRUST_LEVELS: Final[frozenset[str]] = frozenset({"high", "medium", "low"})
# Окна счётчиков поиска №2: шесть годовых отрезков с 2020-09-01 по 2026-09-01.
# Границы полуоткрытые: [from, to). Ключ окна — год его начала.
#
# Почему годовые, а не сразу рабочие: рабочие окна (all, now, before, recent24)
# пересекаются, и запрашивать их по отдельности — значит платить за одни и те же
# документы дважды. Годовые отрезки не пересекаются и покрывают весь период, а любое
# рабочее окно собирается из них сложением, без сети. Заодно остаётся годовой ряд,
# по которому видно форму кривой, а не только два её конца.
YEAR_WINDOWS: Final[dict[str, tuple[date, date]]] = {
    str(year): (date(year, 9, 1), date(year + 1, 9, 1))
    for year in range(COLLECTION_START.year, CUTOFF_DATE.year)
}

# Предыдущая шестилетка, ровно такой же длины, как период сбора. Окно без нижней
# границы было бы у каждого источника своим: arXiv существует с 1991, TechCrunch
# с 2005, OpenAlex индексирует и XIX век. Дробь тогда сравнивала бы тридцатилетнюю
# предысторию одной технологии с пятнадцатилетней другой — та же болезнь, что была
# у пулового фона в growth. Шесть лет против шести сравнимы, и все три источника
# в 2014 году уже работали.
PREV6_START: Final[date] = date(2014, 9, 1)

# Окна счётчиков: предыдущая шестилетка плюс шесть годовых окон периода сбора.
# Корпусные итоги берутся только по годовым (YEAR_WINDOWS): share_prev6 —
# отношение чисел из одних и тех же источников, нормировать его не на что.
COUNTER_WINDOWS: Final[dict[str, tuple[date, date]]] = {
    "prev6": (PREV6_START, COLLECTION_START),
    **YEAR_WINDOWS,
}
ALLOWED_COUNTER_WINDOWS: Final[frozenset[str]] = frozenset(COUNTER_WINDOWS)

# Из каких окон складывается каждое рабочее. Складывает model/counters.py.
# all — только период сбора, pre2020 в него не входит и живёт отдельным признаком.
AGGREGATE_WINDOWS: Final[dict[str, tuple[str, ...]]] = {
    "all": ("2020", "2021", "2022", "2023", "2024", "2025"),
    "now": ("2025",),
    "before": ("2023",),
    "recent24": ("2024", "2025"),
    "prev6": ("prev6",),
}

# Имена окон, допустимые у корпусного итога источника. Поиск №1 просит before и now,
# поиск №2 — годовые; окно all остаётся именем сложенного итога за весь период.
# prev6 сюда не входит: корпусный итог за него не считается.
ALLOWED_WINDOWS: Final[frozenset[str]] = (
    frozenset({"before", "now", "all"}) | frozenset(YEAR_WINDOWS)
)


def _windows_version(windows: dict[str, tuple[date, date]]) -> str:
    """Отпечаток набора окон: те же даты — та же версия, сдвинули границу — другая."""
    payload = json.dumps(
        {name: [str(start), str(end)] for name, (start, end) in windows.items()},
        sort_keys=True,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:12]


# Версия набора окон. В ключ кэша не входит: окно закодировано в самой строке
# счётчика колонкой period, и дублировать его в отпечатке терминов нельзя — тогда
# добавление одного окна выбрасывало бы все ранее собранные. Остаётся отметкой
# в метаданных прогона: по ней видно, с какими границами собраны числа.
COUNTER_WINDOWS_VERSION: Final[str] = _windows_version(COUNTER_WINDOWS)

# Фильтр типа записи для OpenAlex: в 2026 ретро-индексация книг раздула корпус окна now
# втрое против before. Применяется и к счётчику технологии, и к итогам источника —
# иначе числитель и знаменатель считаются по разным множествам (pipeline.md 0.2).
OPENALEX_TYPE_FILTER: Final[str] = "article"

DEFAULT_RECENT_DAYS: Final[int] = 180
DEFAULT_MAX_WORKERS: Final[int] = 8
# Поиск №1 (search_recent): потолок документов на один подзапрос в одном источнике.
DEFAULT_RECENT_DOCS_PER_SUBQUERY: Final[int] = 25

# Корпусные итоги живут сутки: за день корпус источника меняется незначительно,
# но недоступный вчера источник должен получить новую попытку.
SOURCE_TOTALS_TTL_HOURS: Final[int] = 24


def inclusive_end(exclusive_end: date) -> date:
    """Last UTC calendar day included in [start, exclusive_end)."""
    return exclusive_end - timedelta(days=1)
