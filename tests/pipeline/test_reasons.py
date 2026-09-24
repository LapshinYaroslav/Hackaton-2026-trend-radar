"""Шаблоны причин и объяснений: по тесту на каждый."""
import math

import pytest

from pipeline import reasons

COUNTERS = {"2020": {"openalex": 100, "arxiv": 20}, "2025": {"openalex": 30, "arxiv": 5, "techcrunch": 9}}
FEATURES = {"volume": 5.0, "growth_research": 0.1, "recency": 0.47, "share_news_wordmatch": 0.1,
            "share_prev6": 0.62, "age_first_arxiv": 9.0}


def only(name: str, value: float) -> dict:
    """Вклады, где выделяется один признак."""
    return {key: (value if key == name else 0.0) for key in FEATURES}


def test_n_research_counts_openalex_and_arxiv_in_period() -> None:
    assert reasons.n_research(COUNTERS) == 155


@pytest.mark.parametrize("name, expected", [
    ("share_news_wordmatch", "Большой объём научной литературы: 155 публикаций — технология уже хорошо изучена"),
    ("recency", "Интерес не нарастает: за последние два года 47% всех упоминаний"),
    ("age_first_arxiv", "Термин давно в науке: первый препринт 9 лет назад"),
    ("share_prev6", "Значительная часть публикаций вышла до 2020 года (62%)"),
    ("volume", reasons.GENERIC_BELOW),
    ("growth_research", reasons.GENERIC_BELOW),
])
def test_reason_below_each_feature(name, expected) -> None:
    assert reasons.reason_below(FEATURES, only(name, -1.0), COUNTERS) == expected


@pytest.mark.parametrize("name, expected", [
    ("share_news_wordmatch", "Про технологию пока мало научных публикаций: 155"),
    ("recency", "Упоминания свежие: 47% за последние два года"),
    ("age_first_arxiv", "Молодой термин: первый препринт 9 лет назад"),
    ("share_prev6", "Почти нет публикаций до 2020 года"),
    ("volume", reasons.GENERIC_ABOVE),
    ("growth_research", reasons.GENERIC_ABOVE),
])
def test_explanation_top_each_feature(name, expected) -> None:
    assert reasons.explanation_top(FEATURES, only(name, 1.0), COUNTERS) == [expected]


def test_reason_below_takes_most_negative() -> None:
    contributions = {**only("recency", -0.4), "share_prev6": -0.9}
    assert reasons.reason_below(FEATURES, contributions, COUNTERS).startswith("Значительная часть")


def test_explanation_top_two_largest_positive() -> None:
    contributions = {**only("recency", 0.5), "share_news_wordmatch": 2.0, "share_prev6": 0.1}
    got = reasons.explanation_top(FEATURES, contributions, COUNTERS)
    assert got == ["Про технологию пока мало научных публикаций: 155",
                   "Упоминания свежие: 47% за последние два года"]


def test_missing_value_is_not_put_into_text() -> None:
    """Возраст пропущен (модель подставила медиану): в текст он не идёт, берётся следующий."""
    features = {**FEATURES, "age_first_arxiv": math.nan}
    contributions = {**only("age_first_arxiv", -2.0), "recency": -0.3}
    assert reasons.reason_below(features, contributions, COUNTERS).startswith("Интерес не нарастает")


@pytest.mark.parametrize("reason", ["no_trace", "trace_unknown", "cap", "bad_name", "no_counters", "beyond_top"])
def test_special_reasons_have_text(reason) -> None:
    assert reasons.SPECIAL[reason]
