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


# Патентный признак s2a2-v1: числа патентов и публикаций, правильные формы слов.
S2A2 = {"share_news_wordmatch": 0.1, "recency": 0.47, "share_patent": 0.02,
        "age_first_arxiv": 9.0, "share_prev6": 0.62, "growth_research": 0.1}


@pytest.mark.parametrize("number, expected", [
    (0, "0 патентов"), (1, "1 патент"), (3, "3 патента"), (11, "11 патентов"), (21, "21 патент"),
    (690, "690 патентов"), (1223, "1 223 патента"), (112, "112 патентов")])
def test_plural_forms_and_thousands(number, expected) -> None:
    assert reasons.plural(number, "патент", "патента", "патентов") == expected


def test_share_patent_for_signal_names_patents_and_publications() -> None:
    contributions = {key: (0.9 if key == "share_patent" else 0.0) for key in S2A2}
    texts = reasons.explanation_top(S2A2, contributions, COUNTERS, n_pat=3)
    assert texts == ["Патентов мало относительно научных работ: 3 патента на 155 публикаций"]


def test_share_patent_against_signal_names_patents_and_publications() -> None:
    contributions = {key: (-0.9 if key == "share_patent" else 0.0) for key in S2A2}
    counters = {"2024": {"openalex": 1200, "arxiv": 1}}
    text = reasons.reason_below(S2A2, contributions, counters, n_pat=690)
    assert text == "Технология активно патентуется: 690 патентов на 1 201 публикацию"


def test_share_patent_without_count_is_not_used() -> None:
    """Сбой Роспатента: n_pat нет, число патентов не выдумывается — берётся следующий признак."""
    features = {**S2A2, "share_patent": float("nan")}
    contributions = {key: (-0.9 if key == "share_patent" else 0.0) for key in S2A2}
    contributions["recency"] = -0.1
    text = reasons.reason_below(features, contributions, COUNTERS, n_pat=None)
    assert text.startswith("Интерес не нарастает")


def test_every_s2a2_feature_has_templates_and_volume_is_not_among_them() -> None:
    """У каждого признака s2a2-v1 есть шаблон «за» и «против»; volume в наборе нет."""
    from model.config import FEATURES_S2A2
    assert "volume" not in FEATURES_S2A2
    assert set(FEATURES_S2A2) <= set(reasons.POSITIVE) and set(FEATURES_S2A2) <= set(reasons.NEGATIVE)
