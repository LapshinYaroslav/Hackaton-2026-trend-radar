"""Шаблоны причин и объяснений: по тесту на каждый."""
import math
import re

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
    ("share_news_wordmatch", "Слабый интерес рынка"),
    ("recency", "Пик внимания позади"),
    ("age_first_arxiv", "Термин давно известен в науке"),
    ("share_prev6", "Технология не новая: много публикаций до 2020 года"),
    ("volume", reasons.GENERIC_BELOW),
    ("growth_research", "Научный интерес не растёт"),
])
def test_reason_below_each_feature(name, expected) -> None:
    assert reasons.reason_below(FEATURES, only(name, -1.0), COUNTERS) == expected


@pytest.mark.parametrize("name", list(reasons.NEGATIVE))
def test_reason_below_has_no_digits(name) -> None:
    """Причина отсева без чисел из данных, в том числе патентная с известным n_pat; год 2020 — часть формулировки."""
    features = {**FEATURES, "share_patent": 0.9}
    contributions = {key: (-1.0 if key == name else 0.0) for key in features}
    text = reasons.reason_below(features, contributions, COUNTERS, n_pat=690)
    assert not any(char.isdigit() for char in text.replace("2020", ""))


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
    assert reasons.reason_below(FEATURES, contributions, COUNTERS) == "Технология не новая: много публикаций до 2020 года"


def test_explanation_top_two_largest_positive() -> None:
    contributions = {**only("recency", 0.5), "share_news_wordmatch": 2.0, "share_prev6": 0.1}
    got = reasons.explanation_top(FEATURES, contributions, COUNTERS)
    assert got == ["Про технологию пока мало научных публикаций: 155",
                   "Упоминания свежие: 47% за последние два года"]


def test_missing_value_is_not_put_into_text() -> None:
    """Возраст пропущен (модель подставила медиану): в текст он не идёт, берётся следующий."""
    features = {**FEATURES, "age_first_arxiv": math.nan}
    contributions = {**only("age_first_arxiv", -2.0), "recency": -0.3}
    assert reasons.reason_below(features, contributions, COUNTERS) == "Пик внимания позади"


@pytest.mark.parametrize("reason", ["no_trace", "trace_unknown", "cap", "time_budget", "bad_name",
                                    "no_counters", "beyond_top"])
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


def test_share_patent_against_signal_without_numbers() -> None:
    contributions = {key: (-0.9 if key == "share_patent" else 0.0) for key in S2A2}
    counters = {"2024": {"openalex": 1200, "arxiv": 1}}
    text = reasons.reason_below(S2A2, contributions, counters, n_pat=690)
    assert text == "Высокая доля патентов относительно научных работ"


def test_share_patent_without_count_is_not_used() -> None:
    """Сбой Роспатента: n_pat нет, число патентов не выдумывается — берётся следующий признак."""
    features = {**S2A2, "share_patent": float("nan")}
    contributions = {key: (-0.9 if key == "share_patent" else 0.0) for key in S2A2}
    contributions["recency"] = -0.1
    text = reasons.reason_below(features, contributions, COUNTERS, n_pat=None)
    assert text == "Пик внимания позади"


def test_every_s2a2_feature_has_templates_and_volume_is_not_among_them() -> None:
    """У каждого признака s2a2-v1 есть шаблон «за» и «против»; volume в наборе нет."""
    from model.config import FEATURES_S2A2
    assert "volume" not in FEATURES_S2A2
    assert set(FEATURES_S2A2) <= set(reasons.POSITIVE) and set(FEATURES_S2A2) <= set(reasons.NEGATIVE)


WHY_FEATURES = {"recency": 1.0, "age_first_arxiv": 1.0, "share_patent": 0.0, "share_prev6": 0.0,
            "share_news_wordmatch": 0.81, "growth_research": 0.3}


def test_why_words_has_no_digits_and_top_three() -> None:
    contributions = {"recency": 2.5, "age_first_arxiv": 0.7, "share_patent": 0.4, "share_prev6": 0.15,
                     "share_news_wordmatch": -1.3, "growth_research": 0.0}
    text = reasons.why_words(WHY_FEATURES, contributions, n_pat=0)
    assert not re.search(r"\d", text)
    assert text == ("Почти всё, что о ней написано, появилось за последние два года. "
                    "Сам термин совсем молодой: первые научные препринты о нём вышли недавно. "
                    "Патентов пока нет: исследования уже идут, а до коммерческого применения дело не дошло.")


@pytest.mark.parametrize("name, value, n_pat, fragment", [
    ("recency", 0.75, None, "Большая часть"),
    ("recency", 0.62, None, "оживился"),
    ("age_first_arxiv", 5.0, None, "несколько лет назад"),
    ("age_first_arxiv", 9.0, None, "сравнительно новый"),
    ("share_patent", 0.03, 7, "Патентов пока мало"),
    ("share_patent", 0.3, 50, "ниже, чем обычно"),
    ("share_prev6", 0.3, None, "после 2020"),
    ("share_news_wordmatch", 0.81, None, "пресса уже обращает"),
    ("growth_research", 1.2, None, "растёт быстрее"),
])
def test_why_words_levels(name, value, n_pat, fragment) -> None:
    assert fragment in reasons.why_words({name: value}, {name: 1.0}, n_pat=n_pat)


def test_why_words_skips_unknown_and_patent_without_count() -> None:
    features = {"recency": None, "share_patent": 0.0, "share_prev6": 0.0}
    text = reasons.why_words(features, {"recency": 3.0, "share_patent": 2.0, "share_prev6": 0.1}, n_pat=None)
    assert text == "До 2020 года о технологии почти не писали."


@pytest.mark.parametrize("contributions", [{}, {"recency": -1.0}, {"volume": 2.0}])
def test_why_words_fallback(contributions) -> None:
    assert reasons.why_words({"recency": 0.5, "volume": 3.0}, contributions) == reasons.WHY_FALLBACK
