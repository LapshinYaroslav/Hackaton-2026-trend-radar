"""Признаки по счётчикам (pipeline.md 0.2).

Переписано с документов на счётчики. Соответствие старым проверкам задачи 001:
  growth_example_a, growth_example_b, volume_140, zero_documents_in_before_window,
  no_documents, source_without_totals, result_key_order  — сохранены как были, изменён
    только вход: вместо строк-документов счётчики по окнам;
  shares_on_mixed_types      — переписана под нормированные доли (см. test_shares_*);
  unknown_source_type_raises — стала unknown_source_raises: типов на входе больше нет,
    но отнесение источника к научным или рыночным проверяется так же строго;
  now_window_boundaries, before_window_boundaries, cutoff_date_excluded_everywhere,
  period_start_boundary      — границы окон теперь задаёт сборщик, а не эта функция.
    Проверка переехала в tests/collector/test_counters.py::test_counter_windows_match_pipeline,
    где даты всех четырёх окон сверены с документом;
  missing_date_is_ignored    — проверка потеряла предмет: у счётчика нет дат отдельных
    документов, источник не может вернуть запись без даты внутри окна с датами.
"""
import math

import pandas as pd
import pytest

from model.features import (FEATURE_NAMES, compute_features, growth_by_source,
                            share_patent)


def _counters(rows: list[tuple[str, str, int]]) -> pd.DataFrame:
    return pd.DataFrame(rows, columns=["source", "window", "n"])


def _totals(rows: list[tuple[str, str, int]]) -> pd.DataFrame:
    return pd.DataFrame(rows, columns=["source", "window", "n_total"])


def test_growth_example_a():
    counters = _counters([("openalex", "all", 72), ("openalex", "before", 12),
                          ("openalex", "now", 60)])
    totals = _totals([("openalex", "before", 1000), ("openalex", "now", 1500)])

    features = compute_features(counters, totals, expected_sources=None)

    assert features["growth"] == pytest.approx(1.1405, abs=1e-3)


def test_growth_example_b():
    counters = _counters([("openalex", "all", 1600), ("openalex", "before", 700),
                          ("openalex", "now", 900)])
    totals = _totals([("openalex", "before", 1000), ("openalex", "now", 1500)])

    features = compute_features(counters, totals, expected_sources=None)

    assert features["growth"] == pytest.approx(-0.1545, abs=1e-3)


def test_volume_140_documents():
    counters = _counters([("openalex", "all", 140)])

    features = compute_features(counters, _totals([]), expected_sources=None)

    assert features["volume"] == pytest.approx(4.9488, abs=1e-3)


def test_zero_documents_in_before_window():
    """Пустое окно before не роняет growth: сглаживание +1 в числителе и знаменателе."""
    counters = _counters([("openalex", "all", 5), ("openalex", "now", 5),
                          ("openalex", "before", 0)])
    totals = _totals([("openalex", "before", 100), ("openalex", "now", 100)])

    features = compute_features(counters, totals, expected_sources=None)

    assert math.isfinite(features["growth"])


def test_no_documents():
    counters = _counters([("openalex", "all", 0), ("openalex", "now", 0),
                          ("openalex", "before", 0), ("openalex", "recent24", 0)])

    features = compute_features(counters, _totals([]), expected_sources=None)

    assert features["volume"] == 0
    assert math.isnan(features["recency"])
    assert math.isnan(features["growth"])
    assert math.isnan(features["share_research"])
    assert math.isnan(features["share_market"])


def test_empty_counters_table():
    """Источник не ответил ни по одному окну — строк нет вовсе, а не нули."""
    features = compute_features(_counters([]), _totals([]), expected_sources=None)

    assert features["volume"] == 0
    assert math.isnan(features["growth"])


def test_recency_share_of_last_24_months():
    counters = _counters([("openalex", "all", 200), ("openalex", "recent24", 50)])

    features = compute_features(counters, _totals([]), expected_sources=None)

    assert features["recency"] == pytest.approx(0.25, abs=1e-9)


def test_shares_are_raw_counts():
    """Доли считаются по сырым числам: сто статей против одной новости дают 100/101.

    Раньше здесь проверялась нормировка r = n / N, и те же данные давали ровно 0.5.
    Нормировка отменена в задаче 6: она вводилась под многотермовый запрос, когда
    OpenAlex отдавал тысячи документов, а после перехода на один термин стала вредна.
    Корпус OpenAlex в 667 раз больше корпуса TechCrunch, и одна новость перевешивала
    сотни статей: у edge to cloud routing 7 статей против 56 новостей давали долю науки
    не 0.111, а 0.0002. Измерено на 26 технологиях: вырожденных значений 13 из 26
    против 9 у сырых долей.
    """
    counters = _counters([("openalex", "all", 100), ("techcrunch", "all", 1)])
    totals = _totals([("openalex", "all", 1_000_000), ("techcrunch", "all", 10_000)])

    features = compute_features(counters, totals, expected_sources=None)

    assert features["share_research"] == pytest.approx(100 / 101, abs=1e-9)
    assert features["share_market"] == pytest.approx(1 / 101, abs=1e-9)


def test_shares_sum_to_one_across_three_sources():
    counters = _counters([("openalex", "all", 100), ("arxiv", "all", 50),
                          ("techcrunch", "all", 1)])
    totals = _totals([("openalex", "all", 1_000_000), ("arxiv", "all", 500_000),
                      ("techcrunch", "all", 10_000)])

    features = compute_features(counters, totals, expected_sources=None)

    assert features["share_research"] + features["share_market"] == pytest.approx(1.0, abs=1e-9)
    assert features["share_research"] == pytest.approx(150 / 151, abs=1e-9)


def test_zero_market_documents_give_share_research_one():
    """Ноль новостей — это не пропуск: доля науки ровно 1, доля рынка ровно 0, не NaN.

    Таких технологий будет много: TechCrunch пишет далеко не про каждую.
    """
    counters = _counters([("openalex", "all", 100), ("techcrunch", "all", 0)])
    totals = _totals([("openalex", "all", 1_000_000), ("techcrunch", "all", 10_000)])

    features = compute_features(counters, totals, expected_sources=None)

    assert features["share_research"] == 1.0
    assert features["share_market"] == 0.0
    assert not math.isnan(features["share_market"])


def test_zero_research_documents_give_share_market_one():
    counters = _counters([("openalex", "all", 0), ("techcrunch", "all", 3)])
    totals = _totals([("openalex", "all", 1_000_000), ("techcrunch", "all", 10_000)])

    features = compute_features(counters, totals, expected_sources=None)

    assert features["share_market"] == 1.0
    assert features["share_research"] == 0.0


def test_unknown_source_raises():
    """Источник, не отнесённый ни к научным, ни к рыночным, — ошибка, а не тихий пропуск."""
    counters = _counters([("patentsview", "all", 10)])

    with pytest.raises(ValueError, match="patentsview"):
        compute_features(counters, _totals([]), expected_sources=None)


def test_source_without_growth_totals():
    """Нет итогов за окна роста — growth пустой, остальные признаки считаются."""
    counters = _counters([("openalex", "all", 10), ("openalex", "now", 4),
                          ("openalex", "before", 2)])
    totals = _totals([("openalex", "all", 1_000_000)])

    features = compute_features(counters, totals, expected_sources=None)

    assert math.isnan(features["growth"])
    assert features["volume"] == pytest.approx(math.log(11), abs=1e-3)
    assert features["share_research"] == pytest.approx(1.0, abs=1e-3)


def test_source_without_corpus_total_still_counts_in_shares():
    """Корпусные итоги на доли больше не влияют: они нужны только growth.

    До задачи 6 источник без корпусного итога выпадал из долей целиком, и здесь
    share_market равнялся единице. Теперь считаются сами документы.
    """
    counters = _counters([("openalex", "all", 10), ("techcrunch", "all", 5)])
    totals = _totals([("techcrunch", "all", 10_000)])

    features = compute_features(counters, totals, expected_sources=None)

    assert features["share_research"] == pytest.approx(10 / 15, abs=1e-9)
    assert features["volume"] == pytest.approx(math.log(16), abs=1e-3)


def test_growth_ignores_source_without_both_windows():
    """Источник с итогом за одно окно из двух в поправку на фон не входит."""
    counters = _counters([("openalex", "all", 72), ("openalex", "before", 12),
                          ("openalex", "now", 60), ("techcrunch", "all", 2),
                          ("techcrunch", "before", 1), ("techcrunch", "now", 1)])
    totals = _totals([("openalex", "before", 1000), ("openalex", "now", 1500),
                      ("techcrunch", "now", 500)])

    features = compute_features(counters, totals, expected_sources=None)

    assert features["growth"] == pytest.approx(1.1405, abs=1e-3)


def test_zero_corpus_total_gives_no_growth():
    """Нулевой корпус — это «источник не ответил», а не «документов не было»."""
    counters = _counters([("openalex", "all", 10), ("openalex", "now", 5),
                          ("openalex", "before", 5)])
    totals = _totals([("openalex", "before", 0), ("openalex", "now", 100)])

    features = compute_features(counters, totals, expected_sources=None)

    assert math.isnan(features["growth"])


def test_result_key_order():
    features = compute_features(_counters([]), _totals([]), expected_sources=None)

    assert list(features.keys()) == FEATURE_NAMES


def test_share_prev6_counts_both_halves():
    """Доля старых работ: 1 из 4, суммируется по источникам."""
    counters = _counters([("openalex", "prev6", 30), ("arxiv", "prev6", 10),
                          ("openalex", "all", 90), ("arxiv", "all", 30)])

    features = compute_features(counters, _totals([]), expected_sources=None)

    assert features["share_prev6"] == pytest.approx(0.25, abs=1e-9)


def test_share_prev6_is_nan_without_documents_anywhere():
    """Ни до 2020, ни после — считать нечего, это пропуск, а не ноль."""
    counters = _counters([("openalex", "prev6", 0), ("openalex", "all", 0)])

    features = compute_features(counters, _totals([]), expected_sources=None)

    assert math.isnan(features["share_prev6"])


def test_dead_topic_has_share_prev6_one():
    """Писали до 2020 и перестали: доля ровно 1, и это ответ, а не пропуск.

    Именно этот случай отличает угасшую тему от зарождающейся: у обеих volume мал.
    """
    counters = _counters([("openalex", "prev6", 50), ("openalex", "all", 0)])

    features = compute_features(counters, _totals([]), expected_sources=None)

    assert features["volume"] == 0
    assert features["share_prev6"] == 1.0
    assert math.isnan(features["growth"])


def test_young_topic_has_share_prev6_zero():
    counters = _counters([("openalex", "prev6", 0), ("openalex", "all", 40)])
    totals = _totals([("openalex", "all", 1_000_000)])

    features = compute_features(counters, totals, expected_sources=None)

    assert features["share_prev6"] == 0.0


def test_prev6_does_not_leak_into_volume():
    """Старые работы в объём периода сбора не входят: volume считается по all."""
    without = compute_features(_counters([("openalex", "all", 40)]), _totals([]),
                               expected_sources=None)
    with_old = compute_features(
        _counters([("openalex", "all", 40), ("openalex", "prev6", 5000)]), _totals([]),
        expected_sources=None)

    assert with_old["volume"] == without["volume"]


def test_growth_is_weighted_average_over_sources():
    """Приёмочный тест на живых числах neuromorphic computing (20.09.2026).

    Три источника с разным фоном: openalex +0.3395, arxiv +0.3575, techcrunch -0.3594.
    Взвешенный рост почти совпал с прежним пуловым (+0.0355) — на нормальной технологии
    правка ничего не меняет, меняется поведение в вырожденных случаях.
    """
    counters = _counters([("openalex", "now", 1206), ("openalex", "before", 796),
                          ("arxiv", "now", 83), ("arxiv", "before", 90),
                          ("techcrunch", "now", 2), ("techcrunch", "before", 1),
                          ("openalex", "all", 4463)])
    totals = _totals([("openalex", "now", 8_231_350), ("openalex", "before", 5_861_775),
                      ("arxiv", "now", 333_181), ("arxiv", "before", 233_029),
                      ("techcrunch", "now", 6_002), ("techcrunch", "before", 8_598)])

    detail = growth_by_source(counters, totals).set_index("source")

    assert detail.loc["openalex", "g"] == pytest.approx(0.0755, abs=1e-4)
    assert detail.loc["arxiv", "g"] == pytest.approx(-0.4376, abs=1e-4)
    assert detail.loc["techcrunch", "g"] == pytest.approx(0.7649, abs=1e-4)
    assert list(detail["w"]) == [173.0, 2002.0, 3.0]
    assert compute_features(counters, totals, expected_sources=None)["growth"] == pytest.approx(0.0357, abs=1e-4)


def test_growth_uses_own_background_of_each_source():
    """Технологии, живущей только в TechCrunch, фон OpenAlex не применяется.

    Числитель одинаков, а ответ разный: с пуловым знаменателем рост занижался на
    величину расхождения фонов, около 0.70.
    """
    counters = _counters([("techcrunch", "now", 20), ("techcrunch", "before", 10),
                          ("techcrunch", "all", 30)])
    totals = _totals([("openalex", "now", 8_231_350), ("openalex", "before", 5_861_775),
                      ("techcrunch", "now", 6_002), ("techcrunch", "before", 8_598)])

    features = compute_features(counters, totals, expected_sources=None)
    own = math.log(21 / 11) - math.log(6002 / 8598)
    pooled = math.log(21 / 11) - math.log((8_231_350 + 6_002) / (5_861_775 + 8_598))

    assert features["growth"] == pytest.approx(own, abs=1e-9)
    assert own - pooled == pytest.approx(0.699, abs=1e-3)


def test_big_source_outweighs_a_single_news_item():
    """Вес — документы, а не источники: три новости не перевешивают две тысячи статей."""
    counters = _counters([("openalex", "now", 1000), ("openalex", "before", 1000),
                          ("techcrunch", "now", 2), ("techcrunch", "before", 0),
                          ("openalex", "all", 2000)])
    totals = _totals([("openalex", "now", 1_000_000), ("openalex", "before", 1_000_000),
                      ("techcrunch", "now", 10_000), ("techcrunch", "before", 10_000)])

    detail = growth_by_source(counters, totals).set_index("source")
    growth = compute_features(counters, totals, expected_sources=None)["growth"]

    assert detail.loc["techcrunch", "g"] == pytest.approx(math.log(3), abs=1e-9)
    assert growth == pytest.approx(math.log(3) * 2 / 2002, abs=1e-6)
    assert growth < 0.002


def test_source_without_documents_in_growth_windows_still_gives_background():
    """Документов в окнах роста нет ни у кого: ответ — минус фон, а не пропуск."""
    counters = _counters([("openalex", "all", 5), ("openalex", "now", 0),
                          ("openalex", "before", 0)])
    totals = _totals([("openalex", "now", 1500), ("openalex", "before", 1000)])

    features = compute_features(counters, totals, expected_sources=None)

    assert features["growth"] == pytest.approx(-math.log(1.5), abs=1e-9)


def test_incomplete_coverage_refuses_to_compute():
    """Источник не ответил по части окон — признаки не считаются, а не считаются неполными.

    На этапе 2 arXiv под троттлингом ответил на 25 запросов из 182, и n_sources_growth
    показал AUC 0.750 — признак мерил, каким технологиям повезло получить ответ API.
    На полном покрытии тот же признак даёт 0.500.
    """
    counters = _counters([("openalex", "all", 10), ("openalex", "now", 5),
                          ("openalex", "before", 3), ("openalex", "recent24", 7),
                          ("openalex", "prev6", 2), ("arxiv", "all", 4)])

    with pytest.raises(ValueError, match="неполное покрытие"):
        compute_features(counters, _totals([]), expected_sources={"openalex", "arxiv"})


def test_full_coverage_computes():
    """Все ожидаемые источники ответили по всем рабочим окнам — считаем как обычно."""
    rows = [(source, window, 5) for source in ("openalex", "arxiv")
            for window in ("all", "now", "before", "recent24", "prev6")]

    features = compute_features(_counters(rows), _totals([]),
                                expected_sources={"openalex", "arxiv"})

    assert features["volume"] == pytest.approx(math.log(11), abs=1e-3)


def test_coverage_parameter_is_required():
    """Вызов без expected_sources — ошибка: забыть проверку покрытия нельзя.

    Явный None означает осознанный пропуск — так сделаны тесты формул выше.
    """
    with pytest.raises(TypeError):
        compute_features(_counters([("openalex", "all", 10)]), _totals([]))

    features = compute_features(_counters([("openalex", "all", 10)]), _totals([]),
                                expected_sources=None)
    assert features["volume"] == pytest.approx(math.log(11), abs=1e-3)


def test_share_patent_is_fraction_of_patents_plus_research():
    assert share_patent(30, 90) == pytest.approx(0.25, abs=1e-12)


def test_share_patent_without_patents_is_zero():
    assert share_patent(0, 17) == 0.0


def test_share_patent_only_patents_is_one():
    assert share_patent(5, 0) == 1.0


def test_share_patent_without_any_documents_is_nan():
    assert math.isnan(share_patent(0, 0))
