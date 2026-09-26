"""Признаки кандидата обязаны совпадать с обучающей таблицей: один путь расчёта.

Главный тест здесь — test_matches_training_table_on_every_technology. Он ловит
расхождение путей: если режим запроса начнёт считать хоть немного иначе, чем
считалась таблица, на которой обучалась модель, тест упадёт.
"""
import json
import math

import numpy as np
import pandas as pd
import pytest

from model.candidate import (candidate_features, counters_by_window,
                             features_from_counters)
from model.config import (FEATURES, FEATURES_S2A2, MODEL_VERSION, NEWS_SHARE,
                          ROSPATENT_DATASETS)
from model.corpus import load_training_patents
from model.first_mention import TECHNOLOGIES
from model.training_table import training_table
from tests.data_required import needs_counters, needs_training

TOLERANCE = 1e-9


@pytest.fixture(scope="module")
def technologies() -> pd.DataFrame:
    return pd.read_csv(TECHNOLOGIES, encoding="utf-8-sig")


@pytest.fixture(scope="module")
def stored() -> pd.DataFrame:
    frame, _ = training_table()
    return frame.set_index("tech_id")


def _call(row: pd.Series, **options) -> dict:
    """Вызов как в режиме запроса: термины те же, что уходили в источники."""
    return candidate_features(str(row["name_en"]),
                              terms=json.loads(row["terms"]),
                              context_terms=json.loads(row["context_terms"]),
                              area=str(row["area"]), **options)


@needs_training
@needs_counters
@pytest.mark.parametrize("version,columns", [(MODEL_VERSION, FEATURES),
                                             ("s2a2-v1", FEATURES_S2A2)])
def test_matches_training_table_on_every_technology(technologies, stored, version,
                                                    columns) -> None:
    """Для каждой технологии выборки функция даёт ровно те же шесть чисел своей версии.

    n_pat у s2a2-v1 — из снимка обучения, как при сборке обучающей таблицы.
    """
    patents = load_training_patents(ROSPATENT_DATASETS).set_index("tech_id")["n_pat"]
    checked, problems = 0, []
    for _, row in technologies.iterrows():
        tech_id = str(row["tech_id"])
        if tech_id not in stored.index:
            continue
        answer = _call(row, version=version, n_pat=int(patents.loc[tech_id]))
        if not answer["complete"]:
            problems.append(f"{tech_id}: счётчиков не нашлось")
            continue
        checked += 1
        assert list(answer["features"]) == list(columns)
        for name in columns:
            got, want = answer["features"][name], float(stored.loc[tech_id, name])
            if math.isnan(want) and math.isnan(got):
                continue
            if math.isnan(want) != math.isnan(got) or abs(got - want) > TOLERANCE:
                problems.append(f"{tech_id}.{name}: {got!r} против {want!r}")
    assert not problems, problems[:10]
    assert checked >= 156, f"сверено только {checked} технологий"


def test_feature_order_matches_the_model(technologies) -> None:
    """По умолчанию — боевая s2a2-v1: порядок ключей совпадает с её коэффициентами."""
    answer = _call(technologies.iloc[0])
    assert list(answer["features"]) == list(FEATURES_S2A2)


def test_share_patent_without_n_pat_is_a_gap(technologies) -> None:
    """Нет n_pat (сбой Роспатента или он выключен) — share_patent NaN, а не ноль."""
    answer = _call(technologies.iloc[0])
    assert math.isnan(answer["features"]["share_patent"])


@needs_counters
def test_share_patent_zero_patents_is_zero(technologies) -> None:
    """Ответ total = 0 при ненулевой науке — ровно ноль."""
    answer = _call(technologies.iloc[0], n_pat=0)
    assert answer["features"]["share_patent"] == 0.0


@needs_counters
def test_counters_and_sources_come_back(technologies) -> None:
    """Вместе с числами возвращаются счётчики по окнам и список источников."""
    answer = _call(technologies.iloc[0])
    assert answer["sources"] and set(answer["sources"]) <= {"openalex", "arxiv", "techcrunch"}
    assert answer["counters"], "счётчики по окнам пусты"
    for window, by_source in answer["counters"].items():
        assert set(by_source) <= set(answer["sources"])
        assert all(isinstance(value, int) for value in by_source.values())


def test_windows_are_the_seven_collection_windows(technologies) -> None:
    """Окна те же семь, по которым опрашиваются источники."""
    answer = _call(technologies.iloc[0])
    assert set(answer["counters"]) <= {"prev6", "2020", "2021", "2022", "2023", "2024", "2025"}


def test_unknown_technology_does_not_raise(technologies) -> None:
    """Технология без счётчиков возвращает пропуски и complete=False, а не исключение."""
    answer = candidate_features("технологии с таким названием точно нет в кэше")
    assert answer["complete"] is False
    assert answer["sources"] == [] and answer["counters"] == {}
    assert all(math.isnan(value) for value in answer["features"].values())


def test_missing_arxiv_gives_a_gap_not_zero() -> None:
    """Нет ни одного препринта — возраст неизвестен, а не равен нулю."""
    counters = pd.DataFrame(
        [{"source": source, "window": window, "n": 0 if source == "arxiv" else 5}
         for source in ("openalex", "arxiv", "techcrunch")
         for window in ("prev6", "2020", "2021", "2022", "2023", "2024", "2025")])
    got = features_from_counters(counters, version=MODEL_VERSION)
    assert math.isnan(got["age_first_arxiv"])
    assert not math.isnan(got["volume"])


def test_counters_by_window_keeps_collection_order() -> None:
    """Окна выдаются в порядке сбора: prev6 первым, дальше годы."""
    counters = pd.DataFrame([{"source": "arxiv", "window": w, "n": 1}
                             for w in ("2022", "prev6", "2020")])
    assert list(counters_by_window(counters)) == ["prev6", "2020", "2022"]


def test_news_share_uses_the_canonical_name(technologies) -> None:
    """Доля новостей возвращается под новым именем, не под share_market."""
    answer = _call(technologies.iloc[0])
    assert NEWS_SHARE in answer["features"] and "share_market" not in answer["features"]
