"""Подбор гиперпараметров: выбор не должен видеть отложенные строки."""
import numpy as np
import pandas as pd
import pytest

from model.config import FEATURES, N_REPEATS, N_SPLITS
from scripts.tuning import (COMBOS, CURRENT, GRID_C, GRID_WEIGHT, _area_splits,
                          _grouped_splits, choose_inside, nested_cv, nested_loao,
                          pipeline_for)


@pytest.fixture(scope="module")
def frame() -> pd.DataFrame:
    """Синтетика с тем же устройством, что обучающая таблица: области и группы."""
    rng = np.random.default_rng(42)
    rows = 120
    table = pd.DataFrame({name: rng.normal(size=rows) for name in FEATURES})
    table.loc[:, "area"] = [f"A{i % 4}" for i in range(rows)]
    table.loc[:, "label"] = (table[FEATURES[0]] + rng.normal(scale=0.5, size=rows) > 0
                             ).astype(int)
    table.loc[:, "group"] = [f"g{i // 2}" for i in range(rows)]
    return table


def test_grid_is_the_one_that_was_asked_for() -> None:
    """Сетка C и class_weight зафиксирована и перебирается целиком."""
    assert GRID_C == [0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0]
    assert GRID_WEIGHT == ["balanced", None]
    assert len(COMBOS) == 14


def test_c_actually_reaches_the_model() -> None:
    """Параметр доходит до регрессии, а не теряется по дороге."""
    model = pipeline_for(0.03, None)
    assert model.named_steps["logistic"].C == 0.03
    assert model.named_steps["logistic"].class_weight is None


def test_inner_splits_never_share_a_group(frame: pd.DataFrame) -> None:
    """Внутренние фолды режутся по группам: одна технология не попадёт в обе части."""
    rows = np.arange(len(frame))
    for fit_part, hold in _grouped_splits(frame, rows, 42):
        left = set(frame["group"].iloc[rows[fit_part]])
        right = set(frame["group"].iloc[rows[hold]])
        assert not left & right


def test_inner_selection_uses_only_the_given_rows(frame: pd.DataFrame) -> None:
    """Выбор считается по переданным строкам: остальные на него не влияют.

    Тот же вызов на той же обучающей части обязан дать тот же ответ, даже если
    отложенные строки в таблице заменить на мусор.
    """
    rows = np.arange(0, 90)
    splits = _grouped_splits(frame, rows, 42)
    first = choose_inside(frame, rows, "accuracy", splits)
    spoiled = frame.copy()
    spoiled.loc[90:, FEATURES] = 999.0
    spoiled.loc[90:, "label"] = 1 - spoiled.loc[90:, "label"]
    second = choose_inside(spoiled, rows, "accuracy", splits)
    assert first == second


def test_area_splits_hold_out_whole_areas(frame: pd.DataFrame) -> None:
    """Внутренний LOAO откладывает область целиком."""
    rows = np.flatnonzero((frame["area"] != "A0").to_numpy())
    for fit_part, hold in _area_splits(frame, rows):
        left = set(frame["area"].iloc[rows[fit_part]])
        right = set(frame["area"].iloc[rows[hold]])
        assert not left & right
        assert len(right) == 1


def test_nested_cv_reports_one_row_per_repeat(frame: pd.DataFrame) -> None:
    """Метрики отчитываются по повторам, а матрица ошибок сходится с числом строк."""
    table, picks = nested_cv(frame, "accuracy", combos=[CURRENT])
    assert len(table) == N_REPEATS
    assert len(picks) == N_REPEATS * N_SPLITS
    counts = table[["TP", "FP", "FN", "TN"]].sum(axis=1)
    assert (counts == len(frame)).all()


def test_nested_loao_never_trains_on_the_held_out_area(frame: pd.DataFrame) -> None:
    """Каждая область проверяется моделью, обученной без неё."""
    table = nested_loao(frame, "accuracy", combos=[CURRENT])
    assert sorted(table["область"]) == sorted(frame["area"].unique())
    assert (table[["TP", "FP", "FN", "TN"]].sum(axis=1)
            == frame["area"].value_counts().sort_index().to_numpy()).all()


def test_searching_the_grid_is_not_worse_than_a_single_point(frame: pd.DataFrame) -> None:
    """Перебор по сетке на обучающей части находит не худший внутренний результат."""
    rows = np.arange(len(frame))
    splits = _grouped_splits(frame, rows, 42)
    _, _, fixed = choose_inside(frame, rows, "accuracy", splits, combos=[CURRENT])
    _, _, searched = choose_inside(frame, rows, "accuracy", splits)
    assert searched >= fixed
