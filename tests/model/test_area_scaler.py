"""Нормировщик по областям: что он делает с обучением, с новой областью и с вырождением."""
import numpy as np
import pandas as pd
import pytest

from model.area_scaler import AreaScaler, centre_scale

COLUMNS = ["a", "b"]


def _frame(areas: list[str], a: list[float], b: list[float]) -> pd.DataFrame:
    """Маленькая таблица: две колонки признаков и область."""
    return pd.DataFrame({"a": a, "b": b, "area": areas})


def _train() -> pd.DataFrame:
    """Две области с заведомо разным масштабом: у Y значения в десять раз больше."""
    return _frame(["X"] * 4 + ["Y"] * 4,
                  [1.0, 2.0, 3.0, 4.0, 10.0, 20.0, 30.0, 40.0],
                  [0.0, 1.0, 2.0, 3.0, 100.0, 200.0, 300.0, 400.0])


def test_area_z_centres_each_area_separately() -> None:
    """После нормировки внутри области у каждой области среднее 0 и sd 1."""
    frame = _train()
    result = AreaScaler(COLUMNS, "area_z").fit_transform(frame)
    for rows in ([0, 1, 2, 3], [4, 5, 6, 7]):
        assert result[rows].mean(axis=0) == pytest.approx(0.0, abs=1e-12)
        assert result[rows].std(axis=0) == pytest.approx(1.0, abs=1e-12)


def test_global_mode_ignores_area() -> None:
    """При глобальной нормировке среднее 0 у всей таблицы, а не у каждой области."""
    result = AreaScaler(COLUMNS, "global").fit_transform(_train())
    assert result.mean(axis=0) == pytest.approx(0.0, abs=1e-12)
    assert result[[0, 1, 2, 3]].mean(axis=0) != pytest.approx(0.0, abs=1e-6)


def test_unknown_area_falls_back_to_overall() -> None:
    """Область, которой не было в обучении, нормируется общим центром и масштабом."""
    scaler = AreaScaler(COLUMNS, "area_z").fit(_train())
    new = _frame(["Z"] * 4, [5.0, 6.0, 7.0, 8.0], [5.0, 6.0, 7.0, 8.0])
    centre, scale = centre_scale(_train()[COLUMNS], "area_z")
    assert scaler.transform(new) == pytest.approx((new[COLUMNS].to_numpy() - centre) / scale)


def test_transductive_normalises_new_area_by_itself() -> None:
    """С transductive=True новая область центрируется по собственным строкам."""
    scaler = AreaScaler(COLUMNS, "area_z", transductive=True).fit(_train())
    new = _frame(["Z"] * 4, [5.0, 6.0, 7.0, 8.0], [5.0, 6.0, 7.0, 8.0])
    result = scaler.transform(new)
    assert result.mean(axis=0) == pytest.approx(0.0, abs=1e-12)
    assert result.std(axis=0) == pytest.approx(1.0, abs=1e-12)


def test_robust_mode_puts_median_at_zero() -> None:
    """Робастный вариант вычитает медиану: у медианной строки получается 0."""
    frame = _frame(["X"] * 5, [1.0, 2.0, 3.0, 4.0, 100.0], [1.0, 2.0, 3.0, 4.0, 5.0])
    result = AreaScaler(COLUMNS, "area_robust").fit_transform(frame)
    assert float(np.median(result[:, 0])) == pytest.approx(0.0, abs=1e-12)
    assert np.isfinite(result).all()


def test_constant_column_does_not_explode() -> None:
    """Нулевой разброс не должен давать деления на ноль."""
    frame = _frame(["X"] * 4, [7.0, 7.0, 7.0, 7.0], [1.0, 2.0, 3.0, 4.0])
    for mode in ("global", "area_z", "area_robust"):
        result = AreaScaler(COLUMNS, mode).fit_transform(frame)
        assert np.isfinite(result).all()
        assert result[:, 0] == pytest.approx(0.0, abs=1e-12)


def test_small_area_is_not_remembered() -> None:
    """Область из двух строк слишком мала: её центр и масштаб не запоминаются."""
    frame = _frame(["X"] * 4 + ["Y"] * 2, [1.0, 2.0, 3.0, 4.0, 9.0, 9.5],
                   [1.0, 2.0, 3.0, 4.0, 9.0, 9.5])
    scaler = AreaScaler(COLUMNS, "area_z").fit(frame)
    assert set(scaler.by_area_) == {"X"}


def test_missing_values_do_not_leak_into_statistics() -> None:
    """Пропуски исключаются из центра и масштаба, а не считаются нулями."""
    frame = _frame(["X"] * 4, [1.0, 2.0, 3.0, np.nan], [1.0, 2.0, 3.0, 4.0])
    centre, scale = centre_scale(frame[COLUMNS], "area_z")
    assert centre[0] == pytest.approx(2.0)
    assert np.isfinite(scale).all()
