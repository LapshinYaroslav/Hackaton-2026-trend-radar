"""Нормировка признаков внутри области. Живёт в model/, потому что её пикает артефакт.

joblib сохраняет класс ссылкой на его модуль. Если бы нормировщик остался в
experiments/, обученная модель не загружалась бы без экспериментального кода.
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

from model.config import MIN_AREA_ROWS


def centre_scale(frame: pd.DataFrame, mode: str = "area_z") -> tuple[np.ndarray, np.ndarray]:
    """Центр и масштаб по столбцам: среднее и sd либо медиана и IQR/1.349.

    Столбец, у которого внутри области все значения пропущены, даёт центр 0 и
    масштаб 1: нормировать нечем. Значения остаются пропусками и заполняются
    дальше по пайплайну медианой обучающей части. numpy предупреждает о пустом
    срезе, поэтому предупреждение здесь глушится намеренно.
    """
    values = frame.to_numpy(dtype=float)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        if mode == "area_robust":
            centre = np.nanmedian(values, axis=0)
            first, third = np.nanpercentile(values, [25, 75], axis=0)
            scale = (third - first) / 1.349
        else:
            centre = np.nanmean(values, axis=0)
            scale = np.nanstd(values, axis=0)
    centre = np.where(np.isfinite(centre), centre, 0.0)
    scale = np.where(np.isfinite(scale) & (scale > 1e-9), scale, 1.0)
    return centre, scale


class AreaScaler(BaseEstimator, TransformerMixin):
    """Нормировка признаков: глобальная или внутри области.

    На вход DataFrame с колонками признаков и колонкой area, на выход массив
    признаков без area. Центр и масштаб считаются только по обучающей части.
    Область, которой в обучении не было, нормируется глобальным центром и
    масштабом; при transductive=True — собственными значениями блока (метки
    при этом не используются, только признаки).
    """

    def __init__(self, columns: list[str], mode: str = "global",
                 transductive: bool = False) -> None:
        self.columns = columns
        self.mode = mode
        self.transductive = transductive

    def fit(self, X: pd.DataFrame, y=None) -> "AreaScaler":
        """Запоминает общий центр и масштаб и по одному набору на область."""
        self.overall_ = centre_scale(X[self.columns], self.mode)
        self.by_area_ = {}
        if self.mode != "global":
            for area, block in X.groupby("area"):
                if len(block) >= MIN_AREA_ROWS:
                    self.by_area_[str(area)] = centre_scale(block[self.columns], self.mode)
        return self

    def stats_for(self, area: str) -> tuple[np.ndarray, np.ndarray]:
        """Центр и масштаб для области. Незнакомая область получает общие."""
        return self.by_area_.get(str(area), self.overall_) if self.mode != "global" \
            else self.overall_

    def transform(self, X: pd.DataFrame) -> np.ndarray:
        """Вычитает центр и делит на масштаб — свой для каждой области."""
        values = X[self.columns].to_numpy(dtype=float)
        result = (values - self.overall_[0]) / self.overall_[1]
        if self.mode == "global":
            return result
        areas = X["area"].astype(str).to_numpy()
        for area in np.unique(areas):
            mask = areas == area
            stats = self.by_area_.get(area)
            if stats is None and self.transductive and int(mask.sum()) >= MIN_AREA_ROWS:
                stats = centre_scale(X.loc[mask, self.columns], self.mode)
            if stats is not None:
                result[mask] = (values[mask] - stats[0]) / stats[1]
        return result
