"""Подбор гиперпараметров внутри фолдов: ни один выбор не сделан по проверочным данным.

Схема. Внешний фолд отдаёт обучающую часть; на ней запускается ВТОРАЯ кросс-валидация,
и только по её предсказаниям выбираются C, class_weight и порог. Выбранная комбинация
обучается заново на всей обучающей части и применяется к отложенной. Отложенные строки
не участвуют ни в одном выборе — ни признаков, ни порога, ни гиперпараметров.

Для leave-one-area-out то же самое: внутренний перебор идёт по пяти обучающим областям
(каждая по очереди отложена внутри обучающей части), шестая область не видна.

Запуск: python -m scripts.tuning
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import Pipeline

from model.area_scaler import AreaScaler
from model.config import (FEATURES, N_INNER_SPLITS, N_REPEATS, N_SPLITS, NORMALIZATION,
                          SEED, THRESHOLD_GRID)
from model.train import score_at

GRID_C = [0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0]
GRID_WEIGHT = ["balanced", None]
CRITERIA = ["f1", "accuracy"]
# Текущая конфигурация: C по умолчанию sklearn, веса классов уравнены.
CURRENT = (1.0, "balanced")
COMBOS = [(value, weight) for value in GRID_C for weight in GRID_WEIGHT]


def pipeline_for(penalty: float, class_weight: str | None,
                 mode: str = NORMALIZATION) -> Pipeline:
    """Тот же пайплайн, что в обучении, но с заданными гиперпараметрами."""
    return Pipeline([
        ("area_scaler", AreaScaler(FEATURES, mode)),
        ("imputer", SimpleImputer(strategy="median")),
        ("logistic", LogisticRegression(C=penalty, class_weight=class_weight,
                                        max_iter=5000, random_state=SEED)),
    ])


def inner_probabilities(frame: pd.DataFrame, rows: np.ndarray, combo: tuple,
                        splits: list[tuple[np.ndarray, np.ndarray]],
                        mode: str = NORMALIZATION) -> np.ndarray:
    """Предсказания вне обучения внутри обучающей части при заданной комбинации."""
    features, labels = frame[FEATURES + ["area"]], frame["label"]
    probability = np.full(len(rows), np.nan)
    for fit_part, hold in splits:
        model = pipeline_for(*combo, mode=mode).fit(features.iloc[rows[fit_part]],
                                         labels.iloc[rows[fit_part]])
        probability[hold] = model.predict_proba(features.iloc[rows[hold]])[:, 1]
    return probability


def choose_inside(frame: pd.DataFrame, rows: np.ndarray, criterion: str,
                  splits: list[tuple[np.ndarray, np.ndarray]],
                  combos: list[tuple] | None = None,
                  mode: str = NORMALIZATION) -> tuple[tuple, float, float]:
    """Лучшая комбинация и порог по внутренней кросс-валидации обучающей части."""
    labels = frame["label"].iloc[rows].to_numpy()
    best = (None, 0.5, -np.inf)
    for combo in (combos if combos is not None else COMBOS):
        probability = inner_probabilities(frame, rows, combo, splits, mode)
        scores = [score_at(labels, probability, float(point))[criterion]
                  for point in THRESHOLD_GRID]
        position = int(np.nanargmax(scores))
        if scores[position] > best[2]:
            best = (combo, float(THRESHOLD_GRID[position]), float(scores[position]))
    return best


def _grouped_splits(frame: pd.DataFrame, rows: np.ndarray,
                    seed: int) -> list[tuple[np.ndarray, np.ndarray]]:
    """Разбиение обучающей части на внутренние фолды по группам."""
    splitter = StratifiedGroupKFold(n_splits=N_INNER_SPLITS, shuffle=True,
                                    random_state=seed)
    block = frame.iloc[rows]
    return list(splitter.split(block[FEATURES], block["label"], block["group"]))


def _area_splits(frame: pd.DataFrame, rows: np.ndarray) -> list[tuple[np.ndarray, np.ndarray]]:
    """Разбиение обучающей части по областям: каждая по очереди отложена."""
    areas = frame["area"].iloc[rows].to_numpy()
    return [(np.flatnonzero(areas != area), np.flatnonzero(areas == area))
            for area in sorted(set(areas))]


def nested_cv(frame: pd.DataFrame, criterion: str,
              combos: list[tuple] | None = None,
              mode: str = NORMALIZATION) -> tuple[pd.DataFrame, list[dict]]:
    """Вложенная кросс-валидация: метрики по повторам и что выбиралось внутри."""
    features, labels, groups = frame[FEATURES + ["area"]], frame["label"], frame["group"]
    truth = labels.to_numpy()
    rows, picks = [], []
    for repeat in range(N_REPEATS):
        outer = StratifiedGroupKFold(n_splits=N_SPLITS, shuffle=True,
                                     random_state=SEED + repeat)
        predicted = np.full(len(frame), np.nan)
        for train, test in outer.split(features, labels, groups):
            splits = _grouped_splits(frame, train, SEED + repeat)
            combo, threshold, _ = choose_inside(frame, train, criterion, splits,
                                                combos, mode)
            model = pipeline_for(*combo, mode=mode).fit(features.iloc[train],
                                                       labels.iloc[train])
            predicted[test] = model.predict_proba(features.iloc[test])[:, 1] >= threshold
            picks.append({"повтор": repeat, "C": combo[0], "class_weight": combo[1],
                          "порог": threshold})
        got = score_at(truth, predicted.astype(float), 0.5)
        rows.append({"повтор": repeat, **got})
    return pd.DataFrame(rows), picks


def nested_loao(frame: pd.DataFrame, criterion: str,
                combos: list[tuple] | None = None,
                mode: str = NORMALIZATION) -> pd.DataFrame:
    """Leave-one-area-out: выбор идёт по пяти обучающим областям, шестая не видна."""
    features, labels = frame[FEATURES + ["area"]], frame["label"]
    rows = []
    for area in sorted(frame["area"].astype(str).unique()):
        test = np.flatnonzero((frame["area"].astype(str) == area).to_numpy())
        train = np.flatnonzero((frame["area"].astype(str) != area).to_numpy())
        combo, threshold, _ = choose_inside(frame, train, criterion,
                                            _area_splits(frame, train), combos, mode)
        model = pipeline_for(*combo, mode=mode).fit(features.iloc[train],
                                                   labels.iloc[train])
        probability = model.predict_proba(features.iloc[test])[:, 1]
        got = score_at(labels.iloc[test].to_numpy(), probability, threshold)
        rows.append({"область": area, "C": combo[0],
                     "class_weight": combo[1] or "None", "порог": threshold, **got})
    return pd.DataFrame(rows)


def main() -> None:
    """Короткий прогон для проверки, что схема работает."""
    from model.train import training_table
    frame = training_table().reset_index(drop=True)
    table, picks = nested_cv(frame, "accuracy", combos=[CURRENT])
    print("текущая конфигурация, CV:",
          {key: round(float(table[key].mean()), 3)
           for key in ("precision", "recall", "f1", "accuracy")})
    print("выбранных порогов:", len(picks))


if __name__ == "__main__":
    main()
