"""Общее для отчётов и гейта: разметка таблиц, округление, расчёт AUC.

md и rounded перенесены дословно из experiments/explore/analyze.py. AUC живёт
здесь, а не в гейте, потому что его считает ещё и отчёт про режим отката:
две реализации разошлись бы незаметно.
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold

from model.config import FEATURES, N_REPEATS, N_SPLITS, NORMALIZATION, SEED
from model.train import build_pipeline


def md(frame: pd.DataFrame, index: bool = False) -> str:
    """Таблица в разметке Markdown. Своя, чтобы не тянуть tabulate в зависимости."""
    data = frame.reset_index() if index else frame
    header = [str(name) for name in data.columns]
    body = [[("" if value is None or (isinstance(value, float) and math.isnan(value))
              else str(value)) for value in row] for row in data.itertuples(index=False)]
    lines = ["| " + " | ".join(header) + " |",
             "|" + "|".join("---" for _ in header) + "|"]
    lines += ["| " + " | ".join(row) + " |" for row in body]
    return "\n".join(lines)


def rounded(value: float | None, digits: int = 3) -> float | None:
    """Округление, не ломающееся на NaN: пропуск должен быть пустой клеткой."""
    return round(float(value), digits) if value is not None and math.isfinite(value) else None


def auc_by_cross_validation(frame: pd.DataFrame, mode: str = NORMALIZATION) -> float:
    """AUC вне обучения: среднее по фолдам внутри повтора, затем по повторам."""
    features, labels, groups = frame[FEATURES + ["area"]], frame["label"], frame["group"]
    per_repeat = []
    for repeat in range(N_REPEATS):
        splitter = StratifiedGroupKFold(n_splits=N_SPLITS, shuffle=True,
                                        random_state=SEED + repeat)
        folds = [roc_auc_score(labels.iloc[test],
                               build_pipeline(mode).fit(features.iloc[train],
                                                        labels.iloc[train])
                               .predict_proba(features.iloc[test])[:, 1])
                 for train, test in splitter.split(features, labels, groups)]
        per_repeat.append(float(np.mean(folds)))
    return float(np.mean(per_repeat))


def auc_by_area(frame: pd.DataFrame, mode: str = NORMALIZATION) -> float:
    """AUC на области, исключённой из обучения, усреднённая по шести областям."""
    features, labels = frame[FEATURES + ["area"]], frame["label"]
    scores = []
    for area in sorted(frame["area"].astype(str).unique()):
        test = np.flatnonzero((frame["area"].astype(str) == area).to_numpy())
        train = np.flatnonzero((frame["area"].astype(str) != area).to_numpy())
        fitted = build_pipeline(mode).fit(features.iloc[train], labels.iloc[train])
        scores.append(roc_auc_score(labels.iloc[test],
                                    fitted.predict_proba(features.iloc[test])[:, 1]))
    return float(np.mean(scores))
