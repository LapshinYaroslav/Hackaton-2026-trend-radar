"""Собирает артефакт s2a2-v1, если обученного файла нет (каталог в gitignore).

Веса и порог — из docs/methodology.md, п. 8.2 / 9.3b. Центр и масштаб — единичные:
живой скоринг стартует, область без своей статистики идёт в общий откат.
Если рядом есть таблица обучения, вызывается настоящий model.train.
"""

from __future__ import annotations

import logging
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from model.area_scaler import AreaScaler
from model.config import CUTOFF_DATE, DEFAULT_VERSION, NORMALIZATION, SEED, VERSIONS
from model.train import artifact_paths, existing_artifact, save

logger = logging.getLogger(__name__)

# docs/methodology.md, таблица весов s2a2-v1 (160 строк, area_z, C=1, balanced).
S2A2_WEIGHTS = {
    "share_patent": -0.661,
    "growth_research": 0.003,
    "recency": 1.049,
    "share_news_wordmatch": 1.335,
    "share_prev6": -0.100,
    "age_first_arxiv": -0.418,
}
S2A2_INTERCEPT = 0.381
S2A2_THRESHOLD = 0.400


def write_published(version: str = DEFAULT_VERSION) -> Path:
    """joblib + JSON из опубликованных коэффициентов, без обучающей таблицы."""
    columns = list(VERSIONS[version])
    scaler = AreaScaler(columns, NORMALIZATION)
    zeros = np.zeros(len(columns), dtype=float)
    ones = np.ones(len(columns), dtype=float)
    scaler.overall_ = (zeros, ones)
    scaler.by_area_ = {}
    dummy = pd.DataFrame(
        [{name: 0.0 for name in columns} | {"area": "Роботы"} for _ in range(4)]
    )
    labels = np.array([1, 0, 1, 0])
    pipeline = Pipeline(
        [
            ("area_scaler", scaler),
            ("imputer", SimpleImputer(strategy="median")),
            (
                "logistic",
                LogisticRegression(class_weight="balanced", max_iter=200, random_state=SEED),
            ),
        ]
    )
    # imputer.statistics_ нужны predict(); логистик в выдаче не участвует.
    pipeline.named_steps["imputer"].fit(dummy[columns])
    pipeline.named_steps["logistic"].fit(dummy[columns], labels)
    weights = {name: float(S2A2_WEIGHTS.get(name, 0.0)) for name in columns}
    meta = {
        "model_version": version,
        "cutoff_date": CUTOFF_DATE,
        "normalization": NORMALIZATION,
        "features": columns,
        "threshold": S2A2_THRESHOLD,
        "intercept": S2A2_INTERCEPT,
        "coefficients": weights,
        "imputer_median_scaled": {
            "share_patent": 0.15,
            "growth_research": 0.0,
            "recency": 0.40,
            "share_news_wordmatch": 0.35,
            "share_prev6": 0.10,
            "age_first_arxiv": 3.0,
        },
        "overall": {"centre": dict(zip(columns, [0.0] * len(columns))),
                    "scale": dict(zip(columns, [1.0] * len(columns)))},
        "by_area": {},
        "bootstrap": True,
        "bootstrap_note": "веса из methodology.md; нормировка единичная, пока нет обучения",
    }
    model_path, _ = save(pipeline, meta)
    logger.info("bootstrap model written to %s", model_path)
    return model_path


def try_train(version: str = DEFAULT_VERSION) -> Path | None:
    """Обучает по таблице, если experiments/ось признаков доступны."""
    try:
        from model.train import train, training_table
        frame = training_table()
        if frame is None or len(frame) < 8:
            return None
        pipeline, meta = train(frame, version)
        model_path, _ = save(pipeline, meta)
        return model_path
    except Exception as exc:
        logger.info("live train skipped: %s", exc)
        return None


def ensure_artifact(version: str = DEFAULT_VERSION) -> Path:
    """Путь к joblib: уже лежит, обучен, либо собран из опубликованных весов."""
    path = existing_artifact(version)
    if path.exists():
        return path
    trained = try_train(version)
    if trained is not None and trained.exists():
        return trained
    return write_published(version)
