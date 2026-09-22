"""Обучение итоговой модели на всех 160 строках и сохранение артефакта.

Что делается:
  1. Порог выбирается кросс-валидацией: по предсказаниям вне обучения, усреднённым
     по повторам, берётся точка максимума accuracy на сетке. Это гиперпараметр,
     и выбирается он по кросс-валидации, а не по обучающим предсказаниям.
  2. Модель обучается на всех 160 строках.
  3. Артефакт кладётся в data/model/: joblib с пайплайном и JSON с коэффициентами,
     порогом, центрами и масштабами по областям.

Честная оценка качества — не здесь: она считается вложенной кросс-валидацией
в model/release_report.py, где порог подбирается только на обучающей части.

Запуск: python -m model.train
"""
from __future__ import annotations

import json
from datetime import date, datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import Pipeline

from model.area_scaler import AreaScaler
from model.config import (CUTOFF_DATE, FEATURES, MODEL_VERSION, N_REPEATS, N_SPLITS,
                          NORMALIZATION, SEED, THRESHOLD_CRITERION, THRESHOLD_GRID,
                          canonical)

ROOT = Path(__file__).resolve().parents[1]
ARTIFACT_DIR = ROOT / "data" / "model"
MODEL_PATH = ARTIFACT_DIR / f"trend_radar_{MODEL_VERSION}.joblib"
JSON_PATH = ARTIFACT_DIR / f"trend_radar_{MODEL_VERSION}.json"


def build_pipeline(mode: str = NORMALIZATION) -> Pipeline:
    """Нормировка, заполнение медианой, логистическая регрессия.

    mode="area_z" — центр и масштаб внутри области, основной путь.
    mode="global" — общие центр и масштаб по всей выборке: так модель работает,
    когда область кандидата неизвестна. Метрики этого режима считает
    scripts/fallback_report.py, они другие и заявляются отдельно.
    """
    return Pipeline([
        ("area_scaler", AreaScaler(FEATURES, mode)),
        ("imputer", SimpleImputer(strategy="median")),
        ("logistic", LogisticRegression(class_weight="balanced", max_iter=2000,
                                        random_state=SEED)),
    ])


def out_of_fold(frame: pd.DataFrame) -> np.ndarray:
    """Вероятности вне обучения: строка × повтор. По столбцу на повтор."""
    features = frame[FEATURES + ["area"]]
    labels, groups = frame["label"], frame["group"]
    matrix = np.full((len(frame), N_REPEATS), np.nan)
    for repeat in range(N_REPEATS):
        splitter = StratifiedGroupKFold(n_splits=N_SPLITS, shuffle=True,
                                        random_state=SEED + repeat)
        for train, test in splitter.split(features, labels, groups):
            fitted = build_pipeline().fit(features.iloc[train], labels.iloc[train])
            matrix[test, repeat] = fitted.predict_proba(features.iloc[test])[:, 1]
    return matrix


def score_at(labels: np.ndarray, probability: np.ndarray, threshold: float) -> dict:
    """Матрица ошибок и четыре метрики при пороге."""
    predicted = probability >= threshold
    tp = int(((predicted == 1) & (labels == 1)).sum())
    fp = int(((predicted == 1) & (labels == 0)).sum())
    fn = int(((predicted == 0) & (labels == 1)).sum())
    tn = int(((predicted == 0) & (labels == 0)).sum())
    precision = tp / (tp + fp) if tp + fp else float("nan")
    recall = tp / (tp + fn) if tp + fn else float("nan")
    f1 = (2 * precision * recall / (precision + recall)
          if precision == precision and precision + recall > 0 else float("nan"))
    return {"TP": tp, "FP": fp, "FN": fn, "TN": tn, "precision": precision,
            "recall": recall, "f1": f1, "accuracy": (tp + tn) / len(labels)}


def choose_threshold(labels: np.ndarray, probability: np.ndarray) -> float:
    """Точка максимума выбранной метрики на сетке порогов."""
    values = [score_at(labels, probability, float(point))[THRESHOLD_CRITERION]
              for point in THRESHOLD_GRID]
    return float(THRESHOLD_GRID[int(np.nanargmax(values))])


def describe(pipeline: Pipeline, frame: pd.DataFrame, threshold: float) -> dict:
    """Всё, что нужно для объяснений и воспроизведения, в обычном словаре."""
    scaler = pipeline.named_steps["area_scaler"]
    logistic = pipeline.named_steps["logistic"]
    imputer = pipeline.named_steps["imputer"]
    return {
        "model_version": MODEL_VERSION,
        "cutoff_date": CUTOFF_DATE,
        "trained_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "random_state": SEED,
        "normalization": NORMALIZATION,
        "features": list(FEATURES),
        "threshold": threshold,
        "threshold_criterion": THRESHOLD_CRITERION,
        "intercept": float(logistic.intercept_[0]),
        "coefficients": {name: float(weight)
                         for name, weight in zip(FEATURES, logistic.coef_[0])},
        "imputer_median_scaled": {name: float(value) for name, value
                                  in zip(FEATURES, imputer.statistics_)},
        "overall": {"centre": dict(zip(FEATURES, map(float, scaler.overall_[0]))),
                    "scale": dict(zip(FEATURES, map(float, scaler.overall_[1])))},
        "by_area": {area: {"centre": dict(zip(FEATURES, map(float, centre))),
                           "scale": dict(zip(FEATURES, map(float, scale)))}
                    for area, (centre, scale) in scaler.by_area_.items()},
        "training": {"rows": len(frame), "signals": int(frame["label"].sum()),
                     "negatives": int((frame["label"] == 0).sum()),
                     "groups": int(frame["group"].nunique()),
                     "areas": sorted(frame["area"].astype(str).unique()),
                     "missing_by_feature": {name: int(frame[name].isna().sum())
                                            for name in FEATURES}},
    }


def train(frame: pd.DataFrame) -> tuple[Pipeline, dict]:
    """Выбирает порог по кросс-валидации и обучает модель на всех строках."""
    frame = canonical(frame).reset_index(drop=True)
    labels = frame["label"].to_numpy()
    threshold = choose_threshold(labels, np.nanmean(out_of_fold(frame), axis=1))
    pipeline = build_pipeline().fit(frame[FEATURES + ["area"]], frame["label"])
    return pipeline, describe(pipeline, frame, threshold)


def save(pipeline: Pipeline, meta: dict) -> tuple[Path, Path]:
    """Кладёт joblib и JSON рядом друг с другом."""
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump({"pipeline": pipeline, "meta": meta}, MODEL_PATH)
    JSON_PATH.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    return MODEL_PATH, JSON_PATH


def training_table() -> pd.DataFrame:
    """Обучающая таблица: признаки прогона плюс возраст первого препринта."""
    from model.training_table import training_table as assemble
    frame, _ = assemble()
    return canonical(frame)


def main() -> None:
    """Обучает и сохраняет артефакт."""
    frame = training_table()
    pipeline, meta = train(frame)
    model_path, json_path = save(pipeline, meta)
    print(f"порог: {meta['threshold']}")
    print(f"модель: {model_path.relative_to(ROOT)}")
    print(f"коэффициенты: {json_path.relative_to(ROOT)}")
    for name, weight in sorted(meta["coefficients"].items(), key=lambda kv: -abs(kv[1])):
        print(f"  {name:22s} {weight:+.4f}")


if __name__ == "__main__":
    main()
