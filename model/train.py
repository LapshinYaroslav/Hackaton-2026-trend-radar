"""Обучение модели выбранной версии на всех 160 строках и сохранение артефакта.

Что делается:
  1. Порог выбирается кросс-валидацией: по предсказаниям вне обучения, усреднённым
     по повторам, берётся точка максимума accuracy на сетке. Это гиперпараметр,
     и выбирается он по кросс-валидации, а не по обучающим предсказаниям.
  2. Модель обучается на всех 160 строках.
  3. Артефакт кладётся в data/model/<версия>/: joblib с пайплайном и JSON с
     коэффициентами, порогом, центрами и масштабами по областям. Версии живут в разных
     каталогах, обучение одной не перезаписывает другую.

Честная оценка качества — не здесь: она считается вложенной кросс-валидацией
в model/release_report.py, где порог подбирается только на обучающей части.

Запуск: python -m model.train [--version s2a2-v1]
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import Pipeline

from model.area_scaler import AreaScaler
from model.config import (CUTOFF_DATE, DEFAULT_VERSION, FEATURES, MODEL_VERSION, N_REPEATS,
                          N_SPLITS, NORMALIZATION, SEED, THRESHOLD_CRITERION, THRESHOLD_GRID,
                          VERSIONS, canonical)

ROOT = Path(__file__).resolve().parents[1]
ARTIFACT_DIR = ROOT / "data" / "model"


def artifact_paths(version: str) -> tuple[Path, Path]:
    """joblib и JSON версии: data/model/<версия>/trend_radar_<версия>.*"""
    folder = ARTIFACT_DIR / version
    return folder / f"trend_radar_{version}.joblib", folder / f"trend_radar_{version}.json"


def existing_artifact(version: str) -> Path:
    """Путь к joblib для чтения. У s2a1-v1 запасной — прежний data/model/trend_radar_s2a1-v1.joblib,
    где артефакт лежал до разделения версий по каталогам."""
    path, _ = artifact_paths(version)
    legacy = ARTIFACT_DIR / f"trend_radar_{version}.joblib"
    return legacy if version == MODEL_VERSION and not path.exists() and legacy.exists() else path


# Пути s2a1-v1: их читают отчёты этой версии (scripts/release_report.py).
MODEL_PATH, JSON_PATH = artifact_paths(MODEL_VERSION)


def build_pipeline(mode: str = NORMALIZATION, features: list[str] = FEATURES) -> Pipeline:
    """Нормировка, заполнение медианой, логистическая регрессия.

    mode="area_z" — центр и масштаб внутри области, основной путь.
    mode="global" — общие центр и масштаб по всей выборке: так модель работает,
    когда область кандидата неизвестна. Метрики этого режима считает
    scripts/fallback_report.py, они другие и заявляются отдельно.
    features — набор признаков; по умолчанию шесть признаков s2a1-v1.
    """
    return Pipeline([
        ("area_scaler", AreaScaler(features, mode)),
        ("imputer", SimpleImputer(strategy="median")),
        ("logistic", LogisticRegression(class_weight="balanced", max_iter=2000,
                                        random_state=SEED)),
    ])


def out_of_fold(frame: pd.DataFrame, columns: list[str] = FEATURES) -> np.ndarray:
    """Вероятности вне обучения: строка × повтор. По столбцу на повтор."""
    features = frame[columns + ["area"]]
    labels, groups = frame["label"], frame["group"]
    matrix = np.full((len(frame), N_REPEATS), np.nan)
    for repeat in range(N_REPEATS):
        splitter = StratifiedGroupKFold(n_splits=N_SPLITS, shuffle=True,
                                        random_state=SEED + repeat)
        for train, test in splitter.split(features, labels, groups):
            fitted = build_pipeline(features=columns).fit(features.iloc[train],
                                                          labels.iloc[train])
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


def describe(pipeline: Pipeline, frame: pd.DataFrame, threshold: float,
             version: str = MODEL_VERSION) -> dict:
    """Всё, что нужно для объяснений и воспроизведения, в обычном словаре."""
    columns = VERSIONS[version]
    scaler = pipeline.named_steps["area_scaler"]
    logistic = pipeline.named_steps["logistic"]
    imputer = pipeline.named_steps["imputer"]
    return {
        "model_version": version,
        "cutoff_date": CUTOFF_DATE,
        "trained_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "random_state": SEED,
        "normalization": NORMALIZATION,
        "features": list(columns),
        "threshold": threshold,
        "threshold_criterion": THRESHOLD_CRITERION,
        "intercept": float(logistic.intercept_[0]),
        "coefficients": {name: float(weight)
                         for name, weight in zip(columns, logistic.coef_[0])},
        "imputer_median_scaled": {name: float(value) for name, value
                                  in zip(columns, imputer.statistics_)},
        "overall": {"centre": dict(zip(columns, map(float, scaler.overall_[0]))),
                    "scale": dict(zip(columns, map(float, scaler.overall_[1])))},
        "by_area": {area: {"centre": dict(zip(columns, map(float, centre))),
                           "scale": dict(zip(columns, map(float, scale)))}
                    for area, (centre, scale) in scaler.by_area_.items()},
        "training": {"rows": len(frame), "signals": int(frame["label"].sum()),
                     "negatives": int((frame["label"] == 0).sum()),
                     "groups": int(frame["group"].nunique()),
                     "areas": sorted(frame["area"].astype(str).unique()),
                     "missing_by_feature": {name: int(frame[name].isna().sum())
                                            for name in columns}},
    }


def train(frame: pd.DataFrame, version: str = DEFAULT_VERSION) -> tuple[Pipeline, dict]:
    """Выбирает порог по кросс-валидации и обучает модель версии на всех строках."""
    columns = VERSIONS[version]
    frame = canonical(frame).reset_index(drop=True)
    labels = frame["label"].to_numpy()
    threshold = choose_threshold(labels, np.nanmean(out_of_fold(frame, columns), axis=1))
    pipeline = build_pipeline(features=columns).fit(frame[columns + ["area"]], frame["label"])
    return pipeline, describe(pipeline, frame, threshold, version)


def save(pipeline: Pipeline, meta: dict) -> tuple[Path, Path]:
    """Кладёт joblib и JSON в каталог своей версии."""
    model_path, json_path = artifact_paths(meta["model_version"])
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"pipeline": pipeline, "meta": meta}, model_path)
    json_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    return model_path, json_path


def training_table() -> pd.DataFrame:
    """Обучающая таблица: признаки прогона плюс возраст первого препринта."""
    from model.training_table import training_table as assemble
    frame, _ = assemble()
    return canonical(frame)


def main() -> None:
    """Обучает и сохраняет артефакт версии из --version (по умолчанию боевой)."""
    import argparse
    parser = argparse.ArgumentParser(description="Обучение модели")
    parser.add_argument("--version", default=DEFAULT_VERSION, choices=sorted(VERSIONS))
    frame = training_table()
    pipeline, meta = train(frame, parser.parse_args().version)
    model_path, json_path = save(pipeline, meta)
    print(f"порог: {meta['threshold']}")
    print(f"модель: {model_path.relative_to(ROOT)}")
    print(f"коэффициенты: {json_path.relative_to(ROOT)}")
    for name, weight in sorted(meta["coefficients"].items(), key=lambda kv: -abs(kv[1])):
        print(f"  {name:22s} {weight:+.4f}")


if __name__ == "__main__":
    main()
