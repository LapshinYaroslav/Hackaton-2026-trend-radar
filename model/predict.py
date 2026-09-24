"""Предсказание с разбором логита по признакам — для объяснений в выдаче.

Логит складывается из свободного члена и вкладов признаков:

    logit = intercept + sum_j w_j * z_j,   z_j = (x_j - centre_j) / scale_j

centre и scale берутся по области кандидата; у незнакомой области — общие.
Пропуск признака заменяется медианой обучающей части, посчитанной уже в
нормированной шкале, — ровно так же, как это делает пайплайн при обучении.
Вклад признака w_j * z_j и есть то, что показывается пользователю: он в тех же
единицах, что логит, и знак говорит, в какую сторону признак сдвинул ответ.

Запуск: python -m model.predict
"""
from __future__ import annotations

import math
from collections.abc import Mapping
from pathlib import Path

import joblib

from model.config import DEFAULT_VERSION, EXPLANATIONS, canonical
from model.train import existing_artifact

_CACHE: dict[Path, dict] = {}


def load(path: Path | None = None, version: str = DEFAULT_VERSION) -> dict:
    """Артефакт версии с пайплайном и метаданными. Читается с диска один раз.

    По умолчанию — боевая версия model.config.DEFAULT_VERSION; path перекрывает версию.
    """
    path = path or existing_artifact(version)
    if path not in _CACHE:
        _CACHE[path] = joblib.load(path)
    return _CACHE[path]


def _scaled(value: float | None, centre: float, scale: float, fallback: float) -> float:
    """Нормированное значение признака. Пропуск заменяется медианой обучения."""
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return fallback
    return (float(value) - centre) / scale


def predict(features: Mapping[str, float | None], area: str = "",
            artifact: dict | None = None) -> tuple[float, dict[str, float]]:
    """Вероятность сигнала и вклад каждого признака в логит.

    Сумма вкладов плюс intercept из метаданных даёт логит; сигмоида от него —
    возвращаемая вероятность. Имена признаков принимаются и старые, и новые.
    """
    artifact = artifact or load()
    meta = artifact["meta"]
    scaler = artifact["pipeline"].named_steps["area_scaler"]
    centre, scale = scaler.stats_for(area)
    values = canonical_mapping(features)
    contributions = {}
    for index, name in enumerate(meta["features"]):
        normalized = _scaled(values.get(name), float(centre[index]), float(scale[index]),
                             meta["imputer_median_scaled"][name])
        contributions[name] = meta["coefficients"][name] * normalized
    logit = meta["intercept"] + sum(contributions.values())
    return 1.0 / (1.0 + math.exp(-logit)), contributions


def canonical_mapping(features: Mapping[str, float | None]) -> dict[str, float | None]:
    """Словарь признаков с каноническими именами: старые имена тоже принимаются."""
    import pandas as pd
    return canonical(pd.DataFrame([dict(features)])).iloc[0].to_dict()


def explain(features: Mapping[str, float | None], area: str = "",
            artifact: dict | None = None, top: int = 3) -> dict:
    """Вероятность, решение по порогу и самые весомые вклады — готовый ответ выдаче."""
    artifact = artifact or load()
    meta = artifact["meta"]
    probability, contributions = predict(features, area, artifact)
    ranked = sorted(contributions.items(), key=lambda pair: -abs(pair[1]))[:top]
    return {
        "score": probability,
        "is_signal": probability >= meta["threshold"],
        "threshold": meta["threshold"],
        "model_version": meta["model_version"],
        "cutoff_date": meta["cutoff_date"],
        "top_features": [{"name": name, "value": features.get(name),
                          "contribution": value,
                          "explanation_ru": EXPLANATIONS.get(name, name)}
                         for name, value in ranked],
    }


def main() -> None:
    """Показывает разбор на первой строке обучающей таблицы."""
    from model.train import training_table
    frame = training_table()
    row = frame.iloc[0]
    meta = load()["meta"]
    probability, contributions = predict(row.to_dict(), str(row["area"]))
    print(f"{row['tech_id']} {row['name_en']} ({row['area']})")
    print(f"  вероятность {probability:.4f}, порог {meta['threshold']}")
    print(f"  intercept {meta['intercept']:+.4f}")
    for name, value in sorted(contributions.items(), key=lambda pair: -abs(pair[1])):
        print(f"  {name:22s} {value:+.4f}")


if __name__ == "__main__":
    main()
