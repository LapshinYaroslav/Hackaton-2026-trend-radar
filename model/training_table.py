"""Обучающая таблица: шесть признаков модели по 160 размеченным технологиям.

Пять признаков лежат готовыми в таблице признаков прогона, шестой —
age_first_arxiv — восстанавливается из кэша счётчиков. Ничего не считается
заново: сборка нужна только для того, чтобы отвязать обучение от
исследовательского кода, где эта склейка жила раньше.

Запуск: python -m model.training_table
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from model.config import FEATURES, canonical
from model.first_mention import TECHNOLOGIES, build as build_first_mention

ROOT = Path(__file__).resolve().parents[1]
FEATURES_DIR = ROOT / "data" / "experiments"


def load_features() -> tuple[pd.DataFrame, Path]:
    """Последняя таблица признаков прогона с каноническими именами колонок."""
    source = sorted(FEATURES_DIR.glob("axis_features_*.csv"))[-1]
    frame = canonical(pd.read_csv(source, encoding="utf-8-sig")).reset_index(drop=True)
    return frame, source


def training_table() -> tuple[pd.DataFrame, Path]:
    """Таблица признаков плюс возраст первого препринта, по tech_id."""
    frame, source = load_features()
    technologies = pd.read_csv(TECHNOLOGIES, encoding="utf-8-sig")
    ages = build_first_mention(technologies)[["tech_id", "age_first_arxiv"]]
    return frame.merge(ages, on="tech_id", how="left"), source


def main() -> None:
    """Печатает состав таблицы и покрытие признаков."""
    frame, source = training_table()
    print(f"{source.name}: {len(frame)} строк, "
          f"{int(frame['label'].sum())} сигналов, "
          f"{int((frame['label'] == 0).sum())} мейнстримов, "
          f"{frame['group'].nunique()} групп")
    for name in FEATURES:
        print(f"  {name:22s} заполнено {int(frame[name].notna().sum()):3d} из {len(frame)}")


if __name__ == "__main__":
    main()
