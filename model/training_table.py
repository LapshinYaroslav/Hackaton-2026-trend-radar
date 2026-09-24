"""Обучающая таблица: признаки обеих версий модели по 160 размеченным технологиям.

Пять признаков лежат готовыми в таблице признаков прогона, age_first_arxiv
восстанавливается из кэша счётчиков, share_patent (s2a2-v1) — из снимка патентных
счётчиков evidence/rospatent_training_counts.json и научного счётчика n_research таблицы. Ничего не считается
заново: сборка нужна только для того, чтобы отвязать обучение от
исследовательского кода, где эта склейка жила раньше.

Запуск: python -m model.training_table
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from model.config import FEATURES, ROSPATENT_DATASETS, canonical
from model.corpus import load_training_patents
from model.features import share_patent
from model.first_mention import TECHNOLOGIES, build as build_first_mention

ROOT = Path(__file__).resolve().parents[1]
FEATURES_DIR = ROOT / "data" / "experiments"


def load_features() -> tuple[pd.DataFrame, Path]:
    """Последняя таблица признаков прогона с каноническими именами колонок."""
    source = sorted(FEATURES_DIR.glob("axis_features_*.csv"))[-1]
    frame = canonical(pd.read_csv(source, encoding="utf-8-sig")).reset_index(drop=True)
    return frame, source


def with_patents(frame: pd.DataFrame) -> pd.DataFrame:
    """n_pat из снимка обучения и share_patent = n_pat / (n_pat + n_research)."""
    patents = load_training_patents(ROSPATENT_DATASETS)[["tech_id", "n_pat"]]
    frame = frame.merge(patents, on="tech_id", how="left", validate="one_to_one")
    if frame["n_pat"].isna().any():
        missing = frame.loc[frame["n_pat"].isna(), "tech_id"].tolist()
        raise ValueError(f"в снимке Роспатента нет технологий {missing}")
    frame.loc[:, "share_patent"] = [share_patent(int(pat), int(research)) for pat, research
                                    in zip(frame["n_pat"], frame["n_research"])]
    return frame


def training_table() -> tuple[pd.DataFrame, Path]:
    """Таблица признаков плюс возраст первого препринта и патентный признак, по tech_id."""
    frame, source = load_features()
    technologies = pd.read_csv(TECHNOLOGIES, encoding="utf-8-sig")
    ages = build_first_mention(technologies)[["tech_id", "age_first_arxiv"]]
    return with_patents(frame.merge(ages, on="tech_id", how="left")), source


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
