"""Числа для сдачи: честные метрики, подбор гиперпараметров внутри фолдов, коэффициенты.

Ни один выбор — ни гиперпараметров, ни порога — не сделан по проверочным данным:
всё выбирается внутренней кросс-валидацией на обучающей части фолда. Рядом стоит
текущая конфигурация (C=1, class_weight=balanced), у которой подбирается только порог.

Запуск: python -m model.release_report
"""
from __future__ import annotations

import json
from collections import Counter
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.report_tools import md, rounded
from model.config import EXPLANATIONS, FEATURES, MODEL_VERSION, NEWS_SHARE
from model.train import JSON_PATH, MODEL_PATH, training_table
from scripts.tuning import CRITERIA, CURRENT, nested_cv, nested_loao

ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "reports"
SETUPS = [("текущая (C=1, balanced)", [CURRENT]), ("перебор внутри фолдов", None)]


def cv_row(table: pd.DataFrame, title: str, criterion: str) -> dict:
    """Строка сводки: среднее ± std по повторам и средняя матрица ошибок."""
    row = {"конфигурация": title, "критерий порога": criterion}
    for key, name in (("precision", "Precision"), ("recall", "Recall"),
                      ("f1", "F1"), ("accuracy", "Accuracy")):
        values = table[key].to_numpy(dtype=float)
        row[name] = f"{values.mean():.3f} ± {values.std():.3f}"
    for key in ("TP", "FP", "FN", "TN"):
        row[key] = round(float(table[key].mean()), 1)
    return row


def loao_row(table: pd.DataFrame, title: str, criterion: str) -> dict:
    """Строка сводки по областям: среднее по шести и суммарная матрица ошибок."""
    row = {"конфигурация": title, "критерий порога": criterion}
    for key, name in (("precision", "Precision"), ("recall", "Recall"),
                      ("f1", "F1"), ("accuracy", "Accuracy")):
        row[name] = rounded(float(table[key].mean()))
    for key in ("TP", "FP", "FN", "TN"):
        row[key] = int(table[key].sum())
    return row


def picks_table(picks: list[dict], title: str, criterion: str) -> pd.DataFrame:
    """Что именно выбирал внутренний перебор: частоты по 50 фолдам."""
    combos = Counter((item["C"], item["class_weight"] or "None") for item in picks)
    thresholds = [item["порог"] for item in picks]
    rows = []
    for (penalty, weight), count in sorted(combos.items(), key=lambda kv: -kv[1]):
        rows.append({"конфигурация": title, "критерий порога": criterion,
                     "C": penalty, "class_weight": weight,
                     "выбрано фолдов": count,
                     "доля": rounded(count / len(picks))})
    rows[0]["порог: медиана"] = rounded(float(np.median(thresholds)))
    rows[0]["порог: мин-макс"] = f"{min(thresholds)} … {max(thresholds)}"
    return pd.DataFrame(rows)


def coefficients_table(meta: dict) -> pd.DataFrame:
    """Коэффициенты обученной на всех строках модели, по убыванию модуля."""
    rows = [{"признак": name, "вес": rounded(meta["coefficients"][name], 4),
             "знак": "+" if meta["coefficients"][name] > 0 else "−",
             "модуль веса": rounded(abs(meta["coefficients"][name]), 4),
             "пропусков в обучении": meta["training"]["missing_by_feature"][name],
             "что это": EXPLANATIONS.get(name, "")}
            for name in FEATURES]
    return pd.DataFrame(rows).sort_values("модуль веса", ascending=False)


def main() -> None:
    """Считает обе конфигурации по обоим критериям и пишет отчёт."""
    frame = training_table().reset_index(drop=True)
    meta = json.loads(JSON_PATH.read_text(encoding="utf-8"))

    cv_rows, loao_rows, picks_frames, loao_tables = [], [], [], []
    for criterion in CRITERIA:
        for title, combos in SETUPS:
            table, picks = nested_cv(frame, criterion, combos)
            cv_rows.append(cv_row(table, title, criterion))
            picks_frames.append(picks_table(picks, title, criterion))
            areas = nested_loao(frame, criterion, combos)
            loao_rows.append(loao_row(areas, title, criterion))
            loao_tables.append((title, criterion, areas))

    REPORTS.mkdir(parents=True, exist_ok=True)
    path = REPORTS / f"model_release_{date.today():%Y%m%d}.md"
    with path.open("w", encoding="utf-8") as out:
        print(f"# Закрепление модели {MODEL_VERSION}\n", file=out)
        print(f"Шесть признаков: {', '.join(FEATURES)}. Нормировка внутри области "
              f"с откатом на общие центр и масштаб для новой области. Обучение на "
              f"{meta['training']['rows']} строках "
              f"({meta['training']['signals']} сигналов, "
              f"{meta['training']['negatives']} мейнстримов, "
              f"{meta['training']['groups']} групп). Дата среза "
              f"{meta['cutoff_date']}, random_state {meta['random_state']}.\n", file=out)
        print(f"Артефакт: `{MODEL_PATH.relative_to(ROOT)}`, коэффициенты и центры — "
              f"`{JSON_PATH.relative_to(ROOT)}`.\n", file=out)
        print("**Про честность чисел.** Гиперпараметры (C, class_weight) и порог "
              "выбираются внутренней кросс-валидацией на обучающей части каждого "
              "фолда. Отложенные строки не участвуют ни в одном выборе. Для "
              "leave-one-area-out внутренний перебор идёт по пяти обучающим областям, "
              "шестая не видна вовсе.\n", file=out)
        print("**Переименование.** `share_market` → "
              f"`{NEWS_SHARE}`: признак считает долю документов TechCrunch, найденных "
              "по СЛОВАМ названия, среди всех найденных документов (новости плюс "
              "научные публикации). Точную фразу содержат 6.5 % найденных документов, "
              "все слова названия — 83 %. Старое имя принимается как алиас.\n", file=out)

        print("## 1. Кросс-валидация, среднее ± std по 10 повторам\n", file=out)
        print(pd.DataFrame(cv_rows).pipe(md), file=out)
        print("\n## 2. Leave-one-area-out, среднее по шести областям\n", file=out)
        print("Матрица ошибок здесь суммарная по всем 160 строкам.\n", file=out)
        print(pd.DataFrame(loao_rows).pipe(md), file=out)

        print("\n### 2.1. По областям\n", file=out)
        for title, criterion, table in loao_tables:
            print(f"**{title}, порог под {criterion}**\n", file=out)
            print(table.assign(**{key: table[key].map(rounded) for key in
                                  ("precision", "recall", "f1", "accuracy")}).pipe(md),
                  file=out)
            print("", file=out)

        print("## 3. Что выбирал внутренний перебор\n", file=out)
        print("Пятьдесят внешних фолдов: пять разбиений на десять повторов.\n", file=out)
        print(pd.concat(picks_frames).pipe(md), file=out)

        print("\n## 4. Коэффициенты модели, обученной на всех строках\n", file=out)
        print(f"Порог {meta['threshold']} выбран по кросс-валидации, критерий "
              f"{meta['threshold_criterion']}. Свободный член "
              f"{meta['intercept']:+.4f}. Веса — в нормированной шкале: признаки "
              f"приведены к среднему 0 и sd 1 внутри своей области, поэтому веса "
              f"сравнимы между собой.\n", file=out)
        print(coefficients_table(meta).pipe(md), file=out)
        print(f"\nОбласти с собственными центром и масштабом: "
              f"{', '.join(sorted(meta['by_area']))}.\n", file=out)
    print(f"записано: {path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
