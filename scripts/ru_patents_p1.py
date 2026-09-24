"""Задача П1: наборы PA, PB, PC против V0 — без сети, протокол задачи М.

V0 — шесть признаков s2a1-v1. PA: volume → share_ru; PB: volume → share_patent;
PC: volume → share_ru, growth_research → share_patent. Набор с признаком, остановленным
в П0, не считается. Нормировка внутри области, C=1, balanced, StratifiedGroupKFold
5 × 10 по группам с сидами SEED + повтор (общими для всех наборов — разности парные),
LOAO, порог по accuracy только на обучающей части, пропуски — медиана внутри фолда.

Запуск: python -m scripts.ru_patents_p1
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from model.config import FEATURES
from model.train import build_pipeline, choose_threshold, out_of_fold
from scripts.report_tools import auc_per_area, auc_per_repeat
from scripts.ru_patents_p0 import append_sheets, subsets, tables as p0_tables
from scripts.tuning import CURRENT, nested_cv, nested_loao

METRICS = ["precision", "recall", "f1", "accuracy"]


def replaced(swaps: dict[str, str]) -> list[str]:
    """Набор V0 с заменой признаков на тех же местах."""
    return [swaps.get(name, name) for name in FEATURES]


ALL_SETS = {
    "V0": list(FEATURES),
    "PA": replaced({"volume": "share_ru"}),
    "PB": replaced({"volume": "share_patent"}),
    "PC": replaced({"volume": "share_ru", "growth_research": "share_patent"}),
}


def active_sets(stopped: set[str]) -> dict[str, list[str]]:
    """Наборы без признаков, остановленных в П0. V0 остаётся всегда."""
    return {name: columns for name, columns in ALL_SETS.items()
            if name == "V0" or not stopped & set(columns)}


def evaluate(frame: pd.DataFrame, columns: list[str]) -> dict:
    """Метрики CV по повторам, AUC по повторам, LOAO по областям."""
    cv, _ = nested_cv(frame, "accuracy", combos=[CURRENT], columns=columns)
    cv.loc[:, "auc"] = auc_per_repeat(frame, columns=columns)
    loao = nested_loao(frame, "accuracy", combos=[CURRENT], columns=columns)
    loao.loc[:, "auc"] = loao["область"].map(auc_per_area(frame, columns=columns))
    return {"cv": cv, "loao": loao}


def summary_rows(name: str, rows: int, result: dict) -> list[dict]:
    """CV: среднее ± std по повторам; LOAO: среднее по шести областям."""
    cv = {key: f"{result['cv'][key].mean():.3f} ± {result['cv'][key].std(ddof=0):.3f}"
          for key in METRICS + ["auc"]}
    loao = {key: round(float(result["loao"][key].mean()), 3) for key in METRICS + ["auc"]}
    return [{"набор": name, "режим": "CV", "строк": rows, **cv},
            {"набор": name, "режим": "LOAO", "строк": rows, **loao}]


def paired_rows(results: dict, names: list[str], rows: int) -> list[dict]:
    """Разности набор − V0 по повторам CV: среднее, std, в скольких повторах лучше."""
    out = []
    for name in names:
        for key in ("auc", "accuracy"):
            diff = (results[(name, rows)]["cv"][key].to_numpy()
                    - results[("V0", rows)]["cv"][key].to_numpy())
            out.append({"разность": f"{name} − V0", "строк": rows, "метрика": key,
                        "среднее": round(float(diff.mean()), 4),
                        "std": round(float(diff.std(ddof=0)), 4),
                        "лучше из 10": int((diff > 0).sum()), "равно": int((diff == 0).sum())})
    return out


def weights(frame: pd.DataFrame, sets: dict[str, list[str]]) -> pd.DataFrame:
    """Веса, свободный член и порог при обучении на всех 160 строках."""
    rows = []
    for name, columns in sets.items():
        fitted = build_pipeline(features=columns).fit(frame[columns + ["area"]], frame["label"])
        logistic = fitted.named_steps["logistic"]
        threshold = choose_threshold(frame["label"].to_numpy(),
                                     np.nanmean(out_of_fold(frame, columns), axis=1))
        values = {**dict(zip(columns, logistic.coef_[0])),
                  "intercept": logistic.intercept_[0], "порог": threshold}
        rows.append({"набор": name, **{key: round(float(value), 6)
                                       for key, value in values.items()}})
    return pd.DataFrame(rows)


def area_auc(results: dict) -> pd.DataFrame:
    """LOAO AUC по областям: строка — выборка и область, колонки — наборы."""
    parts = [result["loao"][["область", "auc"]].assign(строк=rows, набор=name)
             for (name, rows), result in results.items()]
    table = pd.concat(parts).pivot_table(index=["строк", "область"], columns="набор",
                                         values="auc", sort=False)
    return table.round(3).reset_index()


def tables(frame: pd.DataFrame, stopped: set[str]) -> dict[str, pd.DataFrame]:
    """Все таблицы П1 в форматах листов М1."""
    sets = active_sets(stopped)
    frames = subsets(frame)
    results = {(name, rows): evaluate(block, columns)
               for rows, block in frames.items() for name, columns in sets.items()}
    summary = [row for (name, rows), result in results.items()
               for row in summary_rows(name, rows, result)]
    others = [name for name in sets if name != "V0"]
    paired = [row for rows in frames for row in paired_rows(results, others, rows)]
    return {"П1 наборы": pd.DataFrame([{"набор": name, "признаки": ", ".join(columns)}
                                       for name, columns in sets.items()]),
            "П1 сводка": pd.DataFrame(summary), "П1 парные разности": pd.DataFrame(paired),
            "П1 веса": weights(frame, sets), "П1 LOAO AUC по областям": area_auc(results)}


def main() -> None:
    """П0 заново из кэша (без сети), затем П1; печать и дозапись листов П1."""
    p0, frame = p0_tables()
    rule = p0["П0 правило остановки"]
    stopped = set(rule.loc[(rule["условие"] == "ИТОГ: остановлен") & rule["выполнено"], "признак"])
    print(f"остановлены в П0: {sorted(stopped) or 'нет'}")
    sheets = tables(frame, stopped)
    for title, table in sheets.items():
        print(f"\n{title}\n{table.to_string(index=False)}")
    names = append_sheets(sheets)
    print(f"\nлистов в report_tables.xlsx: {len(names)}")


if __name__ == "__main__":
    main()
