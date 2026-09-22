"""Метрики режима отката: та же модель на общих центре и масштабе.

Зачем. Все заявленные числа измерены с нормировкой внутри области. В открытом
запросе область кандидата часто неизвестна, и тогда работает откат на общие
центр и масштаб по всей обучающей выборке. Это другой режим, и числа у него
свои — их и надо показывать, когда область не выбрана.

Фолды те же: StratifiedGroupKFold с теми же random_state, что у основного пути,
и тот же вложенный подбор порога на обучающей части. Отличается ровно одно —
режим нормировки.

Запуск: python -m scripts.fallback_report
"""
from __future__ import annotations

from datetime import date
from pathlib import Path

import pandas as pd

from model.config import FEATURES, MODEL_VERSION, N_REPEATS, SEED
from model.training_table import training_table
from scripts.report_tools import auc_by_area, auc_by_cross_validation, md, rounded
from scripts.tuning import CURRENT, nested_cv, nested_loao

ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = ROOT / "evidence"
MODES = {"area_z": "по области (основной путь)",
         "global": "общие центр и масштаб (откат)"}
CRITERION = "accuracy"


def measure(frame: pd.DataFrame, mode: str) -> dict:
    """Все числа одного режима: AUC и четыре метрики по CV и по областям."""
    cv_table, _ = nested_cv(frame, CRITERION, combos=[CURRENT], mode=mode)
    loao_table = nested_loao(frame, CRITERION, combos=[CURRENT], mode=mode)
    return {
        "auc_cv": auc_by_cross_validation(frame, mode),
        "auc_loao": auc_by_area(frame, mode),
        "cv": {key: (float(cv_table[key].mean()), float(cv_table[key].std()))
               for key in ("precision", "recall", "f1", "accuracy")},
        "loao": {key: float(loao_table[key].mean())
                 for key in ("precision", "recall", "f1", "accuracy")},
        "loao_rows": loao_table,
    }


def cv_row(mode: str, got: dict) -> dict:
    """Строка сводки по кросс-валидации: среднее ± std по повторам."""
    row = {"нормировка": MODES[mode], "AUC": rounded(got["auc_cv"])}
    for key, name in (("precision", "Precision"), ("recall", "Recall"),
                      ("f1", "F1"), ("accuracy", "Accuracy")):
        mean, spread = got["cv"][key]
        row[name] = f"{mean:.3f} ± {spread:.3f}"
    return row


def loao_row(mode: str, got: dict) -> dict:
    """Строка сводки по leave-one-area-out: среднее по шести областям."""
    row = {"нормировка": MODES[mode], "AUC": rounded(got["auc_loao"])}
    for key, name in (("precision", "Precision"), ("recall", "Recall"),
                      ("f1", "F1"), ("accuracy", "Accuracy")):
        row[name] = rounded(got["loao"][key])
    return row


def by_area(measured: dict[str, dict]) -> pd.DataFrame:
    """Accuracy по каждой области в обоих режимах рядом."""
    rows = []
    for area in measured["area_z"]["loao_rows"]["область"]:
        row = {"область": area}
        for mode in MODES:
            table = measured[mode]["loao_rows"]
            found = table.loc[table["область"] == area].iloc[0]
            row[MODES[mode]] = rounded(float(found["accuracy"]))
        row["разница"] = rounded(row[MODES["global"]] - row[MODES["area_z"]])
        rows.append(row)
    return pd.DataFrame(rows)


def main() -> None:
    """Считает оба режима и пишет отчёт в evidence/."""
    frame, source = training_table()
    frame = frame.reset_index(drop=True)
    measured = {mode: measure(frame, mode) for mode in MODES}

    EVIDENCE.mkdir(parents=True, exist_ok=True)
    path = EVIDENCE / f"fallback_mode_{date.today():%Y%m%d}.md"
    with path.open("w", encoding="utf-8") as out:
        print(f"# Режим отката: метрики без нормировки по области\n", file=out)
        print(f"Модель {MODEL_VERSION}, шесть признаков: {', '.join(FEATURES)}. "
              f"Таблица `{source.name}`, {len(frame)} строк, "
              f"{int(frame['label'].sum())} сигналов, "
              f"{int((frame['label'] == 0).sum())} мейнстримов.\n", file=out)
        print("**Зачем эти числа.** Признаки нормируются внутри области, и все "
              "заявленные метрики измерены так. В открытом запросе область кандидата "
              "часто неизвестна — тогда берутся общие центр и масштаб по всей "
              "обучающей выборке. Это другой режим, и показывать при демо без "
              "выбранной области надо его числа, а не основные.\n", file=out)
        print(f"Фолды те же: StratifiedGroupKFold, random_state={SEED}, "
              f"{N_REPEATS} повторов. Порог подбирается внутренней кросс-валидацией "
              f"на обучающей части, критерий {CRITERION}. Отличается ровно одно — "
              f"режим нормировки.\n", file=out)

        print("## 1. Кросс-валидация, среднее ± std по повторам\n", file=out)
        print(pd.DataFrame([cv_row(mode, measured[mode]) for mode in MODES]).pipe(md),
              file=out)
        print("\n## 2. Leave-one-area-out, среднее по шести областям\n", file=out)
        print(pd.DataFrame([loao_row(mode, measured[mode]) for mode in MODES]).pipe(md),
              file=out)
        print("\n## 3. Accuracy по областям\n", file=out)
        print(by_area(measured).pipe(md), file=out)
    print(f"записано: {path.relative_to(ROOT)}")
    for mode in MODES:
        got = measured[mode]
        print(f"  {MODES[mode]:38s} AUC CV {got['auc_cv']:.3f}  "
              f"Accuracy CV {got['cv']['accuracy'][0]:.3f}  "
              f"LOAO {got['loao']['accuracy']:.3f}")


if __name__ == "__main__":
    main()
