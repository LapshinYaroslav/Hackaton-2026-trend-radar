"""Годовые счётчики источников в рабочие окна признаков.

Источник опрашивается по шести непересекающимся годам, а признакам нужны окна
all, now, before, recent24, которые друг друга перекрывают. Складываем годы здесь,
без сети: один и тот же год участвует в нескольких окнах, но оплачен один раз.

Работает одинаково для счётчиков технологии (колонка n) и для корпусных итогов
источника (колонка n_total) — таблицы одной формы, отличается имя колонки со значением.
"""
import pandas as pd

from collector.constants import AGGREGATE_WINDOWS, ALLOWED_COUNTER_WINDOWS


def _check_years(rows: pd.DataFrame) -> None:
    """Проверяет, что на входе именно годовые окна, а не уже сложенные."""
    unknown = sorted(set(rows["window"].astype(str)) - ALLOWED_COUNTER_WINDOWS)
    if unknown:
        raise ValueError(
            f"Не годовые окна на входе агрегации: {unknown}. Ожидаются "
            f"{sorted(ALLOWED_COUNTER_WINDOWS)}. Повторное сложение уже сложенного "
            f"удвоило бы числа и осталось бы незамеченным."
        )


def aggregate_windows(
    rows: pd.DataFrame,
    value_column: str = "n",
    windows: dict[str, tuple[str, ...]] | None = None,
) -> pd.DataFrame:
    """Складывает годовые счётчики в рабочие окна. Колонки на выходе те же, что на входе.

    Год, которого нет у источника, делает неизвестными все окна, в которые он входит:
    такая строка не выводится вовсе. Частичная сумма молча занизила бы volume, и
    «мало публикаций» стало бы неотличимо от «источник не ответил».
    """
    _check_years(rows)
    parts = windows or AGGREGATE_WINDOWS
    columns = ["source", "window", value_column]
    if rows.empty:
        return pd.DataFrame(columns=columns)

    by_year = rows.groupby(["source", "window"])[value_column].sum()
    result: list[dict[str, object]] = []
    for source, values in by_year.groupby(level="source"):
        have = {str(window): int(value) for (_, window), value in values.items()}
        for name, years in parts.items():
            if not set(years) <= have.keys():
                continue
            result.append({"source": str(source), "window": name,
                           value_column: sum(have[year] for year in years)})
    return pd.DataFrame(result, columns=columns)
