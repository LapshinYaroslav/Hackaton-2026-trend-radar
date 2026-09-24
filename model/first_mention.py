"""Возраст первого упоминания A1-A3 из кэша счётчиков. Без сети.

Счётчики лежат по парам (фраза поиска, источник) и разбиты на семь окон:
prev6 = [2014-09-01, 2020-09-01) и шесть годовых до даты среза 2026-09-01.
Возраст восстанавливается по первому окну с ненулевым счётчиком, поэтому он
известен с точностью до года, а не до дня.

OpenAlex здесь не используется: после ретро-индексации он отдаёт книгу 1934 года
с датой публикации 2026 (находка 014, model/features.py), и дата первого документа
у него не значит ничего.

Запуск: python -m model.first_mention
"""
import json
import math
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
COUNTERS = ROOT / "data" / "interim" / "collector" / "cache" / "counters"
TECHNOLOGIES = ROOT / "data" / "interim" / "technologies.csv"

# Порядок окон от старого к новому. Ключ годового окна — год его начала.
WINDOW_ORDER = ["prev6", "2020", "2021", "2022", "2023", "2024", "2025"]
CUTOFF_YEAR = 2026
# Окно prev6 накрывает шесть лет, и внутри него дата неизвестна: возраст цензурирован
# справа, известно только «больше шести». Значение 9.0 — середина отрезка 6..12.
# На AUC выбор числа не влияет: оно всё равно больше любого годового возраста, а AUC
# зависит только от порядка. На среднее и медиану влияет, поэтому число названо здесь.
CENSORED_AGE = 9.0
SOURCE_COLUMNS = {"arxiv": "age_first_arxiv", "techcrunch": "age_first_press"}


def load_counters() -> pd.DataFrame:
    """Все счётчики из кэша: фраза поиска, источник, окно, число документов."""
    rows = []
    for path in sorted(COUNTERS.glob("*.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        for counter in payload.get("counters", []):
            rows.append({"tech_key": payload["tech_key"], "source": counter["source"],
                         "window": counter["window"], "n": counter["n"]})
    return pd.DataFrame(rows, columns=["tech_key", "source", "window", "n"])


def age_of_window(window: str | float) -> float:
    """Возраст в годах от начала окна до даты среза. prev6 цензурирован."""
    if not isinstance(window, str):
        return math.nan
    if window == "prev6":
        return CENSORED_AGE
    return float(CUTOFF_YEAR - int(window))


def first_window_of(counters: pd.DataFrame, source: str) -> str | float:
    """Первое окно с ненулевым счётчиком у ОДНОЙ технологии. Нет документов — пропуск.

    Окна перебираются от старого к новому, поэтому возвращается именно первое
    упоминание, а не самое заметное.
    """
    block = counters.loc[counters["source"] == source]
    if block.empty:
        return math.nan
    by_window = block.groupby("window")["n"].sum()
    for window in WINDOW_ORDER:
        if float(by_window.get(window, 0.0)) > 0:
            return window
    return math.nan


def age_from_counters(counters: pd.DataFrame, source: str = "arxiv") -> float:
    """Возраст первого упоминания по счётчикам одной технологии.

    Вход — годовые счётчики до агрегации: рабочие окна перекрываются, и по ним
    первое окно не определить.
    """
    return age_of_window(first_window_of(counters, source))


def first_windows(counters: pd.DataFrame, source: str) -> pd.Series:
    """То же по каждой фразе сразу. Считает та же функция, что и для одной технологии."""
    return pd.Series({key: first_window_of(block, source)
                      for key, block in counters.groupby("tech_key", sort=True)},
                     dtype=object)


def build(technologies: pd.DataFrame | None = None) -> pd.DataFrame:
    """Таблица tech_id, A1, A2, A3 плюс сами окна первого упоминания.

    Счётчики ключуются фразой поиска: технологии с одинаковой фразой делят строку
    кэша, поэтому соединение идёт по tech_key, а не по tech_id.
    """
    table = (technologies if technologies is not None
             else pd.read_csv(TECHNOLOGIES, encoding="utf-8-sig"))
    counters = load_counters()
    result = table[["tech_id", "tech_key"]].copy()
    for source, column in SOURCE_COLUMNS.items():
        windows = first_windows(counters, source)
        result.loc[:, column.replace("age_first", "window_first")] = (
            result["tech_key"].map(windows))
        result.loc[:, column] = result["tech_key"].map(windows).map(age_of_window)
    result.loc[:, "lag_press_minus_arxiv"] = (result["age_first_arxiv"]
                                              - result["age_first_press"])
    return result.drop(columns=["tech_key"])


def main() -> None:
    """Печатает покрытие: где первое упоминание нашлось, а где его нет вовсе."""
    frame = build()
    print(frame.head(10).to_string(index=False))
    for column in ("age_first_arxiv", "age_first_press", "lag_press_minus_arxiv"):
        print(f"{column}: заполнено {int(frame[column].notna().sum())} из {len(frame)}")


if __name__ == "__main__":
    main()
