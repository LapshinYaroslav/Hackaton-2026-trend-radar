"""Сложение годовых счётчиков в рабочие окна."""
import pandas as pd
import pytest

from collector.constants import AGGREGATE_WINDOWS, COUNTER_WINDOWS
from model.counters import aggregate_windows

YEARS = ["2020", "2021", "2022", "2023", "2024", "2025"]


def _years(source: str, values: list[int], column: str = "n") -> pd.DataFrame:
    return pd.DataFrame({"source": source, "window": YEARS, column: values})


def _value(frame: pd.DataFrame, source: str, window: str, column: str = "n") -> int:
    rows = frame.loc[(frame["source"] == source) & (frame["window"] == window), column]
    return int(rows.iloc[0])


def test_aggregate_windows_match_constants():
    """Границы рабочих окон после сложения — те же даты, что были до годовых окон."""
    assert COUNTER_WINDOWS[AGGREGATE_WINDOWS["now"][0]] == (pd.Timestamp("2025-09-01").date(),
                                                            pd.Timestamp("2026-09-01").date())
    assert COUNTER_WINDOWS[AGGREGATE_WINDOWS["before"][0]] == (pd.Timestamp("2023-09-01").date(),
                                                               pd.Timestamp("2024-09-01").date())
    first, last = AGGREGATE_WINDOWS["recent24"]
    assert COUNTER_WINDOWS[first][0] == pd.Timestamp("2024-09-01").date()
    assert COUNTER_WINDOWS[last][1] == pd.Timestamp("2026-09-01").date()
    assert list(AGGREGATE_WINDOWS["all"]) == YEARS


def test_sums_years_into_windows():
    rows = _years("openalex", [1, 2, 4, 8, 16, 32])

    result = aggregate_windows(rows)

    assert _value(result, "openalex", "all") == 63
    assert _value(result, "openalex", "now") == 32
    assert _value(result, "openalex", "before") == 8
    assert _value(result, "openalex", "recent24") == 48


def test_sources_are_summed_separately():
    rows = pd.concat([_years("openalex", [1, 1, 1, 1, 1, 1]),
                      _years("techcrunch", [0, 0, 0, 0, 0, 5])], ignore_index=True)

    result = aggregate_windows(rows)

    assert _value(result, "openalex", "all") == 6
    assert _value(result, "techcrunch", "all") == 5
    assert _value(result, "techcrunch", "before") == 0


def test_missing_year_removes_windows_that_need_it():
    """Нет 2022 года — нет окна all, но now и before считаются: они его не включают."""
    rows = _years("openalex", [1, 2, 4, 8, 16, 32])
    rows = rows.loc[rows["window"] != "2022"]

    result = aggregate_windows(rows)

    windows = set(result.loc[result["source"] == "openalex", "window"])
    assert windows == {"now", "before", "recent24"}


def test_zero_year_is_not_a_missing_year():
    """Ноль — это ответ источника, окно считается."""
    rows = _years("arxiv", [0, 0, 0, 0, 0, 0])

    result = aggregate_windows(rows)

    assert _value(result, "arxiv", "all") == 0
    # prev6 в этих строках нет, поэтому и в результате его быть не должно.
    assert set(result["window"]) == set(AGGREGATE_WINDOWS) - {"prev6"}


def test_works_on_corpus_totals_column():
    rows = _years("openalex", [10, 20, 30, 40, 50, 60], column="n_total")

    result = aggregate_windows(rows, value_column="n_total")

    assert list(result.columns) == ["source", "window", "n_total"]
    assert _value(result, "openalex", "all", column="n_total") == 210


def test_already_aggregated_input_raises():
    """Повторное сложение удвоило бы числа: ловим на входе, а не по странным признакам."""
    rows = pd.DataFrame({"source": ["openalex"], "window": ["all"], "n": [100]})

    with pytest.raises(ValueError, match="all"):
        aggregate_windows(rows)


def test_empty_input_gives_empty_frame():
    rows = pd.DataFrame({"source": [], "window": [], "n": []})

    result = aggregate_windows(rows)

    assert result.empty
    assert list(result.columns) == ["source", "window", "n"]


def test_duplicate_year_rows_are_summed_not_dropped():
    """Две строки за один год у одного источника — сумма, а не молчаливая потеря одной."""
    rows = pd.DataFrame({"source": ["openalex"] * 7,
                         "window": YEARS + ["2025"],
                         "n": [1, 1, 1, 1, 1, 1, 9]})

    result = aggregate_windows(rows)

    assert _value(result, "openalex", "now") == 10
    assert _value(result, "openalex", "all") == 15
