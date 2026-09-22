"""Возраст первого упоминания: какое окно считается первым и что делать с пропусками."""
import math

import pandas as pd
import pytest

from model.first_mention import (CENSORED_AGE, WINDOW_ORDER, age_of_window,
                                               build, first_windows)


def _counters(rows: dict[tuple[str, str], dict[str, int]]) -> pd.DataFrame:
    """Счётчики из словаря {(фраза, источник): {окно: n}} со всеми семью окнами."""
    out = []
    for (key, source), windows in rows.items():
        for window in WINDOW_ORDER:
            out.append({"tech_key": key, "source": source, "window": window,
                        "n": windows.get(window, 0)})
    return pd.DataFrame(out)


def test_first_window_is_the_oldest_nonzero() -> None:
    """Первым считается самое старое окно с ненулевым счётчиком, а не самое большое."""
    counters = _counters({("a", "arxiv"): {"2021": 1, "2023": 99, "2025": 5}})
    assert first_windows(counters, "arxiv").loc["a"] == "2021"


def test_window_before_2020_is_censored() -> None:
    """Окно prev6 накрывает шесть лет: точный возраст неизвестен, берётся метка цензуры."""
    counters = _counters({("a", "arxiv"): {"prev6": 3, "2025": 40}})
    assert first_windows(counters, "arxiv").loc["a"] == "prev6"
    assert age_of_window("prev6") == CENSORED_AGE


def test_no_documents_at_all_is_a_gap() -> None:
    """Ноль во всех семи окнах — это пропуск, а не нулевой возраст."""
    counters = _counters({("a", "arxiv"): {}})
    assert pd.isna(first_windows(counters, "arxiv").loc["a"])
    assert math.isnan(age_of_window(float("nan")))


def test_age_counts_years_to_the_cutoff() -> None:
    """Возраст — годы от начала окна до даты среза 2026-09-01."""
    assert age_of_window("2025") == 1.0
    assert age_of_window("2020") == 6.0


def test_censored_age_is_above_every_yearly_age() -> None:
    """Цензурированное значение обязано быть старше любого годового: иначе порядок врёт."""
    assert CENSORED_AGE > max(age_of_window(w) for w in WINDOW_ORDER if w != "prev6")


def test_lag_is_press_year_minus_arxiv_year(monkeypatch: pytest.MonkeyPatch) -> None:
    """A3 = возраст arXiv минус возраст прессы, то есть год прессы минус год arXiv."""
    counters = _counters({("a", "arxiv"): {"2021": 1}, ("a", "techcrunch"): {"2024": 1}})
    monkeypatch.setattr("model.first_mention.load_counters", lambda: counters)
    got = build(pd.DataFrame({"tech_id": ["t1"], "tech_key": ["a"]}))
    assert got.loc[0, "age_first_arxiv"] == 5.0
    assert got.loc[0, "age_first_press"] == 2.0
    assert got.loc[0, "lag_press_minus_arxiv"] == 3.0


def test_same_phrase_shares_one_cache_row(monkeypatch: pytest.MonkeyPatch) -> None:
    """Две технологии с одной фразой получают одинаковый возраст: кэш ключуется фразой."""
    counters = _counters({("a", "arxiv"): {"2022": 1}, ("a", "techcrunch"): {"2022": 1}})
    monkeypatch.setattr("model.first_mention.load_counters", lambda: counters)
    got = build(pd.DataFrame({"tech_id": ["t1", "t2"], "tech_key": ["a", "a"]}))
    assert got["age_first_arxiv"].tolist() == [4.0, 4.0]


def test_missing_source_gives_gap_not_zero(monkeypatch: pytest.MonkeyPatch) -> None:
    """Если у источника нет ни одного документа, A3 тоже пропуск, а не разность с нулём."""
    counters = _counters({("a", "arxiv"): {"2022": 1}, ("a", "techcrunch"): {}})
    monkeypatch.setattr("model.first_mention.load_counters", lambda: counters)
    got = build(pd.DataFrame({"tech_id": ["t1"], "tech_key": ["a"]}))
    assert got.loc[0, "age_first_arxiv"] == 4.0
    assert pd.isna(got.loc[0, "age_first_press"])
    assert pd.isna(got.loc[0, "lag_press_minus_arxiv"])
