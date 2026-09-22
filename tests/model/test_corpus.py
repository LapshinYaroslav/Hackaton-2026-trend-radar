"""Корпусные итоги: отсутствие или неполнота — отказ, а не тихий переход на кэш."""
import json

import pandas as pd
import pytest

from collector.constants import YEAR_WINDOWS
from model.corpus import (EXPECTED_SOURCES, CorpusTotalsError, load_training_totals,
                          TRAINING_TOTALS)


def _rows(sources=None, windows=None, n_total: int = 1000) -> list[dict]:
    """Полный набор пар источник-окно, если не сказано иное."""
    return [{"source": source, "window": window, "n_total": n_total, "available": True}
            for source in (sources or sorted(EXPECTED_SOURCES))
            for window in (windows or sorted(YEAR_WINDOWS))]


def _write(tmp_path, rows) -> object:
    path = tmp_path / "totals.json"
    path.write_text(json.dumps(rows, ensure_ascii=False), encoding="utf-8")
    return path


def test_real_file_is_present_and_complete() -> None:
    """Файл в репозитории существует и проходит проверку состава."""
    frame = load_training_totals()
    assert len(frame) == len(EXPECTED_SOURCES) * len(YEAR_WINDOWS)
    assert set(frame["source"]) == set(EXPECTED_SOURCES)
    assert TRAINING_TOTALS.exists()


def test_missing_file_is_a_clear_refusal(tmp_path) -> None:
    """Нет файла — понятная ошибка, а не пустая таблица и не живой кэш."""
    with pytest.raises(CorpusTotalsError) as error:
        load_training_totals(tmp_path / "нет-такого.json")
    assert "живому кэшу" in str(error.value)


def test_missing_window_is_named_in_the_error(tmp_path) -> None:
    """Не хватает окна — ошибка называет, какой именно пары нет."""
    rows = [row for row in _rows() if row["window"] != "2023"]
    with pytest.raises(CorpusTotalsError) as error:
        load_training_totals(_write(tmp_path, rows))
    assert "2023" in str(error.value)


def test_missing_source_is_named_in_the_error(tmp_path) -> None:
    """Не хватает источника — ошибка называет его."""
    rows = [row for row in _rows() if row["source"] != "techcrunch"]
    with pytest.raises(CorpusTotalsError) as error:
        load_training_totals(_write(tmp_path, rows))
    assert "techcrunch" in str(error.value)


def test_unavailable_rows_count_as_missing(tmp_path) -> None:
    """Строка с available=false — это «источник не ответил», а не ноль."""
    rows = _rows()
    for row in rows:
        if row["source"] == "arxiv":
            row["available"] = False
    with pytest.raises(CorpusTotalsError):
        load_training_totals(_write(tmp_path, rows))


def test_zero_corpus_is_refused(tmp_path) -> None:
    """Нулевой корпус в знаменателе поправки недопустим."""
    rows = _rows()
    rows[0]["n_total"] = 0
    with pytest.raises(CorpusTotalsError) as error:
        load_training_totals(_write(tmp_path, rows))
    assert "нулевой корпус" in str(error.value).lower()


def test_broken_json_is_a_clear_refusal(tmp_path) -> None:
    """Битый файл — понятная ошибка, а не traceback из json."""
    path = tmp_path / "totals.json"
    path.write_text("{не json", encoding="utf-8")
    with pytest.raises(CorpusTotalsError):
        load_training_totals(path)


def test_extra_windows_do_not_break_the_check(tmp_path) -> None:
    """Лишнее окно сверх набора не мешает: проверяется наличие нужных, не отсутствие иных."""
    rows = _rows() + [{"source": "arxiv", "window": "prev6", "n_total": 5,
                       "available": True}]
    assert len(load_training_totals(_write(tmp_path, rows))) == len(rows)
