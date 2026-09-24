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


# Снимок патентных счётчиков обучения (s2a2-v1).
from model.config import ROSPATENT_DATASETS  # noqa: E402
from model.corpus import (PatentSnapshotError, TRAINING_PATENTS,  # noqa: E402
                          load_training_patents)


def _snapshot(tmp_path, items, datasets=None):
    path = tmp_path / "patents.json"
    path.write_text(json.dumps({"datasets": datasets or ROSPATENT_DATASETS, "items": items}),
                    encoding="utf-8")
    return path


def _items(count=160, n_pat=3):
    return [{"tech_id": f"t{i}", "phrase": f"p{i}", "n_pat": n_pat} for i in range(count)]


def test_real_patent_snapshot_has_160_rows() -> None:
    frame = load_training_patents(ROSPATENT_DATASETS)
    assert len(frame) == 160 and frame["tech_id"].is_unique and (frame["n_pat"] >= 0).all()
    assert TRAINING_PATENTS.exists()


def test_missing_patent_snapshot_is_a_refusal(tmp_path) -> None:
    with pytest.raises(PatentSnapshotError):
        load_training_patents(ROSPATENT_DATASETS, tmp_path / "нет.json")


def test_incomplete_patent_snapshot_is_a_refusal(tmp_path) -> None:
    with pytest.raises(PatentSnapshotError):
        load_training_patents(ROSPATENT_DATASETS, _snapshot(tmp_path, _items(159)))


def test_negative_count_in_snapshot_is_a_refusal(tmp_path) -> None:
    items = _items()
    items[5]["n_pat"] = -1
    with pytest.raises(PatentSnapshotError):
        load_training_patents(ROSPATENT_DATASETS, _snapshot(tmp_path, items))


def test_snapshot_of_other_datasets_is_a_refusal(tmp_path) -> None:
    with pytest.raises(PatentSnapshotError):
        load_training_patents(ROSPATENT_DATASETS, _snapshot(tmp_path, _items(), ["us"]))


def test_training_share_patent_equals_p0_on_every_row() -> None:
    """share_patent обучающей таблицы (снимок) совпадает с П0 (кэш сбора) на всех 160 строках."""
    import numpy as np

    from model.train import training_table
    from scripts.ru_patents_p0 import feature_table
    stored = training_table().set_index("tech_id")["share_patent"]
    p0 = feature_table().set_index("tech_id")["share_patent"].loc[stored.index]
    assert len(stored) == 160
    assert (stored.isna() == p0.isna()).all()
    assert np.nanmax((stored - p0).abs().to_numpy()) <= 1e-12
