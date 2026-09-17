from pathlib import Path

import pandas as pd
import pytest

from training.dataset import (
    NEGATIVE_COLUMNS,
    TABLE_COLUMNS,
    build_training_table,
    load_groups,
    load_negatives,
)


def _write_csv(tmp_path: Path, name: str, rows: list[dict], columns: list[str]) -> Path:
    """Пишет CSV во временный файл с кодировкой utf-8-sig."""
    path = tmp_path / name
    pd.DataFrame(rows, columns=columns).to_csv(path, index=False, encoding="utf-8-sig")
    return path


def _valid_negative_row(**overrides) -> dict:
    row = {
        "id": "n1",
        "name": "Технология X",
        "area": "Edge",
        "negative_type": "mature",
        "pair_with": "",
        "criterion": "тест",
        "evidence_url": "https://example.com",
    }
    row.update(overrides)
    return row


def _make_signals(rows: list[tuple[int, str, str]]) -> pd.DataFrame:
    return pd.DataFrame(rows, columns=["id", "name", "area"])


def _make_negatives(rows: list[tuple]) -> pd.DataFrame:
    df = pd.DataFrame(rows, columns=["id", "name", "area", "negative_type", "pair_with"])
    df["pair_with"] = df["pair_with"].astype("Int64")
    return df


# --- load_negatives ---


def test_load_negatives_valid_two_rows(tmp_path):
    rows = [
        _valid_negative_row(id="n1", name="Технология A", pair_with="9"),
        _valid_negative_row(id="n2", name="Технология B", pair_with=""),
    ]
    path = _write_csv(tmp_path, "negatives.csv", rows, NEGATIVE_COLUMNS)

    result = load_negatives(path)

    assert len(result) == 2
    assert (result["label"] == 0).all()
    assert str(result["pair_with"].dtype) == "Int64"
    assert result.loc[result["id"] == "n1", "pair_with"].iloc[0] == 9
    assert pd.isna(result.loc[result["id"] == "n2", "pair_with"].iloc[0])


def test_load_negatives_header_only(tmp_path):
    path = _write_csv(tmp_path, "negatives.csv", [], NEGATIVE_COLUMNS)

    result = load_negatives(path)

    assert len(result) == 0


def test_load_negatives_missing_column_raises(tmp_path):
    columns = [c for c in NEGATIVE_COLUMNS if c != "evidence_url"]
    rows = [{k: v for k, v in _valid_negative_row().items() if k != "evidence_url"}]
    path = _write_csv(tmp_path, "negatives.csv", rows, columns)

    with pytest.raises(ValueError):
        load_negatives(path)


@pytest.mark.parametrize(
    "overrides",
    [
        {"area": "Космос"},
        {"negative_type": "old"},
        {"evidence_url": ""},
        {"evidence_url": "www.site.ru"},
        {"pair_with": "abc"},
        {"id": "1"},
    ],
)
def test_load_negatives_invalid_field_raises(tmp_path, overrides):
    rows = [_valid_negative_row(**overrides)]
    path = _write_csv(tmp_path, "negatives.csv", rows, NEGATIVE_COLUMNS)

    with pytest.raises(ValueError):
        load_negatives(path)


def test_load_negatives_duplicate_id_raises(tmp_path):
    rows = [
        _valid_negative_row(id="n1", name="Технология A"),
        _valid_negative_row(id="n1", name="Технология B"),
    ]
    path = _write_csv(tmp_path, "negatives.csv", rows, NEGATIVE_COLUMNS)

    with pytest.raises(ValueError):
        load_negatives(path)


def test_load_negatives_duplicate_name_raises(tmp_path):
    rows = [
        _valid_negative_row(id="n1", name="Облачные  GPU"),
        _valid_negative_row(id="n2", name=" облачные gpu"),
    ]
    path = _write_csv(tmp_path, "negatives.csv", rows, NEGATIVE_COLUMNS)

    with pytest.raises(ValueError):
        load_negatives(path)


def test_load_negatives_error_lists_both_rows(tmp_path):
    rows = [
        _valid_negative_row(id="n1", name="A", area="Космос"),
        _valid_negative_row(id="n2", name="B"),
        _valid_negative_row(id="n3", name="C", negative_type="old"),
    ]
    path = _write_csv(tmp_path, "negatives.csv", rows, NEGATIVE_COLUMNS)

    with pytest.raises(ValueError) as exc_info:
        load_negatives(path)

    message = str(exc_info.value)
    assert "2" in message
    assert "4" in message


# --- load_groups ---


def test_load_groups_valid(tmp_path):
    rows = [
        {"id": "8", "group": "neuromorphic", "reason": "r"},
        {"id": "45", "group": "neuromorphic", "reason": "r"},
    ]
    path = _write_csv(tmp_path, "groups.csv", rows, ["id", "group", "reason"])

    result = load_groups(path)

    assert result == {8: "neuromorphic", 45: "neuromorphic"}


def test_load_groups_duplicate_id_raises(tmp_path):
    rows = [
        {"id": "8", "group": "a", "reason": "r"},
        {"id": "8", "group": "b", "reason": "r"},
    ]
    path = _write_csv(tmp_path, "groups.csv", rows, ["id", "group", "reason"])

    with pytest.raises(ValueError):
        load_groups(path)


# --- build_training_table ---


def test_build_training_table_columns_and_order():
    signals = _make_signals([(1, "Сигнал A", "Edge"), (2, "Сигнал B", "Роботы")])
    negatives = _make_negatives([("n1", "Не-сигнал A", "Edge", "mature", pd.NA)])

    result = build_training_table(signals, negatives, {})

    assert list(result.columns) == TABLE_COLUMNS
    assert result["tech_id"].is_unique
    assert result["label"].tolist() == [1, 1, 0]


def test_build_training_table_groups():
    signals = _make_signals([(1, "Сигнал A", "Edge"), (2, "Сигнал B", "Роботы")])
    negatives = _make_negatives([("n1", "Не-сигнал A", "Edge", "mature", pd.NA)])
    groups = {1: "photonic"}

    result = build_training_table(signals, negatives, groups)

    assert result.loc[result["tech_id"] == "s1", "group"].iloc[0] == "photonic"
    assert result.loc[result["tech_id"] == "s2", "group"].iloc[0] == "s2"
    assert result.loc[result["tech_id"] == "n1", "group"].iloc[0] == "n1"


def test_build_training_table_unknown_group_id_raises():
    signals = _make_signals([(1, "Сигнал A", "Edge")])
    negatives = _make_negatives([])
    groups = {99: "x"}

    with pytest.raises(ValueError):
        build_training_table(signals, negatives, groups)


def test_build_training_table_unknown_pair_with_raises():
    signals = _make_signals([(1, "Сигнал A", "Edge")])
    negatives = _make_negatives([("n1", "Не-сигнал A", "Edge", "mature", 99)])

    with pytest.raises(ValueError):
        build_training_table(signals, negatives, {})


def test_build_training_table_name_collision_raises():
    signals = _make_signals([(1, "Сигнал A", "Edge")])
    negatives = _make_negatives([("n1", "сигнал a", "Edge", "mature", pd.NA)])

    with pytest.raises(ValueError):
        build_training_table(signals, negatives, {})


def test_build_training_table_empty_negatives():
    signals = _make_signals([(1, "Сигнал A", "Edge")])
    negatives = _make_negatives([])

    result = build_training_table(signals, negatives, {})

    assert len(result) == 1
    assert result["label"].tolist() == [1]
