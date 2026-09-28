"""Посев строится из файлов проекта: 100 сигналов + 60 негативов."""

from __future__ import annotations

from collections import Counter

from db.loaders import (
    AREAS,
    load_negatives,
    load_signal_groups,
    load_signals_xlsx,
    load_stoplist,
    load_technologies_csv,
    merge_training,
    parse_links,
    split_companies,
)


def test_parse_links_markdown() -> None:
    text = "[A](https://a.example/x), [B](https://b.example/y)"
    assert parse_links(text) == [
        {"title": "A", "url": "https://a.example/x"},
        {"title": "B", "url": "https://b.example/y"},
    ]


def test_split_companies() -> None:
    assert split_companies("Keycard, Cyata; Fabrix Security") == [
        "Keycard",
        "Cyata",
        "Fabrix Security",
    ]


def test_training_table_is_160_balanced() -> None:
    table = merge_training(
        load_technologies_csv(),
        load_negatives(),
        load_signal_groups(),
        load_signals_xlsx(),
    )
    assert len(table) == 160
    labels = Counter(row["label"] for row in table)
    assert labels[1] == 100
    assert labels[0] == 60
    areas = {name for name, *_ in AREAS}
    assert {row["area"] for row in table} == areas
    negatives = [row for row in table if row["label"] == 0]
    per_area = Counter(row["area"] for row in negatives)
    assert all(count == 10 for count in per_area.values())
    assert sum(1 for row in negatives if row.get("pair_with")) == 30
    assert all(row.get("evidence") for row in negatives)


def test_signals_xlsx_enriches_when_present() -> None:
    signals = load_signals_xlsx()
    if not signals:
        return
    assert len(signals) == 100
    first = next(item for item in signals if item["tech_id"] == "s1")
    assert first["companies"]
    assert first["expert_score"] == 7
    assert first["sources"]
    table = merge_training(
        load_technologies_csv(),
        load_negatives(),
        load_signal_groups(),
        signals,
    )
    s1 = next(row for row in table if row["tech_id"] == "s1")
    assert s1["rationale"]
    assert s1["stage_raw"]
    assert s1["companies"]
    assert s1["evidence"][0]["kind"] == "dataset_source"


def test_stoplist_and_groups_present() -> None:
    assert len(load_stoplist()) >= 50
    groups = load_signal_groups()
    assert {row["group_name"] for row in groups} >= {
        "neuromorphic",
        "federated",
        "photonic",
    }
