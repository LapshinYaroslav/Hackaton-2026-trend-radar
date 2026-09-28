"""Е1: числа по смешанному пулу и порог LOAO как в гейте."""
import numpy as np
import pandas as pd
import pytest

from tests.data_required import needs_model, needs_training

from scripts.e_mix import cached_counters, inner_threshold, numbers, training_rows


def _rows() -> pd.DataFrame:
    rows = pd.DataFrame({"kind": ["сигнал", "пул", "мейнстрим", "пул", "сигнал"],
                         "score": [0.9, 0.8, 0.6, 0.3, 0.2]})
    rows["ранг"] = np.arange(1, 6)
    rows["в ТОП"] = rows["score"] >= 0.5
    return rows


def test_numbers_counts_top_ranks_and_shares() -> None:
    """ТОП только выше порога; медианы рангов и доли — по своим строкам."""
    got = numbers(_rows(), 0.5)
    assert (got["мест в ТОП"], got["сигнал в ТОП"], got["мейнстрим в ТОП"], got["пул в ТОП"]) == (3, 1, 1, 1)
    assert got["медианный ранг: сигнал"] == 3.0 and got["медианный ранг: пул"] == 3.0
    assert got["доля сигналов выше порога"] == 0.5 and got["доля пула выше порога"] == 0.5


def test_numbers_without_pool_rows_gives_nan_not_error() -> None:
    """Пустой пул: медиана и доля — NaN, счётчики — нули."""
    got = numbers(_rows().loc[lambda x: x["kind"] != "пул"], 0.5)
    assert got["кандидатов пула"] == 0 and np.isnan(got["медианный ранг: пул"])


@needs_training
def test_variant_a_threshold_equals_gate() -> None:
    """Вариант А: порог LOAO совпадает с choose_inside гейта."""
    from model.config import FEATURES_S2A2
    from scripts.tuning import CURRENT, _area_splits, choose_inside
    frame = training_rows()
    train = np.flatnonzero((frame["area"] != "Финтех").to_numpy())
    gate = choose_inside(frame, train, "accuracy", _area_splits(frame, train), [CURRENT], columns=FEATURES_S2A2)[1]
    assert inner_threshold(frame, train, own_stats=False) == gate


def test_missing_counters_is_an_error() -> None:
    """Нет счётчиков в кэше — ошибка, а не пустые признаки."""
    with pytest.raises(FileNotFoundError):
        cached_counters("такого кандидата нет в кэше", "openalex")


@needs_model
def test_inputs_in_evidence_reproduce_stored_scores() -> None:
    """evidence/e1_inputs/pools.json: 9 пулов, признаки s2a2-v1 из счётчиков, score прогона воспроизводится."""
    import json
    from model.config import FEATURES_S2A2
    from scripts.e_mix import INPUTS, POOLS, pool
    data = json.loads(INPUTS.read_text(encoding="utf-8"))
    assert list(data) == list(POOLS)
    frame, check = pool("Д агро A")
    assert len(frame) == check["кандидатов"] == 38 and set(FEATURES_S2A2) <= set(frame.columns)
    assert check["макс. |Δscore|"] < 0.01
