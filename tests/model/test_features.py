import math

import pandas as pd
import pytest

from model.features import FEATURE_NAMES, compute_features


def _make_documents(rows: list[tuple[str, str, str]]) -> pd.DataFrame:
    df = pd.DataFrame(rows, columns=["published_at", "source", "source_type"])
    df["published_at"] = pd.to_datetime(df["published_at"])
    return df


def _make_source_totals(rows: list[tuple[str, str, int]]) -> pd.DataFrame:
    return pd.DataFrame(rows, columns=["source", "window", "n_total"])


def test_growth_example_a():
    rows = [("2023-10-01", "s1", "paper")] * 12 + [("2025-10-01", "s1", "paper")] * 60
    documents = _make_documents(rows)
    source_totals = _make_source_totals([("s1", "before", 1000), ("s1", "now", 1500)])

    features = compute_features(documents, source_totals)

    assert features["growth"] == pytest.approx(1.1405, abs=1e-3)


def test_growth_example_b():
    rows = [("2023-10-01", "s1", "paper")] * 700 + [("2025-10-01", "s1", "paper")] * 900
    documents = _make_documents(rows)
    source_totals = _make_source_totals([("s1", "before", 1000), ("s1", "now", 1500)])

    features = compute_features(documents, source_totals)

    assert features["growth"] == pytest.approx(-0.1545, abs=1e-3)


def test_volume_140_documents():
    rows = [("2022-01-01", "s1", "paper")] * 140
    documents = _make_documents(rows)
    source_totals = _make_source_totals([])

    features = compute_features(documents, source_totals)

    assert features["volume"] == pytest.approx(4.9488, abs=1e-3)


@pytest.mark.parametrize(
    "date, is_included",
    [
        ("2025-09-01", True),
        ("2025-08-31", False),
        ("2026-08-31", True),
    ],
)
def test_now_window_boundaries(date, is_included):
    documents = _make_documents([(date, "s1", "paper")])
    source_totals = _make_source_totals([("s1", "now", 100), ("s1", "before", 100)])

    features = compute_features(documents, source_totals)

    expected = math.log(2) if is_included else 0.0
    assert features["growth"] == pytest.approx(expected, abs=1e-3)


def test_cutoff_date_excluded_everywhere():
    documents = _make_documents([("2026-09-01", "s1", "paper")])
    source_totals = _make_source_totals([])

    features = compute_features(documents, source_totals)

    assert features["volume"] == 0
    assert math.isnan(features["recency"])
    assert math.isnan(features["growth"])
    assert math.isnan(features["share_research"])
    assert math.isnan(features["share_market"])


@pytest.mark.parametrize(
    "date, is_included",
    [
        ("2023-09-01", True),
        ("2024-08-31", True),
        ("2024-09-01", False),
    ],
)
def test_before_window_boundaries(date, is_included):
    documents = _make_documents([(date, "s1", "paper")])
    source_totals = _make_source_totals([("s1", "now", 100), ("s1", "before", 100)])

    features = compute_features(documents, source_totals)

    expected = math.log(0.5) if is_included else 0.0
    assert features["growth"] == pytest.approx(expected, abs=1e-3)


def test_period_start_boundary():
    documents_before_start = _make_documents([("2020-08-31", "s1", "paper")])
    documents_at_start = _make_documents([("2020-09-01", "s1", "paper")])
    source_totals = _make_source_totals([])

    features_before_start = compute_features(documents_before_start, source_totals)
    features_at_start = compute_features(documents_at_start, source_totals)

    assert features_before_start["volume"] == 0
    assert features_at_start["volume"] == pytest.approx(math.log(2), abs=1e-3)


def test_zero_documents_in_before_window():
    rows = [("2025-10-01", "s1", "paper")] * 5
    documents = _make_documents(rows)
    source_totals = _make_source_totals([("s1", "now", 100), ("s1", "before", 100)])

    features = compute_features(documents, source_totals)

    assert math.isfinite(features["growth"])


def test_no_documents():
    documents = _make_documents([])
    source_totals = _make_source_totals([])

    features = compute_features(documents, source_totals)

    assert features["volume"] == 0
    assert math.isnan(features["recency"])
    assert math.isnan(features["growth"])
    assert math.isnan(features["share_research"])
    assert math.isnan(features["share_market"])


def test_missing_date_is_ignored():
    rows = [("2022-01-01", "s1", "paper")] * 140
    documents = _make_documents(rows)
    documents_with_nat = pd.concat(
        [documents, pd.DataFrame([{"published_at": pd.NaT, "source": "s1", "source_type": "paper"}])],
        ignore_index=True,
    )
    source_totals = _make_source_totals([])

    features_plain = compute_features(documents, source_totals)
    features_with_nat = compute_features(documents_with_nat, source_totals)

    assert features_with_nat["volume"] == pytest.approx(features_plain["volume"], abs=1e-9)


def test_shares_on_mixed_types():
    rows = (
        [("2022-01-01", "s1", "paper")] * 3
        + [("2022-01-01", "s1", "news")] * 1
        + [("2022-01-01", "s1", "blog")] * 1
    )
    documents = _make_documents(rows)
    source_totals = _make_source_totals([])

    features = compute_features(documents, source_totals)

    assert features["share_research"] == pytest.approx(0.6, abs=1e-3)
    assert features["share_market"] == pytest.approx(0.2, abs=1e-3)


def test_unknown_source_type_raises():
    documents = _make_documents([("2022-01-01", "s1", "tweet")])
    source_totals = _make_source_totals([])

    with pytest.raises(ValueError):
        compute_features(documents, source_totals)


def test_source_without_totals():
    rows = [("2022-01-01", "s_missing", "paper")] * 10
    documents = _make_documents(rows)
    source_totals = _make_source_totals([("s_other", "now", 100), ("s_other", "before", 100)])

    features = compute_features(documents, source_totals)

    assert math.isnan(features["growth"])
    assert features["volume"] == pytest.approx(math.log(11), abs=1e-3)
    assert features["share_research"] == pytest.approx(1.0, abs=1e-3)


def test_result_key_order():
    documents = _make_documents([])
    source_totals = _make_source_totals([])

    features = compute_features(documents, source_totals)

    assert list(features.keys()) == FEATURE_NAMES
