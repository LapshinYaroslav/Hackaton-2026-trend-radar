import math

import pandas as pd

CUTOFF = pd.Timestamp("2026-09-01")
PERIOD_START = pd.Timestamp("2020-09-01")

SOURCE_TYPES = {"paper", "preprint", "patent", "news", "press_release",
                "product", "report", "standard", "blog"}
RESEARCH_TYPES = {"paper", "preprint", "patent"}
MARKET_TYPES = {"news", "press_release", "product"}

FEATURE_NAMES = ["volume", "growth", "recency", "share_research", "share_market"]


def _check_source_types(documents: pd.DataFrame) -> None:
    """Проверяет, что все source_type — из SOURCE_TYPES."""
    unknown = sorted(set(documents["source_type"]) - SOURCE_TYPES)
    if unknown:
        raise ValueError(f"Неизвестные source_type: {unknown}")


def _filter_period(documents: pd.DataFrame, cutoff: pd.Timestamp) -> pd.DataFrame:
    """Оставляет документы с датой в [PERIOD_START, cutoff), убирает NaT."""
    dates = documents["published_at"]
    mask = dates.notna() & (dates >= PERIOD_START) & (dates < cutoff)
    return documents.loc[mask].copy()


def _volume(documents: pd.DataFrame) -> float:
    """ln(1 + число документов)."""
    return math.log1p(len(documents))


def _share(documents: pd.DataFrame, types: set[str]) -> float:
    """Доля документов с source_type из types. NaN, если документов нет."""
    n_all = len(documents)
    if n_all == 0:
        return math.nan
    return documents["source_type"].isin(types).sum() / n_all


def _window_mask(dates: pd.Series, start: pd.Timestamp, end: pd.Timestamp) -> pd.Series:
    """Полуоткрытый интервал [start, end): start входит, end не входит."""
    return (dates >= start) & (dates < end)


def _recency(documents: pd.DataFrame, cutoff: pd.Timestamp) -> float:
    """Доля документов за последние 24 месяца. NaN, если документов нет."""
    n_all = len(documents)
    if n_all == 0:
        return math.nan
    mask = _window_mask(documents["published_at"], cutoff - pd.DateOffset(months=24), cutoff)
    return mask.sum() / n_all


def _growth(documents: pd.DataFrame, source_totals: pd.DataFrame, cutoff: pd.Timestamp) -> float:
    """ln((n_now+1)/(n_before+1)) - ln(N_now/N_before) по общим источникам."""
    if len(documents) == 0:
        return math.nan

    totals_now = source_totals.loc[source_totals["window"] == "now"].set_index("source")["n_total"]
    totals_before = source_totals.loc[source_totals["window"] == "before"].set_index("source")["n_total"]

    doc_sources = set(documents["source"])
    valid_sources = doc_sources & set(totals_now.index) & set(totals_before.index)
    if not valid_sources:
        return math.nan

    docs_valid = documents.loc[documents["source"].isin(valid_sources)]
    now_start, now_end = cutoff - pd.DateOffset(months=12), cutoff
    before_start, before_end = cutoff - pd.DateOffset(months=36), cutoff - pd.DateOffset(months=24)
    n_now = _window_mask(docs_valid["published_at"], now_start, now_end).sum()
    n_before = _window_mask(docs_valid["published_at"], before_start, before_end).sum()

    n_total_now = totals_now.loc[list(valid_sources)].sum()
    n_total_before = totals_before.loc[list(valid_sources)].sum()

    return math.log((n_now + 1) / (n_before + 1)) - math.log(n_total_now / n_total_before)


def compute_features(documents: pd.DataFrame,
                      source_totals: pd.DataFrame,
                      cutoff: pd.Timestamp = CUTOFF) -> dict[str, float]:
    """Признаки volume, growth, recency, share_research, share_market по документам технологии."""
    _check_source_types(documents)
    docs = _filter_period(documents, cutoff)

    values = {
        "volume": _volume(docs),
        "growth": _growth(docs, source_totals, cutoff),
        "recency": _recency(docs, cutoff),
        "share_research": _share(docs, RESEARCH_TYPES),
        "share_market": _share(docs, MARKET_TYPES),
    }
    return {name: values[name] for name in FEATURE_NAMES}