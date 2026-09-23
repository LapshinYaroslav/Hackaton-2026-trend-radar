from datetime import date, datetime, timedelta, timezone
import math

import pytest

from collector.adapters.base import totals_signature
from collector.api import DocumentCollector
from collector.constants import (
    SOURCE_TOTALS_TTL_HOURS,
    WINDOW_BEFORE_END,
    WINDOW_BEFORE_START,
    WINDOW_NOW_END,
    WINDOW_NOW_START,
)
from collector.db import MemoryCache
from collector.exceptions import InvalidSourceTypeError
from collector.models import Candidate, Document, SourceTotal, Technology
from tests.collector.fakes import FakeAdapter, doc

BOTH_WINDOWS = {
    (WINDOW_BEFORE_START, WINDOW_BEFORE_END): 1000,
    (WINDOW_NOW_START, WINDOW_NOW_END): 1250,
}


def _collector(*adapters: FakeAdapter) -> DocumentCollector:
    return DocumentCollector(adapters=list(adapters))


def implied_features(result) -> dict[str, float]:
    """Contract with Yaroslav: collector does not compute this; empty docs => volume=0, rest NaN."""
    if not result.documents:
        return {
            "volume": 0.0,
            "growth": math.nan,
            "freshness": math.nan,
            "share_research": math.nan,
        }
    raise AssertionError("implied_features is only for the empty-document contract")


def test_s1_s2_source_type_dates_and_totals() -> None:
    s1 = FakeAdapter(
        "s1",
        "paper",
        documents=[
            doc(
                source="s1",
                source_type="paper",
                published_at="2024-01-15",
                url="https://s1.example/a",
            )
        ],
        totals=BOTH_WINDOWS,
    )
    s2 = FakeAdapter(
        "s2",
        "news",
        documents=[
            doc(
                source="s2",
                source_type="news",
                published_at="2025-12-01",
                url="https://s2.example/b",
                trust_level="medium",
            )
        ],
        totals={
            (WINDOW_BEFORE_START, WINDOW_BEFORE_END): 400,
            (WINDOW_NOW_START, WINDOW_NOW_END): 480,
        },
    )
    collector = _collector(s1, s2)
    totals = collector.probe_source_totals()
    by_source = {(row.source, row.window): row for row in totals}

    assert by_source[("s1", "before")].n_total == 1000
    assert by_source[("s1", "now")].n_total == 1250
    assert by_source[("s1", "before")].available is True
    assert by_source[("s2", "before")].n_total == 400
    assert by_source[("s2", "now")].n_total == 480

    result = collector.collect_history(
        Candidate(candidate_id="c1", name_en="photonic inference processors")
    )
    dates = {item.url: item.published_at for item in result.documents}
    types = {item.url: item.source_type for item in result.documents}
    assert dates["https://s1.example/a"] == date(2024, 1, 15)
    assert dates["https://s2.example/b"] == date(2025, 12, 1)
    assert types["https://s1.example/a"] == "paper"
    assert types["https://s2.example/b"] == "news"
    assert result.to_contract_dict()["candidate_id"] == "c1"


def test_invalid_source_type_fails_immediately() -> None:
    adapter = FakeAdapter("s1", "paper", fail_source_type="social")
    collector = _collector(adapter)
    with pytest.raises(InvalidSourceTypeError) as exc:
        collector.collect_history(Candidate(candidate_id="c1", name_en="x"))
    assert exc.value.source_type == "social"


def test_document_model_rejects_unknown_source_type() -> None:
    with pytest.raises(InvalidSourceTypeError):
        Document(
            published_at="2024-01-01",
            source="s1",
            source_type="tweet",
            title="x",
            url="https://s1.example/x",
            trust_level="high",
            organizations=[],
            text="",
        )


def test_empty_result_volume_zero_other_nan() -> None:
    s1 = FakeAdapter("s1", "paper", documents=[], totals=BOTH_WINDOWS)
    collector = _collector(s1)
    result = collector.collect_history(Candidate(candidate_id="c-empty", name_en="unknown tech"))
    payload = result.to_contract_dict()
    assert payload["documents"] == []
    assert payload["source_totals"] == [
        {"source": "s1", "window": "before", "n_total": 1000},
        {"source": "s1", "window": "now", "n_total": 1250},
    ]
    features = implied_features(result)
    assert features["volume"] == 0
    assert math.isnan(features["growth"])
    assert math.isnan(features["freshness"])
    assert math.isnan(features["share_research"])


def test_documents_without_date_or_outside_window_are_dropped() -> None:
    s1 = FakeAdapter(
        "s1",
        "paper",
        documents=[
            doc(source="s1", source_type="paper", published_at="2019-12-31", url="https://s1/old"),
            doc(source="s1", source_type="paper", published_at="2026-09-01", url="https://s1/cutoff"),
            doc(source="s1", source_type="paper", published_at="2021-06-01", url="https://s1/ok"),
        ],
        totals=BOTH_WINDOWS,
    )
    result = _collector(s1).collect_history(Candidate(candidate_id="c1", name_en="x"))
    assert [item.url for item in result.documents] == ["https://s1/ok"]


def test_source_without_both_windows_is_omitted_from_growth_totals() -> None:
    s1 = FakeAdapter("s1", "paper", totals=BOTH_WINDOWS)
    s2 = FakeAdapter(
        "s2",
        "blog",
        totals={(WINDOW_NOW_START, WINDOW_NOW_END): 10},
    )
    totals = _collector(s1, s2).probe_source_totals()
    assert all(row.available for row in totals if row.source == "s1")
    assert all(not row.available for row in totals if row.source == "s2")
    contract = _collector(s1, s2).collect_history(
        Candidate(candidate_id="c1", name_en="x")
    ).to_contract_dict()
    assert {row["source"] for row in contract["source_totals"]} == {"s1"}


def test_training_and_query_share_the_same_result_shape() -> None:
    s1 = FakeAdapter(
        "s1",
        "paper",
        documents=[
            doc(source="s1", source_type="paper", published_at="2022-01-01", url="https://s1/1")
        ],
        totals=BOTH_WINDOWS,
    )
    collector = _collector(s1)
    trained = collector.collect_training(
        [Technology(tech_id="c42", name_en="photonic inference processors")]
    )[0]
    queried = collector.collect_history(
        Candidate(
            candidate_id="c42",
            query_id="q7",
            name_en="photonic inference processors",
            aliases=["optical AI accelerator"],
        )
    )
    assert trained.to_contract_dict()["documents"] == queried.to_contract_dict()["documents"]
    assert trained.to_contract_dict()["source_totals"] == queried.to_contract_dict()["source_totals"]
    assert queried.cache_hit is True


def test_search_recent_does_not_apply_feature_window() -> None:
    s1 = FakeAdapter(
        "s1",
        "news",
        documents=[
            doc(
                source="s1",
                source_type="news",
                published_at="2026-09-10",
                url="https://s1/fresh",
                trust_level="medium",
            )
        ],
        totals=BOTH_WINDOWS,
    )
    recent = _collector(s1).search_recent(
        ["photonic processors"],
        date_from=date(2026, 9, 1),
        date_to_exclusive=date(2026, 9, 18),
    )
    history = _collector(s1).collect_history(Candidate(candidate_id="c1", name_en="photonic processors"))
    assert [item.url for item in recent.documents] == ["https://s1/fresh"]
    assert history.documents == []


def test_weak_sources_alone_lower_confirmation() -> None:
    s1 = FakeAdapter(
        "s1",
        "blog",
        documents=[
            doc(
                source="s1",
                source_type="blog",
                published_at="2024-05-01",
                url="https://s1/blog",
                trust_level="low",
            )
        ],
        totals=BOTH_WINDOWS,
    )
    result = _collector(s1).collect_history(Candidate(candidate_id="c1", name_en="x"))
    assert result.independent_confirmation is False


def test_candidate_cap_and_mainstream_cut_before_search2() -> None:
    s1 = FakeAdapter("s1", "paper", totals=BOTH_WINDOWS)
    collector = _collector(s1)
    candidates = [
        Candidate(candidate_id="m1", name_en="transformer"),
        Candidate(candidate_id="c1", name_en="photonic inference"),
        Candidate(candidate_id="c2", name_en="optical analog compute"),
    ]
    results = collector.collect_histories(
        candidates,
        max_candidates=1,
        is_mainstream=lambda item: item.candidate_id == "m1",
    )
    assert results[0].skipped_as_mainstream is True
    assert results[1].skipped_as_mainstream is False
    assert results[2].skipped_as_mainstream is True
    assert s1.search_calls
    searched_ids = {call[0] for call in s1.search_calls}
    assert "photonic inference" in searched_ids
    assert "transformer" not in searched_ids


def _cache_with_totals(age_hours: float, n_total: int = 7) -> MemoryCache:
    """Кэш с готовыми итогами источника s1 заданного возраста.

    Подпись способа подсчёта берётся у того же FakeAdapter: проверяем именно
    протухание по времени, а не смену способа.
    """
    cache = MemoryCache()
    stamp = datetime.now(timezone.utc) - timedelta(hours=age_hours)
    cache.put_source_totals(
        [
            SourceTotal(source="s1", window=window, n_total=n_total, collected_at=stamp,
                        type_filter=None, totals_signature=totals_signature(FakeAdapter("s1", "paper")))
            for window in ("before", "now")
        ]
    )
    return cache


def test_fresh_totals_are_taken_from_cache() -> None:
    s1 = FakeAdapter("s1", "paper", totals=BOTH_WINDOWS)
    totals = DocumentCollector(
        adapters=[s1], cache=_cache_with_totals(age_hours=1)
    ).probe_source_totals()

    assert s1.count_calls == []
    assert {row.n_total for row in totals} == {7}


def test_stale_totals_are_probed_again() -> None:
    """Итоги старше суток перепрашиваются: корпус источника за это время меняется."""
    s1 = FakeAdapter("s1", "paper", totals=BOTH_WINDOWS)
    totals = DocumentCollector(
        adapters=[s1], cache=_cache_with_totals(age_hours=SOURCE_TOTALS_TTL_HOURS + 1)
    ).probe_source_totals()

    assert len(s1.count_calls) == 2
    assert {row.n_total for row in totals} == {1000, 1250}


def test_unavailable_source_is_retried_by_a_new_run() -> None:
    """Одна неудачная проба не выключает источник из growth навсегда."""
    cache = MemoryCache()
    broken = FakeAdapter("s1", "paper", totals={})
    DocumentCollector(adapters=[broken], cache=cache).probe_source_totals()
    assert all(not row.available for row in cache.get_source_totals() or [])

    fixed = FakeAdapter("s1", "paper", totals=BOTH_WINDOWS)
    totals = DocumentCollector(adapters=[fixed], cache=cache).probe_source_totals()

    assert len(fixed.count_calls) == 2
    assert all(row.available for row in totals)


def test_broken_source_is_probed_once_per_process() -> None:
    """Внутри одного прогона недоступный источник не перепрашивается на каждой технологии."""
    broken = FakeAdapter("s1", "paper", totals={})
    collector = DocumentCollector(adapters=[broken], cache=MemoryCache())
    for _ in range(3):
        collector.probe_source_totals()
    assert len(broken.count_calls) == 2

    collector.probe_source_totals(force=True)
    assert len(broken.count_calls) == 4


def _totals_rows(source: str, windows: tuple[str, ...], n_total: int,
                 age_hours: float = 1.0) -> list[SourceTotal]:
    """Готовые строки итогов одного источника заданного возраста."""
    stamp = datetime.now(timezone.utc) - timedelta(hours=age_hours)
    return [SourceTotal(source=source, window=window, n_total=n_total, collected_at=stamp,
                        type_filter=None,
                        totals_signature=totals_signature(FakeAdapter(source, "paper")))
            for window in windows]


def test_partial_run_keeps_totals_of_other_sources() -> None:
    """Прогон по одному источнику не стирает из кэша итоги остальных.

    Без слияния фоновый добор одного arXiv унёс бы корпуса OpenAlex и TechCrunch,
    и growth перестал бы считаться у всех технологий сразу: поправку на фон брать
    было бы неоткуда.
    """
    cache = MemoryCache()
    cache.put_source_totals(_totals_rows("s2", ("before", "now"), n_total=500))

    DocumentCollector(adapters=[FakeAdapter("s1", "paper", totals=BOTH_WINDOWS)],
                      cache=cache).probe_source_totals()

    saved = {(row.source, row.window): row.n_total for row in cache.get_source_totals() or []}
    assert saved[("s2", "before")] == 500
    assert saved[("s2", "now")] == 500
    assert saved[("s1", "before")] == 1000


def test_partial_run_keeps_other_windows_of_the_same_source() -> None:
    """Опрос по окнам роста не стирает годовые итоги того же источника."""
    years = ("2020", "2021", "2022", "2023", "2024", "2025")
    cache = MemoryCache()
    cache.put_source_totals(_totals_rows("s1", years, n_total=42))

    DocumentCollector(adapters=[FakeAdapter("s1", "paper", totals=BOTH_WINDOWS)],
                      cache=cache).probe_source_totals()

    saved = {(row.source, row.window): row.n_total for row in cache.get_source_totals() or []}
    assert all(saved[("s1", year)] == 42 for year in years)
    assert saved[("s1", "now")] == 1250


def test_fresh_total_replaces_stale_one_without_duplicating_it() -> None:
    """Пара (источник, окно) остаётся в единственном экземпляре, побеждает свежая."""
    cache = MemoryCache()
    cache.put_source_totals(_totals_rows("s1", ("before", "now"), n_total=7,
                                         age_hours=SOURCE_TOTALS_TTL_HOURS + 1))

    DocumentCollector(adapters=[FakeAdapter("s1", "paper", totals=BOTH_WINDOWS)],
                      cache=cache).probe_source_totals()

    saved = cache.get_source_totals() or []
    assert len(saved) == 2
    assert {(row.window, row.n_total) for row in saved} == {("before", 1000), ("now", 1250)}


def _recent_sources() -> tuple[FakeAdapter, FakeAdapter, FakeAdapter]:
    """Три источника поиска №1 с настоящими именами: маршрутизация идёт по имени."""
    fresh = [doc(source="x", source_type="paper", published_at="2026-09-10", url="https://x/1")]
    return (FakeAdapter("openalex", "paper", documents=fresh),
            FakeAdapter("arxiv", "preprint", documents=fresh),
            FakeAdapter("techcrunch", "news", documents=fresh))


def test_search_recent_routes_by_language() -> None:
    """ru — только OpenAlex с language:ru; en — все три, arXiv со словами по отдельности."""
    openalex, arxiv, techcrunch = _recent_sources()
    DocumentCollector(adapters=[openalex, arxiv, techcrunch]).search_recent(
        [{"subquery_id": "q1-ru-1", "language": "ru", "text": "квантовые сенсоры"},
         {"subquery_id": "q1-en-1", "language": "en", "text": "quantum sensing"}],
        date_from=date(2026, 9, 1), date_to_exclusive=date(2026, 9, 18))

    assert [call[0] for call in openalex.search_calls] == ["квантовые сенсоры", "quantum sensing"]
    assert openalex.search_options == [{"language": "ru"}, {}]
    assert [call[0] for call in arxiv.search_calls] == ["quantum sensing"]
    assert arxiv.search_options == [{"words": True}]
    assert [call[0] for call in techcrunch.search_calls] == ["quantum sensing"]
    assert techcrunch.search_calls[0][3] == 25


def test_search_recent_documents_carry_subquery_ids() -> None:
    """Документ, найденный двумя подзапросами, один, и у него оба subquery_id."""
    openalex, _, _ = _recent_sources()
    result = DocumentCollector(adapters=[openalex]).search_recent(
        [{"subquery_id": "q1-en-1", "language": "en", "text": "quantum sensing"},
         {"subquery_id": "q1-en-2", "language": "en", "text": "quantum sensors"}],
        date_from=date(2026, 9, 1), date_to_exclusive=date(2026, 9, 18))

    documents = result.to_dict()["documents"]
    assert len(documents) == 1
    assert documents[0]["subquery_ids"] == ["q1-en-1", "q1-en-2"]


def test_search_recent_rejects_unknown_language() -> None:
    """Язык без маршрута — ошибка, а не тихая отправка во все источники."""
    with pytest.raises(ValueError):
        DocumentCollector(adapters=list(_recent_sources())).search_recent(
            [{"subquery_id": "q1-de-1", "language": "de", "text": "quanten sensorik"}])
