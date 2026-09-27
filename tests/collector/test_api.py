from datetime import date, datetime, timedelta, timezone

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
from collector.models import Document, SourceTotal
from tests.collector.fakes import FakeAdapter, doc

BOTH_WINDOWS = {
    (WINDOW_BEFORE_START, WINDOW_BEFORE_END): 1000,
    (WINDOW_NOW_START, WINDOW_NOW_END): 1250,
}


def _collector(*adapters: FakeAdapter) -> DocumentCollector:
    return DocumentCollector(adapters=list(adapters))


def test_invalid_source_type_fails_immediately() -> None:
    adapter = FakeAdapter("s1", "paper", fail_source_type="social")
    collector = _collector(adapter)
    with pytest.raises(InvalidSourceTypeError) as exc:
        collector.search_recent(["x"])
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
