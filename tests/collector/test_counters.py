"""Структуры счётчиков поиска №2 (pipeline.md 0.2)."""
from dataclasses import fields
from datetime import date

import pytest

from collector.adapters.base import totals_signature
from collector.api import DocumentCollector, terms_hash
from collector.constants import (
    COUNTER_WINDOWS,
    PREV6_START,
    YEAR_WINDOWS,
    WINDOW_BEFORE_END,
    WINDOW_BEFORE_START,
    WINDOW_NOW_END,
    WINDOW_NOW_START,
)
from collector.db import MemoryCache
from collector.models import (
    Candidate,
    Counter,
    CounterResult,
    RecentSearchResult,
    SourceTotal,
    build_search_terms,
)
from tests.collector.fakes import FakeAdapter

BOTH_WINDOWS = {
    (WINDOW_BEFORE_START, WINDOW_BEFORE_END): 5861775,
    (WINDOW_NOW_START, WINDOW_NOW_END): 8231350,
}
# Корпус источника по тем же годовым окнам, что и счётчики технологии.
YEAR_TOTALS = {bounds: 1_000_000 + index for index, bounds in enumerate(COUNTER_WINDOWS.values())}


def _counter(**changes) -> Counter:
    payload = {
        "tech_key": "neuromorphic computing",
        "source": "openalex",
        "window": "2025",
        "date_from": date(2025, 9, 1),
        "date_to": date(2026, 9, 1),
        "n": 39,
        "type_filter": "article",
        "query_variant": "phrase|article",
    }
    return Counter(**(payload | changes))


def test_counter_windows_are_six_years_without_gaps() -> None:
    """Шесть годовых окон подряд, полуоткрытые [from, to), от начала сбора до среза.

    Непересечение проверяется явно: рабочие окна складываются из этих чисел, и год,
    попавший в набор дважды, удвоил бы volume незаметно.
    """
    bounds = [YEAR_WINDOWS[name] for name in sorted(YEAR_WINDOWS)]

    assert len(bounds) == 6
    assert bounds[0][0] == date(2020, 9, 1)
    assert bounds[-1][1] == date(2026, 9, 1)
    assert all(end == nxt for (_, end), (nxt, _) in zip(bounds, bounds[1:]))


def test_prev6_window_ends_where_collection_starts() -> None:
    """Окно prev6 упирается в начало сбора и с годовыми не пересекается."""
    assert COUNTER_WINDOWS["prev6"] == (PREV6_START, date(2020, 9, 1))
    assert set(COUNTER_WINDOWS) == {"prev6"} | set(YEAR_WINDOWS)


def test_counter_dict_matches_contract() -> None:
    """Форма счётчика — как в разделе «Счётчик» документа."""
    assert _counter().to_dict() == {
        "tech_key": "neuromorphic computing",
        "source": "openalex",
        "window": "2025",
        "date_from": "2025-09-01",
        "date_to": "2026-09-01",
        "type_filter": "article",
        "query_variant": "phrase|article",
        "n": 39,
    }


@pytest.mark.parametrize(
    "changes",
    [
        {"window": "quarter"},
        {"n": -1},
        {"date_from": date(2026, 9, 1), "date_to": date(2025, 9, 1)},
        {"tech_key": "  "},
    ],
)
def test_counter_rejects_broken_values(changes: dict) -> None:
    with pytest.raises(ValueError):
        _counter(**changes)


def test_two_searches_cannot_be_mixed_up() -> None:
    """Защита по типам: у счётчиков нет документов, у поиска №1 нет счётчиков и итогов."""
    counter_fields = {item.name for item in fields(CounterResult)}
    recent_fields = {item.name for item in fields(RecentSearchResult)}

    assert "documents" not in counter_fields
    assert {"counters", "source_totals"} & recent_fields == set()


def test_counter_result_hides_unavailable_totals() -> None:
    """Источник без итогов за оба окна в контракт не попадает, как и у документов."""
    result = CounterResult(
        candidate_id="c42",
        tech_key="neuromorphic computing",
        terms_hash="abc",
        counters=[_counter()],
        source_totals=[
            SourceTotal(source="openalex", window="now", n_total=8231350),
            SourceTotal(source="techcrunch", window="now", n_total=0, available=False),
        ],
    )
    payload = result.to_contract_dict()

    assert [row["source"] for row in payload["source_totals"]] == ["openalex"]
    assert payload["counters"][0]["n"] == 39
    assert "documents" not in payload


def test_candidate_from_dict_reads_both_lists() -> None:
    """Контракт кандидата из документа разбирается целиком."""
    candidate = Candidate.from_dict(
        {
            "candidate_id": "c42",
            "query_id": "q7",
            "name_ru": "Фотонные процессоры инференса",
            "name_en": "photonic inference processor",
            "terms": ["photonic inference processor", " optical AI accelerator "],
            "context_terms": ["photonic computing", ""],
        }
    )
    assert candidate.terms == ["photonic inference processor", "optical AI accelerator"]
    assert candidate.context_terms == ["photonic computing"]


YEARLY = {"prev6": 13, "y2020": 21, "y2021": 34, "y2022": 55, "y2023": 90,
          "y2024": 145, "y2025": 233}


def _search_terms() -> dict:
    return {
        "terms": ["neuromorphic computing", "neuromorphic chip"],
        "context_terms": ["spiking neural network", "brain-inspired computing"],
    }


def _counting_adapter(source: str = "s1", **counts) -> FakeAdapter:
    """Источник, который отвечает на счётчик разными числами по годам: y2020=..."""
    by_window = {COUNTER_WINDOWS[name.lstrip("y")]: value for name, value in counts.items()}
    return FakeAdapter(source, "paper", totals=YEAR_TOTALS, counts=by_window)


def test_count_history_returns_one_row_per_source_and_year() -> None:
    """Шесть годовых окон на источник, документов в ответе нет."""
    adapter = _counting_adapter(**YEARLY)
    collector = DocumentCollector(adapters=[adapter], cache=MemoryCache())

    result = collector.count_history(Candidate(candidate_id="c1", name_en="neuromorphic computing",
                                               **_search_terms()))
    by_window = {row.window: row.n for row in result.counters}

    assert by_window == {"prev6": 13, "2020": 21, "2021": 34, "2022": 55, "2023": 90,
                        "2024": 145, "2025": 233}
    assert result.tech_key == "neuromorphic computing"
    assert "documents" not in result.to_contract_dict()
    assert all(row.date_from < row.date_to for row in result.counters)


def test_silent_source_gives_no_row() -> None:
    """Источник не ответил по окну — строки нет, ноль не выдумывается."""
    adapter = _counting_adapter(y2020=379, y2025=83)
    result = DocumentCollector(adapters=[adapter], cache=MemoryCache()).count_history(
        Candidate(candidate_id="c1", name_en="x", **_search_terms())
    )
    assert {row.window for row in result.counters} == {"2020", "2025"}


def test_counters_come_from_cache_on_second_call() -> None:
    """Повторный вызов не трогает сеть."""
    adapter = _counting_adapter(**YEARLY)
    collector = DocumentCollector(adapters=[adapter], cache=MemoryCache())
    candidate = Candidate(candidate_id="c1", name_en="neuromorphic computing", **_search_terms())

    first = collector.count_history(candidate)
    calls_after_first = len(adapter.match_calls)
    second = collector.count_history(candidate)

    assert first.cache_hit is False and second.cache_hit is True
    assert len(adapter.match_calls) == calls_after_first
    assert [row.n for row in second.counters] == [row.n for row in first.counters]


def test_new_terms_invalidate_cached_counters() -> None:
    """Уточнили термины — ключ другой — счётчики пересчитываются (находка 012)."""
    adapter = _counting_adapter(**YEARLY)
    collector = DocumentCollector(adapters=[adapter], cache=MemoryCache())
    name = "neuromorphic computing"

    first = collector.count_history(Candidate(candidate_id="c1", name_en=name, **_search_terms()))
    refined = collector.count_history(
        Candidate(
            candidate_id="c1",
            name_en=name,
            terms=["neuromorphic computing", "memristor crossbar"],
            context_terms=["spiking neural network", "brain-inspired computing"],
        )
    )

    assert refined.terms_hash != first.terms_hash
    assert refined.cache_hit is False
    assert len(adapter.match_calls) == 2 * len(COUNTER_WINDOWS)


def test_terms_hash_ignores_order_and_case() -> None:
    """Тот же набор терминов в другом порядке — тот же ключ, кэш срабатывает."""
    adapter = _counting_adapter(**YEARLY)
    collector = DocumentCollector(adapters=[adapter], cache=MemoryCache())
    name = "neuromorphic computing"

    first = collector.count_history(Candidate(candidate_id="c1", name_en=name, **_search_terms()))
    same = collector.count_history(
        Candidate(
            candidate_id="c1",
            name_en=name,
            terms=["Neuromorphic Chip", "NEUROMORPHIC COMPUTING"],
            context_terms=["brain-inspired computing", "spiking neural network"],
        )
    )
    assert same.terms_hash == first.terms_hash
    assert same.cache_hit is True


def test_unusable_terms_fail_loudly() -> None:
    """Из пустых терминов запрос не собрать: ошибка до любого обращения к сети.

    Раньше сюда подавался один годный термин — тогда без второго блока запрос не
    собирался. После задачи 3Д одно каноническое название и есть запрос, поэтому
    непригодным остался только вход без единого непустого термина.
    """
    adapter = _counting_adapter(y2020=379)
    collector = DocumentCollector(adapters=[adapter], cache=MemoryCache())

    with pytest.raises(ValueError, match="запрос не собрался"):
        collector.count_history(
            Candidate(candidate_id="c1", name_en="x", terms=["   ", ""])
        )
    assert adapter.match_calls == []


def test_changed_type_filter_invalidates_cached_totals() -> None:
    """Сменили фильтр типа — старый знаменатель growth из кэша не берётся."""
    cache = MemoryCache()
    old = FakeAdapter("s1", "paper", totals=BOTH_WINDOWS, type_filter=None)
    DocumentCollector(adapters=[old], cache=cache).probe_source_totals()
    assert all(row.type_filter is None for row in cache.get_source_totals() or [])

    new = FakeAdapter("s1", "paper", totals=BOTH_WINDOWS, type_filter="article")
    totals = DocumentCollector(adapters=[new], cache=cache).probe_source_totals()

    assert len(new.count_calls) == 2
    assert all(row.type_filter == "article" for row in totals)


def test_same_filter_still_comes_from_cache() -> None:
    """Фильтр не менялся — итоги по-прежнему берутся из кэша, лишних вызовов нет."""
    cache = MemoryCache()
    first = FakeAdapter("s1", "paper", totals=BOTH_WINDOWS, type_filter="article")
    DocumentCollector(adapters=[first], cache=cache).probe_source_totals()

    second = FakeAdapter("s1", "paper", totals=BOTH_WINDOWS, type_filter="article")
    DocumentCollector(adapters=[second], cache=cache).probe_source_totals()

    assert second.count_calls == []


def test_changed_totals_request_invalidates_cache() -> None:
    """Сменился запрос за итогом, а не фильтр — кэш всё равно обесценивается.

    Ровно этот случай прошёл мимо проверки по фильтру: у arXiv итоги переехали
    из OpenAlex на собственный totalResults, а type_filter как был None, так и остался.
    Подпись считается из самого запроса, поэтому ловится без ручной отметки о версии.
    """
    cache = MemoryCache()
    old = FakeAdapter("arxiv", "preprint", totals=BOTH_WINDOWS,
                      totals_endpoint="https://api.openalex.org/works")
    DocumentCollector(adapters=[old], cache=cache).probe_source_totals()

    new = FakeAdapter("arxiv", "preprint", totals=BOTH_WINDOWS,
                      totals_endpoint="http://export.arxiv.org/api/query")
    totals = DocumentCollector(adapters=[new], cache=cache).probe_source_totals()

    assert len(new.count_calls) == 2
    assert {row.totals_signature for row in totals} == {totals_signature(new)}


def test_totals_signature_follows_the_request() -> None:
    """Подпись выводится из запроса: тот же запрос — та же подпись, другой фильтр — другая."""
    same_a = FakeAdapter("s1", "paper", type_filter="article")
    same_b = FakeAdapter("s1", "paper", type_filter="article")
    other = FakeAdapter("s1", "paper", type_filter=None)

    assert totals_signature(same_a) == totals_signature(same_b)
    assert totals_signature(same_a) != totals_signature(other)


def test_totals_signature_ignores_secrets() -> None:
    """Ключ API в подпись не входит: смена ключа не должна обнулять кэш и не должна храниться."""
    from collector.adapters.base import SIGNATURE_SKIP_PARAMS

    assert "api_key" in SIGNATURE_SKIP_PARAMS
    assert "mailto" in SIGNATURE_SKIP_PARAMS


def test_terms_hash_does_not_depend_on_window_set(monkeypatch) -> None:
    """Набор окон в отпечаток терминов не входит: окно живёт в ключе строки (period)."""
    search = build_search_terms(**_search_terms())
    before = terms_hash(search)

    monkeypatch.setattr("collector.api.COUNTER_WINDOWS_VERSION", "другая-версия-окон",
                        raising=False)
    assert terms_hash(search) == before


def test_new_window_is_counted_alone_and_old_numbers_stay(monkeypatch) -> None:
    """Добавили окно — доспрашивается только оно, собранные годы из кэша не пересчитываются.

    Раньше набор окон входил в отпечаток: одно новое окно меняло ключ и выбрасывало
    все ранее собранные строки по всем окнам.
    """
    adapter = _counting_adapter(**YEARLY)
    collector = DocumentCollector(adapters=[adapter], cache=MemoryCache())
    candidate = Candidate(candidate_id="c1", name_en="neuromorphic computing", **_search_terms())

    without_last = {name: bounds for name, bounds in COUNTER_WINDOWS.items()
                    if name != "2025"}
    monkeypatch.setattr("collector.api.COUNTER_WINDOWS", without_last)
    first = collector.count_history(candidate)
    calls_after_five = len(adapter.match_calls)

    monkeypatch.setattr("collector.api.COUNTER_WINDOWS", COUNTER_WINDOWS)
    second = collector.count_history(candidate)

    assert calls_after_five == len(COUNTER_WINDOWS) - 1
    assert len(adapter.match_calls) == len(COUNTER_WINDOWS), "доспрошено одно новое окно"
    assert second.terms_hash == first.terms_hash
    assert {row.window for row in second.counters} == set(COUNTER_WINDOWS)
    assert {row.window: row.n for row in second.counters}["2020"] == YEARLY["y2020"]


def test_repeated_run_after_new_window_is_silent() -> None:
    """Окна добрали — повторный прогон снова не трогает сеть."""
    adapter = _counting_adapter(**YEARLY)
    collector = DocumentCollector(adapters=[adapter], cache=MemoryCache())
    candidate = Candidate(candidate_id="c1", name_en="neuromorphic computing", **_search_terms())

    collector.count_history(candidate)
    calls = len(adapter.match_calls)
    again = collector.count_history(candidate)

    assert again.cache_hit is True
    assert len(adapter.match_calls) == calls


def test_empty_context_gives_single_or_block() -> None:
    """Пустой список контекста — осознанный отказ от второго блока, а не сломанный ответ."""
    search = build_search_terms(["datacenter fpga accelerator", "fpga inference accelerator"], [])

    assert search.usable
    assert search.query == '("datacenter fpga accelerator" OR "fpga inference accelerator")'
    assert "AND" not in search.query


def test_single_term_without_context_is_a_query() -> None:
    """Одно каноническое название без второго блока — рабочий запрос поиска №2."""
    search = build_search_terms(["neuromorphic chip"], [])

    assert search.usable
    assert search.query == '("neuromorphic chip")'


def test_terms_without_a_single_usable_one_are_not_a_query() -> None:
    """Непригодны термины, среди которых нет ни одного непустого."""
    search = build_search_terms(["  ", ""], [])

    assert not search.usable


def test_missing_context_field_is_still_an_error() -> None:
    """Поля контекста нет совсем — это сломанный ответ, а не отказ от блока."""
    search = build_search_terms(["a b", "c d"], None)

    assert not search.usable


def test_silent_source_is_asked_again_next_run() -> None:
    """Источник промолчал по окну — на следующем прогоне его спросят снова.

    Молчание источника и его ноль — разные вещи. Замороженное в кэше молчание давало бы
    признаки по неполным данным: ровно так на этапе 2 arXiv ответил на 25 запросов из 182,
    и доля научных источников посчиталась почти без него.
    """
    silent = FakeAdapter("arxiv", "preprint", totals=YEAR_TOTALS, counts={})
    loud = _counting_adapter("openalex", **YEARLY)
    cache = MemoryCache()
    candidate = Candidate(candidate_id="c1", name_en="x", **_search_terms())

    first = DocumentCollector(adapters=[loud, silent], cache=cache).count_history(candidate)
    assert {row.source for row in first.counters} == {"openalex"}

    talkative = _counting_adapter("arxiv", **YEARLY)
    second = DocumentCollector(adapters=[loud, talkative], cache=cache).count_history(candidate)

    assert {row.source for row in second.counters} == {"openalex", "arxiv"}
    assert len(talkative.match_calls) == len(COUNTER_WINDOWS), "спрошен только молчавший источник"
    assert second.cache_hit is False


def test_nothing_is_asked_twice_when_everything_is_cached() -> None:
    """Все пары источник-окно на месте — сеть не трогаем."""
    adapters = [_counting_adapter("openalex", **YEARLY), _counting_adapter("arxiv", **YEARLY)]
    cache = MemoryCache()
    candidate = Candidate(candidate_id="c1", name_en="x", **_search_terms())

    DocumentCollector(adapters=adapters, cache=cache).count_history(candidate)
    calls = [len(adapter.match_calls) for adapter in adapters]
    again = DocumentCollector(adapters=adapters, cache=cache).count_history(candidate)

    assert again.cache_hit is True
    assert [len(adapter.match_calls) for adapter in adapters] == calls
