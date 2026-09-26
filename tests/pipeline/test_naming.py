"""След названия (pipeline/naming.py), кэш поиска №1 и ключ кэша шага 4. Без сети."""
from datetime import date

from collector.exceptions import AdapterError
from pipeline import naming, search_cache
from search import extract_terms as et
from tests.collector.fakes import FakeAdapter, doc


class Trace:
    """OpenAlex-заглушка: число работ по фразе, None — ошибка источника."""

    def __init__(self, works: dict[str, int | None]):
        self.works, self.calls = works, []

    def count_institutions(self, search, date_from, date_to_exclusive):
        phrase = search.terms[0]
        self.calls.append(phrase)
        if self.works.get(phrase, 0) is None:
            raise AdapterError("openalex", "429")
        return {"works": self.works.get(phrase, 0), "institutions": 2, "capped": False}


def test_trace_counts_works_and_institutions() -> None:
    got = naming.trace("edge ai chip", Trace({"edge ai chip": 12}))
    assert got == {"name": "edge ai chip", "n_works": 12, "n_institutions": 2, "n_institutions_capped": False}


def test_zero_trace_and_openalex_failure() -> None:
    assert naming.trace("phantom lattice", Trace({}))["n_works"] == 0
    assert naming.trace("edge ai chip", Trace({"edge ai chip": None}))["n_works"] is None  # trace_unknown


def test_search_cache_second_call_from_disk(tmp_path) -> None:
    fresh = [doc(source="openalex", source_type="paper", published_at="2026-05-01", url="u1")]
    adapter = FakeAdapter("openalex", "paper", documents=fresh)
    cached = search_cache.CachedSearch(adapter, cache_dir=tmp_path)
    first = cached.search("edge ai", date(2026, 3, 1), date(2026, 9, 24), limit=25)
    again = cached.search("edge ai", date(2026, 3, 1), date(2026, 9, 24), limit=25)
    other = cached.search("edge ai", date(2026, 3, 1), date(2026, 9, 24), limit=25, language="ru")
    assert first == again == other and len(adapter.search_calls) == 2 and cached.hits == 1


def test_extract_cache_key_depends_on_model() -> None:
    assert et._cache_key("пачка", "gpt://f/yandexgpt-5-pro") != et._cache_key("пачка", "gpt://f/yandexgpt-5-lite")
