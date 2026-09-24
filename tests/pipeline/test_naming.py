"""Нормализатор названий кандидатов и кэши оркестратора без сети."""
from datetime import date
from unittest.mock import patch

import pytest

from collector.exceptions import AdapterError
from experiments.subqueries import cache as llm_cache
from experiments.subqueries import llm
from experiments.technologies import terms as T
from pipeline import naming, search_cache
from search import extract_candidates as ec
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


def answers(*replies: str):
    """Заглушка terms.ask_name: ответы по очереди, запоминает repeat и note."""
    seen = []

    def ask(name_ru, area, repeat=0, note=""):
        seen.append((repeat, note))
        return replies[min(repeat, len(replies) - 1)], True
    return ask, seen


def variants(*names: str) -> str:
    return "\n".join(f"VARIANT: {name}" for name in names)


def test_choose_max_and_tie_first() -> None:
    items = [{"name": "a b", "n_works": 3}, {"name": "c d", "n_works": 7}, {"name": "e f", "n_works": 7}]
    assert naming.choose(items)["name"] == "c d"
    assert naming.choose([{"name": "a b", "n_works": 0}, {"name": "c d", "n_works": 0}])["name"] == "a b"


def test_normalize_picks_max_trace() -> None:
    ask, _ = answers(variants("edge ai chip", "edge inference chip", "tinyml accelerator"))
    with patch.object(T, "ask_name", ask):
        got = naming.normalize_one("чипы", "Edge", Trace({"edge inference chip": 40, "edge ai chip": 12}))
    assert got["outcome"] == "ok" and got["chosen"]["name"] == "edge inference chip"
    assert [v["n_works"] for v in got["variants"]] == [12, 40, 0]


def test_all_zero_retries_with_note_then_no_trace() -> None:
    """Все три без работ: повтор с retry_note, как при обучении; после всех попыток — no_trace."""
    ask, seen = answers(variants("phantom one", "phantom two", "phantom three"))
    with patch.object(T, "ask_name", ask):
        got = naming.normalize_one("фантом", "Edge", Trace({}))
    assert got["outcome"] == "no_trace" and got["chosen"] is None
    assert [repeat for repeat, _ in seen] == list(range(naming.ATTEMPTS))
    assert "не встречается в научной литературе" in seen[1][1]


def test_format_violation_retries_with_note() -> None:
    ask, seen = answers("VARIANT: one", variants("edge ai chip", "edge inference chip", "tinyml accelerator"))
    with patch.object(T, "ask_name", ask):
        got = naming.normalize_one("чипы", "Edge", Trace({"edge ai chip": 5}))
    assert got["outcome"] == "ok" and got["attempts"] == 2
    assert "строк VARIANT 1" in seen[1][1]


def test_company_stoplist_is_off() -> None:
    """Термин из стоп-листа компаний проходит валидацию: стоп-лист выключен подменой шаблона."""
    names, problems = T.check_candidates(variants("sovereign ai", "physical intelligence", "edge ai chip"),
                                         naming.NO_STOPLIST)
    assert problems == [] and names[0] == "sovereign ai"


def test_openalex_failure_is_trace_unknown() -> None:
    ask, _ = answers(variants("edge ai chip", "edge inference chip", "tinyml accelerator"))
    with patch.object(T, "ask_name", ask):
        got = naming.normalize_one("чипы", "Edge", Trace({"edge ai chip": None, "edge inference chip": None,
                                                          "tinyml accelerator": None}))
    assert got["outcome"] == "trace_unknown" and got["chosen"] is None


def test_normalizer_cache_skips_second_call(tmp_path) -> None:
    """Тот же name_ru и area: второй раз ответ из кэша, модель не вызывается."""
    reply = {"text": variants("a b", "c d", "e f"), "error": None}
    with patch.object(llm_cache, "CACHE_DIR", tmp_path), \
         patch.object(llm, "build_model_uri", lambda: "gpt://t/yandexgpt-5-pro"), \
         patch.object(llm, "ask_llm", return_value=reply) as model:
        T.ask_name("чипы", "Edge")
        T.ask_name("чипы", "Edge")
        T.ask_name("чипы", "Роботы")
    assert model.call_count == 2


def test_search_cache_second_call_from_disk(tmp_path) -> None:
    fresh = [doc(source="openalex", source_type="paper", published_at="2026-05-01", url="u1")]
    adapter = FakeAdapter("openalex", "paper", documents=fresh)
    cached = search_cache.CachedSearch(adapter, cache_dir=tmp_path)
    first = cached.search("edge ai", date(2026, 3, 1), date(2026, 9, 24), limit=25)
    again = cached.search("edge ai", date(2026, 3, 1), date(2026, 9, 24), limit=25)
    other = cached.search("edge ai", date(2026, 3, 1), date(2026, 9, 24), limit=25, language="ru")
    assert first == again == other and len(adapter.search_calls) == 2 and cached.hits == 1


def test_extract_cache_key_depends_on_model() -> None:
    assert ec._cache_key("пачка", "gpt://f/yandexgpt-5-pro") != ec._cache_key("пачка", "gpt://f/yandexgpt-5-lite")
