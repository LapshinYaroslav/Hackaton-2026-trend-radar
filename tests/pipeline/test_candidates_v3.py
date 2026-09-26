"""Задача Г: генерация кандидатов v3 — промпт subq-v3 и состав шага 4 (единственный вариант с задачи З)."""
from datetime import date, timedelta
import hashlib

import pytest

import search.subqueries as sq
from pipeline.run_query import step4_documents
from tests.collector.fakes import doc
from tests.pipeline import test_run_query as base

BASE_ADAPTERS = base.adapters  # до подмены в тестах: иначе рекурсия
# Снимок subq-v3 на задаче З: sha256 системных промптов и ключ кэша. Ключ тот же, что до удаления subq-v2,
# поэтому кэш подзапросов прежних прогонов v3 остаётся рабочим.
V3_PROMPT_SHA = {"ru": "a416f4425edcfad27044334273be0c8fda139b7e0e1f903ddf7940f2219149a3", "en": "cce6a39441effa095b3192f93b872ee59497b08a4924ac05aabe3e73fc2cc41a"}
V3_CACHE_KEY = "2dc88b63317905669119454eaa3b15cf32f68c7c81e0f241374c1c482f0d6af7"
V2_CACHE_KEY = "b577fdb502c81d99eaca24201541953a6f5bd0ef9e251b8823737363c043608e"
URI = "gpt://folder/yandexgpt-5-pro/latest"


def item(source: str, number: int, subquery: str = "q-en-1", language: str = "en") -> dict:
    """Документ поиска №1 в виде словаря, как его видит оркестратор."""
    return {"source": source, "url": f"https://{source}/{number}", "language": language,
            "subquery_ids": [subquery]}


SUBQUERIES = [{"subquery_id": "q-en-1", "language": "en"}, {"subquery_id": "q-ru-1", "language": "ru"}]


@pytest.mark.parametrize("language", ["ru", "en"])
def test_v3_prompt_is_unchanged(language):
    assert hashlib.sha256(sq.build_system_prompt(language).encode("utf-8")).hexdigest() == V3_PROMPT_SHA[language]


def test_v3_cache_key_is_unchanged_and_not_v2():
    assert sq.cache_key("Тема X", URI) == V3_CACHE_KEY != V2_CACHE_KEY


@pytest.mark.parametrize("language, count", [("ru", 5), ("en", 8)])
def test_v3_prompt_has_directions_and_form(language, count):
    prompt = sq.build_system_prompt(language)
    assert sq.DIRECTIONS_V3 in prompt and f"составь {count} поисковых подзапросов" in prompt
    assert sq.LANGUAGE_RULES[language] in prompt and f'{{"{language}": ["...", "..."]}}' in prompt
    assert "водородная" not in prompt and "раздел учебника" not in prompt


def test_v3_orders_sources_and_caps_openalex_at_half():
    documents = ([item("openalex", n) for n in range(10)] + [item("techcrunch", 20)]
                 + [item("arxiv", 30), item("arxiv", 31)] + [item("openalex", 40, "q-ru-1", "ru")])
    picked = step4_documents(documents, SUBQUERIES)
    assert [doc["source"] for doc in picked] == ["arxiv", "arxiv", "techcrunch"] + ["openalex"] * 3
    assert [doc["url"] for doc in picked[3:]] == [f"https://openalex/{n}" for n in range(3)]
    assert sum(doc["source"] == "openalex" for doc in picked) <= len(picked) / 2


def test_v3_drops_russian_openalex_by_subquery_or_language():
    documents = [item("arxiv", 1), item("arxiv", 2), item("openalex", 3, "q-ru-1", "en"),
                 item("openalex", 4, "q-en-1", "ru"), item("openalex", 5)]
    assert [doc["url"] for doc in step4_documents(documents, SUBQUERIES)] == \
        ["https://arxiv/1", "https://arxiv/2", "https://openalex/5"]


def test_v3_edge_cases():
    assert step4_documents([], SUBQUERIES) == []
    assert step4_documents([item("openalex", 1)], SUBQUERIES) == []


class RuOpenAlex(base.QueryFake):
    """OpenAlex, у которого русские подзапросы (language=ru) находят свои, русские документы."""

    def __init__(self, en_docs, ru_docs):
        super().__init__("openalex", "paper", documents=en_docs)
        self.ru_docs = ru_docs

    def search(self, query, date_from, date_to_exclusive, *, limit, **options):
        found = super().search(query, date_from, date_to_exclusive, limit=limit, **options)
        return self.ru_docs[:limit] if options.get("language") == "ru" else found


def adapters_with_russian():
    """Как base.adapters, но OpenAlex отдаёт русскоязычные документы на русские подзапросы."""
    fresh = (date.today() - timedelta(days=20)).isoformat()
    ru = [doc(source="openalex", source_type="paper", published_at=fresh, url=f"https://openalex/ru{i}",
              title=f"openalex ru doc {i}", language="ru") for i in range(2)]
    plain = BASE_ADAPTERS()
    return [RuOpenAlex(plain[0].documents, ru), *plain[1:]]


def test_run_query_russian_documents_stay_in_pool(tmp_path, monkeypatch):
    monkeypatch.setattr(base, "adapters", adapters_with_russian)
    out, _, prompts = base.run(tmp_path)
    pool, text = out["stats"]["documents_by_source"], "\n".join(prompts)
    step4 = out["stats"]["documents_for_candidates_by_source"]
    assert pool["openalex_ru"] == 2 and out["candidates_version"] == "v3"
    assert step4["openalex_ru"] == 0 and "openalex ru doc" not in text
    assert step4["openalex"] <= sum(step4.values()) / 2
    assert text.index("arxiv doc") < text.index("techcrunch doc") < text.index("openalex doc")
