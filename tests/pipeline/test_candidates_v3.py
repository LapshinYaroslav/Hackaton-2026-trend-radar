"""Задача Г: генерация кандидатов v3 (по умолчанию) — промпт subq-v3 и состав шага 4; v2 — опция."""
import hashlib
from datetime import date, timedelta

import pytest

import search.subqueries as sq
from pipeline.run_query import step4_documents
from tests.collector.fakes import doc
from tests.pipeline import test_run_query as base

BASE_ADAPTERS = base.adapters  # до подмены в тестах: иначе рекурсия
# Снимок subq-v2 до задачи Г: sha256 системных промптов и ключ кэша.
V2_PROMPT_SHA = {"ru": "b3d8e097cac1b7568373e875a83654ff2426f80afed6634a5c82e0318623baab",
                 "en": "c11c94133adb262f3e2962f544d9796c7fd9d8e48cd4600e19777100654f9f1b"}
V2_CACHE_KEY = "b577fdb502c81d99eaca24201541953a6f5bd0ef9e251b8823737363c043608e"
URI = "gpt://folder/yandexgpt-5-pro/latest"


def item(source: str, number: int, subquery: str = "q-en-1", language: str = "en") -> dict:
    """Документ поиска №1 в виде словаря, как его видит оркестратор."""
    return {"source": source, "url": f"https://{source}/{number}", "language": language,
            "subquery_ids": [subquery]}


SUBQUERIES = [{"subquery_id": "q-en-1", "language": "en"}, {"subquery_id": "q-ru-1", "language": "ru"}]


@pytest.mark.parametrize("language", ["ru", "en"])
def test_v2_prompt_is_unchanged(language):
    for prompt in (sq.build_system_prompt(language), sq.build_system_prompt(language, sq.PROMPT_VERSION)):
        assert hashlib.sha256(prompt.encode("utf-8")).hexdigest() == V2_PROMPT_SHA[language]


def test_v2_cache_key_is_unchanged_and_v3_differs():
    assert sq.cache_key("Тема X", URI) == V2_CACHE_KEY
    assert sq.cache_key("Тема X", URI, sq.PROMPT_VERSION_V3) != V2_CACHE_KEY


@pytest.mark.parametrize("language, count", [("ru", 5), ("en", 8)])
def test_v3_prompt_has_new_directions_and_v2_form(language, count):
    prompt = sq.build_system_prompt(language, sq.PROMPT_VERSION_V3)
    assert sq.DIRECTIONS_V3 in prompt and f"составь {count} поисковых подзапросов" in prompt
    assert sq.LANGUAGE_RULES[language] in prompt and f'{{"{language}": ["...", "..."]}}' in prompt
    assert "водородная" not in prompt and "раздел учебника" not in prompt


def test_unknown_prompt_version_rejected():
    with pytest.raises(ValueError):
        sq.generate_subqueries("тема", "q1", prompt_version="subq-v9")


def test_v2_step4_is_all_documents_in_order():
    documents = [item("openalex", 1), item("arxiv", 2), item("openalex", 3, "q-ru-1", "ru"), item("techcrunch", 4)]
    assert step4_documents(documents, SUBQUERIES) == documents
    assert step4_documents(documents, SUBQUERIES, "v2") == documents


def test_v3_orders_sources_and_caps_openalex_at_half():
    documents = ([item("openalex", n) for n in range(10)] + [item("techcrunch", 20)]
                 + [item("arxiv", 30), item("arxiv", 31)] + [item("openalex", 40, "q-ru-1", "ru")])
    picked = step4_documents(documents, SUBQUERIES, "v3")
    assert [doc["source"] for doc in picked] == ["arxiv", "arxiv", "techcrunch"] + ["openalex"] * 3
    assert [doc["url"] for doc in picked[3:]] == [f"https://openalex/{n}" for n in range(3)]
    assert sum(doc["source"] == "openalex" for doc in picked) <= len(picked) / 2


def test_v3_drops_russian_openalex_by_subquery_or_language():
    documents = [item("arxiv", 1), item("arxiv", 2), item("openalex", 3, "q-ru-1", "en"),
                 item("openalex", 4, "q-en-1", "ru"), item("openalex", 5)]
    assert [doc["url"] for doc in step4_documents(documents, SUBQUERIES, "v3")] == \
        ["https://arxiv/1", "https://arxiv/2", "https://openalex/5"]


def test_v3_edge_cases():
    assert step4_documents([], SUBQUERIES, "v3") == []
    assert step4_documents([item("openalex", 1)], SUBQUERIES, "v3") == []


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


@pytest.mark.parametrize("version", ["v2", "v3"])
def test_run_query_russian_documents_stay_in_pool(tmp_path, monkeypatch, version):
    monkeypatch.setattr(base, "adapters", adapters_with_russian)
    out, _, prompts = base.run(tmp_path, candidates_version=version)
    pool, text = out["stats"]["documents_by_source"], "\n".join(prompts)
    step4 = out["stats"]["documents_for_candidates_by_source"]
    assert pool["openalex_ru"] == 2 and out["candidates_version"] == version
    if version == "v3":
        assert step4["openalex_ru"] == 0 and "openalex ru doc" not in text
        assert step4["openalex"] <= sum(step4.values()) / 2
        assert text.index("arxiv doc") < text.index("techcrunch doc") < text.index("openalex doc")
    else:  # v2: на шаг 4 идут все документы пула
        assert step4 == pool and "openalex ru doc" in text


def test_run_query_rejects_unknown_candidates_version(tmp_path):
    with pytest.raises(ValueError):
        base.run(tmp_path, candidates_version="v9")
