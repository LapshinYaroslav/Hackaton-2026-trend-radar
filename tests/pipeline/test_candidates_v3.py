"""Генерация кандидатов v3 (задача Г): промпт подзапросов (с задачи О3 — subq-v4) и состав шага 4."""
from datetime import date, timedelta
import hashlib

import pytest

import search.subqueries as sq
from pipeline import run_query as rq
from pipeline.run_query import step4_documents
from tests.collector.fakes import doc
from tests.pipeline import test_run_query as base

BASE_ADAPTERS = base.adapters  # до подмены в тестах: иначе рекурсия
# Снимок subq-v4 (задача О3): sha256 системных промптов и ключ кэша (тема, версия, модель, температура 0).
# V3_CACHE_KEY — ключ subq-v3 при T=0 (задача О2): v4 обязан от него отличаться, кэш v3 не подхватывается.
V4_PROMPT_SHA = {"ru": "0aaeb7aed353ac7a1921cb4d7a6a8cda0b8f1cc47f685fab85104f60aa28e647", "en": "1cf82d75c5b81e9a3ec116454d1ffe506c61cfca352df9606a09ab1b544dc124"}
V4_CACHE_KEY = "d73e6387379bb766f9e67563a3422b0a2ce3f88e768da0ffdeb6bce1d7426bfb"
V4_T03_CACHE_KEY = "3d35641f2dbcc186514e547fbd8bdff75fb3ea4326d150f2c9ae4b96aac36128"
V3_CACHE_KEY = "0ea5022925abee6add50e935b71fa496d41273ef8c5dc513282d53fa92a130c2"
V2_CACHE_KEY = "b577fdb502c81d99eaca24201541953a6f5bd0ef9e251b8823737363c043608e"
URI = "gpt://folder/yandexgpt-5-pro/latest"


def item(source: str, number: int, subquery: str = "q-en-1", language: str = "en") -> dict:
    """Документ поиска №1 в виде словаря, как его видит оркестратор."""
    return {"source": source, "url": f"https://{source}/{number}", "language": language,
            "subquery_ids": [subquery]}


SUBQUERIES = [{"subquery_id": "q-en-1", "language": "en"}, {"subquery_id": "q-ru-1", "language": "ru"}]


@pytest.mark.parametrize("language", ["ru", "en"])
def test_v4_prompt_is_unchanged(language):
    assert hashlib.sha256(sq.build_system_prompt(language).encode("utf-8")).hexdigest() == V4_PROMPT_SHA[language]


def test_v4_cache_key_is_unchanged_and_depends_on_version():
    assert sq.PROMPT_VERSION == "subq-v4"
    assert sq.cache_key("Тема X", URI) == V4_CACHE_KEY
    assert sq.cache_key("Тема X", URI, version="subq-v3") == V3_CACHE_KEY != V2_CACHE_KEY != V4_CACHE_KEY


def test_cache_key_depends_on_temperature():
    assert sq.SUBQUERY_TEMPERATURE == 0
    assert sq.cache_key("Тема X", URI, temperature=0.3) == V4_T03_CACHE_KEY != V4_CACHE_KEY


@pytest.mark.parametrize("language, count", [("ru", 5), ("en", 8)])
def test_v4_prompt_has_examples_and_form(language, count):
    prompt = sq.build_system_prompt(language)
    assert sq.EXAMPLES_V4 in prompt and f"составь {count} поисковых подзапросов" in prompt
    assert "„laser weeding robots“" in prompt and "„precision agriculture“ (устоявшийся раздел)" in prompt
    assert sq.LANGUAGE_RULES[language] in prompt and f'{{"{language}": ["...", "..."]}}' in prompt
    assert not hasattr(sq, "DIRECTIONS_V3")


def test_v3_orders_sources_and_caps_openalex_at_half():
    documents = ([item("openalex", n) for n in range(10)] + [item("techcrunch", 20)]
                 + [item("arxiv", 30), item("arxiv", 31)] + [item("openalex", 40, "q-ru-1", "ru")])
    picked = step4_documents(documents, SUBQUERIES, minimum=0)
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
    assert step4_documents([item("openalex", 1)], SUBQUERIES, minimum=0) == []
    assert step4_documents([item("openalex", 1)], SUBQUERIES) == [item("openalex", 1)]


def test_x2b_rich_topic_unchanged():
    """Богатая тема: по пропорции набирается 300 ≥ 200 — добора нет, английских OpenAlex не больше головы."""
    documents = ([item("arxiv", n) for n in range(100)] + [item("techcrunch", 100 + n) for n in range(50)]
                 + [item("openalex", 200 + n) for n in range(400)])
    picked = step4_documents(documents, SUBQUERIES)
    assert picked == step4_documents(documents, SUBQUERIES, minimum=0) and len(picked) == 300
    assert rq.step4_topped_up(documents, SUBQUERIES, picked) == 0


def test_x2b_poor_topic_topped_up_to_200_in_order():
    """Бедная тема: 20 arXiv + 10 TechCrunch, по пропорции 60 — английские OpenAlex добираются до 200 по порядку;
    русские не идут."""
    documents = ([item("openalex", n) for n in range(250)] + [item("openalex", 999, "q-ru-1", "ru")]
                 + [item("arxiv", 300 + n) for n in range(20)] + [item("techcrunch", 400 + n) for n in range(10)])
    picked = step4_documents(documents, SUBQUERIES)
    assert len(picked) == 200 and [doc["url"] for doc in picked[30:]] == [f"https://openalex/{n}" for n in range(170)]
    assert rq.step4_topped_up(documents, SUBQUERIES, picked) == 140
    few = [item("arxiv", 1)] + [item("openalex", n) for n in range(5)]
    assert len(step4_documents(few, SUBQUERIES)) == 6  # меньше 200 всего — берётся всё, что есть


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
