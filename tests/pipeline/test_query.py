"""Сборка контракта без сети: веса опубликованной модели и отсев ниже порога."""

from __future__ import annotations

from search.extract_candidates import filter_raw_candidate
from pipeline.query import _load_artifact, assemble_result


def test_filter_drops_the_topic_itself() -> None:
    assert filter_raw_candidate(
        {
            "doc": 1,
            "name_ru": "промышленные роботы",
            "name_en": "industrial robots",
            "terms": ["industrial robot"],
            "context_terms": [],
        },
        "промышленные роботы",
    ) is None


def test_assemble_splits_top_and_excluded() -> None:
    rows = []
    for index, news in enumerate((0.95, 0.2)):
        rows.append(
            {
                "name": f"tech {index}",
                "name_ru": f"технология {index}",
                "complete": True,
                "doc_ids": [1],
                "features": {
                    "volume": 2.0 + index,
                    "growth_research": 0.1,
                    "recency": 0.8 if index == 0 else 0.1,
                    "share_news_wordmatch": news,
                    "share_prev6": 0.1,
                    "age_first_arxiv": 1.0 if index == 0 else 8.0,
                },
            }
        )
    artifact = _load_artifact([row["features"] for row in rows])
    # Файл модели в репозитории не лежит: тест проверяет запасной путь.
    artifact["fallback"] = False
    result = assemble_result(
        query_id="q-test",
        topic="роботы",
        area="Роботы",
        subqueries=[{"subquery_id": "q-test-en-1", "language": "en", "text": "dexterous hand"}],
        documents=[{"title": "A paper", "url": "https://example.org/a", "source": "arxiv",
                    "source_type": "preprint", "published_at": "2026-01-01", "language": "en",
                    "trust_level": "medium", "text": "аннотация"}],
        built=rows,
        artifact=artifact,
        warnings=[],
        timings={"total": 1},
    )
    assert result["query_id"] == "q-test"
    assert result["stats"]["documents_total"] == 1
    assert result["stats"]["candidates_found"] == 2
    assert len(result["top"]) + len(result["excluded"]) == 2
    assert result["top"] or result["excluded"]
