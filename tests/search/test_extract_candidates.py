"""Тесты шага 4: фильтры и дедуп без реального вызова YandexGPT."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

from search.extract_candidates import (
    dedupe_candidates,
    extract_candidates,
    filter_raw_candidate,
    parse_llm_json,
    snippet_from_doc,
)

FIXTURES = Path(__file__).resolve().parent / "fixtures" / "extract_docs.json"


def test_snippet_truncates_text() -> None:
    doc = {"title": "T", "text": "a" * 2000}
    snip = snippet_from_doc(doc, limit=100)
    assert snip.startswith("T\n")
    assert len(snip) <= 102


def test_parse_llm_json_strips_fence() -> None:
    raw = '```json\n{"candidates": []}\n```'
    assert parse_llm_json(raw)["candidates"] == []


def test_filter_rejects_short_and_topic() -> None:
    assert filter_raw_candidate({"doc": 1, "name_ru": "ИИ", "name_en": "AI"}, "тема") is None
    assert (
        filter_raw_candidate(
            {
                "doc": 1,
                "name_ru": "технологии в медицине",
                "name_en": "medical technologies",
            },
            "технологии в медицине",
        )
        is None
    )
    ok = filter_raw_candidate(
        {
            "doc": 2,
            "name_ru": "фотонные процессоры инференса",
            "name_en": "photonic inference processor",
            "terms": ["optical AI accelerator"],
            "context_terms": ["photonic computing"],
        },
        "технологии в медицине",
    )
    assert ok is not None
    assert ok["doc"] == 2


def test_dedupe_merges_near_duplicates() -> None:
    items = [
        {
            "doc": 1,
            "name_ru": "фотонные процессоры инференса",
            "name_en": "photonic inference processor",
            "terms": ["photonic inference processor"],
            "context_terms": [],
        },
        {
            "doc": 2,
            "name_ru": "фотонный процессор инференса",
            "name_en": "photonic inference processors",
            "terms": ["optical AI accelerator"],
            "context_terms": ["silicon photonics"],
        },
        {
            "doc": 3,
            "name_ru": "графенные биосенсоры",
            "name_en": "graphene biosensors",
            "terms": ["graphene biosensor"],
            "context_terms": [],
        },
    ]
    merged = dedupe_candidates(items, threshold=0.55)
    assert len(merged) <= 2
    photonic = next(m for m in merged if "photonic" in m["name_en"].casefold())
    assert photonic["doc_count"] >= 2
    assert "optical AI accelerator" in photonic["terms"]


def test_extract_candidates_with_mocked_llm() -> None:
    docs = json.loads(FIXTURES.read_text(encoding="utf-8"))
    fake_response = {
        "text": json.dumps(
            {
                "candidates": [
                    {
                        "doc": 1,
                        "name_ru": "фотонные процессоры инференса",
                        "name_en": "photonic inference processor",
                        "terms": ["photonic inference processor", "optical AI accelerator"],
                        "context_terms": ["photonic computing"],
                    },
                    {
                        "doc": 2,
                        "name_ru": "фотонные процессоры инференса",
                        "name_en": "photonic inference processor",
                        "terms": ["optical AI accelerator"],
                        "context_terms": [],
                    },
                    {
                        "doc": 3,
                        "name_ru": "графенные биосенсоры",
                        "name_en": "graphene biosensors",
                        "terms": ["graphene biosensor"],
                        "context_terms": ["home monitoring"],
                    },
                    {
                        "doc": 4,
                        "name_ru": "ИИ",
                        "name_en": "AI",
                        "terms": [],
                        "context_terms": [],
                    },
                ]
            },
            ensure_ascii=False,
        ),
        "error": None,
        "model_uri": "gpt://test/yandexgpt/latest",
    }

    with patch("search.extract_candidates.ask_llm", return_value=fake_response):
        result = extract_candidates(
            docs,
            topic="технологии в медицине",
            query_id="q-test",
            use_cache=False,
            similarity_threshold=0.55,
        )

    assert result["query_id"] == "q-test"
    ids = {c["candidate_id"] for c in result["candidates"]}
    assert len(result["candidates"]) >= 2
    assert all(cid.startswith("q-test-c") for cid in ids)
    # короткое «ИИ» отфильтровано
    assert all(c["name_en"].casefold() != "ai" for c in result["candidates"])
    assert "terms" in result["candidates"][0]
    assert "context_terms" in result["candidates"][0]
    assert "aliases" in result["candidates"][0]
