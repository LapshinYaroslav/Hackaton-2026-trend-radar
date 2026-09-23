"""Тесты общих частей шага 4: фрагмент документа и дедуп без вызова YandexGPT."""

from __future__ import annotations

from search.extract_candidates import dedupe_candidates, snippet_from_doc


def test_snippet_truncates_text() -> None:
    doc = {"title": "T", "text": "a" * 2000}
    snip = snippet_from_doc(doc, limit=100)
    assert snip.startswith("T\n")
    assert len(snip) <= 102



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
