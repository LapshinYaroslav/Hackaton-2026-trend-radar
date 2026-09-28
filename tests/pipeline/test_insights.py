from pipeline.catalog import catalog_candidates
from pipeline.insights import build_insight, extractive_insight


def test_extractive_insight_uses_documents() -> None:
    item = {
        "rank": 1,
        "name_ru": "Кисть робота",
        "name_en": "dexterous robotic hand",
        "explanation_ru": ["Молодой термин"],
        "sources": [
            {
                "title": "A new robotic hand",
                "url": "https://arxiv.org/abs/1",
                "language": "en",
                "source_type": "preprint",
                "text": "The hand grasps unseen objects in a warehouse.",
            }
        ],
    }
    payload = extractive_insight(item)
    assert "warehouse" in payload["description_ru"] or "Молодой" in payload["description_ru"]
    assert payload["cases_ru"]
    assert payload["sources"][0]["summary_note"]
    assert payload["sources"][0]["language"] == "en"


def test_build_insight_without_llm() -> None:
    item = {
        "rank": 2,
        "name_en": "soft gripper",
        "name_ru": "мягкий схват",
        "sources": [
            {
                "title": "Soft gripper ships",
                "url": "https://techcrunch.com/x",
                "language": "en",
                "source_type": "news",
                "text": "A startup ships a soft gripper for food packing.",
            }
        ],
    }
    payload = build_insight(item, use_llm=False)
    assert payload["status"] == "done"
    assert payload["description_ru"]
    assert payload["sources"][0]["summary_ru"]


def test_catalog_lists_found_names() -> None:
    rows = catalog_candidates(
        [{"name_en": "alpha bot", "name_ru": "альфа"}, {"name_en": "beta bot"}],
        [{"name_en": "alpha bot", "name_ru": "альфа", "score": 0.9}],
        [{"name_en": "old robot", "score": 0.1, "skipped_reason": "below_threshold"}],
    )
    assert [row["stage"] for row in rows] == ["top", "excluded", "found"]
    assert rows[2]["name_en"] == "beta bot"
