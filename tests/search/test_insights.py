"""Инсайт: проверка ссылок, доверие и кэш. Живой YandexGPT не вызывается."""

from __future__ import annotations

import json
from unittest.mock import patch

from search.insights import (
    NOT_FOUND,
    generate_insight,
    status_explanation,
    trust_for,
    validate_sections,
)

CARD = {
    "name_ru": "Данные телеуправления",
    "name_en": "robotic teleoperation data",
    "score": 0.91,
    "explanation_ru": ["Про технологию пока мало научных публикаций: 14"],
    "sources": [
        {
            "title": "Scaling teleoperation",
            "url": "https://example.org/doc-1",
            "published_at": "2026-07-02",
            "source_type": "preprint",
            "language": "en",
            "text": "Early dataset for robot teleoperation.",
        },
        {
            "title": "Startup raises seed",
            "url": "https://example.org/doc-2",
            "published_at": "2026-05-18",
            "source_type": "news",
            "language": "en",
            "text": "A company sells robot training data.",
        },
    ],
}


def _ok_payload() -> str:
    return json.dumps(
        {
            "description": {"text": "Собирают данные телеуправления. [1]", "refs": [1]},
            "advantage": {"text": "Можно учить роботов на чужих демонстрациях. [2]", "refs": [2]},
            "case": {"text": NOT_FOUND, "refs": []},
            "analyst_assessment": {"text": NOT_FOUND, "refs": []},
            "summaries": [
                {"n": 1, "summary_ru": "Статья про сбор данных телеуправления."},
                {"n": 2, "summary_ru": "Новость про продажу данных для роботов."},
            ],
        },
        ensure_ascii=False,
    )


def test_trust_table() -> None:
    assert trust_for("paper")[0] == "high"
    assert trust_for("preprint")[0] == "medium"
    assert trust_for("news")[0] == "medium"
    assert trust_for("blog")[0] == "low"
    assert trust_for("press_release")[0] == "low"
    assert trust_for("product")[0] == "low"


def test_status_explanation_uses_score_from_code() -> None:
    text = status_explanation(CARD)
    assert text.startswith("Уверенность модели 0.91.")
    assert "14" in text


def test_rejects_missing_ref_and_unknown_doc() -> None:
    missing = {"description": {"text": "просто факт", "refs": []}}
    assert validate_sections(missing, 1)
    unknown = {"description": {"text": "факт [9]", "refs": [9]}}
    assert any("несуществующий" in item for item in validate_sections(unknown, 2))


def test_generate_retries_bad_ref_once(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr("search.insights.CACHE_DIR", tmp_path)
    bad = json.dumps(
        {
            "description": {"text": "выдумано [9]", "refs": [9]},
            "advantage": {"text": NOT_FOUND, "refs": []},
            "case": {"text": NOT_FOUND, "refs": []},
            "analyst_assessment": {"text": NOT_FOUND, "refs": []},
            "summaries": [],
        },
        ensure_ascii=False,
    )
    answers = iter([{"text": bad, "error": None}, {"text": _ok_payload(), "error": None}])

    def _fake(*args, **kwargs):
        return next(answers)

    with patch("search.insights.ask_llm", side_effect=_fake) as mocked:
        content = generate_insight(CARD, use_cache=False)
    assert mocked.call_count == 2
    assert content["description"]["refs"] == [1]
    assert content["sources"][0]["trust_level"] == "medium"
    assert content["sources"][0]["summary_note"]
    assert content["low_trust_warning"] is None


def test_low_trust_warning(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr("search.insights.CACHE_DIR", tmp_path)
    card = {
        **CARD,
        "sources": [
            {
                "title": "Блог",
                "url": "https://example.org/blog",
                "published_at": "2026-01-01",
                "source_type": "blog",
                "language": "ru",
                "text": "Демо без проверки.",
            }
        ],
    }
    payload = json.dumps(
        {
            "description": {"text": "Показали демо. [1]", "refs": [1]},
            "advantage": {"text": NOT_FOUND, "refs": []},
            "case": {"text": NOT_FOUND, "refs": []},
            "analyst_assessment": {"text": NOT_FOUND, "refs": []},
            "summaries": [],
        },
        ensure_ascii=False,
    )
    with patch("search.insights.ask_llm", return_value={"text": payload, "error": None}):
        content = generate_insight(card, use_cache=False)
    assert content["low_trust_warning"]
    assert content["sources"][0]["summary_note"] is None


def test_cache_skips_second_llm_call(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr("search.insights.CACHE_DIR", tmp_path)
    with patch("search.insights.ask_llm", return_value={"text": _ok_payload(), "error": None}) as mocked:
        generate_insight(CARD)
        generate_insight(CARD)
    assert mocked.call_count == 1
