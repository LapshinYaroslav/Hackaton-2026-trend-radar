"""Область по теме: разбор ответа модели и запасной путь по словам, без сети."""

from search.area import _FAILED, area_from_keywords, parse_area


def test_parse_known_area_and_other() -> None:
    assert parse_area("Роботы") == "Роботы"
    assert parse_area("  финтех\nпояснение") == "Финтех"
    assert parse_area("Другое") is None
    assert parse_area("не знаю") is _FAILED


def test_keywords_prefer_robots_over_industry() -> None:
    assert area_from_keywords("промышленные роботы") == "Роботы"
    assert area_from_keywords("платежи в банке") == "Финтех"
    assert area_from_keywords("терапия рака") is None
