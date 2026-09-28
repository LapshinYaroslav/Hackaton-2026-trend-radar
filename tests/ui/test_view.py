"""Экран = JSON пайплайна: строки UI сверяются с examples/*.json, прочитанными напрямую."""
import copy
import json
from pathlib import Path

import pytest

from ui import view

ROOT = Path(__file__).resolve().parents[2]
FILES = sorted((ROOT / "examples").glob("*.json"))


def raw(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


@pytest.mark.parametrize("path", FILES, ids=[f.stem for f in FILES])
def test_screen_equals_pipeline_json(path: Path) -> None:
    data = raw(path)
    shown = view.check_result(copy.deepcopy(data), data["query_id"])
    assert [(r["rank"], r["name_ru"], r["name_en"], r["score"], r["n_pat"]) for r in view.top_rows(shown)] == \
        [(t["rank"], t["name_ru"], t["name_en"], t["score"], t["n_pat"]) for t in data["top"]]
    assert [(r["name_en"], r["score"], r["reason_ru"]) for r in view.excluded_rows(shown)] == \
        [(e["name_en"], e["score"], e["reason_ru"]) for e in data["excluded"]]
    # то же, что строка CLI «ТОП: N, исключено: M» и плашки
    assert view.stats_view(shown) == {"documents_analyzed": data["stats"]["documents_total"],  # поля ещё нет
                                      "documents_total": data["stats"]["documents_total"],
                                      "candidates_found": data["stats"]["candidates_found"],
                                      "above_075": data["stats"]["above_075"],
                                      "top": len(data["top"]), "excluded": len(data["excluded"])}


@pytest.mark.parametrize("spoil, message", [
    (lambda d: d.pop("timings"), "нет поля timings"),
    (lambda d: d["stats"].pop("above_075"), "нет stats.above_075"),
    (lambda d: d.update(query_id="q-other"), "query_id"),
    (lambda d: d["top"].pop(0), "ранги ТОП"),
    (lambda d: d["top"][0].update(score=0.1), "ниже порога"),
    (lambda d: d["top"].append(copy.deepcopy(d["top"][0])), "ранги ТОП"),
])
def test_not_pipeline_answer_is_error(spoil, message: str) -> None:
    data = raw(FILES[0])
    query_id = data["query_id"]
    spoil(data)
    with pytest.raises(ValueError, match=message):
        view.check_result(data, query_id)


@pytest.mark.parametrize("data", [None, [], "текст"])
def test_non_object_is_error(data) -> None:
    with pytest.raises(ValueError, match="не JSON-объект"):
        view.check_result(data, "q1")


def test_candidates_list_comes_from_candidates_field() -> None:
    data = {"candidates": [{"stage": "top", "name_ru": "а", "name_en": "a", "score": 0.9, "reason_ru": None},
                           {"stage": "found", "name_ru": "б", "name_en": "b", "score": None, "reason_ru": None}]}
    assert [r["name_en"] for r in view.candidate_rows(data)] == ["a", "b"]
    assert view.candidate_rows({}) is None
    assert view.candidate_rows({"candidates": []}) == []


def test_score_text() -> None:
    assert view.score_text(0.945832902678657) == "0.946"
    assert view.score_text(None) == "—"


def test_insight_view_reads_api_format() -> None:
    shown = view.insight_view({"status": "done", "description_ru": "описание", "advantages_ru": ["быстрее"],
                               "cases_ru": [], "sources": [{"url": "u"}]})
    assert shown["description"] == "описание" and shown["advantages"] == ["быстрее"] and shown["sources"]
    assert view.insight_view({"status": "pending"}) is None
    with pytest.raises(ValueError, match="сбой"):
        view.insight_view({"status": "error", "error": "сбой"})


@pytest.mark.parametrize("payload, share, text", [
    ({}, 0.0, "Запуск · 0 %"),
    ({"progress_stage": "Сбор счётчиков (arXiv, OpenAlex, TechCrunch, Роспатент)", "progress_done": 37,
      "progress_total": 100, "eta_s": 540.0}, 0.37, "Сбор счётчиков (arXiv, OpenAlex, TechCrunch, Роспатент) · 37 % · осталось ~9 мин"),
    ({"progress_stage": "Склейка дублей", "progress_done": 99, "progress_total": 100, "eta_s": 12.5}, 0.99,
     "Склейка дублей · 99 % · осталось меньше минуты"),
    ({"progress_stage": "Готово", "progress_done": 130, "progress_total": 100, "eta_s": None}, 1.0, "Готово · 100 %"),
    ({"progress_stage": "Подзапросы", "progress_done": -5, "progress_total": 0}, 0.0, "Подзапросы · 0 %"),
])
def test_progress_view(payload, share, text) -> None:
    assert view.progress_view(payload) == {"share": share, "text": text}


def test_stats_view_prefers_documents_analyzed() -> None:
    data = {"stats": {"documents_analyzed": 580376, "documents_total": 397, "candidates_found": 121, "above_075": 12},
            "top": [], "excluded": []}
    assert view.stats_view(data)["documents_analyzed"] == 580376
