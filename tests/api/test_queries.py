"""API отдаёт только результат run_query: без мока, с проверкой происхождения и неизменности после done."""

from __future__ import annotations

import copy

import pytest

from tests.conftest import ROBOTS
from ui import view


def test_other_area_is_null(finished, pipeline_calls) -> None:
    body = finished("что-то новое", "Другое")
    assert body["area"] is None and pipeline_calls[-1]["area"] is None


def test_unknown_area_is_rejected(client) -> None:
    assert client.post("/queries", json={"topic": "тема", "area": "Космос"}).status_code == 422


def test_list_queries_remembers_created(client) -> None:
    created = client.post("/queries", json={"topic": "история теста", "area": "Edge"})
    query_id = created.json()["query_id"]
    listed = client.get("/queries").json()["items"]
    match = next(item for item in listed if item["query_id"] == query_id)
    assert match["topic"] == "история теста"
    assert match["area"] == "Edge"


def test_answer_is_run_query_output(finished) -> None:
    body = finished()
    shown = view.check_result(body, body["query_id"])
    assert view.top_rows(shown) == view.top_rows(ROBOTS)
    assert view.stats_view(shown) == view.stats_view(ROBOTS)
    assert "_documents" not in body and "mock" not in body


def test_answer_after_done_does_not_change(client, finished, pipeline_calls) -> None:
    body = finished()
    pipeline_calls[-1]["result"]["top"][0]["score"] = 0.0  # тот, кто держит ссылку на result, ответ не меняет
    pipeline_calls[-1]["result"]["top"].clear()
    again = client.get(f"/queries/{body['query_id']}").json()
    assert again == body


def test_pipeline_error_is_explicit(finished, monkeypatch) -> None:
    def broken(*args, **kwargs):
        raise RuntimeError("OpenAlex не ответил")

    monkeypatch.setattr("pipeline.run_query.run_query", broken)
    body = finished()
    assert body["status"] == "error" and "OpenAlex не ответил" in body["error"]
    assert "top" not in body


def test_health_has_no_mock(client) -> None:
    assert "mock" not in client.get("/health").json()


@pytest.mark.parametrize("name", ["load_example", "use_mock", "_run_mock", "contract_path"])
def test_mock_mode_is_gone(name: str) -> None:
    import api.main

    assert not hasattr(api.main, name)


def test_progress_is_pipeline_percent(finished, monkeypatch) -> None:
    """API передаёт run_query on_progress и кладёт pct, этап и остаток как есть: полоса не обнуляется по этапам."""
    from api import store

    seen, events = [], [{"pct": 3, "stage_ru": "Подзапросы", "eta_s": 870.0},
                        {"pct": 41, "stage_ru": "Сбор счётчиков (arXiv, OpenAlex, TechCrunch, Роспатент)", "eta_s": 480.0},
                        {"pct": 97, "stage_ru": "Догрузка источников", "eta_s": 20.0},
                        {"pct": 100, "stage_ru": "Готово", "eta_s": 0.0}]

    def fake(topic, area=None, **kwargs):
        assert "progress" not in kwargs
        for event in events:
            kwargs["on_progress"](event)
            job = store.get_query(kwargs["query_id"])
            seen.append((job["progress_done"], job["progress_total"], job["progress_stage"], job["eta_s"]))
        return copy.deepcopy(ROBOTS)

    monkeypatch.setattr("pipeline.run_query.run_query", fake)
    assert finished()["status"] == "done"
    assert seen == [(e["pct"], 100, e["stage_ru"], e["eta_s"]) for e in events]
