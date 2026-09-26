"""Мок API без Docker: пример контракта приходит через POST/GET."""

from __future__ import annotations

import os

os.environ["QUERY_MOCK_SECONDS"] = "0"

from fastapi.testclient import TestClient

from api.main import app

client = TestClient(app)


def test_query_returns_example_contract() -> None:
    created = client.post("/queries", json={"topic": "роботы для цеха", "area": "Роботы"})
    assert created.status_code == 200
    query_id = created.json()["query_id"]

    body = client.get(f"/queries/{query_id}").json()
    # поток может ещё не дописать done при 0 секунд — опросим пару раз
    for _ in range(20):
        body = client.get(f"/queries/{query_id}").json()
        if body["status"] == "done":
            break
    assert body["status"] == "done"
    assert body["topic"] == "роботы для цеха"
    assert body["area"] == "Роботы"
    assert body["top"][0]["name_ru"]
    assert "explanation_ru" in body["top"][0]
    assert body["excluded"]
    assert body["stats"]["candidates_found"] == 64
    assert "_note" not in body


def test_other_area_is_null() -> None:
    created = client.post("/queries", json={"topic": "что-то новое", "area": "Другое"})
    query_id = created.json()["query_id"]
    body = {"status": "running"}
    for _ in range(20):
        body = client.get(f"/queries/{query_id}").json()
        if body["status"] == "done":
            break
    assert body["area"] is None


def test_list_queries_remembers_created() -> None:
    created = client.post("/queries", json={"topic": "история теста", "area": "Edge"})
    query_id = created.json()["query_id"]
    listed = client.get("/queries").json()["items"]
    ids = {item["query_id"] for item in listed}
    assert query_id in ids
    match = next(item for item in listed if item["query_id"] == query_id)
    assert match["topic"] == "история теста"
    assert match["area"] == "Edge"


def test_insight_is_pending() -> None:
    created = client.post("/queries", json={"topic": "тема", "area": "Финтех"})
    query_id = created.json()["query_id"]
    for _ in range(20):
        if client.get(f"/queries/{query_id}").json()["status"] == "done":
            break
    insight = client.get(f"/queries/{query_id}/insights/1").json()
    assert insight["status"] == "pending"
