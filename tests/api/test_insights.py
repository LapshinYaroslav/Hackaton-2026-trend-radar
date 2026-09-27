"""Инсайт по клику заполняет текст, а не оставляет pending."""

from __future__ import annotations

import os

os.environ["QUERY_MOCK"] = "1"
os.environ["QUERY_MOCK_SECONDS"] = "0"

from fastapi.testclient import TestClient

from api.main import app

client = TestClient(app)


def _done_query() -> str:
    created = client.post("/queries", json={"topic": "роботы для цеха", "area": "Роботы"})
    query_id = created.json()["query_id"]
    body = {}
    for _ in range(20):
        body = client.get(f"/queries/{query_id}").json()
        if body["status"] == "done":
            break
    assert body["status"] == "done"
    return query_id


def test_insight_is_generated_from_documents() -> None:
    query_id = _done_query()
    rank = client.get(f"/queries/{query_id}").json()["top"][0]["rank"]
    insight = client.get(f"/queries/{query_id}/insights/{rank}").json()
    assert insight["status"] == "done"
    assert insight["description_ru"]
    assert insight["sources"]
    assert any(source.get("summary_note") for source in insight["sources"])
    again = client.get(f"/queries/{query_id}/insights/{rank}").json()
    assert again["description_ru"] == insight["description_ru"]
