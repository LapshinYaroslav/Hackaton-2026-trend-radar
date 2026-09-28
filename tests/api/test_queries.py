"""Мок API без Docker: пример контракта приходит через POST/GET."""

from __future__ import annotations

import os

os.environ["QUERY_MOCK"] = "1"
os.environ["QUERY_MOCK_SECONDS"] = "0"

from fastapi.testclient import TestClient

from api.main import app

client = TestClient(app)


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
