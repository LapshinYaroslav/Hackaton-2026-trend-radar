"""API без мока: run_query подменён готовым прогоном пайплайна (examples/robots.json), сеть и LLM выключены."""
import copy
import json
import time
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

ROOT = Path(__file__).resolve().parents[1]
ROBOTS = json.loads((ROOT / "examples" / "robots.json").read_text(encoding="utf-8"))


@pytest.fixture
def pipeline_calls(monkeypatch) -> list[dict]:
    """Вызовы подменённого run_query: аргументы и возвращённый объект (result)."""
    calls: list[dict] = []

    def fake_run_query(topic, **kwargs):
        result = copy.deepcopy(ROBOTS)
        calls.append({"topic": topic, "kwargs": kwargs, "result": result})
        return result

    monkeypatch.delenv("DATABASE_URL", raising=False)
    monkeypatch.setattr("pipeline.run_query.run_query", fake_run_query)
    monkeypatch.setattr("model.bootstrap.ensure_artifact", lambda *args, **kwargs: None)
    monkeypatch.setattr("pipeline.run_query.check_config", lambda *args, **kwargs: None)
    monkeypatch.setattr("pipeline.insights._llm_insight", lambda item, docs: None)
    return calls


@pytest.fixture
def client(pipeline_calls) -> TestClient:
    from api.main import app

    return TestClient(app)


@pytest.fixture
def finished(client):
    """POST темы и опрос GET до done или error; возвращает тело последнего ответа."""
    def run(topic: str = "роботы для промышленности") -> dict:
        query_id = client.post("/queries", json={"topic": topic}).json()["query_id"]
        deadline = time.monotonic() + 10
        while time.monotonic() < deadline:
            body = client.get(f"/queries/{query_id}").json()
            if body["status"] in ("done", "error"):
                return body
            time.sleep(0.02)
        raise AssertionError(f"запрос {query_id} не завершился за 10 с")
    return run
