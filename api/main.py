"""
Временная заглушка API (зона Славы).
Отдаёт мок из docs/contracts, чтобы Docker Compose и UI можно было поднять до готовности пайплайна.
Когда появится настоящий FastAPI — этот файл заменяется/расширяется через PR.
"""

from pathlib import Path
import json

from fastapi import FastAPI, Query

CONTRACT_PATH = Path(__file__).resolve().parent / "docs" / "contracts" / "api_response.json"

app = FastAPI(title="Trend Radar API (stub)", version="0.1.0")


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.get("/search")
def search(q: str = Query(default="", description="Технологическое направление")) -> dict:
    with CONTRACT_PATH.open(encoding="utf-8") as f:
        data = json.load(f)
    data = dict(data)
    if q.strip():
        data["query"] = q.strip()
    data["status"] = "done"
    return data
