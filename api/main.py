"""
API запросов на новом контракте docs/contracts/query_result.example.json.

Пока оркестратора нет: POST создаёт запрос и через несколько секунд
кладёт пример из контракта. Если задан DATABASE_URL — схема, посев
обучающей выборки и история запросов пишутся в Postgres.
"""

from __future__ import annotations

import copy
import json
import os
import threading
import time
import uuid
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from api import store

STAGES = (
    "Подзапросы",
    "Поиск документов",
    "Извлечение кандидатов",
    "Проверка",
    "Сбор статистики",
    "Оценка",
)

AREAS = {
    "Edge",
    "Защита ИИ",
    "Индустриальный ИИ",
    "Инфраструктура ИИ",
    "Роботы",
    "Финтех",
}


def contract_path() -> Path:
    here = Path(__file__).resolve().parent
    candidates = (
        here / "docs" / "contracts" / "query_result.example.json",
        here.parent / "docs" / "contracts" / "query_result.example.json",
    )
    for path in candidates:
        if path.is_file():
            return path
    raise FileNotFoundError("нет docs/contracts/query_result.example.json")


def load_example() -> dict:
    data = json.loads(contract_path().read_text(encoding="utf-8"))
    data.pop("_note", None)
    return data


def mock_seconds() -> float:
    raw = os.getenv("QUERY_MOCK_SECONDS", "3").strip()
    try:
        return max(0.0, float(raw))
    except ValueError:
        return 3.0


class CreateQuery(BaseModel):
    topic: str = Field(min_length=1)
    area: Optional[str] = None


app = FastAPI(title="Trend Radar API", version="0.2.0")
_lock = threading.Lock()


@app.on_event("startup")
def _startup() -> None:
    try:
        store.ensure_ready()
    except Exception:  # noqa: BLE001
        # Без живой БД API остаётся на моке и памяти процесса.
        pass


def _public(job: dict) -> dict:
    payload = {
        "query_id": job["query_id"],
        "topic": job["topic"],
        "area": job["area"],
        "status": job["status"],
        "progress_stage": job["progress_stage"],
        "progress_done": job["progress_done"],
        "progress_total": job["progress_total"],
    }
    if job["status"] == "done" and job.get("result"):
        payload.update(job["result"])
        payload["query_id"] = job["query_id"]
        payload["topic"] = job["topic"]
        payload["area"] = job["area"]
        payload["status"] = "done"
    if job["status"] == "error":
        payload["error"] = job.get("error") or "ошибка расчёта"
    return payload


def _run(query_id: str) -> None:
    seconds = mock_seconds()
    step = seconds / len(STAGES) if STAGES else 0
    try:
        for index, stage in enumerate(STAGES, start=1):
            with _lock:
                store.update_progress(
                    query_id,
                    progress_stage=stage,
                    progress_done=index,
                    progress_total=len(STAGES),
                    status="running",
                )
            if step:
                time.sleep(step)
        example = load_example()
        with _lock:
            job = store.get_query(query_id) or {}
            example["query_id"] = query_id
            example["topic"] = job.get("topic")
            example["area"] = job.get("area")
            store.update_progress(
                query_id,
                progress_stage=STAGES[-1],
                progress_done=len(STAGES),
                progress_total=len(STAGES),
                status="done",
                model_version=example.get("model_version"),
                threshold=example.get("threshold"),
                cutoff_date=example.get("cutoff_date"),
            )
            store.finish_query(query_id, example)
    except Exception as exc:  # noqa: BLE001
        with _lock:
            store.update_progress(query_id, status="error", error=str(exc))
            store.finish_query(query_id, None, error=str(exc))


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "database": bool(store.database_url())}


@app.post("/queries")
def create_query(body: CreateQuery) -> dict:
    topic = body.topic.strip()
    if not topic:
        raise HTTPException(status_code=422, detail="пустая тема")
    area = body.area
    if area in (None, "", "Другое"):
        area = None
    elif area not in AREAS:
        raise HTTPException(status_code=422, detail="неизвестная область")
    query_id = f"q-{uuid.uuid4().hex[:8]}"
    job = {
        "query_id": query_id,
        "topic": topic,
        "area": area,
        "status": "running",
        "progress_stage": STAGES[0],
        "progress_done": 0,
        "progress_total": len(STAGES),
        "result": None,
        "error": None,
        "created_at": None,
    }
    with _lock:
        store.create_query(job)
    threading.Thread(target=_run, args=(query_id,), daemon=True).start()
    return {"query_id": query_id}


@app.get("/queries")
def list_queries(limit: int = 30) -> dict:
    return {"items": store.list_queries(limit=max(1, min(limit, 100)))}


@app.get("/queries/{query_id}")
def get_query(query_id: str) -> dict:
    with _lock:
        job = store.get_query(query_id)
        if job is None:
            raise HTTPException(status_code=404, detail="запрос не найден")
        return _public(copy.deepcopy(job))


@app.get("/queries/{query_id}/insights/{rank}")
def get_insight(query_id: str, rank: int) -> dict:
    """Пока без LLM: инсайт не на критическом пути (задача 2.9)."""
    with _lock:
        job = store.get_query(query_id)
        if job is None:
            raise HTTPException(status_code=404, detail="запрос не найден")
        status = job["status"]
    if status != "done":
        return {"query_id": query_id, "rank": rank, "status": "pending"}
    return store.insight_status(query_id, rank)


@app.get("/catalog/balance")
def catalog_balance() -> dict:
    return {"areas": store.training_balance()}


@app.get("/catalog/technologies")
def catalog_technologies(label: Optional[int] = None) -> dict:
    if label not in (None, 0, 1):
        raise HTTPException(status_code=422, detail="label: 0, 1 или пусто")
    return {"items": store.list_technologies(label)}
