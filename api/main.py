"""
API запросов: POST запускает pipeline.run_query, GET отдаёт контракт выдачи.

Ответ — только результат run_query: мок-режима и файла-примера нет.
Если задан DATABASE_URL — схема, посев и история пишутся в Postgres.
"""

from __future__ import annotations

import copy
import threading
import uuid
from pathlib import Path

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from dotenv import load_dotenv

from api import store

load_dotenv(Path(__file__).resolve().parent.parent / ".env")

# Прогресс — события pipeline.progress (on_progress): pct от бюджета времени, не убывает, 100 — только в конце.
# Хранится в progress_done при progress_total = 100 (схема queries не меняется); eta_s — оценка остатка, в памяти.
START_STAGE = "Запуск"


class CreateQuery(BaseModel):
    topic: str = Field(min_length=1)


app = FastAPI(title="Trend Radar API", version="0.2.0")
_lock = threading.Lock()


@app.on_event("startup")
def _startup() -> None:
    from model.bootstrap import ensure_artifact
    from pipeline.run_query import check_config

    check_config()  # нет ключа — API не стартует
    ensure_artifact()  # нет обученной модели — API не стартует
    store.ensure_ready()  # DATABASE_URL задан, а база не отвечает — API не стартует


def _public(job: dict) -> dict:
    payload = {
        "query_id": job["query_id"],
        "topic": job["topic"],
        "status": job["status"],
        "progress_stage": job["progress_stage"],
        "progress_done": job["progress_done"],
        "progress_total": job["progress_total"],
        "eta_s": job.get("eta_s"),
    }
    if job["status"] == "done" and job.get("result"):
        payload.update(store.public_result(job["result"]) or {})
        payload.pop("_documents", None)
        payload["query_id"] = job["query_id"]
        payload["topic"] = job["topic"]
        payload.pop("area", None)  # прогоны, сохранённые до удаления областей
        payload["status"] = "done"
    if job["status"] == "error":
        payload["error"] = job.get("error") or "ошибка расчёта"
    return payload


def _on_progress(query_id: str):
    """Событие pipeline.progress -> этап по-русски, проценты и оценка остатка в записи запроса."""
    def on_progress(event: dict) -> None:
        with _lock:
            store.update_progress(
                query_id,
                status="running",
                progress_stage=event["stage_ru"],
                progress_done=int(event["pct"]),
                progress_total=100,
                eta_s=event.get("eta_s"),
            )

    return on_progress


def _run_live(query_id: str) -> None:
    from model.bootstrap import ensure_artifact
    from pipeline.run_query import run_query

    ensure_artifact()
    with _lock:
        job = store.get_query(query_id) or {}
        topic = job.get("topic") or ""
        store.update_progress(
            query_id,
            status="running",
            progress_stage=START_STAGE,
            progress_done=0,
            progress_total=100,
        )
    result = run_query(topic, query_id=query_id, on_progress=_on_progress(query_id))
    result["query_id"] = query_id
    result["topic"] = topic
    with _lock:
        # копия: после done ответ не меняется, даже если кто-то держит ссылку на result
        store.finish_query(query_id, copy.deepcopy(result))


def _run(query_id: str) -> None:
    try:
        _run_live(query_id)
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
    query_id = f"q-{uuid.uuid4().hex[:8]}"
    job = {
        "query_id": query_id,
        "topic": topic,
        "status": "running",
        "progress_stage": START_STAGE,
        "progress_done": 0,
        "progress_total": 100,
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
    with _lock:
        job = store.get_query(query_id)
    if job is None:
        raise HTTPException(status_code=404, detail="запрос не найден")
    if job.get("status") != "done" or not job.get("result"):
        raise HTTPException(status_code=409, detail="запрос ещё считается")
    result = job["result"]
    item = next((row for row in (result.get("top") or []) if row.get("rank") == rank), None)
    if item is None:
        raise HTTPException(status_code=404, detail="карточка не найдена")
    stored = store.get_insight(query_id, rank)
    if stored and stored.get("status") == "done" and stored.get("description_ru"):
        return {**stored, "query_id": query_id, "rank": rank}
    from pipeline.insights import build_insight

    # Инсайт строится в run_query до выдачи; build_insight здесь — только для прогонов, сделанных раньше.
    payload = dict(item.get("insight") or build_insight(item, result.get("_documents") or []))
    payload["query_id"] = query_id
    payload["rank"] = rank
    store.save_insight(query_id, rank, payload)
    return payload


@app.get("/catalog/balance")
def catalog_balance() -> dict:
    return {"areas": store.training_balance()}
