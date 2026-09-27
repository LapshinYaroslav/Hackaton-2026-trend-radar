"""
API запросов: POST запускает pipeline.run_query, GET отдаёт контракт выдачи.

QUERY_MOCK=1 — пример из файла (тесты). Без флага тема пользователя идёт в оркестратор.
Если задан DATABASE_URL — схема, посев и история пишутся в Postgres.
"""

from __future__ import annotations

import copy
import json
import logging
import os
import threading
import time
import uuid
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from api import store

logger = logging.getLogger(__name__)

STAGES = (
    "Подзапросы",
    "Поиск документов",
    "Извлечение кандидатов",
    "Проверка",
    "Сбор статистики",
    "Оценка",
)

STAGE_RU = {
    "subqueries": "Подзапросы",
    "search": "Поиск документов",
    "candidates": "Извлечение кандидатов",
    "naming": "Проверка",
    "counters": "Сбор статистики",
    "ranking": "Оценка",
    "dedup": "Оценка",
    "translate": "Оценка",
}

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


def use_mock() -> bool:
    """Тесты и явный мок. Без QUERY_MOCK запрос идёт в run_query."""
    return os.getenv("QUERY_MOCK", "").strip().lower() in {"1", "true", "yes"}


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
        pass
    if not use_mock():
        try:
            from model.bootstrap import ensure_artifact

            ensure_artifact()
        except Exception as exc:
            logger.warning("model artifact: %s", exc)


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
        payload.update(store.public_result(job["result"]) or {})
        payload.pop("_documents", None)
        payload["query_id"] = job["query_id"]
        payload["topic"] = job["topic"]
        payload["area"] = job["area"]
        payload["status"] = "done"
    if job["status"] == "error":
        payload["error"] = job.get("error") or "ошибка расчёта"
    return payload


def _progress(query_id: str):
    def on_progress(stage: str, done: int, total: int) -> None:
        with _lock:
            store.update_progress(
                query_id,
                status="running",
                progress_stage=STAGE_RU.get(stage, stage),
                progress_done=done,
                progress_total=max(int(total or 1), 1),
            )

    return on_progress


def _enrich_later(query_id: str) -> None:
    with _lock:
        job = store.get_query(query_id)
        result = (job or {}).get("result")
    if not result:
        return
    try:
        from pipeline.enrich import enrich_top

        updated = enrich_top(result)
        with _lock:
            store.patch_result(query_id, updated)
    except Exception as exc:
        logger.warning("enrich %s: %s", query_id, exc)
        result["enrichment"] = "error"
        with _lock:
            store.patch_result(query_id, result)


def _run_live(query_id: str) -> None:
    from model.bootstrap import ensure_artifact
    from pipeline.run_query import run_query

    ensure_artifact()
    with _lock:
        job = store.get_query(query_id) or {}
        topic = job.get("topic") or ""
        area = job.get("area")
        store.update_progress(
            query_id,
            status="running",
            progress_stage=STAGES[0],
            progress_done=0,
            progress_total=len(STAGES),
        )
    result = run_query(topic, area=area, query_id=query_id, progress=_progress(query_id))
    result["query_id"] = query_id
    result["topic"] = topic
    result["area"] = area
    result.setdefault("enrichment", "pending")
    with _lock:
        store.finish_query(query_id, result)
    threading.Thread(target=_enrich_later, args=(query_id,), daemon=True).start()


def _run_mock(query_id: str) -> None:
    seconds = mock_seconds()
    step = seconds / len(STAGES) if STAGES else 0
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
        example.setdefault("enrichment", "done")
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


def _run(query_id: str) -> None:
    try:
        if use_mock():
            _run_mock(query_id)
        else:
            _run_live(query_id)
    except Exception as exc:  # noqa: BLE001
        with _lock:
            store.update_progress(query_id, status="error", error=str(exc))
            store.finish_query(query_id, None, error=str(exc))


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "database": bool(store.database_url()), "mock": use_mock()}


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

    payload = build_insight(item, result.get("_documents") or [])
    payload["query_id"] = query_id
    payload["rank"] = rank
    store.save_insight(query_id, rank, payload)
    return payload


@app.get("/catalog/balance")
def catalog_balance() -> dict:
    return {"areas": store.training_balance()}
