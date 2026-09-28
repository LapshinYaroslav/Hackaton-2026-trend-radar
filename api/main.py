"""
API запросов на новом контракте docs/contracts/query_result.example.json.

Пока оркестратора нет: POST создаёт запрос и через несколько секунд
кладёт в память пример из контракта (без PostgreSQL и без Docker).
Когда появится оркестратор Ярослава и save/load Славы — меняется только
место, откуда берётся готовый JSON.
"""

from __future__ import annotations

import copy
import json
import os
import sys
import threading
import time
import uuid
from pathlib import Path
from typing import Optional

from api import history

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

load_dotenv()

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
_jobs: dict[str, dict] = {}
_lock = threading.Lock()


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


def _ensure_area(query_id: str, *, use_llm: bool) -> None:
    """Пустую область заполняем по теме. Явное значение с клиента не трогаем."""
    from search.area import detect_area

    with _lock:
        job = _jobs[query_id]
        topic, area = job["topic"], job["area"]
        if not area:
            job["progress_stage"] = "Область"
    if area:
        return
    found = detect_area(topic, use_llm=use_llm)
    with _lock:
        _jobs[query_id]["area"] = found
    _persist(query_id)


def _live(query_id: str) -> None:
    from pipeline.query import run_query

    with _lock:
        job = _jobs[query_id]
        topic = job["topic"]
        area = job["area"]

    def on_progress(stage: str, done: int, total: int) -> None:
        with _lock:
            current = _jobs[query_id]
            current["progress_stage"] = stage
            current["progress_done"] = done
            current["progress_total"] = total
            current["status"] = "running"
        _persist(query_id)

    result = run_query(topic, area, query_id, on_progress=on_progress)
    with _lock:
        job = _jobs[query_id]
        job["result"] = result
        job["status"] = "done"
        job["progress_stage"] = STAGES[-1]
        job["progress_done"] = len(STAGES)
        job["progress_total"] = len(STAGES)
    _persist(query_id)


def _run(query_id: str) -> None:
    live = os.getenv("QUERY_MODE", "mock").strip().lower() == "live"
    try:
        _ensure_area(query_id, use_llm=live)
    except Exception as exc:  # noqa: BLE001
        with _lock:
            job = _jobs[query_id]
            job["status"] = "error"
            job["error"] = str(exc)
        _persist(query_id)
        return
    if live:
        try:
            _live(query_id)
        except Exception as exc:  # noqa: BLE001
            with _lock:
                job = _jobs[query_id]
                job["status"] = "error"
                job["error"] = str(exc)
            _persist(query_id)
        return
    seconds = mock_seconds()
    step = seconds / len(STAGES) if STAGES else 0
    try:
        for index, stage in enumerate(STAGES, start=1):
            with _lock:
                job = _jobs[query_id]
                job["progress_stage"] = stage
                job["progress_done"] = index
                job["progress_total"] = len(STAGES)
                job["status"] = "running"
            _persist(query_id)
            if step:
                time.sleep(step)
        example = load_example()
        with _lock:
            job = _jobs[query_id]
            example["query_id"] = query_id
            example["topic"] = job["topic"]
            example["area"] = job["area"]
            job["result"] = example
            job["status"] = "done"
            job["progress_stage"] = STAGES[-1]
            job["progress_done"] = len(STAGES)
            job["progress_total"] = len(STAGES)
        _persist(query_id)
    except Exception as exc:  # noqa: BLE001
        with _lock:
            job = _jobs[query_id]
            job["status"] = "error"
            job["error"] = str(exc)
        _persist(query_id)


def _persist(query_id: str) -> None:
    with _lock:
        job = copy.deepcopy(_jobs.get(query_id) or {})
    if not job:
        return
    try:
        history.sync(job)
    except Exception as exc:  # noqa: BLE001
        print(f"история запросов: {exc}", file=sys.stderr)


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


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
    with _lock:
        _jobs[query_id] = {
            "query_id": query_id,
            "topic": topic,
            "area": area,
            "status": "running",
            "progress_stage": STAGES[0],
            "progress_done": 0,
            "progress_total": len(STAGES),
            "result": None,
            "error": None,
        }
    _persist(query_id)
    threading.Thread(target=_run, args=(query_id,), daemon=True).start()
    return {"query_id": query_id}


@app.get("/queries")
def list_queries(limit: int = 30) -> dict:
    with _lock:
        memory = copy.deepcopy(_jobs)
    return {"items": history.list_queries(memory, limit=limit)}


@app.get("/queries/{query_id}")
def get_query(query_id: str) -> dict:
    with _lock:
        job = _jobs.get(query_id)
    if job is None:
        try:
            job = history.load(query_id)
        except Exception as exc:  # noqa: BLE001
            print(f"история запросов: {exc}", file=sys.stderr)
            job = None
        if job is None:
            raise HTTPException(status_code=404, detail="запрос не найден")
        with _lock:
            _jobs.setdefault(query_id, job)
            job = _jobs[query_id]
    with _lock:
        return _public(copy.deepcopy(job))


def _find_card(job: dict, rank: int) -> Optional[dict]:
    result = job.get("result") or {}
    for item in result.get("top") or []:
        if item.get("rank") == rank:
            return item
    return None


def _fill_insight(query_id: str, rank: int) -> None:
    from search.insights import generate_insight

    with _lock:
        job = _jobs.get(query_id)
        card = _find_card(job, rank) if job else None
        slot = (job or {}).get("insights", {}).get(rank)
    if card is None or slot is None:
        return
    try:
        content = generate_insight(card)
        with _lock:
            _jobs[query_id]["insights"][rank] = {"status": "ready", "content": content, "error": None}
    except Exception as exc:  # noqa: BLE001
        with _lock:
            _jobs[query_id]["insights"][rank] = {"status": "error", "content": None, "error": str(exc)}


@app.get("/queries/{query_id}/insights/{rank}")
def get_insight(query_id: str, rank: int) -> dict:
    """Готовый отчёт, либо pending, пока модель пишет текст по документам."""
    with _lock:
        job = _jobs.get(query_id)
    if job is None:
        try:
            job = history.load(query_id)
        except Exception as exc:  # noqa: BLE001
            print(f"история запросов: {exc}", file=sys.stderr)
            job = None
        if job is not None:
            with _lock:
                _jobs.setdefault(query_id, job)
                job = _jobs[query_id]
    if job is None:
        raise HTTPException(status_code=404, detail="запрос не найден")
    with _lock:
        job = _jobs.get(query_id) or job
        if job["status"] != "done":
            return {"query_id": query_id, "rank": rank, "status": "pending"}
        if _find_card(job, rank) is None:
            raise HTTPException(status_code=404, detail="в ТОП нет карточки с таким номером")
        insights = job.setdefault("insights", {})
        slot = insights.get(rank)
        if slot and slot["status"] in {"ready", "error"}:
            return {"query_id": query_id, "rank": rank, **slot}
        if slot is None or not slot.get("started"):
            insights[rank] = {"status": "pending", "content": None, "error": None, "started": True}
            threading.Thread(target=_fill_insight, args=(query_id, rank), daemon=True).start()
        return {"query_id": query_id, "rank": rank, "status": "pending"}
