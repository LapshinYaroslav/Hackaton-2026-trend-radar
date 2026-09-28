"""История запросов в таблице queries (схема Славы).

Локальный API вне Docker ходит на localhost: в .env хост `db` — имя сервиса compose.
Если базы нет, список берётся из памяти процесса.
"""

from __future__ import annotations

import json
import os
from datetime import date, datetime
from pathlib import Path
from typing import Any

def database_url() -> str | None:
    raw = (os.getenv("DATABASE_URL") or "").strip()
    if not raw:
        return None
    if not Path("/.dockerenv").exists():
        raw = raw.replace("@db:", "@127.0.0.1:").replace("@db/", "@127.0.0.1/")
    return raw


def _connect():
    import psycopg2

    return psycopg2.connect(database_url(), connect_timeout=2)


def _json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, default=_json_default)


def _json_default(value: Any) -> str:
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    return str(value)


def sync(job: dict[str, Any]) -> None:
    """Создаёт или обновляет строку. Ошибка соединения не должна ронять запрос."""
    if not database_url():
        return
    result = job.get("result") if job.get("status") == "done" else None
    stats = (result or {}).get("stats") or {}
    warnings = (result or {}).get("warnings") or []
    finished_at = datetime.now().astimezone() if job.get("status") in {"done", "error"} else None
    cutoff = (result or {}).get("cutoff_date")
    threshold = (result or {}).get("threshold")
    with _connect() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO queries (
                    query_id, topic, area, status, progress_stage, progress_done,
                    progress_total, model_version, threshold, cutoff_date,
                    stats, warnings, error, result, started_at, finished_at
                ) VALUES (
                    %s, %s, %s, %s, %s, %s,
                    %s, %s, %s, %s,
                    %s::jsonb, %s::jsonb, %s, %s::jsonb, now(), %s
                )
                ON CONFLICT (query_id) DO UPDATE SET
                    topic = EXCLUDED.topic,
                    area = EXCLUDED.area,
                    status = EXCLUDED.status,
                    progress_stage = EXCLUDED.progress_stage,
                    progress_done = EXCLUDED.progress_done,
                    progress_total = EXCLUDED.progress_total,
                    model_version = COALESCE(EXCLUDED.model_version, queries.model_version),
                    threshold = COALESCE(EXCLUDED.threshold, queries.threshold),
                    cutoff_date = COALESCE(EXCLUDED.cutoff_date, queries.cutoff_date),
                    stats = CASE
                        WHEN EXCLUDED.status = 'done' THEN EXCLUDED.stats
                        ELSE queries.stats
                    END,
                    warnings = CASE
                        WHEN EXCLUDED.status = 'done' THEN EXCLUDED.warnings
                        ELSE queries.warnings
                    END,
                    error = EXCLUDED.error,
                    result = COALESCE(EXCLUDED.result, queries.result),
                    finished_at = COALESCE(EXCLUDED.finished_at, queries.finished_at)
                """,
                (
                    job["query_id"],
                    job["topic"],
                    job.get("area"),
                    job.get("status") or "running",
                    job.get("progress_stage"),
                    int(job.get("progress_done") or 0),
                    int(job.get("progress_total") or 6),
                    (result or {}).get("model_version"),
                    threshold,
                    cutoff,
                    _json(stats),
                    _json(warnings),
                    job.get("error"),
                    _json(result) if result is not None else None,
                    finished_at,
                ),
            )
        conn.commit()


def load(query_id: str) -> dict[str, Any] | None:
    if not database_url():
        return None
    with _connect() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT query_id, topic, area, status, progress_stage, progress_done,
                       progress_total, error, result
                FROM queries WHERE query_id = %s
                """,
                (query_id,),
            )
            row = cur.fetchone()
    if row is None:
        return None
    result = row[8]
    if isinstance(result, str):
        result = json.loads(result)
    return {
        "query_id": row[0],
        "topic": row[1],
        "area": row[2],
        "status": row[3],
        "progress_stage": row[4],
        "progress_done": row[5] or 0,
        "progress_total": row[6] or 6,
        "error": row[7],
        "result": result if row[3] == "done" else None,
        "insights": {},
    }


def list_queries(memory: dict[str, dict], limit: int = 30) -> list[dict[str, Any]]:
    capped = max(1, min(int(limit), 100))
    if database_url():
        try:
            return _list_db(capped)
        except Exception:
            pass
    return _list_memory(memory, capped)


def _list_db(limit: int) -> list[dict[str, Any]]:
    with _connect() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT query_id, topic, area, status, created_at,
                       stats->>'candidates_found', stats->>'above_075'
                FROM queries
                ORDER BY created_at DESC
                LIMIT %s
                """,
                (limit,),
            )
            rows = cur.fetchall()
    return [_row_item(row) for row in rows]


def _row_item(row: tuple) -> dict[str, Any]:
    created = row[4]
    return {
        "query_id": row[0],
        "topic": row[1],
        "area": row[2],
        "status": row[3],
        "created_at": created.isoformat() if hasattr(created, "isoformat") else created,
        "candidates_found": _as_int(row[5]),
        "above_075": _as_int(row[6]),
    }


def _list_memory(memory: dict[str, dict], limit: int) -> list[dict[str, Any]]:
    jobs = sorted(memory.values(), key=lambda job: str(job.get("query_id") or ""), reverse=True)
    items = []
    for job in jobs[:limit]:
        stats = (job.get("result") or {}).get("stats") or {}
        items.append(
            {
                "query_id": job["query_id"],
                "topic": job["topic"],
                "area": job.get("area"),
                "status": job.get("status"),
                "created_at": None,
                "candidates_found": stats.get("candidates_found"),
                "above_075": stats.get("above_075"),
            }
        )
    return items


def _as_int(value: Any) -> int | None:
    if value is None or value == "":
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None
