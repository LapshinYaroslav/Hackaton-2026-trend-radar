"""История запросов: Postgres, если есть DATABASE_URL, иначе память процесса."""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from typing import Any

_memory: dict[str, dict[str, Any]] = {}


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, default=str)


def database_url() -> str | None:
    value = (os.environ.get("DATABASE_URL") or "").strip()
    return value or None


def _connect():
    import psycopg2

    return psycopg2.connect(database_url())


def ensure_ready() -> dict[str, Any] | None:
    """Схема + посев при первом старте с Postgres. Без DATABASE_URL — None."""
    dsn = database_url()
    if not dsn:
        return None
    from db.apply import apply_schema, connect
    from db.seed import seed

    conn = connect(dsn)
    try:
        apply_schema(conn)
        counts = seed(conn)
    finally:
        conn.close()
    return counts


def create_query(job: dict[str, Any]) -> None:
    job.setdefault("created_at", _now())
    _memory[job["query_id"]] = job
    if not database_url():
        return
    with _connect() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO queries (
                    query_id, topic, area, status, progress_stage,
                    progress_done, progress_total, created_at, started_at
                ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (query_id) DO UPDATE SET
                    status = EXCLUDED.status,
                    progress_stage = EXCLUDED.progress_stage
                """,
                (
                    job["query_id"],
                    job["topic"],
                    job.get("area"),
                    job["status"],
                    job.get("progress_stage"),
                    job.get("progress_done") or 0,
                    job.get("progress_total") or 6,
                    job.get("created_at") or _now(),
                    job.get("started_at") or _now(),
                ),
            )
        conn.commit()


def update_progress(query_id: str, **fields: Any) -> None:
    job = _memory.get(query_id)
    if job is not None:
        job.update(fields)
    if not database_url():
        return
    allowed = {
        "status",
        "progress_stage",
        "progress_done",
        "progress_total",
        "error",
        "model_version",
        "threshold",
        "cutoff_date",
    }
    sets = []
    values = []
    for key, value in fields.items():
        if key not in allowed:
            continue
        sets.append(f"{key} = %s")
        values.append(value)
    if not sets:
        return
    values.append(query_id)
    with _connect() as conn:
        with conn.cursor() as cur:
            cur.execute(f"UPDATE queries SET {', '.join(sets)} WHERE query_id = %s", values)
        conn.commit()


def finish_query(query_id: str, result: dict[str, Any] | None, *, error: str | None = None) -> None:
    job = _memory.get(query_id)
    if job is not None:
        job["result"] = result
        job["error"] = error
        job["status"] = "error" if error else "done"
        job["finished_at"] = _now()
    if not database_url():
        return
    status = "error" if error else "done"
    with _connect() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                UPDATE queries SET
                    status = %s,
                    error = %s,
                    result = %s::jsonb,
                    stats = %s::jsonb,
                    timings = %s::jsonb,
                    warnings = %s::jsonb,
                    model_version = %s,
                    threshold = %s,
                    cutoff_date = %s,
                    finished_at = now()
                WHERE query_id = %s
                """,
                (
                    status,
                    error,
                    _json(result) if result is not None else None,
                    _json((result or {}).get("stats") or {}),
                    _json((result or {}).get("timings") or {}),
                    _json((result or {}).get("warnings") or []),
                    (result or {}).get("model_version"),
                    (result or {}).get("threshold"),
                    (result or {}).get("cutoff_date"),
                    query_id,
                ),
            )
            if result and not error:
                _replace_children(cur, query_id, result)
        conn.commit()


def _replace_children(cur, query_id: str, result: dict[str, Any]) -> None:
    cur.execute("DELETE FROM subqueries WHERE query_id = %s", (query_id,))
    cur.execute("DELETE FROM insights WHERE query_id = %s", (query_id,))
    cur.execute("DELETE FROM candidates WHERE query_id = %s", (query_id,))
    for position, item in enumerate(result.get("subqueries") or []):
        cur.execute(
            """
            INSERT INTO subqueries (subquery_id, query_id, language, text, position)
            VALUES (%s, %s, %s, %s, %s)
            ON CONFLICT (subquery_id) DO UPDATE SET
                text = EXCLUDED.text, language = EXCLUDED.language
            """,
            (
                item.get("subquery_id") or f"{query_id}-{position}",
                query_id,
                item.get("language") or "en",
                item.get("text") or "",
                position,
            ),
        )
    for bucket in ("top", "excluded"):
        for item in result.get(bucket) or []:
            _insert_candidate(cur, query_id, bucket, item)


def _insert_candidate(cur, query_id: str, bucket: str, item: dict[str, Any]) -> None:
    cur.execute(
        """
        INSERT INTO candidates (
            query_id, candidate_id, name_ru, name_en, tech_key, bucket, rank, score,
            is_signal, skipped_reason, reason_ru, explanation_ru, contributions,
            counters, features, n_pat, share_patent, rospatent_failed, note_ru, model_version
        ) VALUES (
            %s, %s, %s, %s, %s, %s, %s, %s,
            %s, %s, %s, %s::jsonb, %s::jsonb,
            %s::jsonb, %s::jsonb, %s, %s, %s, %s, %s
        )
        RETURNING id
        """,
        (
            query_id,
            item.get("candidate_id"),
            item.get("name_ru"),
            item.get("name_en") or item.get("name") or "",
            item.get("tech_key") or item.get("name_en"),
            bucket,
            item.get("rank"),
            item.get("score"),
            item.get("is_signal"),
            item.get("skipped_reason"),
            item.get("reason_ru"),
            _json(item.get("explanation_ru") or []),
            _json(item.get("contributions") or {}),
            _json(item.get("counters") or {}),
            _json(item.get("features") or {}),
            item.get("n_pat"),
            item.get("share_patent"),
            item.get("rospatent_failed"),
            item.get("note_ru"),
            item.get("model_version"),
        ),
    )
    candidate_pk = cur.fetchone()[0]
    for source in item.get("sources") or []:
        cur.execute(
            """
            INSERT INTO candidate_sources (
                candidate_pk, title, url, published_at, source, source_type, language, trust_level
            ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
            """,
            (
                candidate_pk,
                source.get("title"),
                source.get("url"),
                source.get("published_at"),
                source.get("source"),
                source.get("source_type"),
                source.get("language"),
                source.get("trust_level"),
            ),
        )
    if bucket == "top" and item.get("rank") is not None:
        cur.execute(
            """
            INSERT INTO insights (query_id, rank, candidate_pk, status)
            VALUES (%s, %s, %s, 'pending')
            ON CONFLICT (query_id, rank) DO NOTHING
            """,
            (query_id, item["rank"], candidate_pk),
        )
    tech_key = item.get("tech_key") or item.get("name_en")
    model_version = item.get("model_version")
    if tech_key and model_version and item.get("score") is not None:
        cur.execute(
            """
            INSERT INTO scores (
                tech_key, model_version, query_id, candidate_id, score, is_signal,
                threshold, contributions, top_features
            ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s::jsonb, %s::jsonb)
            ON CONFLICT (tech_key, model_version) DO UPDATE SET
                query_id = EXCLUDED.query_id,
                score = EXCLUDED.score,
                is_signal = EXCLUDED.is_signal,
                contributions = EXCLUDED.contributions,
                top_features = EXCLUDED.top_features,
                computed_at = now()
            """,
            (
                tech_key,
                model_version,
                query_id,
                item.get("candidate_id"),
                item.get("score"),
                item.get("is_signal"),
                item.get("threshold"),
                _json(item.get("contributions") or {}),
                _json(item.get("top_features") or item.get("explanation_ru") or []),
            ),
        )


def get_query(query_id: str) -> dict[str, Any] | None:
    job = _memory.get(query_id)
    if job is not None:
        return job
    if not database_url():
        return None
    with _connect() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT query_id, topic, area, status, progress_stage, progress_done,
                       progress_total, error, result, created_at
                FROM queries WHERE query_id = %s
                """,
                (query_id,),
            )
            row = cur.fetchone()
    if row is None:
        return None
    result = row[8] or {}
    job = {
        "query_id": row[0],
        "topic": row[1],
        "area": row[2],
        "status": row[3],
        "progress_stage": row[4],
        "progress_done": row[5],
        "progress_total": row[6],
        "error": row[7],
        "result": result if row[3] == "done" else None,
        "created_at": row[9],
    }
    _memory[query_id] = job
    return job


def list_queries(limit: int = 30) -> list[dict[str, Any]]:
    if database_url():
        with _connect() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT query_id, topic, area, status, created_at, finished_at,
                           stats->>'candidates_found', stats->>'above_075'
                    FROM queries
                    ORDER BY created_at DESC
                    LIMIT %s
                    """,
                    (limit,),
                )
                rows = cur.fetchall()
        return [
            {
                "query_id": row[0],
                "topic": row[1],
                "area": row[2],
                "status": row[3],
                "created_at": row[4].isoformat() if row[4] else None,
                "finished_at": row[5].isoformat() if row[5] else None,
                "candidates_found": _as_int(row[6]),
                "above_075": _as_int(row[7]),
            }
            for row in rows
        ]
    items = sorted(_memory.values(), key=lambda job: str(job.get("created_at") or ""), reverse=True)
    out = []
    for job in items[:limit]:
        stats = (job.get("result") or {}).get("stats") or {}
        created = job.get("created_at")
        out.append(
            {
                "query_id": job["query_id"],
                "topic": job["topic"],
                "area": job.get("area"),
                "status": job["status"],
                "created_at": created.isoformat() if hasattr(created, "isoformat") else created,
                "finished_at": None,
                "candidates_found": stats.get("candidates_found"),
                "above_075": stats.get("above_075"),
            }
        )
    return out


def list_technologies(label: int | None = None) -> list[dict[str, Any]]:
    if not database_url():
        return []
    sql = """
        SELECT tech_id, name_ru, name_en, area, label, negative_type, expert_score, tech_key
        FROM technologies
    """
    params: tuple = ()
    if label is not None:
        sql += " WHERE label = %s"
        params = (label,)
    sql += " ORDER BY area, tech_id"
    with _connect() as conn:
        with conn.cursor() as cur:
            cur.execute(sql, params)
            rows = cur.fetchall()
    return [
        {
            "tech_id": row[0],
            "name_ru": row[1],
            "name_en": row[2],
            "area": row[3],
            "label": row[4],
            "negative_type": row[5],
            "expert_score": row[6],
            "tech_key": row[7],
        }
        for row in rows
    ]


def training_balance() -> list[dict[str, Any]]:
    if not database_url():
        return []
    with _connect() as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT area, signals, negatives, total FROM v_training_balance ORDER BY area")
            rows = cur.fetchall()
    return [
        {"area": row[0], "signals": row[1], "negatives": row[2], "total": row[3]}
        for row in rows
    ]


def insight_status(query_id: str, rank: int) -> dict[str, Any]:
    if database_url():
        with _connect() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT status, description_ru FROM insights WHERE query_id = %s AND rank = %s",
                    (query_id, rank),
                )
                row = cur.fetchone()
        if row:
            return {"query_id": query_id, "rank": rank, "status": row[0], "description_ru": row[1]}
    return {"query_id": query_id, "rank": rank, "status": "pending"}


def _as_int(value: Any) -> int | None:
    if value is None or value == "":
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None
