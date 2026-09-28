"""Запись документов поиска №1, признаков и оценок в Postgres."""

from __future__ import annotations

import json
import logging
import os
from typing import Any, Iterable, Mapping, Sequence

from collector.api import tech_key as make_tech_key

logger = logging.getLogger(__name__)

SEARCH1_KEY = "search1"


def _database_url() -> str | None:
    value = (os.environ.get("DATABASE_URL") or "").strip()
    return value or None


def _json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, default=str)


def _connect():
    import psycopg2

    return psycopg2.connect(_database_url())


def persist_search_documents(
    query_id: str,
    documents: Sequence[Mapping],
    candidates: Iterable[Mapping] | None = None,
) -> int:
    """Один URL — одна строка documents; кандидаты связываются через tech_documents."""
    if not _database_url() or not documents:
        return 0
    by_url: dict[str, Mapping] = {}
    for doc in documents:
        url = (doc.get("url") or "").strip()
        if url:
            by_url[url] = doc
    links: list[tuple[str, str]] = []
    for item in candidates or []:
        key = make_tech_key(item.get("name_en") or item.get("name") or "")
        if not key:
            continue
        for source in item.get("sources") or []:
            url = (source.get("url") or "").strip()
            if url:
                links.append((key, url))
                by_url.setdefault(url, source)
        for url in item.get("doc_urls") or []:
            if url:
                links.append((key, str(url)))
    written = 0
    with _connect() as conn:
        with conn.cursor() as cur:
            ids: dict[str, int] = {}
            for url, doc in by_url.items():
                ids[url] = _upsert_document(cur, query_id, doc)
                written += 1
            for key, url in links:
                doc_id = ids.get(url)
                if doc_id is None:
                    continue
                cur.execute(
                    """
                    INSERT INTO tech_documents (tech_key, document_id, role)
                    VALUES (%s, %s, 'search1')
                    ON CONFLICT (tech_key, document_id) DO NOTHING
                    """,
                    (key, doc_id),
                )
        conn.commit()
    return written


def persist_features_and_scores(
    query_id: str,
    items: Iterable[Mapping],
    *,
    model_version: str | None = None,
    threshold: float | None = None,
) -> None:
    """Таблица features и кэш scores по ключу технологии."""
    if not _database_url():
        return
    rows = [item for item in items if item.get("name_en") or item.get("tech_key")]
    if not rows:
        return
    with _connect() as conn:
        with conn.cursor() as cur:
            for item in rows:
                key = item.get("tech_key") or make_tech_key(item.get("name_en") or "")
                features = item.get("features") or {}
                if key and features:
                    cur.execute(
                        """
                        INSERT INTO features (tech_key, candidate_id, query_id, payload)
                        VALUES (%s, %s, %s, %s::jsonb)
                        ON CONFLICT (tech_key) DO UPDATE SET
                            candidate_id = EXCLUDED.candidate_id,
                            query_id = EXCLUDED.query_id,
                            payload = EXCLUDED.payload,
                            computed_at = now()
                        """,
                        (key, item.get("candidate_id"), query_id, _json(features)),
                    )
                version = item.get("model_version") or model_version
                if key and version and item.get("score") is not None:
                    cur.execute(
                        """
                        INSERT INTO scores (
                            tech_key, model_version, query_id, candidate_id, score,
                            is_signal, threshold, contributions, top_features
                        ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s::jsonb, %s::jsonb)
                        ON CONFLICT (tech_key, model_version) DO UPDATE SET
                            query_id = EXCLUDED.query_id,
                            score = EXCLUDED.score,
                            is_signal = EXCLUDED.is_signal,
                            threshold = EXCLUDED.threshold,
                            contributions = EXCLUDED.contributions,
                            top_features = EXCLUDED.top_features,
                            computed_at = now()
                        """,
                        (
                            key,
                            version,
                            query_id,
                            item.get("candidate_id"),
                            item.get("score"),
                            item.get("is_signal"),
                            item.get("threshold", threshold),
                            _json(item.get("contributions") or {}),
                            _json(item.get("top_features") or []),
                        ),
                    )
        conn.commit()


def persist_insight(query_id: str, rank: int, payload: Mapping) -> None:
    if not _database_url():
        return
    with _connect() as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO insights (
                    query_id, rank, status, description_ru, advantages_ru,
                    cases_ru, payload, generated_at
                ) VALUES (%s, %s, %s, %s, %s::jsonb, %s::jsonb, %s::jsonb, now())
                ON CONFLICT (query_id, rank) DO UPDATE SET
                    status = EXCLUDED.status,
                    description_ru = EXCLUDED.description_ru,
                    advantages_ru = EXCLUDED.advantages_ru,
                    cases_ru = EXCLUDED.cases_ru,
                    payload = EXCLUDED.payload,
                    generated_at = now()
                """,
                (
                    query_id,
                    rank,
                    payload.get("status") or "done",
                    payload.get("description_ru"),
                    _json(payload.get("advantages_ru") or []),
                    _json(payload.get("cases_ru") or []),
                    _json(dict(payload)),
                ),
            )
        conn.commit()


def _upsert_document(cur, query_id: str, doc: Mapping) -> int:
    url = (doc.get("url") or "").strip()
    published = doc.get("published_at") or "2020-09-01"
    body = doc.get("text") or doc.get("body") or ""
    orgs = list(doc.get("organizations") or [])
    cur.execute("SELECT id FROM documents WHERE url = %s", (url,))
    row = cur.fetchone()
    if row:
        cur.execute(
            """
            UPDATE documents SET
                query_id = COALESCE(%s, query_id),
                title = %s,
                body = CASE WHEN %s <> '' THEN %s ELSE body END,
                source_type = %s,
                language = %s,
                trust_level = %s,
                organizations = %s,
                collected_at = now()
            WHERE id = %s
            """,
            (
                query_id,
                doc.get("title") or url,
                body,
                body,
                doc.get("source_type") or "news",
                doc.get("language") or "en",
                doc.get("trust_level") or "medium",
                orgs,
                row[0],
            ),
        )
        return int(row[0])
    cur.execute(
        """
        INSERT INTO documents (
            tech_key, query_id, published_at, source, source_type,
            title, url, language, trust_level, organizations, body
        ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        RETURNING id
        """,
        (
            SEARCH1_KEY,
            query_id,
            published,
            doc.get("source") or "openalex",
            doc.get("source_type") or "news",
            doc.get("title") or url,
            url,
            doc.get("language") or "en",
            doc.get("trust_level") or "medium",
            orgs,
            body,
        ),
    )
    return int(cur.fetchone()[0])
