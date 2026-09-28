"""Подключение и применение db/schema.sql."""

from __future__ import annotations

import os
from pathlib import Path

SCHEMA_PATH = Path(__file__).resolve().parent / "schema.sql"


def database_url() -> str | None:
    value = (os.environ.get("DATABASE_URL") or "").strip()
    return value or None


def connect(dsn: str):
    import psycopg2

    return psycopg2.connect(dsn)


def apply_schema(conn) -> None:
    sql = SCHEMA_PATH.read_text(encoding="utf-8")
    with conn.cursor() as cur:
        cur.execute(sql)
    conn.commit()
