"""PostgreSQL cache: documents, source_totals, features.

The collector writes documents and source_totals. The features table is filled by
Yaroslav's compute_features; the collector only reads it to skip Search #2.
"""

from __future__ import annotations

from pathlib import Path
from typing import Protocol

from collector.models import Document, SourceTotal

SCHEMA_PATH = Path(__file__).resolve().parents[1] / "db" / "schema.sql"


class DocumentCache(Protocol):
    def get_documents(self, tech_key: str) -> list[Document] | None: ...

    def put_documents(
        self,
        tech_key: str,
        documents: list[Document],
        *,
        candidate_id: str | None = None,
        query_id: str | None = None,
    ) -> None: ...

    def get_source_totals(self) -> list[SourceTotal] | None: ...

    def put_source_totals(self, totals: list[SourceTotal]) -> None: ...

    def has_features(self, tech_key: str) -> bool: ...


class MemoryCache:
    """In-memory stand-in used by tests and demos without PostgreSQL."""

    def __init__(self) -> None:
        self._documents: dict[str, list[Document]] = {}
        self._totals: list[SourceTotal] | None = None
        self._features: set[str] = set()

    def get_documents(self, tech_key: str) -> list[Document] | None:
        docs = self._documents.get(tech_key)
        return list(docs) if docs is not None else None

    def put_documents(
        self,
        tech_key: str,
        documents: list[Document],
        *,
        candidate_id: str | None = None,
        query_id: str | None = None,
    ) -> None:
        del candidate_id, query_id
        self._documents[tech_key] = list(documents)

    def get_source_totals(self) -> list[SourceTotal] | None:
        if self._totals is None:
            return None
        return list(self._totals)

    def put_source_totals(self, totals: list[SourceTotal]) -> None:
        self._totals = list(totals)

    def has_features(self, tech_key: str) -> bool:
        return tech_key in self._features

    def put_features(self, tech_key: str) -> None:
        """Test helper: simulate a row written by compute_features."""
        self._features.add(tech_key)


class PostgresCache:
    def __init__(self, dsn: str, *, ensure_schema: bool = True) -> None:
        self._dsn = dsn
        if ensure_schema:
            self._apply_schema()

    def _connect(self):
        import psycopg2

        return psycopg2.connect(self._dsn)

    def _apply_schema(self) -> None:
        sql = SCHEMA_PATH.read_text(encoding="utf-8")
        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute(sql)
            conn.commit()

    def get_documents(self, tech_key: str) -> list[Document] | None:
        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT published_at, source, source_type, title, url, language,
                           trust_level, organizations, body
                    FROM documents
                    WHERE tech_key = %s
                    ORDER BY published_at, url
                    """,
                    (tech_key,),
                )
                rows = cur.fetchall()
        if not rows:
            return None
        return [
            Document(
                published_at=row[0],
                source=row[1],
                source_type=row[2],
                title=row[3],
                url=row[4],
                language=row[5],
                trust_level=row[6],
                organizations=list(row[7] or []),
                text=row[8] or "",
            )
            for row in rows
        ]

    def put_documents(
        self,
        tech_key: str,
        documents: list[Document],
        *,
        candidate_id: str | None = None,
        query_id: str | None = None,
    ) -> None:
        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute("DELETE FROM documents WHERE tech_key = %s", (tech_key,))
                for doc in documents:
                    cur.execute(
                        """
                        INSERT INTO documents (
                            tech_key, candidate_id, query_id, published_at, source, source_type,
                            title, url, language, trust_level, organizations, body
                        ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                        """,
                        (
                            tech_key,
                            candidate_id,
                            query_id,
                            doc.published_at,
                            doc.source,
                            doc.source_type,
                            doc.title,
                            doc.url,
                            doc.language,
                            doc.trust_level,
                            _as_text_array(doc.organizations),
                            doc.text,
                        ),
                    )
            conn.commit()

    def get_source_totals(self) -> list[SourceTotal] | None:
        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT source, window, n_total, available FROM source_totals")
                rows = cur.fetchall()
        if not rows:
            return None
        return [
            SourceTotal(
                source=row[0],
                window=row[1],
                n_total=row[2],
                available=row[3],
            )
            for row in rows
        ]

    def put_source_totals(self, totals: list[SourceTotal]) -> None:
        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute("DELETE FROM source_totals")
                for row in totals:
                    cur.execute(
                        """
                        INSERT INTO source_totals (source, window, n_total, available)
                        VALUES (%s, %s, %s, %s)
                        """,
                        (row.source, row.window, row.n_total, row.available),
                    )
            conn.commit()

    def has_features(self, tech_key: str) -> bool:
        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT 1 FROM features WHERE tech_key = %s", (tech_key,))
                row = cur.fetchone()
        return row is not None


def _as_text_array(values: list[str]):
    from psycopg2.extensions import AsIs

    if not values:
        return AsIs("'{}'::text[]")
    quoted = ",".join("'" + item.replace("'", "''") + "'" for item in values)
    return AsIs(f"ARRAY[{quoted}]::text[]")


def build_cache(database_url: str | None) -> MemoryCache | PostgresCache:
    if database_url:
        return PostgresCache(database_url)
    return MemoryCache()
