"""PostgreSQL cache of the collector: source_totals and Search #2 counters.

Документы и признаки (documents, features) сборщик больше не пишет и не читает: этот путь
вызывал только CLI сборщика, удалён в задаче Л. Таблицы остаются в db/schema.sql.
"""

from __future__ import annotations

from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Protocol

from collector.constants import ALLOWED_COUNTER_WINDOWS
from collector.models import Counter, SourceTotal

SCHEMA_PATH = Path(__file__).resolve().parents[1] / "db" / "schema.sql"


class DocumentCache(Protocol):
    def get_source_totals(self) -> list[SourceTotal] | None: ...

    def put_source_totals(self, totals: list[SourceTotal]) -> None: ...

    def get_counters(self, tech_key: str, terms_hash: str,
                     source: str) -> list[Counter] | None: ...

    def put_counters(
        self,
        tech_key: str,
        terms_hash: str,
        source: str,
        counters: list[Counter],
        *,
        candidate_id: str | None = None,
        query_id: str | None = None,
    ) -> None:
        """Дописывает строки источника, сливая по паре (окно, query_variant).

        Слияние, а не замена: у одного источника бывает несколько семантик запроса —
        OpenAlex опрашивается и фразой с type:article, и той же строкой без кавычек.
        Сборщик второй семантики не должен уносить числа первой.
        """
        ...


class MemoryCache:
    """In-memory stand-in used by tests and demos without PostgreSQL."""

    def __init__(self) -> None:
        self._totals: list[SourceTotal] | None = None
        self._counters: dict[tuple[str, str, str], dict[tuple[str, str | None], Counter]] = {}

    def get_source_totals(self) -> list[SourceTotal] | None:
        if self._totals is None:
            return None
        return list(self._totals)

    def put_source_totals(self, totals: list[SourceTotal]) -> None:
        self._totals = [_stamped(row) for row in totals]

    def get_counters(self, tech_key: str, terms_hash: str,
                     source: str) -> list[Counter] | None:
        rows = self._counters.get((tech_key, terms_hash, source))
        return list(rows.values()) if rows is not None else None

    def put_counters(
        self,
        tech_key: str,
        terms_hash: str,
        source: str,
        counters: list[Counter],
        *,
        candidate_id: str | None = None,
        query_id: str | None = None,
    ) -> None:
        del candidate_id, query_id
        rows = self._counters.setdefault((tech_key, terms_hash, source), {})
        rows.update({(row.window, row.query_variant): row for row in counters})

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

    def get_source_totals(self) -> list[SourceTotal] | None:
        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    "SELECT source, period, n_total, available, collected_at, "
                    "type_filter, totals_signature FROM source_totals"
                )
                rows = cur.fetchall()
        if not rows:
            return None
        return [
            SourceTotal(
                source=row[0],
                window=row[1],
                n_total=row[2],
                available=row[3],
                collected_at=row[4],
                type_filter=row[5],
                totals_signature=row[6],
            )
            for row in rows
        ]

    def put_source_totals(self, totals: list[SourceTotal]) -> None:
        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute("DELETE FROM source_totals")
                for row in totals:
                    stamped = _stamped(row)
                    cur.execute(
                        """
                        INSERT INTO source_totals
                            (source, period, n_total, available, collected_at,
                             type_filter, totals_signature)
                        VALUES (%s, %s, %s, %s, %s, %s, %s)
                        """,
                        (
                            stamped.source,
                            stamped.window,
                            stamped.n_total,
                            stamped.available,
                            stamped.collected_at,
                            stamped.type_filter,
                            stamped.totals_signature,
                        ),
                    )
            conn.commit()

    def get_counters(self, tech_key: str, terms_hash: str,
                     source: str) -> list[Counter] | None:
        """Счётчики по технологии, набору терминов и источнику."""
        with self._connect() as conn:
            with conn.cursor() as cur:
                cur.execute(
                    """
                    SELECT source, period, date_from, date_to, n, type_filter,
                           query_variant
                    FROM counters
                    WHERE tech_key = %s AND terms_hash = %s AND source = %s
                      AND period = ANY(%s)
                    ORDER BY period, query_variant
                    """,
                    (tech_key, terms_hash, source, sorted(ALLOWED_COUNTER_WINDOWS)),
                )
                rows = cur.fetchall()
        if not rows:
            return None
        return [
            Counter(
                tech_key=tech_key,
                source=row[0],
                window=row[1],
                date_from=row[2],
                date_to=row[3],
                n=row[4],
                type_filter=row[5],
                query_variant=row[6],
            )
            for row in rows
        ]

    def put_counters(
        self,
        tech_key: str,
        terms_hash: str,
        source: str,
        counters: list[Counter],
        *,
        candidate_id: str | None = None,
        query_id: str | None = None,
    ) -> None:
        """Вставка с обновлением по ключу строки. DELETE убран намеренно.

        Прежняя версия стирала все строки технологии и писала заново. При сборе
        по одной семантике запроса это уносило числа остальных: у OpenAlex их
        несколько, и вторая перезаписала бы первую.
        """
        del source
        with self._connect() as conn:
            with conn.cursor() as cur:
                for row in counters:
                    cur.execute(
                        """
                        INSERT INTO counters (
                            tech_key, terms_hash, source, period, query_variant,
                            date_from, date_to, n, type_filter, candidate_id, query_id
                        ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                        ON CONFLICT (tech_key, terms_hash, source, period, query_variant)
                        DO UPDATE SET n = EXCLUDED.n,
                                      date_from = EXCLUDED.date_from,
                                      date_to = EXCLUDED.date_to,
                                      type_filter = EXCLUDED.type_filter,
                                      collected_at = now()
                        """,
                        (
                            tech_key,
                            terms_hash,
                            row.source,
                            row.window,
                            row.query_variant or "",
                            row.date_from,
                            row.date_to,
                            row.n,
                            row.type_filter,
                            candidate_id,
                            query_id,
                        ),
                    )
            conn.commit()

def _stamped(row: SourceTotal) -> SourceTotal:
    """Проставляет время сбора, если его ещё нет. Свежесть строки решает, перепробовать ли источник."""
    if row.collected_at is not None:
        return row
    return replace(row, collected_at=datetime.now(timezone.utc))


def build_cache(database_url: str | None) -> MemoryCache | PostgresCache:
    if database_url:
        return PostgresCache(database_url)
    return MemoryCache()
