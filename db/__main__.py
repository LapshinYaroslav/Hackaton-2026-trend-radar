"""python -m db — применить схему и залить справочники."""

from __future__ import annotations

import argparse
import sys

from db.apply import apply_schema, connect, database_url
from db.seed import seed


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Применить схему Trend Radar и залить данные")
    parser.add_argument("--dsn", default=None, help="postgresql://… (иначе DATABASE_URL)")
    parser.add_argument("--schema-only", action="store_true")
    parser.add_argument("--seed-only", action="store_true")
    args = parser.parse_args(argv)
    dsn = args.dsn or database_url()
    if not dsn:
        print("нет DATABASE_URL и --dsn", file=sys.stderr)
        return 2
    conn = connect(dsn)
    try:
        if not args.seed_only:
            apply_schema(conn)
            print("schema: ok")
        if not args.schema_only:
            counts = seed(conn)
            print("seed:", counts)
    finally:
        conn.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
