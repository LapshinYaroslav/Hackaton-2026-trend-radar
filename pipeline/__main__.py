"""CLI: python -m pipeline "тема" [--area Финтех] [--out result.json] [--no-cache]."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
RUNS_DIR = ROOT / "data" / "interim" / "pipeline_runs"


def print_progress(stage: str, done: int, total: int) -> None:
    """Строка прогресса в stderr, перезаписывается на месте."""
    print(f"\r{stage:12s} {done}/{total}   ", end="" if done < total else "\n", file=sys.stderr, flush=True)


def main(argv: list[str] | None = None) -> int:
    """Разбирает аргументы, запускает run_query, пишет JSON."""
    parser = argparse.ArgumentParser(description="Тема -> ТОП-15 слабых сигналов")
    parser.add_argument("topic", help="тема в свободной форме")
    parser.add_argument("--area", default=None, help="одна из шести областей; без неё — общая нормировка")
    parser.add_argument("--out", default=None, help="куда записать JSON; по умолчанию data/interim/pipeline_runs/")
    parser.add_argument("--no-cache", action="store_true", help="не брать подзапросы и кандидатов из кэша")
    args = parser.parse_args(argv)
    load_dotenv(ROOT / ".env")
    from pipeline.run_query import run_query  # после .env: модули читают настройки при вызове

    result = run_query(args.topic, args.area, use_cache=not args.no_cache, progress=print_progress)
    out = Path(args.out) if args.out else RUNS_DIR / f"{result['query_id']}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"ТОП: {len(result['top'])}, исключено: {len(result['excluded'])}, "
          f"время: {result['timings']['total']} с, файл: {out}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
