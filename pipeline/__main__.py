"""CLI: python -m pipeline "тема" [--area Финтех] [--out result.json] [--no-cache] [--no-rospatent] [--quiet]
[--no-dedup].

Прогресс — одна строка в stdout, перезаписывается: «[ 37%] Сбор счётчиков 812/2160 · осталось ~9 мин».
Если stdout не терминал — новая строка не чаще раза в 10 с (и последняя, 100 %). --quiet — без прогресса.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path
from typing import Callable, TextIO

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
RUNS_DIR = ROOT / "data" / "interim" / "pipeline_runs"
PIPE_INTERVAL_S = 10.0


def progress_line(event: dict) -> str:
    """«[ 37%] Сбор счётчиков 812/2160 · осталось ~9 мин»; без оценки времени — без хвоста."""
    line = f"[{event['pct']:3d}%] {event['stage_ru']} {event['done']}/{event['total']}"
    if event["eta_s"] is not None:
        line += (f" · осталось ~{math.ceil(event['eta_s'] / 60)} мин" if event["eta_s"] >= 60
                 else " · осталось меньше минуты")
    return line


def printer(stream: TextIO, clock: Callable[[], float] = time.monotonic) -> Callable[[dict], None]:
    """Печать событий: в терминал — одна перезаписываемая строка; в файл или трубу — не чаще раза в 10 с."""
    tty, last, width = stream.isatty(), [None], [0]

    def show(event: dict) -> None:
        line, final = progress_line(event), event["stage"] == "done"
        if tty:
            stream.write("\r" + line.ljust(width[0]) + ("\n" if final else ""))
            width[0] = max(width[0], len(line))
        elif final or last[0] is None or clock() - last[0] >= PIPE_INTERVAL_S:
            stream.write(line + "\n")
            last[0] = clock()
        stream.flush()
    return show


def main(argv: list[str] | None = None) -> int:
    """Разбирает аргументы, запускает run_query, пишет JSON."""
    parser = argparse.ArgumentParser(description="Тема -> ТОП-15 слабых сигналов")
    parser.add_argument("topic", help="тема в свободной форме")
    parser.add_argument("--area", default=None, help="одна из шести областей; без неё — общая нормировка")
    parser.add_argument("--out", default=None, help="куда записать JSON; по умолчанию data/interim/pipeline_runs/")
    parser.add_argument("--no-cache", action="store_true", help="не брать подзапросы и кандидатов из кэша")
    parser.add_argument("--no-rospatent", action="store_true",
                        help="без Роспатента (отладка): share_patent у всех кандидатов недоступен")
    parser.add_argument("--quiet", action="store_true", help="без строки прогресса")
    parser.add_argument("--no-dedup", action="store_true", help="без склейки дублей (для сравнения)")
    args = parser.parse_args(argv)
    load_dotenv(ROOT / ".env")
    from pipeline.run_query import run_query  # после .env: модули читают настройки при вызове

    result = run_query(args.topic, args.area, use_cache=not args.no_cache,
                       rospatent=False if args.no_rospatent else None,
                       on_progress=None if args.quiet else printer(sys.stdout), dedup=not args.no_dedup)
    out = Path(args.out) if args.out else RUNS_DIR / f"{result['query_id']}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"ТОП: {len(result['top'])}, исключено: {len(result['excluded'])}, "
          f"время: {result['timings']['total']} с, файл: {out}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
