"""Задача И2: готовые примеры для стенда — четыре темы жюри из Е3, обычный прогон с кэшем. Сеть.

Сеть идёт только за тем, чего нет в кэше, и за переводом. Перед каждым прогоном — остаток суточной квоты
OpenAlex (x-ratelimit-remaining); меньше 1000 — стоп. Результат — examples/<slug>.json: полный ответ
по контракту query_result плюс meta (запрос, область, дата, модель, генерация, время, доля кэша счётчиков).
examples/README.md — таблица примеров, без оценок качества.

Запуск: python -m scripts.build_examples [slug ...]
"""
from __future__ import annotations

import json
import os
import sys
from datetime import datetime
from pathlib import Path

import httpx
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
EXAMPLES = ROOT / "examples"
MIN_QUOTA = 1000
QUERIES = {"ai": ("технологии в ИИ", "Инфраструктура ИИ"),
           "fintech": ("перспективные решения в финтехе", "Финтех"),
           "cybersec": ("слабые сигналы в области кибербезопасности", "Защита ИИ"),
           "robots": ("роботы для промышленности", "Роботы")}


def quota() -> int:
    """Остаток суточной квоты OpenAlex из заголовка x-ratelimit-remaining (один дешёвый запрос)."""
    key = os.getenv("OPENALEX_API_KEY") or os.getenv("OPEN_ALEX") or ""
    response = httpx.get("https://api.openalex.org/works", params={"per_page": 1, "api_key": key}, timeout=30)
    return int(response.headers["x-ratelimit-remaining"])


def cache_share(result: dict) -> float:
    """Доля запросов счётчиков (все источники), взятых из кэша."""
    queues = result["timings"]["queues"]
    hits, total = sum(q["cache_hits"] for q in queues), sum(q["cache_hits"] + q["requests"] for q in queues)
    return round(hits / total, 3) if total else 0.0


def example(slug: str, result: dict, run_date: str) -> dict:
    """Ответ пайплайна и meta для стенда."""
    topic, area = QUERIES[slug]
    meta = {"query": topic, "area": area, "run_date": run_date, "model_version": result["model_version"],
            "candidates_version": result["candidates_version"], "seconds": result["timings"]["total"],
            "counters_cache_share": cache_share(result)}
    return {**result, "meta": meta}


def readme(examples: dict[str, dict]) -> str:
    """Одна таблица: пример, запрос, дата, кандидатов, выше порога, ТОП-15."""
    rows = [f"| `{slug}.json` | {e['meta']['query']} ({e['meta']['area']}) | {e['meta']['run_date']} | "
            f"{e['stats']['candidates_scored']} | {e['stats']['above_threshold']} | {len(e['top'])} |"
            for slug, e in examples.items()]
    return ("# Готовые примеры для стенда\n\nПолный ответ пайплайна по контракту `docs/contracts/query_result.example.json` "
            "плюс `meta`. Открываются без сети.\n\n| пример | запрос (область) | дата прогона | кандидатов оценено | "
            "выше порога | ТОП-15 |\n|---|---|---|---|---|---|\n" + "\n".join(rows) + "\n")


def main(slugs: list[str]) -> None:
    """Прогоны с кэшем по очереди; квота перед каждым; README — по всем файлам examples/."""
    load_dotenv(ROOT / ".env")
    from pipeline.__main__ import printer
    from pipeline.run_query import run_query
    EXAMPLES.mkdir(exist_ok=True)
    for slug in slugs:
        left = quota()
        print(f"{slug}: квота OpenAlex {left}", flush=True)
        if left < MIN_QUOTA:
            raise SystemExit(f"квота OpenAlex {left} < {MIN_QUOTA}: прогоны остановлены")
        topic, area = QUERIES[slug]
        run_date = datetime.now().isoformat(timespec="seconds")
        result = run_query(topic, area, use_cache=True, on_progress=printer(sys.stdout))
        (EXAMPLES / f"{slug}.json").write_text(json.dumps(example(slug, result, run_date), ensure_ascii=False, indent=2),
                                             encoding="utf-8")
        print(f"{slug}: {result['timings']['total']} с, доля кэша счётчиков {cache_share(result)}", flush=True)
    done = {p.stem: json.loads(p.read_text(encoding="utf-8")) for p in sorted(EXAMPLES.glob("*.json"))}
    (EXAMPLES / "README.md").write_text(readme({s: done[s] for s in QUERIES if s in done}), encoding="utf-8")
    print(f"квота OpenAlex после: {quota()}")


if __name__ == "__main__":
    main(sys.argv[1:] or list(QUERIES))
