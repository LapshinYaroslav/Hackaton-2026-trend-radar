"""Задача Р: замеры очереди Роспатента. Сеть — только к Роспатенту.

  limit  — отдаёт ли API total при limit=0 (один запрос по фразе из кэша П, сверка total);
  r2     — очередь при 1, 2 и 4 соединениях на 30 фразах вне кэша, по 10 на уровень;
  r3     — время сквозного прогона пайплайна с Роспатентом (только timings);
  r4     — n_pat для всех оценённых кандидатов прогонов задачи Н.

Прогоны задачи Н слепые: из них читаются только name_en и skipped_reason (оценённый
кандидат — в ТОПе или с причиной below_threshold / beyond_top). score, место и порядок
не читаются и не выводятся; фразы сортируются по алфавиту.

Запуск: python -m scripts.rospatent_queue limit|r2|r3|r4
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import httpx
import numpy as np
import pandas as pd
from dotenv import load_dotenv

from collector import rospatent as rp
from collector.api import tech_key
from model.config import ROSPATENT_DATASETS
from scripts.ru_patents_p0 import append_sheets

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "data" / "interim" / "pipeline_runs"
BLIND_RUNS = ["q20260923170352", "q20260923192323", "q20260923200541"]
SCORED_REASONS = {"below_threshold", "beyond_top"}
LEVELS = (1, 2, 4)
PER_LEVEL = 10


def token() -> str:
    """Ключ Роспатента из .env; в вывод не попадает."""
    load_dotenv(ROOT / ".env")
    return os.environ["ROSPATENT"]


def blind_phrases(run_id: str) -> list[str]:
    """Фразы оценённых кандидатов прогона Н, по алфавиту. Читаются только name_en и причина."""
    data = json.loads((RUNS / f"{run_id}.json").read_text(encoding="utf-8"))
    names = [item["name_en"] for item in data["top"]]
    names += [item["name_en"] for item in data["excluded"]
              if item.get("skipped_reason") in SCORED_REASONS]
    return sorted({tech_key(name) for name in names})


def cached(phrase: str) -> bool:
    return rp.cache_path(rp.request_key(phrase, ROSPATENT_DATASETS)).exists()


def limit_check() -> pd.DataFrame:
    """Один запрос с limit=0 по фразе из кэша П: есть ли total и совпадает ли он."""
    phrase = "neuromorphic chip"
    record = json.loads(rp.cache_path(rp.request_key(phrase, ROSPATENT_DATASETS))
                        .read_text(encoding="utf-8"))
    body = {**{k: v for k, v in rp.request_key(phrase, ROSPATENT_DATASETS).items()
               if k != "source"}, "limit": 0}
    started = time.monotonic()
    response = httpx.post(rp.URL, json=body, timeout=rp.TIMEOUT_S,
                          headers={"Authorization": f"Bearer {token()}"})
    payload = response.json() if response.status_code == 200 else {}
    return pd.DataFrame([{"фраза": phrase, "HTTP": response.status_code,
                          "total при limit=0": payload.get("total"),
                          "hits при limit=0": len(payload.get("hits") or []),
                          "total в кэше П (limit=1)": record["response"]["total"],
                          "с": round(time.monotonic() - started, 2)}])


def queue_row(title: str, results: dict[str, dict], summary: dict) -> dict:
    """Строка замера очереди: время, запросы, кэш, повторы, 429, сбои, время на фразу."""
    seconds = np.array([item["seconds"] for item in results.values() if not item["cached"]])
    return {"очередь": title, "параллельность": summary["parallel"],
            "фраз": summary["candidates"], "кэш": summary["cache_hits"],
            "HTTP-запросов": summary["requests"], "повторов": summary["retries"],
            "ответов 429": summary["n429"], "сбоев": summary["failures"],
            "всего, с": round(summary["finished"] - summary["started"], 2),
            "на фразу медиана, с": round(float(np.median(seconds)), 2) if len(seconds) else None,
            "на фразу максимум, с": round(float(np.max(seconds)), 2) if len(seconds) else None}


def r2_phrases() -> list[str]:
    """30 фраз прогонов Н вне кэша Роспатента: перемешаны (seed 42), по 10 на уровень."""
    pool = sorted({phrase for run_id in BLIND_RUNS for phrase in blind_phrases(run_id)
                   if not cached(phrase)})
    picked = list(np.random.default_rng(42).permutation(pool)[:PER_LEVEL * len(LEVELS)])
    if len(picked) < PER_LEVEL * len(LEVELS):
        raise ValueError(f"вне кэша только {len(pool)} фраз, нужно {PER_LEVEL * len(LEVELS)}")
    return picked


def r2() -> pd.DataFrame:
    """Очередь при 1, 2 и 4 соединениях, разные фразы на каждый уровень."""
    phrases, key, rows = r2_phrases(), token(), []
    for number, level in enumerate(LEVELS):
        part = phrases[number * PER_LEVEL:(number + 1) * PER_LEVEL]
        results, summary = rp.count_all(part, ROSPATENT_DATASETS, key, parallel=level)
        rows.append(queue_row(f"Р2, {level} соединени{'е' if level == 1 else 'я'}",
                              results, summary))
    return pd.DataFrame(rows)


def r4() -> pd.DataFrame:
    """n_pat для оценённых кандидатов прогонов Н; параллельность — ROSPATENT_PARALLEL."""
    key, rows = token(), []
    for run_id in BLIND_RUNS:
        results, summary = rp.count_all(blind_phrases(run_id), ROSPATENT_DATASETS, key,
                                        parallel=rp.ROSPATENT_PARALLEL)
        rows.append(queue_row(run_id, results, summary))
    unique = {phrase for run_id in BLIND_RUNS for phrase in blind_phrases(run_id)}
    rows.append({"очередь": "итого уникальных фраз", "фраз": len(unique),
                 "кэш": sum(cached(phrase) for phrase in unique)})
    return pd.DataFrame(rows)


def r3() -> pd.DataFrame:
    """Время сквозного прогона Р3. Из файла читаются только timings и stats.candidates_scored.

    Прогон — последний файл pipeline_runs с темой «технологии в ИИ»; ТОП не читается.
    Опоздание Роспатента — насколько его очередь закончилась позже самой медленной из
    остальных (0, если раньше).
    """
    runs = sorted(RUNS.glob("q*.json"), key=lambda path: path.stat().st_mtime)
    data = next(json.loads(path.read_text(encoding="utf-8")) for path in reversed(runs)
                if json.loads(path.read_text(encoding="utf-8")).get("topic") == "технологии в ИИ")
    timings, queues = data["timings"], data["timings"]["queues"]
    others = max(queue["finished_s"] for queue in queues if queue["source"] != "rospatent")
    rows = [{"очередь": queue["source"], **{key: queue[key] for key in (
                "started_s", "finished_s", "duration_s", "candidates", "requests", "cache_hits",
                "retries", "n429", "failures")}} for queue in queues]
    patent = next(queue for queue in queues if queue["source"] == "rospatent")
    rows.append({"очередь": "Роспатент позже самой медленной из остальных, с",
                 "duration_s": round(max(0.0, patent["finished_s"] - others), 2)})
    rows.append({"очередь": "этап counters, с", "duration_s": timings["counters"]})
    rows.append({"очередь": "весь прогон, с", "duration_s": timings["total"],
                 "candidates": data["stats"]["candidates_scored"]})
    return pd.DataFrame(rows)


def main() -> None:
    """Команда из argv: limit, r2, r3 или r4. Печать и дозапись листа Р."""
    command = sys.argv[1] if len(sys.argv) > 1 else ""
    sheets = {"limit": ("Р2 limit=0", limit_check), "r2": ("Р2 параллельность", r2),
              "r3": ("Р3 сквозной прогон", r3), "r4": ("Р4 сбор для Н", r4)}
    if command not in sheets:
        raise SystemExit("команда: limit | r2 | r3 | r4")
    title, action = sheets[command]
    table = action()
    print(table.to_string(index=False))
    append_sheets({title: table})


if __name__ == "__main__":
    main()
