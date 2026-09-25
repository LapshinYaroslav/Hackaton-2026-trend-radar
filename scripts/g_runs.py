"""Задача Г2: четыре прогона пайплайна — две свежие темы × candidates_version v2 и v3. Сеть.

Слепота: ТОП-15 и пулы не печатаются. В печать — время, кредиты OpenAlex, число кандидатов,
состав документов шага 4 по источникам и сами подзапросы.

Кредиты OpenAlex — сумма заголовка x-ratelimit-credits-used ответов api.openalex.org (перехват
httpx.Client.send, код сборщика не меняется). Бюджет 3500: прогноз на прогон — максимум прошлых
прогонов (707); если потрачено + прогноз > 3500, следующий прогон не запускается.

Для слепого листа (Г3) рядом с прогоном пишется g_extra_<query_id>.json: документы, переданные на
шаг 4, и doc_ids кандидатов пула (обёртки над extract_terms и apply_cap, логика не меняется).

Запуск: python -m scripts.g_runs
"""
from __future__ import annotations

import json
import threading
import time
from pathlib import Path
from unittest.mock import patch

import httpx
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "data" / "interim" / "pipeline_runs"
REGISTRY = ROOT / "data" / "interim" / "g_runs.json"
THEMES = [("перспективные решения в энергетике", "Индустриальный ИИ"), ("медицинские технологии", "Роботы")]
VERSIONS = ("v3", "v2")  # v3 первым: у него холодный кэш счётчиков, время для правила 4 не занижено
BUDGET, FORECAST_PER_RUN = 3500, 707


class Credits:
    """Счётчик кредитов и запросов OpenAlex по заголовкам ответов; потокобезопасный."""

    def __init__(self) -> None:
        self.lock, self.credits, self.requests = threading.Lock(), 0.0, 0

    def hook(self, original):
        def send(client, request, *args, **kwargs):
            response = original(client, request, *args, **kwargs)
            if request.url.host == "api.openalex.org":
                used = response.headers.get("x-ratelimit-credits-used")
                with self.lock:
                    self.requests += 1
                    self.credits += float(used) if used else 0.0
            return response
        return send


def one_run(topic: str, area: str, version: str, meter: Credits) -> dict:
    """Один прогон run_query; запись JSON прогона и g_extra; числа для отчёта."""
    from pipeline import run_query as rq
    captured = {}
    extract, cap = rq.extract_terms, rq.apply_cap

    def extract_spy(documents, *args, **kwargs):
        captured["documents"] = list(documents)
        return extract(captured["documents"], *args, **kwargs)

    def cap_spy(candidates, documents, *args, **kwargs):
        kept, capped = cap(candidates, documents, *args, **kwargs)
        captured["pool"] = [{"name_en": item["name_en"], "doc_ids": item["doc_ids"]} for item in kept]
        return kept, capped

    before, started = (meter.credits, meter.requests), time.monotonic()
    with patch.object(rq, "extract_terms", extract_spy), patch.object(rq, "apply_cap", cap_spy):
        result = rq.run_query(topic, area, candidates_version=version)
    seconds = round(time.monotonic() - started, 1)
    (RUNS / f"{result['query_id']}.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    (RUNS / f"g_extra_{result['query_id']}.json").write_text(json.dumps(captured, ensure_ascii=False), encoding="utf-8")
    return {"тема": topic, "область": area, "вариант": version, "query_id": result["query_id"],
            "секунд": seconds, "кредитов OpenAlex": round(meter.credits - before[0], 1),
            "запросов OpenAlex": meter.requests - before[1]}


def main() -> None:
    """Четыре прогона с проверкой бюджета перед каждым; реестр — data/interim/g_runs.json."""
    load_dotenv(ROOT / ".env")
    meter = Credits()
    plan = [(topic, area, version) for topic, area in THEMES for version in VERSIONS]
    print(f"прогноз: {len(plan)} × {FORECAST_PER_RUN} = {len(plan) * FORECAST_PER_RUN} кредитов, бюджет {BUDGET}")
    if len(plan) * FORECAST_PER_RUN > BUDGET:
        raise SystemExit("прогноз выше бюджета — прогоны не запускаются")
    registry = []
    with patch.object(httpx.Client, "send", meter.hook(httpx.Client.send)):
        for topic, area, version in plan:
            if meter.credits + FORECAST_PER_RUN > BUDGET:
                print(f"стоп: потрачено {meter.credits}, следующий прогон может выйти за {BUDGET}")
                break
            registry.append(one_run(topic, area, version, meter))
            REGISTRY.write_text(json.dumps(registry, ensure_ascii=False, indent=2), encoding="utf-8")
            print(json.dumps(registry[-1], ensure_ascii=False), flush=True)
    print(f"итого кредитов OpenAlex: {round(meter.credits, 1)}, запросов: {meter.requests}")


if __name__ == "__main__":
    main()
