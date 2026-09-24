"""Задача П0.2: сбор n_ru (OpenAlex) и n_pat (Роспатент) по 160 технологиям. Единственный шаг с сетью.

Фраза — та же, что у научного счётчика обучающей таблицы: terms из data/interim/technologies.csv
(у всех 160 строк один термин без контекста). Окно одно: 2020-09-01…2026-08-31.

  n_ru  — OpenAlex, тот же запрос, что у счётчика (search, type:article) плюс
          authorships.institutions.country_code:RU; group_by — дешёвый вызов, 1 кредит,
          meta.count тот же (проверено в П0.1).
  n_pat — Роспатент, поле total; фраза в кавычках, фильтр date_published, датасеты
          model.config.ROSPATENT_DATASETS.

Сырые ответы — в data/interim/collector/cache/ru_patents/<источник>/<sha ключа>.json;
ключ включает фразу, фильтры и набор датасетов. Повторный запуск в сеть не ходит.
Ключи API не пишутся ни в кэш, ни в вывод.

Запуск: python -m scripts.ru_patents_collect
"""
from __future__ import annotations

import hashlib
import json
import os
import time
from pathlib import Path

import httpx
import pandas as pd
from dotenv import load_dotenv

from collector import rospatent
from collector.query import build_query
from model.config import ROSPATENT_DATASETS

ROOT = Path(__file__).resolve().parents[1]
TECHNOLOGIES = ROOT / "data" / "interim" / "technologies.csv"
CACHE = ROOT / "data" / "interim" / "collector" / "cache" / "ru_patents"
OPENALEX_URL = "https://api.openalex.org/works"
ROSPATENT_URL = "https://searchplatform.rospatent.gov.ru/patsearch/v0.2/search"
OPENALEX_FILTER = ("from_publication_date:2020-09-01,to_publication_date:2026-08-31,"
                   "type:article,authorships.institutions.country_code:RU")
CREDIT_CAP = 2000
PAUSE = {"openalex": 0.12, "rospatent": 0.5}


def phrases() -> list[str]:
    """Уникальные фразы 160 технологий: у каждой ровно один термин."""
    table = pd.read_csv(TECHNOLOGIES, encoding="utf-8-sig")
    terms = table["terms"].map(json.loads)
    if (terms.map(len) != 1).any() or (table["context_terms"].map(json.loads).map(len) > 0).any():
        raise ValueError("ожидался один термин без контекста у каждой технологии")
    return sorted(set(terms.map(lambda items: items[0])))


def request_key(source: str, phrase: str) -> dict:
    """Всё, от чего зависит ответ: источник, запрос, фильтры, датасеты. Без ключей API."""
    if source == "openalex":
        return {"source": source, "search": build_query([phrase], [])[0],
                "filter": OPENALEX_FILTER, "group_by": "publication_year"}
    return rospatent.request_key(phrase, ROSPATENT_DATASETS)


def cache_path(key: dict) -> Path:
    """Файл кэша по sha256 от ключа запроса."""
    digest = hashlib.sha256(json.dumps(key, ensure_ascii=False, sort_keys=True)
                            .encode("utf-8")).hexdigest()
    return CACHE / key["source"] / f"{digest}.json"


def send(key: dict, secrets: dict[str, str],
         statuses: list[int] | None = None) -> httpx.Response:
    """Один запрос к источнику; до трёх попыток при 429 и 5xx. statuses — коды всех попыток."""
    for attempt in range(3):
        if key["source"] == "openalex":
            params = {"search": key["search"], "filter": key["filter"],
                      "group_by": key["group_by"], "per_page": 1,
                      "api_key": secrets["openalex"]}
            response = httpx.get(OPENALEX_URL, params=params, timeout=120)
        else:
            body = {"q": key["q"], "limit": 1, "filter": key["filter"],
                    "datasets": key["datasets"]}
            response = httpx.post(ROSPATENT_URL, json=body, timeout=120,
                                  headers={"Authorization": f"Bearer {secrets['rospatent']}"})
        if statuses is not None:
            statuses.append(response.status_code)
        if response.status_code != 429 and response.status_code < 500:
            return response
        time.sleep(5 * (attempt + 1))
    return response


def fetch(source: str, phrase: str, secrets: dict[str, str], spent: dict) -> dict:
    """Ответ из кэша или из сети; успешный ответ кладётся в кэш вместе с ключом."""
    key = request_key(source, phrase)
    path = cache_path(key)
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    started = time.monotonic()
    try:
        response = send(key, secrets, spent.setdefault("statuses", []))
    except httpx.HTTPError as exc:
        # Текст исключения httpx может содержать URL с api_key — наружу только тип.
        raise RuntimeError(f"{source} {phrase!r}: {type(exc).__name__}") from None
    spent[source] += 1
    spent.setdefault("seconds", []).append(time.monotonic() - started)
    credits = response.headers.get("x-ratelimit-credits-used")
    spent["openalex_credits"] += float(credits) if credits else 0.0
    if response.status_code != 200:
        raise RuntimeError(f"{source} {phrase!r}: HTTP {response.status_code}")
    record = {"key": key, "phrase": phrase, "credits_used": credits,
              "fetched_at": time.strftime("%Y-%m-%dT%H:%M:%S"), "response": response.json()}
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(record, ensure_ascii=False), encoding="utf-8")
    time.sleep(PAUSE[source])
    return record


def main() -> None:
    """Прогноз кредитов, затем сбор обоих источников с печатью расхода."""
    load_dotenv(ROOT / ".env")
    secrets = {"openalex": os.environ.get("OPENALEX_API_KEY") or os.environ["OPEN_ALEX"],
               "rospatent": os.environ["ROSPATENT"]}
    items = phrases()
    missing = sum(not cache_path(request_key("openalex", p)).exists() for p in items)
    print(f"фраз {len(items)}; OpenAlex без кэша {missing}, прогноз {missing} кредитов")
    if missing > CREDIT_CAP:
        raise SystemExit(f"прогноз {missing} выше потолка {CREDIT_CAP}, сбор не начат")
    spent = {"openalex": 0, "rospatent": 0, "openalex_credits": 0.0}
    for number, phrase in enumerate(items, start=1):
        for source in ("openalex", "rospatent"):
            fetch(source, phrase, secrets, spent)
        if number % 20 == 0:
            print(f"  {number}/{len(items)}  запросов OpenAlex {spent['openalex']}, "
                  f"Роспатент {spent['rospatent']}")
    print(f"готово: OpenAlex {spent['openalex']} запросов, {spent['openalex_credits']} кредитов; "
          f"Роспатент {spent['rospatent']} запросов")


if __name__ == "__main__":
    main()
