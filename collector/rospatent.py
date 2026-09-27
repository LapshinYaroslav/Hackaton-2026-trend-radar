"""Счётчик патентов Роспатента по фразе (задачи П и Р): только поле total, документы не выгружаются.

Запрос тот же, что собирал признак share_patent для обучения (снимок обучения —
evidence/rospatent_training_counts.json, scripts/build_rospatent_snapshot.py):
фраза в кавычках, date_published 2020-09-01…2026-08-31, набор датасетов передаётся параметром
(определение признака живёт в model.config.ROSPATENT_DATASETS) и входит в ключ кэша.

Сбой не равен нулю: если после повторов ответа нет, n_pat = None и failed = True. Ноль
пишется только при ответе total = 0. Ключ API не попадает ни в кэш, ни в исключения.
"""
from __future__ import annotations

import hashlib
import json
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Callable, Sequence

import httpx

ROOT = Path(__file__).resolve().parents[1]
CACHE_DIR = ROOT / "data" / "interim" / "collector" / "cache" / "ru_patents" / "rospatent"
URL = "https://searchplatform.rospatent.gov.ru/patsearch/v0.2/search"
WINDOW = {"date_published": {"range": {"gte": "20200901", "lte": "20260831"}}}

# Источник в этапе счётчиков пайплайна. Боевая модель s2a2-v1 использует share_patent,
# поэтому по умолчанию включён; выключение (CLI --no-rospatent) — только для отладки.
ROSPATENT_ENABLED = True
# Одновременных соединений в очереди Роспатента. Р2 (24.09.2026): при 1, 2 и 4 ни одного 429 и сбоя — берём 4.
ROSPATENT_PARALLEL = 4
# Размер выдачи: нужен только total. limit=0 отдаёт total без документов (проверено 24.09.2026).
ROSPATENT_LIMIT = 0
TIMEOUT_S = 15.0
# Паузы перед повторами на ConnectError, таймаут и 5xx: три повтора.
RETRY_DELAYS_S = (1, 2, 4)
# 429: ждать Retry-After, если он есть, иначе столько секунд; не больше трёх раз.
RETRY_429_DEFAULT_S = 5.0
RETRY_429_MAX = 3
Post = Callable[..., httpx.Response]


def request_key(phrase: str, datasets: Sequence[str]) -> dict:
    """Всё, от чего зависит total: фраза, окно, датасеты. Совпадает с ключом задачи П."""
    return {"source": "rospatent", "q": f'"{phrase}"', "filter": WINDOW,
            "datasets": list(datasets)}


def cache_path(key: dict, cache_dir: Path | None = None) -> Path:
    """Файл кэша по sha256 от ключа запроса (тот же формат, что в задаче П)."""
    digest = hashlib.sha256(json.dumps(key, ensure_ascii=False, sort_keys=True)
                            .encode("utf-8")).hexdigest()
    return (cache_dir or CACHE_DIR) / f"{digest}.json"


def retry_after(response: httpx.Response) -> float:
    """Пауза после 429: заголовок Retry-After в секундах, иначе значение по умолчанию."""
    try:
        return max(0.0, float(response.headers.get("retry-after", "")))
    except ValueError:
        return RETRY_429_DEFAULT_S


def send(body: dict, token: str, post: Post, sleep: Callable[[float], None]) -> dict:
    """POST с повторами. Ответ 200 или None, плюс число запросов, повторов и ответов 429.

    Сетевой сбой (ConnectError, таймаут, прочие транспортные) и 5xx — до трёх повторов
    с паузами RETRY_DELAYS_S; 429 — до трёх ожиданий Retry-After; прочие 4xx — без повторов.
    """
    stats = {"response": None, "requests": 0, "retries": 0, "n429": 0}
    faults = 0
    while True:
        stats["requests"] += 1
        try:
            response = post(URL, json=body, headers={"Authorization": f"Bearer {token}"},
                            timeout=TIMEOUT_S)
        except httpx.TransportError:
            response = None
        if response is not None and response.status_code == 200:
            stats["response"] = response
            return stats
        if response is not None and response.status_code == 429:
            stats["n429"] += 1
            if stats["n429"] > RETRY_429_MAX:
                return stats
            sleep(retry_after(response))
        elif response is None or response.status_code >= 500:
            if faults >= len(RETRY_DELAYS_S):
                return stats
            sleep(RETRY_DELAYS_S[faults])
            faults += 1
        else:
            return stats
        stats["retries"] += 1


def count_phrase(phrase: str, datasets: Sequence[str], token: str, *, post: Post,
                 sleep: Callable[[float], None] = time.sleep, cache_dir: Path | None = None,
                 use_cache: bool = True) -> dict:
    """n_pat по фразе: из кэша или из сети; успешный ответ кладётся в кэш."""
    key = request_key(phrase, datasets)
    path = cache_path(key, cache_dir)
    result = {"phrase": phrase, "n_pat": None, "cached": False, "requests": 0, "retries": 0,
              "n429": 0, "failed": False, "seconds": 0.0}
    if use_cache and path.exists():
        record = json.loads(path.read_text(encoding="utf-8"))
        return {**result, "n_pat": int(record["response"]["total"]), "cached": True}
    started = time.monotonic()
    body = {"q": key["q"], "limit": ROSPATENT_LIMIT, "filter": key["filter"],
            "datasets": key["datasets"]}
    stats = send(body, token, post, sleep)
    result.update({name: stats[name] for name in ("requests", "retries", "n429")},
                  seconds=round(time.monotonic() - started, 3))
    total = stats["response"].json().get("total") if stats["response"] is not None else None
    if total is None:
        return {**result, "failed": True}
    record = {"key": key, "phrase": phrase, "credits_used": None,
              "fetched_at": time.strftime("%Y-%m-%dT%H:%M:%S"), "response": stats["response"].json()}
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(record, ensure_ascii=False), encoding="utf-8")
    return {**result, "n_pat": int(total)}


def count_all(phrases: Sequence[str], datasets: Sequence[str], token: str | None, *,
              parallel: int = ROSPATENT_PARALLEL, post: Post | None = None,
              sleep: Callable[[float], None] = time.sleep, cache_dir: Path | None = None,
              use_cache: bool = True) -> tuple[dict[str, dict], dict]:
    """Очередь Роспатента: одна фраза — один запрос, parallel соединений. Результаты и сводка.

    Без токена ни одного запроса не делается: каждая фраза не из кэша — сбой.
    """
    unique = list(dict.fromkeys(phrases))
    started = time.time()
    with httpx.Client(timeout=TIMEOUT_S) as client:
        send_post = post or client.post

        def one(phrase: str) -> dict:
            if token is None and not (use_cache and cache_path(request_key(phrase, datasets),
                                                               cache_dir).exists()):
                return {"phrase": phrase, "n_pat": None, "cached": False, "requests": 0,
                        "retries": 0, "n429": 0, "failed": True, "seconds": 0.0}
            return count_phrase(phrase, datasets, token or "", post=send_post, sleep=sleep,
                                cache_dir=cache_dir, use_cache=use_cache)

        with ThreadPoolExecutor(max_workers=max(1, parallel)) as pool:
            results = {item["phrase"]: item for item in pool.map(one, unique)}
    finished = time.time()
    summary = {"source": "rospatent", "started": started, "finished": finished,
               "candidates": len(unique), "parallel": parallel,
               "requests": sum(item["requests"] for item in results.values()),
               "cache_hits": sum(item["cached"] for item in results.values()),
               "retries": sum(item["retries"] for item in results.values()),
               "n429": sum(item["n429"] for item in results.values()),
               "failures": sum(item["failed"] for item in results.values())}
    return results, summary
