"""Живой запрос: подзапросы, документы, кандидаты, счётчики, оценка.

Ответ — тот же JSON, что docs/contracts/query_result.example.json.
Если файла модели нет, веса берутся из evidence/model_release_20260922.md,
а центр и масштаб считаются по кандидатам этого запроса.
"""

from __future__ import annotations

import math
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import replace
from pathlib import Path
from typing import Callable

import pandas as pd
from dotenv import load_dotenv

from collector.api import MemoryCache, build_collector
from collector.models import Candidate
from collector.query import clean_terms
from collector.settings import Settings
from model.area_scaler import centre_scale
from model.candidate import candidate_features
from model.config import CUTOFF_DATE, FEATURES, MODEL_VERSION
from model.predict import predict
from model.ranking import top_features
from model.train import MODEL_PATH
from search.extract_candidates import extract_candidates
from search.subqueries import generate_subqueries

Progress = Callable[[str, int, int], None]

STAGES = (
    "Подзапросы",
    "Поиск документов",
    "Извлечение кандидатов",
    "Проверка",
    "Сбор статистики",
    "Оценка",
)

# Веса модели, обученной на всех 160 строках. evidence/model_release_20260922.md, раздел 4.
PUBLISHED_COEFFICIENTS = {
    "share_news_wordmatch": 1.4485,
    "recency": 1.096,
    "age_first_arxiv": -0.5174,
    "share_prev6": -0.2461,
    "volume": -0.0236,
    "growth_research": -0.0218,
}
PUBLISHED_INTERCEPT = 0.4084
PUBLISHED_THRESHOLD = 0.325
TOP_LIMIT = 15


def _env_int(name: str, default: int) -> int:
    raw = (os.getenv(name) or "").strip()
    try:
        return max(1, int(raw)) if raw else default
    except ValueError:
        return default


def _progress(callback: Progress | None, stage: str, done: int) -> None:
    if callback:
        callback(stage, done, len(STAGES))


def _collector():
    """Локально хост db из Docker не резолвится — тогда кэш в памяти процесса."""
    load_dotenv()
    settings = Settings.from_env()
    url = settings.database_url or ""
    in_docker = Path("/.dockerenv").exists()
    remote_db = "@db:" in url or url.endswith("@db")
    if url and (in_docker or not remote_db):
        return build_collector(settings=settings)
    settings = replace(settings, database_url=None)
    return build_collector(settings=settings, cache=MemoryCache())


def _terms(candidate: dict) -> tuple[list[str], list[str]]:
    terms, _ = clean_terms(candidate.get("terms") or [candidate.get("name_en") or ""], "terms")
    if not terms and candidate.get("name_en"):
        terms, _ = clean_terms([candidate["name_en"]], "terms")
    context, _ = clean_terms(candidate.get("context_terms") or [], "context_terms")
    if len(context) < 2:
        context = []
    return terms, context


def _fetch_one(collector, candidate: dict, area: str) -> dict:
    terms, context = _terms(candidate)
    name = candidate.get("name_en") or candidate.get("name_ru") or "кандидат"

    def fetch(search):
        item = Candidate(
            candidate_id=str(candidate.get("candidate_id") or name),
            name_en=name,
            query_id=candidate.get("query_id"),
            name_ru=candidate.get("name_ru"),
            terms=list(search.terms),
            context_terms=list(search.context_terms),
        )
        try:
            result = collector.count_history(item)
        except ValueError:
            return pd.DataFrame(columns=["source", "window", "n"])
        rows = [
            {"source": row.source, "window": row.window, "n": row.n}
            for row in result.counters
        ]
        return pd.DataFrame(rows, columns=["source", "window", "n"])

    built = candidate_features(name, terms=terms or [name], context_terms=context, area=area, fetch=fetch)
    built["name_ru"] = candidate.get("name_ru") or name
    built["doc_ids"] = list(candidate.get("doc_ids") or [])
    return built


def _load_artifact(feature_rows: list[dict]):
    if MODEL_PATH.is_file():
        from model.predict import load

        artifact = load()
        artifact["fallback"] = False
        return artifact
    frame = pd.DataFrame(feature_rows)
    for name in FEATURES:
        if name not in frame.columns:
            frame[name] = math.nan
    centre, scale = centre_scale(frame[FEATURES])

    class _Scaler:
        mode = "global"
        overall_ = (centre, scale)
        by_area_: dict = {}

        def stats_for(self, area: str):
            return self.overall_

    class _Pipeline:
        def __init__(self) -> None:
            self.named_steps = {"area_scaler": _Scaler()}

    return {
        "pipeline": _Pipeline(),
        "fallback": True,
        "meta": {
            "features": list(FEATURES),
            "coefficients": dict(PUBLISHED_COEFFICIENTS),
            "intercept": PUBLISHED_INTERCEPT,
            "threshold": PUBLISHED_THRESHOLD,
            "imputer_median_scaled": {name: 0.0 for name in FEATURES},
            "model_version": MODEL_VERSION,
            "cutoff_date": CUTOFF_DATE,
            "by_area": {},
        },
    }


def _sources_for(doc_ids: list, documents: list[dict], limit: int = 6) -> list[dict]:
    picked = []
    for doc_id in doc_ids:
        index = int(doc_id) - 1
        if 0 <= index < len(documents):
            picked.append(documents[index])
        if len(picked) >= limit:
            break
    if not picked:
        picked = documents[:limit]
    cards = []
    for doc in picked:
        cards.append(
            {
                "title": doc.get("title"),
                "url": doc.get("url"),
                "published_at": doc.get("published_at"),
                "source": doc.get("source"),
                "source_type": doc.get("source_type"),
                "language": doc.get("language") or "en",
                "trust_level": doc.get("trust_level") or "medium",
                "text": (doc.get("text") or "")[:1500],
            }
        )
    return cards


def _explanations(features: dict, contributions: dict) -> list[str]:
    lines = []
    for item in top_features(features, contributions):
        value = item.get("value")
        text = item["explanation_ru"]
        if isinstance(value, float):
            text = f"{text}: {value:.2f} (вклад {item['contribution']:+.2f})"
        lines.append(text)
    return lines


def _plain_features(features: dict) -> dict:
    out = {}
    for name, value in features.items():
        if value is None or (isinstance(value, float) and math.isnan(value)):
            out[name] = None
        else:
            out[name] = float(value)
    return out


def assemble_result(
    *,
    query_id: str,
    topic: str,
    area: str | None,
    subqueries: list[dict],
    documents: list[dict],
    built: list[dict],
    artifact: dict,
    warnings: list[str],
    timings: dict,
) -> dict:
    """Считает вероятность и раскладывает кандидатов на ТОП и исключённых."""
    meta = artifact["meta"]
    scored = []
    for item in built:
        row = {
            "name_ru": item.get("name_ru"),
            "name_en": item.get("name"),
            "features": _plain_features(item.get("features") or {}),
            "doc_ids": item.get("doc_ids") or [],
            "complete": bool(item.get("complete")),
            "score": None,
            "contributions": {},
            "explanation_ru": [],
            "skipped_reason": None,
        }
        if not item.get("complete"):
            row["skipped_reason"] = "no_trace"
            row["reason_ru"] = "Счётчиков по этим терминам не нашлось"
            scored.append(row)
            continue
        probability, contributions = predict(item["features"], area or "", artifact)
        row["score"] = probability
        row["contributions"] = contributions
        row["explanation_ru"] = _explanations(item["features"], contributions)
        if probability < meta["threshold"]:
            row["skipped_reason"] = "below_threshold"
            row["reason_ru"] = row["explanation_ru"][0] if row["explanation_ru"] else "Ниже порога модели"
        scored.append(row)

    ready = sorted((item for item in scored if item["score"] is not None), key=lambda item: -item["score"])
    signals = [item for item in ready if item["score"] >= meta["threshold"]]
    below = [item for item in ready if item["score"] < meta["threshold"]]
    missing = [item for item in scored if item["score"] is None]
    top_rows = signals[:TOP_LIMIT]
    excluded = signals[TOP_LIMIT:] + below + missing

    def card(item: dict, rank: int | None) -> dict:
        payload = {
            "name_ru": item["name_ru"],
            "name_en": item["name_en"],
            "score": item["score"],
            "explanation_ru": item["explanation_ru"],
            "contributions": item["contributions"],
            "counters": item["features"],
            "sources": _sources_for(item["doc_ids"], documents),
        }
        if rank is not None:
            payload["rank"] = rank
        return payload

    top = [card(item, rank) for rank, item in enumerate(top_rows, start=1)]
    excluded_cards = []
    for item in excluded:
        excluded_cards.append(
            {
                "name_ru": item["name_ru"],
                "name_en": item["name_en"],
                "score": item["score"],
                "skipped_reason": item["skipped_reason"] or "below_threshold",
                "reason_ru": item.get("reason_ru") or "Ниже порога модели",
            }
        )
    by_source: dict[str, int] = {}
    for doc in documents:
        source = str(doc.get("source") or "unknown")
        by_source[source] = by_source.get(source, 0) + 1
    if artifact.get("fallback"):
        warnings = list(warnings) + [
            "Файл модели не найден, оценка идёт по опубликованным весам. "
            "Центр и масштаб посчитаны по кандидатам этого запроса."
        ]
    return {
        "query_id": query_id,
        "topic": topic,
        "area": area,
        "model_version": meta["model_version"],
        "threshold": meta["threshold"],
        "cutoff_date": meta["cutoff_date"],
        "subqueries": subqueries,
        "stats": {
            "documents_by_source": by_source,
            "documents_total": len(documents),
            "candidates_found": len(built),
            "candidates_scored": sum(1 for item in scored if item["score"] is not None),
            "above_threshold": len(signals),
            "above_075": sum(1 for item in signals if item["score"] >= 0.75),
        },
        "top": top,
        "excluded": excluded_cards,
        "timings": timings,
        "warnings": warnings,
    }


def run_query(
    topic: str,
    area: str | None,
    query_id: str,
    on_progress: Progress | None = None,
) -> dict:
    """Один пользовательский запрос. Сеть: YandexGPT, OpenAlex, arXiv, TechCrunch."""
    load_dotenv()
    started = time.monotonic()
    timings: dict[str, float] = {}
    warnings: list[str] = []
    where = area or ""

    _progress(on_progress, STAGES[0], 1)
    mark = time.monotonic()
    sub = generate_subqueries(topic, query_id)
    timings["subqueries"] = round(time.monotonic() - mark, 1)
    warnings.extend(sub.get("warnings") or [])

    _progress(on_progress, STAGES[1], 2)
    mark = time.monotonic()
    from collector.api import search_recent

    found = search_recent(
        sub["subqueries"],
        collector=_collector(),
        limit=_env_int("QUERY_RECENT_PER_SUBQUERY", 10),
    )
    documents = found.get("documents") or []
    cap_docs = _env_int("QUERY_MAX_DOCS", 60)
    if len(documents) > cap_docs:
        warnings.append(f"в извлечение ушло {cap_docs} документов из {len(documents)}")
        documents = documents[:cap_docs]
    timings["search"] = round(time.monotonic() - mark, 1)

    _progress(on_progress, STAGES[2], 3)
    mark = time.monotonic()
    extracted = extract_candidates(documents, topic, query_id)
    warnings.extend(extracted.get("warnings") or [])
    candidates = extracted.get("candidates") or []
    timings["candidates"] = round(time.monotonic() - mark, 1)

    _progress(on_progress, STAGES[3], 4)
    score_cap = _env_int("QUERY_MAX_SCORE", 10)
    if len(candidates) > score_cap:
        warnings.append(f"оценены первые {score_cap} кандидатов из {len(candidates)}")
        candidates = candidates[:score_cap]

    _progress(on_progress, STAGES[4], 5)
    mark = time.monotonic()
    collector = _collector()
    built: list[dict] = []
    workers = min(6, max(1, len(candidates)))
    if candidates:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = [pool.submit(_fetch_one, collector, item, where) for item in candidates]
            for future in as_completed(futures):
                try:
                    built.append(future.result())
                except Exception as exc:  # noqa: BLE001
                    warnings.append(f"кандидат не посчитан: {exc}")
    timings["counters"] = round(time.monotonic() - mark, 1)

    _progress(on_progress, STAGES[5], 6)
    mark = time.monotonic()
    complete = [item["features"] for item in built if item.get("complete")]
    if not complete:
        artifact = _load_artifact([{name: 0.0 for name in FEATURES}])
    else:
        artifact = _load_artifact(complete)
    result = assemble_result(
        query_id=query_id,
        topic=topic,
        area=area,
        subqueries=sub.get("subqueries") or [],
        documents=documents,
        built=built,
        artifact=artifact,
        warnings=warnings,
        timings=timings,
    )
    timings["ranking"] = round(time.monotonic() - mark, 1)
    timings["total"] = round(time.monotonic() - started, 1)
    result["timings"] = timings
    return result
