"""Оркестратор режима запроса: тема пользователя -> ТОП-15 и исключённые с причинами.

Шаги: подзапросы -> поиск №1 -> кандидаты (шаг 4) -> названия через нормализатор обучения
(pipeline/naming.py: три варианта, выбор по следу в OpenAlex, no_trace) -> слияние по tech_key
-> страховочная проверка названия -> лимит -> счётчики и признаки -> ранжирование.

Запрос кандидата для счётчиков строится так же, как у 160 обучающих технологий: один
термин — tech_key названия от нормализатора, без context_terms (CLAUDE.md, правила ML, п. 2).
Синонимы и контекст, которые предлагает шаг 4, в счётчики не идут.
"""
from __future__ import annotations

import json
import re
import time
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache
from pathlib import Path
from typing import Callable, Sequence

from collector.adapters.base import SourceAdapter
from collector.api import DocumentCollector, default_adapters, tech_key
from collector.db import MemoryCache
from collector.settings import Settings
from model.predict import load
from model.ranking import rank_candidates
from pipeline import naming
from pipeline.fetch import parallel_fetch
from pipeline.search_cache import CachedSearch
from pipeline.reasons import SPECIAL, explanation_top, reason_below
from search.extract_candidates import extract_candidates
from search.extract_terms import extract_terms
from search.subqueries import generate_subqueries

ROOT = Path(__file__).resolve().parents[1]
TECHNOLOGIES = ROOT / "data" / "interim" / "technologies.csv"
TOP_N = 15
MAX_SCORED = 60
HIGH_SCORE = 0.75
MAX_SOURCES = 5
NAME_WORDS = (2, 5)
BAD_NAME_CHARS = '"(),:;/'
LATIN_RE = re.compile(r"[a-zA-Z]")
CYRILLIC_RE = re.compile(r"[а-яёА-ЯЁ]")
EMPTY_AREA_WARNING = "область не выбрана: нормализатор получил пустую область, такое поведение не проверялось"
OUTCOME_REASON = {"no_trace": "no_trace", "bad_format": "bad_name", "trace_unknown": "trace_unknown"}
Progress = Callable[[str, int, int], None]


@lru_cache(maxsize=1)
def training_labels() -> dict[str, dict]:
    """tech_key обучающей технологии -> {id, label}. Первая строка при повторе фразы."""
    import pandas as pd
    table = pd.read_csv(TECHNOLOGIES, encoding="utf-8-sig")
    known: dict[str, dict] = {}
    for row in table.itertuples():
        known.setdefault(tech_key(row.name_en), {"id": row.tech_id,
                                                 "label": "signal" if row.label == 1 else "mainstream"})
    return known


def named_candidates(raw: Sequence[dict], area: str, openalex, progress: Progress) -> tuple[list[dict], list[dict]]:
    """Нормализатор по name_ru каждого кандидата шага 4; нет названия — сразу в исключённые."""
    results = naming.normalize_all([item["name_ru"] for item in raw], area, openalex,
                                   on_done=lambda done, total: progress("naming", done, total))
    kept, dropped = [], []
    for item, result in zip(raw, results):
        chosen = result["chosen"] or {}
        candidate = {"name_ru": item["name_ru"], "name_raw": item["name_en"],
                     "name_en": chosen.get("name") or item["name_en"], "doc_ids": list(item.get("doc_ids") or []),
                     "name_variants": [{k: v[k] for k in ("name", "n_works", "n_institutions", "n_institutions_capped")}
                                       for v in result["variants"]],
                     "name_choice_rule": naming.CHOICE_RULE, "n_works": chosen.get("n_works"),
                     "n_institutions": chosen.get("n_institutions"),
                     "n_institutions_capped": chosen.get("n_institutions_capped")}
        if result["chosen"] is None:
            dropped.append(candidate | {"skipped_reason": OUTCOME_REASON[result["outcome"]]})
        else:
            kept.append(candidate)
    return kept, dropped


def direct_candidates(raw: Sequence[dict], openalex, progress: Progress) -> tuple[list[dict], list[dict]]:
    """Без нормализатора (extract-v2): name_en = термин шага 4, след и организации — тем же вызовом."""
    phrases = [tech_key(item["name_en"]) for item in raw]
    with ThreadPoolExecutor(max_workers=naming.MAX_WORKERS) as pool:
        traces = list(pool.map(lambda phrase: naming.trace(phrase, openalex), phrases))
    progress("naming", len(raw), len(raw))
    kept, dropped = [], []
    for item, trace in zip(raw, traces):
        candidate = {"name_ru": item["name_ru"], "name_raw": item["name_en"], "name_en": item["name_en"],
                     "doc_ids": list(item.get("doc_ids") or []), "name_variants": [trace],
                     "name_choice_rule": "direct", "n_works": trace["n_works"],
                     "n_institutions": trace["n_institutions"],
                     "n_institutions_capped": trace["n_institutions_capped"], "quote": item.get("quote")}
        if trace["n_works"] is None:
            dropped.append(candidate | {"skipped_reason": "trace_unknown"})
        elif trace["n_works"] == 0:
            dropped.append(candidate | {"skipped_reason": "no_trace"})
        else:
            kept.append(candidate)
    return kept, dropped


def merge_candidates(candidates: Sequence[dict]) -> list[dict]:
    """Слияние по tech_key(name_en): первая запись остаётся, doc_ids объединяются."""
    merged: dict[str, dict] = {}
    for item in candidates:
        key = tech_key(item["name_en"])
        if key not in merged:
            merged[key] = {**item, "doc_ids": []}
        merged[key]["doc_ids"] = sorted(set(merged[key]["doc_ids"]) | set(item.get("doc_ids") or []))
    return list(merged.values())


# Общие слова, которые не различают технологии: снимаются перед сравнением основ.
SUBTOPIC_COMMON = frozenset({"method", "methods", "system", "systems", "technique", "techniques", "integration",
                             "optimization", "approach", "framework", "in", "for", "of", "based"})
STEM_LENGTH = 5
WORD_RE = re.compile(r"[a-z0-9\-]+")


def subtopic_key(name_en: str) -> frozenset:
    """Множество основ (первые пять букв) без общих слов."""
    return frozenset(word[:STEM_LENGTH] for word in WORD_RE.findall(name_en.lower())
                     if word not in SUBTOPIC_COMMON)


def merge_subtopics(candidates: Sequence[dict]) -> list[dict]:
    """Слияние подтем: одинаковые множества основ — одна запись, остаётся больший n_works.

    Надмножество не сливается: иначе конкретное уходило бы в общее
    (uwb robot localization -> localization method in robotics).
    """
    groups: dict[frozenset, list[dict]] = {}
    for item in candidates:
        groups.setdefault(subtopic_key(item["name_en"]), []).append(item)
    merged = []
    for items in groups.values():
        keep = max(items, key=lambda item: item.get("n_works") or 0)
        doc_ids = sorted({i for item in items for i in item.get("doc_ids") or []})
        merged.append({**keep, "doc_ids": doc_ids})
    return merged


def good_name(name_en: str) -> bool:
    """2–5 слов, есть латиница, нет кириллицы и символов, ломающих фразовый поиск."""
    words = len(name_en.split())
    return (NAME_WORDS[0] <= words <= NAME_WORDS[1] and bool(LATIN_RE.search(name_en))
            and not CYRILLIC_RE.search(name_en) and not any(c in name_en for c in BAD_NAME_CHARS))


def details(candidate: dict, documents: Sequence[dict]) -> dict:
    """Поля названия и диагностики, общие для ТОПа и исключённых."""
    picked = [documents[i - 1] for i in candidate.get("doc_ids") or [] if 1 <= i <= len(documents)]
    return {"name_raw": candidate.get("name_raw"), "name_variants": candidate.get("name_variants"),
            "name_choice_rule": candidate.get("name_choice_rule"), "n_works": candidate.get("n_works"),
            "n_institutions": candidate.get("n_institutions"),
            "n_institutions_capped": candidate.get("n_institutions_capped"),
            "quote": candidate.get("quote"),
            "n_docs": len(picked), "n_sources": len({doc["source"] for doc in picked}),
            "known_training_label": training_labels().get(tech_key(candidate["name_en"]))}


def excluded(candidate: dict, reason: str, documents: Sequence[dict], score: float | None = None,
             text: str | None = None) -> dict:
    """Запись исключённого кандидата."""
    return {"name_ru": candidate["name_ru"], "name_en": candidate["name_en"], "score": score,
            "skipped_reason": reason, "reason_ru": text or SPECIAL[reason], **details(candidate, documents)}


def apply_cap(candidates: list[dict], documents: Sequence[dict],
              limit: int = MAX_SCORED) -> tuple[list[dict], list[dict]]:
    """Больше limit — остаются кандидаты с наибольшим числом документов, остальные — cap."""
    ranked = sorted(candidates, key=lambda item: -len(item["doc_ids"]))
    return ranked[:limit], [excluded(item, "cap", documents) for item in ranked[limit:]]


def sources_of(doc_ids: Sequence[int], documents: Sequence[dict]) -> list[dict]:
    """До пяти документов поиска №1 из doc_ids кандидата, свежие первыми."""
    fields = ("title", "url", "published_at", "source", "source_type", "language", "trust_level")
    picked = [documents[i - 1] for i in doc_ids if 1 <= i <= len(documents)]
    picked.sort(key=lambda doc: doc.get("published_at") or "", reverse=True)
    return [{name: doc.get(name) for name in fields} for doc in picked[:MAX_SOURCES]]


def origin(doc: dict, subqueries: Sequence[dict]) -> str:
    """Источник документа; OpenAlex, найденный только русскими подзапросами, — openalex_ru."""
    language = {item["subquery_id"]: item["language"] for item in subqueries}
    ids = doc.get("subquery_ids") or []
    only_ru = bool(ids) and all(language.get(i) == "ru" for i in ids)
    return "openalex_ru" if doc["source"] == "openalex" and only_ru else doc["source"]


def document_stats(documents: Sequence[dict], subqueries: Sequence[dict]) -> dict[str, int]:
    """Документы по источникам происхождения (openalex, openalex_ru, arxiv, techcrunch)."""
    counts = {"openalex": 0, "arxiv": 0, "techcrunch": 0, "openalex_ru": 0}
    for doc in documents:
        key = origin(doc, subqueries)
        counts[key] = counts.get(key, 0) + 1
    return counts


def split_ranked(ranked: list[dict], by_key: dict[str, dict], documents: Sequence[dict],
                 threshold: float) -> tuple[list[dict], list[dict]]:
    """ТОП-15 выше порога без добивания; остальные — в исключённые с причиной."""
    top, dropped = [], []
    for item in ranked:
        candidate = by_key[item["name"]]
        if item["score"] is None:
            dropped.append(excluded(candidate, "no_counters", documents))
        elif item["score"] < threshold:
            dropped.append(excluded(candidate, "below_threshold", documents, item["score"],
                                    reason_below(item["features"], item["contributions"], item["counters"])))
        elif len(top) >= TOP_N:
            dropped.append(excluded(candidate, "beyond_top", documents, item["score"]))
        else:
            top.append({"rank": len(top) + 1, "name_ru": candidate["name_ru"], "name_en": candidate["name_en"],
                        "score": item["score"],
                        "explanation_ru": explanation_top(item["features"], item["contributions"], item["counters"]),
                        "contributions": item["contributions"], "counters": item["counters"],
                        **details(candidate, documents), "sources": sources_of(candidate["doc_ids"], documents)})
    return top, dropped


def score_pool(pool: list[dict], area: str | None, adapters: Sequence[SourceAdapter], settings: Settings,
               warnings: list[str], progress: Progress, timings: dict, use_cache: bool = True) -> list[dict]:
    """Счётчики (источники параллельно) и ранжирование; время счётчиков отдельно от ранжирования."""
    mark, fetch_time, done = time.monotonic(), [0.0], [0]

    def tick():
        done[0] += 1
        progress("counters", done[0], len(pool))

    fetch = parallel_fetch(adapters, settings, warnings, on_done=tick, use_cache=use_cache)

    def timed_fetch(search):
        begin = time.monotonic()
        try:
            return fetch(search)
        finally:
            fetch_time[0] += time.monotonic() - begin

    items = [{"name": tech_key(c["name_en"]), "terms": [tech_key(c["name_en"])], "context_terms": [],
              "area": area or ""} for c in pool]
    begin = time.monotonic()
    fetch.prefetch([item["name"] for item in items])  # очереди по источникам, задача Л5.2
    fetch_time[0] += time.monotonic() - begin
    ranked = rank_candidates(items, area=area or "", fetch=timed_fetch)
    timings["counters"] = round(fetch_time[0], 2)
    timings["ranking"] = round(time.monotonic() - mark - fetch_time[0], 2)
    progress("ranking", 1, 1)
    return ranked


def staged(name: str, timings: dict, progress: Progress, action: Callable):
    """Выполняет шаг, пишет его время и прогресс 0/1 -> 1/1."""
    mark = time.monotonic()
    progress(name, 0, 1)
    value = action()
    timings[name] = round(time.monotonic() - mark, 2)
    progress(name, 1, 1)
    return value


def run_query(topic: str, area: str | None = None, *, use_cache: bool = True,
              progress: Progress | None = None, adapters: Sequence[SourceAdapter] | None = None,
              settings: Settings | None = None, candidate_sources: set[str] | None = None,
              extract_model: str | None = "yandexgpt-5-pro", naming_mode: str = "direct",
              counters_cache: bool | None = None, extract_version: str = "v2") -> dict:
    """Тема -> JSON с ТОП-15, исключёнными, статистикой и временем по шагам.

    candidate_sources — из документов каких источников извлекать кандидатов (openalex,
    openalex_ru, arxiv, techcrunch). None — из всех, как раньше. Поиск №1 и его статистика
    при этом по всем источникам; doc_ids кандидатов нумеруют только отобранные документы.
    extract_model — модель шага 4 (extract-v2);
    naming_mode — normalizer (нормализатор обучения) или direct (name_en = термин шага 4).
    counters_cache — файловый кэш счётчиков отдельно от остальных кэшей; None — как use_cache.
    extract_version — v2 (search/extract_terms.py) или v1 (прежний шаг 4, search/extract_candidates.py;
    модель берётся из .env, extract_model не используется; в паре с naming_mode="normalizer").
    По умолчанию — рука R1 задачи К: extract-v2 на yandexgpt-5-pro, без нормализатора.
    """
    if extract_version not in ("v1", "v2"):
        raise ValueError(f"extract_version {extract_version!r}: ожидалось v1 или v2")
    progress = progress or (lambda stage, done, total: None)
    settings = settings or Settings.from_env()
    adapters = list(adapters) if adapters is not None else default_adapters(settings)
    meta, started, timings = load()["meta"], time.monotonic(), {}
    query_id = f"q{datetime.now():%Y%m%d%H%M%S}"

    subq = staged("subqueries", timings, progress, lambda: generate_subqueries(topic, query_id, use_cache=use_cache))
    warnings = list(subq["warnings"])
    searchers = [CachedSearch(adapter, use_cache=use_cache) for adapter in adapters]
    collector = DocumentCollector(adapters=searchers, cache=MemoryCache(), settings=settings)
    documents = staged("search", timings, progress,
                       lambda: collector.search_recent(subq["subqueries"]).to_dict()["documents"])
    all_documents = documents
    if candidate_sources is not None:
        documents = [doc for doc in documents if origin(doc, subq["subqueries"]) in candidate_sources]
    if extract_version == "v1":
        extract = lambda: extract_candidates(documents, topic, query_id, use_cache=use_cache)
    else:
        extract = lambda: extract_terms(documents, topic, query_id, model=extract_model, use_cache=use_cache)
    found = staged("candidates", timings, progress, extract)
    warnings += found["warnings"] + ([EMPTY_AREA_WARNING] if not area else [])

    mark = time.monotonic()
    openalex = next(a for a in adapters if a.source == "openalex")
    if naming_mode == "direct":
        named, unnamed = direct_candidates(found["candidates"], openalex, progress)
    else:
        named, unnamed = named_candidates(found["candidates"], area or "", openalex, progress)
    timings["naming"] = round(time.monotonic() - mark, 2)
    merged = merge_subtopics(merge_candidates(named))
    dropped = [excluded(item, item["skipped_reason"], documents) for item in unnamed]
    dropped += [excluded(item, "bad_name", documents) for item in merged if not good_name(item["name_en"])]
    pool, capped = apply_cap([item for item in merged if good_name(item["name_en"])], documents)
    ranked = score_pool(pool, area, adapters, settings, warnings, progress, timings,
                        use_cache if counters_cache is None else counters_cache)
    top, below = split_ranked(ranked, {tech_key(c["name_en"]): c for c in pool}, documents, meta["threshold"])
    scores = [item["score"] for item in ranked if item["score"] is not None]
    timings["total"] = round(time.monotonic() - started, 2)
    return {
        "query_id": query_id, "topic": topic.strip(), "area": area,
        "model_version": meta["model_version"], "threshold": meta["threshold"], "cutoff_date": meta["cutoff_date"],
        "subqueries": subq["subqueries"],
        "candidate_sources": sorted(candidate_sources) if candidate_sources is not None else None,
        "extract_version": extract_version, "extract_model": extract_model if extract_version == "v2" else None,
        "naming_mode": naming_mode,
        "stats": {"documents_by_source": document_stats(all_documents, subq["subqueries"]),
                  "documents_total": len(all_documents), "documents_for_candidates": len(documents),
                  "candidates_found": len(found["candidates"]),
                  "candidates_named": len(merged),
                  "candidates_scored": len(scores),
                  "above_threshold": sum(score >= meta["threshold"] for score in scores),
                  "above_075": sum(score >= HIGH_SCORE for score in scores)},
        "normalizer_deviations": list(naming.DEVIATIONS),
        "top": top, "excluded": dropped + capped + below,
        "timings": timings, "warnings": warnings,
    }
