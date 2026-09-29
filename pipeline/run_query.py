"""Оркестратор режима запроса: тема пользователя -> ТОП-15 и исключённые с причинами.

Шаги: подзапросы -> поиск №1 -> кандидаты (шаг 4, extract-v4: name_en = термин) -> след названия
в OpenAlex (pipeline/naming.py, no_trace) -> слияние по tech_key -> страховочная проверка названия -> лимит
-> счётчики и признаки -> ранжирование -> склейка дублей -> перевод названий -> догрузка источников ТОП
-> инсайты ТОП (описание для главного экрана и карточки).

Запрос кандидата для счётчиков строится так же, как у 160 обучающих технологий: один
термин — tech_key названия, без context_terms (CLAUDE.md, правила ML, п. 2).
Синонимы и контекст, которые предлагает шаг 4, в счётчики не идут.
"""
from __future__ import annotations

import json
import math
import os
import re
import threading
import time
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache
from pathlib import Path
from typing import Callable, Sequence

from collector import rospatent as rospatent_source
from collector.adapters.base import SourceAdapter
from collector.api import DocumentCollector, default_adapters, tech_key
from collector.constants import AGGREGATE_WINDOWS
from collector.db import build_cache
from collector.settings import Settings
from pipeline.catalog import catalog_candidates
from pipeline.enrich import enrich_top
from pipeline.insights import insights_top
from pipeline.persist import persist_features_and_scores, persist_search_documents
from pipeline.weak_sources import move_weak_only
from model.config import ROSPATENT_DATASETS
from model.features import RESEARCH_SOURCES, share_patent
from model.predict import load
from model.ranking import rank_candidates
from pipeline import dedup as dedup_module, fetch as fetch_module, naming, translate
from pipeline.progress import Report, tracker
from pipeline.fetch import parallel_fetch
from pipeline.search_cache import CachedSearch
from pipeline.reasons import SPECIAL, explanation_top, reason_below, why_words
from search.extract_terms import extract_terms
from search.subqueries import generate_subqueries

ROOT = Path(__file__).resolve().parents[1]
TECHNOLOGIES = ROOT / "data" / "interim" / "technologies.csv"
TOP_N = 15
DUPLICATE_TEXT = "Дубль: тот же класс технологий, что «{}», — показан вариантом названия"
MAX_SCORED = 60
# Бюджет времени прогона от старта запроса (задача Х): переопределяется env TIME_BUDGET_S и флагом --time-budget.
TIME_BUDGET_S = 900
TIME_RESERVE_S = 90  # резерв на ранжирование, склейку и перевод после счётчиков
# Потолок оценки (Х3): бюджет только уменьшает число оценённых. 200 на «ИИ аналитике» дало шум — редкие
# составные «ИИ в X» из одного документа выглядят молодыми и вытесняют нормальные термины.
MAX_SCORED_HARD = 60
# Добор кандидатов (Х2): один раунд новых подзапросов, если названных меньше TARGET_NAMED и прошло
# меньше 35 % бюджета. Выбор и проверки — evidence/rounds_choice.md.
TARGET_NAMED, MAX_ROUNDS, ROUNDS_BEFORE_SHARE = 60, 1, 0.35
STEP4_MIN_DOCUMENTS = 200  # Х2b: меньше — английские OpenAlex добираются сверх пропорции
BATCH_SIZE = 10  # пачка кандидатов между пересчётами прогноза времени
FIRST_ESTIMATE_S, FIRST_ESTIMATE_N, ESTIMATE_BATCHES = 12.0, 5, 3  # до 5 оценённых — 12 с; среднее по 3 пачкам
QUOTA_MARGIN = 1.2
QUOTA_WARNING = ("Квота OpenAlex: остатка {left} запросов хватит примерно на {limit} кандидатов "
                 "с запасом 20 %, оценка ограничена")
HIGH_SCORE = 0.75
# Балл ранжирования (методология 9.9): равные на экране score (3 знака) расходятся на 0…RANK_STEPS шагов по 0.001.
RANK_STEPS, RANK_STEP = 10, 0.001
MAX_SOURCES = 5
NAME_WORDS = (2, 5)
BAD_NAME_CHARS = '"(),:;/'
LATIN_RE = re.compile(r"[a-zA-Z]")
CYRILLIC_RE = re.compile(r"[а-яёА-ЯЁ]")
EMPTY_AREA_WARNING = "область не выбрана: нормализатор получил пустую область, такое поведение не проверялось"
ROSPATENT_OFF_WARNING = ("Роспатент выключен: патентный признак share_patent недоступен у всех кандидатов, "
                         "модель подставила медиану обучения")
TRANSLATE_FAILED_WARNING = ("Перевод не прошёл проверки после всех попыток, в ТОП оставлено английское название: {}")
ENRICH_STOPPED_WARNING = "Догрузка источников остановлена по бюджету времени: у части ТОП только источники поиска №1"
PATENT_FAILED_NOTE = "Патентный признак недоступен: Роспатент не ответил, подставлена медиана обучения"
ROSPATENT_NO_KEY_WARNING = ("Нет ключа ROSPATENT в .env: патентный признак share_patent недоступен у всех кандидатов, "
                            "модель подставила медиану обучения")
# Генерация кандидатов v3 (задача Г): промпт подзапросов subq-v4 (задача О3) и состав документов шага 4 (step4_documents).
CANDIDATES_VERSION = "v3"
# Поля выхода о режиме шага 4 и названий (контракт): в продукте один режим — extract-v4 (задача О2) и direct.
EXTRACT_VERSION, NAMING_MODE = "v4", "direct"
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


def direct_candidates(raw: Sequence[dict], openalex, progress: Progress) -> tuple[list[dict], list[dict]]:
    """Без нормализатора (extract-v4): name_en = термин шага 4, след и организации — тем же вызовом."""
    phrases, done, lock = [tech_key(item["name_en"]) for item in raw], [0], threading.Lock()

    def traced(phrase: str) -> dict:
        trace = naming.trace(phrase, openalex)
        with lock:
            done[0] += 1
            progress("naming", done[0], len(raw))
        return trace
    with ThreadPoolExecutor(max_workers=naming.MAX_WORKERS) as pool:
        traces = list(pool.map(traced, phrases))
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
    return {"name_ru": candidate["name_ru"], "name_en": candidate["name_en"],
            "tech_key": tech_key(candidate["name_en"]), "score": score,
            "skipped_reason": reason, "reason_ru": text or SPECIAL[reason], **details(candidate, documents)}


def per_candidate_s(batches: Sequence[tuple[float, int]]) -> float:
    """Скользящее среднее секунд на кандидата без кэша по последним пачкам; пока их меньше 5 — 12 с."""
    recent = batches[-ESTIMATE_BATCHES:]
    uncached = sum(n for _, n in recent)
    if sum(n for _, n in batches) < FIRST_ESTIMATE_N or not uncached:
        return FIRST_ESTIMATE_S
    return sum(seconds for seconds, _ in recent) / uncached


def fits(cost: int, elapsed: float, budget: float, estimate: float) -> bool:
    """Брать ли пачку: прогноз помещается в бюджет без резерва (минимум совпал с потолком 60 — ветки нет)."""
    limit = budget - TIME_RESERVE_S
    return elapsed < limit and elapsed + estimate * cost <= limit


def quota_limit(done: int, cost: int, before: int | None, after: int | None) -> int | None:
    """Сколько всего кандидатов выдержит остаток квоты OpenAlex с запасом 20 %; расход — по первой пачке."""
    if before is None or after is None or not cost or before <= after:
        return None
    return done + int(after / ((before - after) / cost * QUOTA_MARGIN))


def schedule_counters(fetch, phrases: Sequence[str], elapsed: Callable[[], float], budget: float,
                      warnings: list[str], quota: Callable[[], int | None]) -> tuple[list[str], str | None, dict]:
    """Счётчики пачками в прежнем порядке, пока прогноз времени помещается (задача Х3).

    Жёсткий дедлайн — бюджет без резерва: пачку, не успевшую к нему, prefetch перестаёт ждать, её
    недособранные кандидаты не оцениваются. Возвращает оценённые фразы, причину для остальных
    (time_budget, cap по квоте или None) и сводку расписания.
    """
    limit, done, batches, stop, scored = min(len(phrases), MAX_SCORED_HARD), 0, [], None, []
    info = {"quota_before": quota(), "quota_after_first": None, "quota_limit": None}
    while done < limit:
        batch = phrases[done:min(done + BATCH_SIZE, limit)]
        cost = sum(not fetch.is_cached(phrase) for phrase in batch)
        if not fits(cost, elapsed(), budget, per_candidate_s(batches)):
            stop = "time_budget"
            break
        mark = elapsed()
        ready = fetch.prefetch(batch, timeout=budget - TIME_RESERVE_S - mark)
        scored += [phrase for phrase in batch if phrase in ready]
        batches.append((elapsed() - mark, cost))
        done += len(batch)
        if len(ready) < len(batch):  # дедлайн внутри пачки
            stop = "time_budget"
            break
        if cost and info["quota_after_first"] is None and info["quota_before"] is not None:
            info["quota_after_first"] = quota()
            room = quota_limit(done, cost, info["quota_before"], info["quota_after_first"])
            if room is not None and room < limit:
                limit, info["quota_limit"] = max(done, room), room
                warnings.append(QUOTA_WARNING.format(left=info["quota_after_first"], limit=limit))
    if stop is None and done < len(phrases):
        stop = "cap"
    return scored, stop, {**info, "per_candidate_s": round(per_candidate_s(batches), 2)}


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


def is_russian(doc: dict, subqueries: Sequence[dict]) -> bool:
    """Русский документ OpenAlex: найден только русскими подзапросами или помечен языком ru."""
    return doc["source"] == "openalex" and (origin(doc, subqueries) == "openalex_ru" or doc.get("language") == "ru")


def step4_documents(documents: Sequence[dict], subqueries: Sequence[dict],
                    minimum: int = STEP4_MIN_DOCUMENTS) -> list[dict]:
    """Документы для шага 4 (v3): все arXiv, затем TechCrunch, затем английские OpenAlex
    в прежнем порядке, не больше, чем arXiv и TechCrunch вместе; русские OpenAlex не идут.

    Х2b: если так набирается меньше minimum, английские OpenAlex добираются сверх пропорции до minimum.
    """
    head = ([doc for doc in documents if doc["source"] == "arxiv"]
            + [doc for doc in documents if doc["source"] == "techcrunch"])
    english = [doc for doc in documents if doc["source"] == "openalex" and not is_russian(doc, subqueries)]
    return head + english[:max(len(head), minimum - len(head))]


def step4_topped_up(documents: Sequence[dict], subqueries: Sequence[dict], picked: Sequence[dict]) -> int:
    """Сколько документов шага 4 добрано правилом Х2b сверх пропорции."""
    return len(picked) - len(step4_documents(documents, subqueries, minimum=0))


def documents_analyzed(documents: Sequence[dict], ranked: Sequence[dict], n_pats: dict[str, int]) -> int:
    """Документы, учтённые прогоном: поиск №1 с догрузкой + публикации в счётчиках оценённых кандидатов
    (все окна 2014–2026, все источники) + патенты Роспатента. Повторы между кандидатами не вычитаются:
    у счётчиков нет идентификаторов документов."""
    counted = sum(int(value) for item in ranked if item.get("score") is not None
                  for window in (item.get("counters") or {}).values() for value in window.values())
    return len(documents) + counted + sum(int(value) for value in n_pats.values())


def document_stats(documents: Sequence[dict], subqueries: Sequence[dict]) -> dict[str, int]:
    """Документы по источникам происхождения (openalex, openalex_ru, arxiv, techcrunch)."""
    counts = {"openalex": 0, "arxiv": 0, "techcrunch": 0, "openalex_ru": 0}
    for doc in documents:
        key = origin(doc, subqueries)
        counts[key] = counts.get(key, 0) + 1
    return counts


def evidence(item: dict, n_pats: dict[str, int | None]) -> int:
    """Подтверждения кандидата: публикации в счётчиках (все окна 2014–2026, все источники) плюс патенты."""
    counted = sum(int(value) for window in (item.get("counters") or {}).values() for value in window.values())
    return counted + int(n_pats.get(item["name"]) or 0)


def with_rank_scores(ranked: list[dict], n_pats: dict[str, int | None]) -> list[dict]:
    """rank_score и порядок по нему; неоценённые — в конце в прежнем порядке.

    Группа — равные до 3 знаков score. Внутри группы по убыванию подтверждений (затем по названию) —
    добавка от RANK_STEPS до 0 шагов по RANK_STEP к округлённому score: до 11 членов значения на экране разные.
    Одиночка — rank_score = score. Порог и плашки считаются по score, не по rank_score.
    """
    scored = [dict(item) for item in ranked if item["score"] is not None]
    groups: dict[float, list[dict]] = {}
    for item in scored:
        groups.setdefault(round(item["score"], 3), []).append(item)
    for base, group in groups.items():
        group.sort(key=lambda item: (-evidence(item, n_pats), item["name"]))
        last = len(group) - 1
        for place, item in enumerate(group):
            if not last:
                item["rank_score"] = item["score"]
            elif last <= RANK_STEPS:
                item["rank_score"] = round(base + RANK_STEP * round(RANK_STEPS * (last - place) / last), 6)
            else:
                item["rank_score"] = round(base + RANK_STEP * RANK_STEPS * (last - place) / last, 6)
    scored.sort(key=lambda item: (-item["rank_score"], -evidence(item, n_pats), item["name"]))
    return scored + [item for item in ranked if item["score"] is None]


def split_ranked(ranked: list[dict], by_key: dict[str, dict], documents: Sequence[dict],
                 threshold: float, n_pats: dict[str, int | None] | None = None,
                 duplicate_of: dict[int, int] | None = None) -> tuple[list[dict], list[dict]]:
    """ТОП-15 выше порога без добивания; остальные — в исключённые с причиной.

    n_pats — число патентов по фразе для текста патентного признака (нет — признак в текст не идёт).
    duplicate_of — {индекс дубля: индекс принятого} (pipeline/dedup.py): дубль не занимает место, уходит
    в исключённые с причиной duplicate_of и в variants принятого; ТОП добирается следующими.
    """
    duplicate_of = duplicate_of or {}
    variants: dict[int, list[dict]] = {}
    for index, home in duplicate_of.items():
        variants.setdefault(home, []).append({"term_en": by_key[ranked[index]["name"]]["name_en"],
                                              "score": ranked[index]["score"]})
    top, dropped = [], []
    for index, item in enumerate(ranked):
        candidate = by_key[item["name"]]
        n_pat = (n_pats or {}).get(item["name"])
        extra = {"variants": variants.get(index, [])} if item["score"] is not None else {}
        scored = {"features": item.get("features") or {}, "contributions": item.get("contributions") or {},
                  "counters": item.get("counters") or {}, "is_signal": item.get("is_signal"),
                  "model_version": item.get("model_version"), "threshold": item.get("threshold")}
        if index in duplicate_of:
            accepted = by_key[ranked[duplicate_of[index]]["name"]]["name_en"]
            dropped.append({**excluded(candidate, "duplicate_of", documents, item["score"],
                                       DUPLICATE_TEXT.format(accepted)), "duplicate_of": accepted,
                            **scored, **extra})
        elif item["score"] is None:
            dropped.append(excluded(candidate, "no_counters", documents))
        elif item["score"] < threshold:
            dropped.append({**excluded(candidate, "below_threshold", documents, item["score"],
                                       reason_below(item["features"], item["contributions"], item["counters"],
                                                    n_pat)), **scored, **extra})
        elif len(top) >= TOP_N:
            dropped.append({**excluded(candidate, "beyond_top", documents, item["score"]), **scored, **extra})
        else:
            top.append({"rank": len(top) + 1, "name_ru": candidate["name_ru"], "name_en": candidate["name_en"],
                        "tech_key": tech_key(candidate["name_en"]),
                        "score": item["score"], "rank_score": item.get("rank_score", item["score"]),
                        "is_signal": item.get("is_signal"),
                        "features": item.get("features") or {},
                        "explanation_ru": explanation_top(item["features"], item["contributions"], item["counters"],
                                                          n_pat=n_pat),
                        "why_ru": why_words(item["features"], item["contributions"], n_pat=n_pat),
                        "contributions": item["contributions"], "counters": item["counters"],
                        "model_version": item.get("model_version"),
                        "threshold": item.get("threshold"),
                        **details(candidate, documents), "sources": sources_of(candidate["doc_ids"], documents),
                        **extra})
    return top, dropped


def find_duplicates(ranked: list[dict], by_key: dict[str, dict], warnings: list[str]) -> dict[int, int]:
    """Склейка дублей по term_en оценённых кандидатов (pipeline/dedup.py); предупреждения — в warnings."""
    rows = [{"name_en": by_key[item["name"]]["name_en"], "score": item["score"]} for item in ranked]
    decide, notes = dedup_module.decider([r["name_en"] for r in rows if r["score"] is not None])
    warnings.extend(notes)
    return dedup_module.dedup(rows, decide)


def patent_fields(counters: dict, result: dict | None) -> dict:
    """n_pat, share_patent и флаг сбоя для выхода кандидата. Оценку модели не меняют.

    n_research — OpenAlex + arXiv по шести годовым окнам 2020–2025, без prev6. Сбой
    Роспатента — n_pat и share_patent null, rospatent_failed true; ноль — только при total = 0.
    NaN доли (нет ни патентов, ни науки) в JSON — null.
    """
    if result is None or result["failed"]:
        return {"n_pat": None, "share_patent": None, "rospatent_failed": True, "note_ru": PATENT_FAILED_NOTE}
    n_research = sum(counters.get(window, {}).get(source, 0)
                     for window in AGGREGATE_WINDOWS["all"] for source in RESEARCH_SOURCES)
    value = share_patent(result["n_pat"], n_research)
    return {"n_pat": result["n_pat"], "share_patent": None if math.isnan(value) else value,
            "rospatent_failed": False}


def rospatent_options(enabled: bool | None) -> dict | None:
    """Параметры очереди Роспатента или None, если источник выключен."""
    if not (rospatent_source.ROSPATENT_ENABLED if enabled is None else enabled):
        return None
    return {"datasets": ROSPATENT_DATASETS, "token": os.environ.get("ROSPATENT"),
            "parallel": rospatent_source.ROSPATENT_PARALLEL}


def score_pool(pool: list[dict], area: str | None, adapters: Sequence[SourceAdapter], settings: Settings,
               warnings: list[str], progress: Progress, timings: dict, use_cache: bool = True,
               rospatent: dict | None = None, report: Report | None = None, budget: float = math.inf,
               elapsed: Callable[[], float] = lambda: 0.0, quota: Callable[[], int | None] = lambda: None
               ) -> tuple[list[dict], dict[str, dict], set[str], str | None, dict]:
    """Счётчики (источники параллельно, пачками по бюджету времени) и ранжирование; время счётчиков отдельно.

    Возвращает ранжирование оценённых, n_pat по фразам (пусто, если Роспатент выключен), оценённые фразы
    (tech_key), причину для остальных и сводку расписания (schedule_counters).
    """
    mark, fetch_time, done = time.monotonic(), [0.0], [0]

    def tick():
        done[0] += 1
        progress("counters", done[0], len(pool))

    units, sent = len(pool) * (len(adapters) + (rospatent is not None)), [0]

    def request_done(n: int) -> None:  # прогресс: единица — запрос «кандидат × источник»
        sent[0] += n
        if report:
            report("counters", sent[0], units)
    if report:
        report("counters", 0, units)
    fetch = parallel_fetch(adapters, settings, warnings, on_done=tick, use_cache=use_cache,
                           rospatent=rospatent, on_request=request_done)

    def timed_fetch(search):
        begin = time.monotonic()
        try:
            return fetch(search)
        finally:
            fetch_time[0] += time.monotonic() - begin

    items = [{"name": tech_key(c["name_en"]), "terms": [tech_key(c["name_en"])], "context_terms": [],
              "area": area or ""} for c in pool]
    begin = time.monotonic()  # очереди по источникам (Л5.2), пачками по бюджету (Х3)
    scored, stop, schedule = schedule_counters(fetch, [item["name"] for item in items], elapsed, budget,
                                               warnings, quota)
    fetch_time[0] += time.monotonic() - begin
    items = [item for item in items if item["name"] in set(scored)]
    for item in items:  # n_pat — в признак share_patent; сбой -> None -> медиана обучения
        found = fetch.patents.get(item["name"])
        item["n_pat"] = found["n_pat"] if found and not found["failed"] else None
    ranked = rank_candidates(items, area=area or "", fetch=timed_fetch)
    timings["counters"] = round(fetch_time[0], 2)
    timings["ranking"] = round(time.monotonic() - mark - fetch_time[0], 2)
    timings["queues"] = list(fetch.queues)
    progress("ranking", 1, 1)
    return ranked, dict(fetch.patents), set(scored), stop, schedule


def name_in_russian(top: list[dict], excluded_entries: list[dict], use_cache: bool,
                    progress: Progress | None = None) -> dict:
    """Русские названия (задача З): перевод у ТОП-15 и оценённых исключённых, у остальных — name_ru шага 4.

    Оценённые исключённые (below_threshold, beyond_top) показываются с причиной по признакам модели.
    """
    scored = [e for e in excluded_entries if e.get("skipped_reason") in ("below_threshold", "beyond_top")]
    shown = top + scored
    numbers = translate.translate_entries(shown, use_cache=use_cache,
                                          on_done=(lambda done: progress("translate", done, len(shown))) if progress else None)
    for entry in excluded_entries:
        if entry not in scored:
            entry.update(name_ru_source="extract" if entry.get("name_ru") else None, name_ru_auto=True)
    return numbers


def both_progress(progress: Progress | None, report: Report | None) -> Progress:
    """Прогресс «этап done/total» для CLI и тот же ход для процентов (счётчики — по запросам, в score_pool)."""
    def call(stage: str, done: int, total: int) -> None:
        if progress:
            progress(stage, done, total)
        if report and stage != "counters":
            report(stage, done, total)
    return call


def budget_seconds(value: float | None = None) -> float:
    """Бюджет прогона в секундах: аргумент, иначе env TIME_BUDGET_S, иначе TIME_BUDGET_S; не меньше нуля."""
    if value is None:
        value = os.environ.get("TIME_BUDGET_S") or TIME_BUDGET_S
    return max(0.0, float(value))


def staged(name: str, timings: dict, progress: Progress, action: Callable):
    """Выполняет шаг, пишет его время и прогресс 0/1 -> 1/1."""
    mark = time.monotonic()
    progress(name, 0, 1)
    value = action()
    timings[name] = round(time.monotonic() - mark, 2)
    progress(name, 1, 1)
    return value


def named_count(named: Sequence[dict]) -> int:
    """Названных кандидатов после слияния — как stats.candidates_named."""
    return len(merge_subtopics(merge_candidates(named)))


def extra_round(number: int, state: dict, collector: DocumentCollector, openalex, progress: Progress) -> dict:
    """Раунд добора (Х2): новые подзапросы -> поиск -> шаг 4 по новым документам -> след новых терминов.

    state — накопленное за прогон (subqueries, all_documents, documents шага 4, raw, named, unnamed, warnings)
    и параметры (topic, query_id, sources, extract_model, subq_cache, extract_cache); дописывается на месте.
    doc_ids новых кандидатов сдвигаются на число прежних документов шага 4. Возвращает строку stats.rounds.
    """
    mark = time.monotonic()
    subq = generate_subqueries(state["topic"], state["query_id"], use_cache=state["subq_cache"],
                               round_=number, used=state["subqueries"])
    state["subqueries"] += subq["subqueries"]
    seen = {doc["url"] for doc in state["all_documents"]}
    fresh = [doc for doc in collector.search_recent(subq["subqueries"]).to_dict()["documents"]
             if doc["url"] not in seen]
    state["all_documents"] += fresh
    usable = [doc for doc in fresh if state["sources"] is None or origin(doc, state["subqueries"]) in state["sources"]]
    picked = step4_documents(usable, state["subqueries"])
    found = (extract_terms(picked, state["topic"], state["query_id"], model=state["extract_model"],
                           use_cache=state["extract_cache"]) if picked else {"candidates": [], "warnings": []})
    offset = len(state["documents"])
    state["documents"] += picked
    raw = [{**item, "doc_ids": [i + offset for i in item.get("doc_ids") or []]} for item in found["candidates"]]
    state["raw"] += raw
    known = {tech_key(item["name_en"]) for item in state["named"] + state["unnamed"]}
    by_key = {tech_key(item["name_en"]): item for item in state["named"]}
    named, unnamed = direct_candidates([item for item in raw if tech_key(item["name_en"]) not in known],
                                       openalex, progress)
    again = [{**by_key[tech_key(item["name_en"])], "doc_ids": item["doc_ids"]} for item in raw
             if tech_key(item["name_en"]) in by_key]  # известный термин: только документы к нему
    before = named_count(state["named"])
    state["named"] += named + again
    state["unnamed"] += unnamed
    state["warnings"] += subq.get("warnings", []) + found["warnings"]
    return {"round": number, "subqueries": [item["text"] for item in subq["subqueries"]],
            "documents_new": len(fresh), "documents_for_candidates": len(picked),
            "step4_topped_up": step4_topped_up(usable, state["subqueries"], picked),
            "candidates_found": len(raw), "named_gain": named_count(state["named"]) - before,
            "seconds": round(time.monotonic() - mark, 2)}


def add_rounds(rounds: list[dict], state: dict, collector: DocumentCollector, openalex, progress: Progress,
               elapsed: Callable[[], float], budget: float) -> None:
    """Раунды добора, пока их не больше MAX_ROUNDS, названных меньше TARGET_NAMED и прошло меньше 35 % бюджета.

    Сбой подзапросов раунда — предупреждение и конец добора.
    """
    while (len(rounds) <= MAX_ROUNDS and named_count(state["named"]) < TARGET_NAMED
           and elapsed() < ROUNDS_BEFORE_SHARE * budget):
        try:
            rounds.append(extra_round(len(rounds), state, collector, openalex, progress))
        except ValueError as exc:
            state["warnings"].append(f"раунд добора {len(rounds)} не выполнен: {exc}")
            return


def run_query(topic: str, area: str | None = None, *, use_cache: bool = True,
              progress: Progress | None = None, adapters: Sequence[SourceAdapter] | None = None,
              settings: Settings | None = None, candidate_sources: set[str] | None = None,
              extract_model: str | None = "yandexgpt-5-pro", counters_cache: bool | None = None,
              rospatent: bool | None = None, on_progress: Callable[[dict], None] | None = None,
              dedup: bool = True, query_id: str | None = None, time_budget: float | None = None,
              extract_cache: bool | None = None) -> dict:
    """Тема -> JSON с ТОП-15, исключёнными, статистикой и временем по шагам.

    candidate_sources — из документов каких источников извлекать кандидатов (openalex,
    openalex_ru, arxiv, techcrunch). None — из всех, как раньше. Поиск №1 и его статистика
    при этом по всем источникам; doc_ids кандидатов нумеруют только отобранные документы.
    extract_model — модель шага 4 (extract-v4, search/extract_terms.py); name_en кандидата — его термин.
    counters_cache — файловый кэш счётчиков отдельно от остальных кэшей; None — как use_cache.
    rospatent — очередь Роспатента в этапе счётчиков; None — как collector.rospatent.ROSPATENT_ENABLED.
    n_pat идёт в признак share_patent модели s2a2-v1; у кандидатов с оценкой — n_pat, share_patent,
    rospatent_failed (при сбое ещё note_ru). Выключенный при s2a2-v1 — предупреждение в warnings.
    Генерация кандидатов — v3 (задача Г): промпт subq-v4 (задача О3) и состав шага 4 из step4_documents.
    on_progress(event) — прогресс в процентах (pipeline/progress.py, задача И1); None — выход не меняется.
    dedup — склейка дублей перед отбором ТОП-15 (pipeline/dedup.py; решение команды — evidence/dedup_check.md).
    time_budget — бюджет прогона в секундах от старта (задача Х); None — env TIME_BUDGET_S или TIME_BUDGET_S (900).
    extract_cache — кэш шага 4 отдельно от остальных кэшей; None — как use_cache.
    Добор кандидатов (Х2): не больше MAX_ROUNDS раундов новых подзапросов, пока названных меньше TARGET_NAMED.
    """
    budget, stopped_at = budget_seconds(time_budget), None
    progress_warnings: list[str] = []
    report, finish = (tracker(on_progress, progress_warnings.append, budget=budget, tail=TIME_RESERVE_S)
                      if on_progress else (None, None))
    progress = both_progress(progress, report)
    settings = settings or Settings.from_env()
    adapters = list(adapters) if adapters is not None else default_adapters(settings)
    from model.bootstrap import ensure_artifact
    ensure_artifact()
    meta, started, timings = load()["meta"], time.monotonic(), {}
    query_id = query_id or f"q{datetime.now():%Y%m%d%H%M%S}"

    subq = staged("subqueries", timings, progress, lambda: generate_subqueries(topic, query_id, use_cache=use_cache))
    warnings = list(subq.get("warnings", []))
    searchers = [CachedSearch(adapter, use_cache=use_cache) for adapter in adapters]
    collector = DocumentCollector(
        adapters=searchers, cache=build_cache(settings.database_url), settings=settings
    )
    documents = staged("search", timings, progress,
                       lambda: collector.search_recent(subq["subqueries"]).to_dict()["documents"])
    all_documents = documents
    if candidate_sources is not None:
        documents = [doc for doc in documents if origin(doc, subq["subqueries"]) in candidate_sources]
    usable, documents = documents, step4_documents(documents, subq["subqueries"])
    extract_cache = use_cache if extract_cache is None else extract_cache
    found = staged("candidates", timings, progress,
                   lambda: extract_terms(documents, topic, query_id, model=extract_model, use_cache=extract_cache))
    warnings += found["warnings"] + ([EMPTY_AREA_WARNING] if not area else [])

    mark = time.monotonic()
    openalex = next(a for a in adapters if a.source == "openalex")
    named, unnamed = direct_candidates(found["candidates"], openalex, progress)
    timings["naming"] = round(time.monotonic() - mark, 2)
    rounds = [{"round": 0, "subqueries": [item["text"] for item in subq["subqueries"]],
               "documents_new": len(all_documents), "documents_for_candidates": len(documents),
               "step4_topped_up": step4_topped_up(usable, subq["subqueries"], documents),
               "candidates_found": len(found["candidates"]), "named_gain": named_count(named),
               "seconds": round(time.monotonic() - started, 2)}]
    state = {"topic": topic, "query_id": query_id, "sources": candidate_sources, "extract_model": extract_model,
             "subq_cache": use_cache, "extract_cache": extract_cache, "subqueries": list(subq["subqueries"]),
             "all_documents": list(all_documents), "documents": list(documents), "raw": list(found["candidates"]),
             "named": named, "unnamed": unnamed, "warnings": warnings}
    add_rounds(rounds, state, collector, openalex, progress, lambda: time.monotonic() - started, budget)
    subq["subqueries"], all_documents, documents = state["subqueries"], state["all_documents"], state["documents"]
    found["candidates"], named, unnamed = state["raw"], state["named"], state["unnamed"]
    merged = merge_subtopics(merge_candidates(named))
    dropped = [excluded(item, item["skipped_reason"], documents) for item in unnamed]
    dropped += [excluded(item, "bad_name", documents) for item in merged if not good_name(item["name_en"])]
    good = [item for item in merged if good_name(item["name_en"])]
    pool, capped = apply_cap(good, documents, limit=MAX_SCORED_HARD)
    options = rospatent_options(rospatent)
    if options is not None and not options["token"]:
        warnings.append(ROSPATENT_NO_KEY_WARNING)
    ranked, patents, scored, stop, schedule = score_pool(
        pool, area, adapters, settings, warnings, progress, timings,
        use_cache if counters_cache is None else counters_cache, options, report, budget=budget,
        elapsed=lambda: time.monotonic() - started, quota=lambda: fetch_module.openalex_quota())
    capped += [excluded(item, stop, documents) for item in pool if tech_key(item["name_en"]) not in scored]
    stopped_at = "counters" if stop == "time_budget" else stopped_at
    n_pats = {name: result["n_pat"] for name, result in patents.items() if not result["failed"]}
    ranked = with_rank_scores(ranked, n_pats)
    by_key = {tech_key(c["name_en"]): c for c in pool}
    duplicate_of = staged("dedup", timings, progress, lambda: find_duplicates(ranked, by_key, warnings)) if dedup else {}
    top, below = split_ranked(ranked, by_key, documents, meta["threshold"], n_pats, duplicate_of)
    top, below = move_weak_only(top, below)
    if options is not None:
        by_name = {item["name"]: item for item in ranked}
        for entry in top + below:
            if entry["score"] is not None:
                key = tech_key(entry["name_en"])
                entry.update(patent_fields(by_name[key]["counters"], patents.get(key)))
    elif "share_patent" in meta["features"]:
        warnings.append(ROSPATENT_OFF_WARNING)
    translation = staged("translate", timings, progress,
                         lambda: name_in_russian(top, dropped + capped + below, use_cache, progress))
    english = [entry["name_en"] for entry in top if entry.get("name_ru_source") == "name_en"]
    warnings += [TRANSLATE_FAILED_WARNING.format(", ".join(english))] if english else []
    mark, enriched = time.monotonic(), {"query_id": query_id, "top": top, "_documents": all_documents}
    enrich_top(enriched, collector, time_left=lambda: budget - (time.monotonic() - started), progress=progress,
               source_limit=MAX_SOURCES)
    timings["enrich"], all_documents = round(time.monotonic() - mark, 2), enriched["_documents"]
    warnings += [ENRICH_STOPPED_WARNING] if enriched["enrichment"] == "partial" else []
    mark = time.monotonic()
    insights_top(top, all_documents, progress=progress)  # описание на главном экране и карточка без ожидания LLM
    timings["insights"] = round(time.monotonic() - mark, 2)
    for entry in top + dropped + capped + below:
        entry["model_version"] = meta["model_version"]
    scores = [item["score"] for item in ranked if item["score"] is not None]
    timings["total"] = round(time.monotonic() - started, 2)
    if finish:
        finish()
    warnings += progress_warnings
    excluded_items = dropped + capped + below
    candidates = catalog_candidates(found["candidates"], top, excluded_items)
    persist_search_documents(query_id, all_documents, top + excluded_items)
    persist_features_and_scores(
        query_id, top + excluded_items, model_version=meta["model_version"], threshold=meta["threshold"]
    )
    return {
        "query_id": query_id, "topic": topic.strip(), "area": area,
        "model_version": meta["model_version"], "threshold": meta["threshold"], "cutoff_date": meta["cutoff_date"],
        "subqueries": subq["subqueries"],
        "candidate_sources": sorted(candidate_sources) if candidate_sources is not None else None,
        "extract_version": EXTRACT_VERSION, "extract_model": extract_model,
        "naming_mode": NAMING_MODE, "candidates_version": CANDIDATES_VERSION,
        "stats": {"documents_by_source": document_stats(all_documents, subq["subqueries"]),
                  "documents_for_candidates_by_source": document_stats(documents, subq["subqueries"]),
                  "documents_total": len(all_documents), "documents_for_candidates": len(documents),
                  "documents_analyzed": documents_analyzed(all_documents, ranked, n_pats),
                  "candidates_found": len(found["candidates"]),
                  "candidates_named": len(merged),
                  "candidates_scored": len(scores),
                  "above_threshold": sum(score >= meta["threshold"] for score in scores),
                  "above_075": sum(score >= HIGH_SCORE for score in scores),
                  "rospatent_enabled": options is not None,
                  "rospatent_failures": sum(result["failed"] for result in patents.values()),
                  "translation": translation,
                  "time_budget_s": budget, "elapsed_s": timings["total"], "stopped_at": stopped_at,
                  "counters_schedule": schedule, "rounds": rounds,
                  "documents_topped_up": sum(item["step4_topped_up"] for item in rounds)},
        "normalizer_deviations": list(naming.DEVIATIONS),
        "top": top, "excluded": excluded_items, "candidates": candidates,
        "enrichment": enriched["enrichment"],
        "_documents": all_documents,
        "timings": timings, "warnings": warnings,
    }
