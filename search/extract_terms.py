"""Шаг 4, версия extract-v2: документы -> устоявшиеся термины сообщества (задача К).

Механика прежнего шага 4 (search/extract_candidates.py) сохранена: фрагменты документов,
пачки, глобальная нумерация doc, кэш ответов, слияние синонимов, отбор по числу документов.
Меняются промпт и формат ответа: [{term_en, term_ru, quote, doc}]. Валидация кодом;
нарушения — в warnings, и пачка переспрашивается один раз при температуре 0.8 с перечнем
нарушений. Нормализатор после этого шага не вызывается: name_en = term_en.
"""
from __future__ import annotations

import json
import re
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Iterable

from search.extract_candidates import (BATCH_SIZE, MAX_TOKENS, _cache_get, _cache_key, _cache_put,
                                       build_user_prompt, dedupe_candidates, select_top, snippet_from_doc)
from search.llm_yandex_gpt import ask_llm, build_model_uri

PROMPT_VERSION = "extract-v2"
PROMPT_PATH = Path(__file__).resolve().parent / "prompts" / "extract_candidates_v2.txt"
FIRST_TEMPERATURE, RETRY_TEMPERATURE = 0.2, 0.8
MAX_PARALLEL_BATCHES = 10
FORBIDDEN_CHARS = '"(),:;/'
BAD_LAST_WORDS = {"integration", "optimization", "approach", "framework", "method", "methods",
                  "technique", "techniques"}
LATIN_RE, CYRILLIC_RE = re.compile(r"[a-zA-Z]"), re.compile(r"[а-яёА-ЯЁ]")
FENCE_RE = re.compile(r"^```(?:json)?\s*|\s*```$", re.IGNORECASE)


def squash(text: str) -> str:
    """Нижний регистр и одиночные пробелы: так сравниваются quote и текст документа."""
    return " ".join(str(text).casefold().split())


def parse_items(text: str | None) -> list:
    """JSON-массив из ответа; объект со списком внутри тоже принимается. Не разобрать — ValueError."""
    data = json.loads(FENCE_RE.sub("", (text or "").strip()).strip())
    if isinstance(data, dict):
        data = next((value for value in data.values() if isinstance(value, list)), None)
    if not isinstance(data, list):
        raise ValueError("ответ не JSON-массив")
    return data


def violation(item: Any, snippets: dict[int, str]) -> str | None:
    """Причина отказа одному объекту ответа или None."""
    if not isinstance(item, dict):
        return "не объект"
    term = str(item.get("term_en") or "").strip()
    words = term.split()
    if not 2 <= len(words) <= 4:
        return f"term_en «{term}»: слов {len(words)}, нужно 2–4"
    if not LATIN_RE.search(term) or CYRILLIC_RE.search(term) or any(c in term for c in FORBIDDEN_CHARS):
        return f"term_en «{term}»: не латиница или запрещённые символы"
    if words[-1].casefold() in BAD_LAST_WORDS:
        return f"term_en «{term}»: последнее слово {words[-1]}"
    try:
        doc = int(item.get("doc"))
    except (TypeError, ValueError):
        return f"term_en «{term}»: нет номера doc"
    if doc not in snippets:
        return f"term_en «{term}»: doc {doc} не из этой пачки"
    quote = str(item.get("quote") or "")
    if not quote.strip() or squash(quote) not in squash(snippets[doc]):
        return f"term_en «{term}»: quote не найден в документе {doc}"
    return None


def ask_batch(topic: str, batch: list[tuple[int, str]], model: str | None, use_cache: bool,
              temperature: float, note: str = "") -> tuple[list, str]:
    """Сырые объекты ответа на одну пачку и ошибка разбора (пустая строка — без ошибки)."""
    system = PROMPT_PATH.read_text(encoding="utf-8").strip() + note
    user = build_user_prompt(topic, batch)
    key = _cache_key(f"{system}\n---\n{user}\n---\n{temperature}", build_model_uri(model))
    cached = _cache_get(key) if use_cache else None
    if cached is not None:
        return list(cached.get("candidates") or []), ""
    answer = ask_llm(system, user, purpose=PROMPT_VERSION, temperature=temperature, json_object=False,
                     max_tokens=MAX_TOKENS, model=model)
    if answer["error"]:
        return [], f"вызов LLM не удался: {answer['error']}"
    try:
        items = parse_items(answer["text"])
    except (ValueError, json.JSONDecodeError) as exc:
        return [], f"разбор ответа не удался: {exc}"
    _cache_put(key, {"candidates": items, "model_uri": answer["model_uri"]})
    return items, ""


def check_batch(items: list, snippets: dict[int, str]) -> tuple[list[dict], list[str]]:
    """Годные объекты в формате кандидата шага 4 и перечень нарушений."""
    good, problems = [], []
    for item in items:
        reason = violation(item, snippets)
        if reason:
            problems.append(reason)
            continue
        term = " ".join(str(item["term_en"]).split())
        good.append({"doc": int(item["doc"]), "name_en": term, "name_ru": str(item.get("term_ru") or term).strip(),
                     "terms": [term], "context_terms": [], "quote": str(item["quote"]).strip()})
    return good, problems


def extract_batch(topic: str, batch: list[tuple[int, str]], model: str | None,
                  use_cache: bool) -> tuple[list[dict], list[str]]:
    """Одна пачка: вызов, проверка; при нарушениях — один повтор при 0.8 с их перечнем."""
    snippets = dict(batch)
    items, error = ask_batch(topic, batch, model, use_cache, FIRST_TEMPERATURE)
    good, problems = check_batch(items, snippets)
    warnings = [error] if error else []
    warnings += [f"отброшен: {p}" for p in problems]
    if error or problems:
        note = "\n\nПрошлый ответ нарушил требования: " + "; ".join((problems or [error])[:15]) + \
               ". Исправь и верни JSON-массив заново."
        retry_items, retry_error = ask_batch(topic, batch, model, use_cache, RETRY_TEMPERATURE, note)
        retry_good, retry_problems = check_batch(retry_items, snippets)
        seen = {(c["name_en"].casefold(), c["doc"]) for c in good}
        good += [c for c in retry_good if (c["name_en"].casefold(), c["doc"]) not in seen]
        warnings += ([retry_error] if retry_error else []) + [f"отброшен после повтора: {p}" for p in retry_problems]
    return good, warnings


def extract_terms(documents: Iterable[dict], topic: str, query_id: str = "q1", *, model: str | None = None,
                  batch_size: int = BATCH_SIZE, max_candidates: int = 10**6, use_cache: bool = True,
                  parallel: bool = True) -> dict:
    """Контракт шага 4 (candidates, warnings, stats) плюс quote у каждого кандидата."""
    docs = list(documents)
    snippets = [(i, text) for i, doc in enumerate(docs, start=1) if (text := snippet_from_doc(doc))]
    batches = [snippets[start:start + batch_size] for start in range(0, len(snippets), batch_size)]
    # Пачки параллельно (задача Л4.2): не больше MAX_PARALLEL_BATCHES, а ask_llm держит общий
    # семафор на 10 вызовов. map возвращает результаты в порядке пачек — выход тот же, что
    # у последовательного прогона. parallel=False — последовательно, для сверки.
    workers = MAX_PARALLEL_BATCHES if parallel else 1
    with ThreadPoolExecutor(max_workers=workers) as pool:
        results = list(pool.map(lambda batch: extract_batch(topic, batch, model, use_cache), batches))
    raw, warnings = [], []
    for good, batch_warnings in results:
        raw += good
        warnings += batch_warnings
    quotes = {}
    for item in raw:
        quotes.setdefault(item["name_en"], item["quote"])
    # dedupe_candidates при одном кандидате возвращает его без doc_ids — восстанавливаем из doc.
    merged = [{**c, "doc_ids": c.get("doc_ids") or [c["doc"]], "doc_count": c.get("doc_count") or 1}
              for c in dedupe_candidates(raw)]
    merged = select_top(merged, limit=max_candidates)
    candidates = [{"candidate_id": f"{query_id}-c{n}", "query_id": query_id, "name_ru": c["name_ru"],
                   "name_en": c["name_en"], "terms": c["terms"], "context_terms": [], "doc_ids": c["doc_ids"],
                   "doc_count": c["doc_count"], "quote": quotes.get(c["name_en"])}
                  for n, c in enumerate(merged, start=1)]
    return {"query_id": query_id, "topic": topic.strip(), "candidates": candidates, "warnings": warnings,
            "stats": {"documents": len(docs), "valid_mentions": len(raw), "returned": len(candidates),
                      "prompt_version": PROMPT_VERSION}}
