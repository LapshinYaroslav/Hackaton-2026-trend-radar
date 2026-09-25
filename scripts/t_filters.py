"""Задача Т: четыре фильтра выдачи после оценки модели, до отбора ТОП-15.

Функции чистые: файлов не читают, на вход получают термины, числа и документы.
Определения и списки слов заданы условием задачи заранее и не меняются.

Ф1 — зрелая технология по объёму научных публикаций.
Ф2 — дубли одного понятия по стеммированному ключу фразы.
Ф3 — шаблонное описание вместо названия.
Ф4 — повторяемость термина в документах пула запроса.
"""
from __future__ import annotations

import re
from collections.abc import Sequence

import numpy as np
import snowballstemmer

STEMMER = snowballstemmer.stemmer("english")

MOD = {"ai-driven", "ai-based", "ai-powered", "ai-enabled", "ai-assisted", "intelligent", "smart",
       "automated", "advanced", "novel", "efficient", "improved", "enhanced"}
HEAD = {"system", "systems", "strategy", "strategies", "framework", "frameworks", "approach",
        "approaches", "method", "methods", "methodology", "technique", "techniques", "solution",
        "solutions", "platform", "preparation", "process", "mechanism", "scheme"}

REASON_MATURE = ("Зрелая технология: {n} научных публикаций за 2020–2026 — больше, чем у 95% "
                 "известных слабых сигналов")
REASON_DUPLICATE = "Дубль понятия „{kept}“"
REASON_TEMPLATE = "Описание, а не название технологии"
REASON_SINGLE = "Термин встречается только в одном источнике — не подтверждён независимыми публикациями"


def plain_words(text: str) -> list[str]:
    """Нижний регистр, `-` и `/` → пробел, слова через схлопнутые пробелы."""
    return re.sub(r"[-/]", " ", (text or "").lower()).split()


def mature_threshold(n_research_signals: Sequence[float]) -> float:
    """T_mature: 95-й процентиль n_research обучающих сигналов, интерполяция linear."""
    return float(np.percentile(np.asarray(n_research_signals, dtype=float), 95, method="linear"))


def is_mature(n_research: float, threshold: float) -> bool:
    """Ф1: исключить, если научных публикаций строго больше порога."""
    return n_research > threshold


def phrase_key(term: str) -> tuple[str, ...]:
    """Ключ фразы для Ф2: слова после plain_words, каждое через Snowball."""
    return tuple(STEMMER.stemWords(plain_words(term)))


def _contains(outer: tuple[str, ...], inner: tuple[str, ...]) -> bool:
    """inner входит в outer как непрерывная последовательность слов."""
    size = len(inner)
    return any(outer[start:start + size] == inner for start in range(len(outer) - size + 1))


def are_duplicates(first: tuple[str, ...], second: tuple[str, ...]) -> bool:
    """Ф2: ключи совпадают или ключ от двух слов входит в другой."""
    if not first or not second:
        return False
    if first == second:
        return True
    short, long_ = sorted((first, second), key=len)
    return len(short) >= 2 and _contains(long_, short)


def duplicate_groups(terms: Sequence[str], scores: Sequence[float]) -> list[list[int]]:
    """Транзитивные группы дублей (индексы); в каждой группе первым идёт максимальный score."""
    keys = [phrase_key(term) for term in terms]
    parent = list(range(len(terms)))

    def root(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for i in range(len(terms)):
        for j in range(i + 1, len(terms)):
            if are_duplicates(keys[i], keys[j]):
                parent[root(j)] = root(i)
    groups: dict[int, list[int]] = {}
    for i in range(len(terms)):
        groups.setdefault(root(i), []).append(i)
    return [sorted(group, key=lambda i: (-scores[i], i)) for group in groups.values()]


def is_template(term: str) -> bool:
    """Ф3: есть слово из MOD и последнее слово из HEAD; дефис внутри слова сохраняется."""
    words = (term or "").lower().split()
    return bool(words) and words[-1] in HEAD and any(word in MOD for word in words)


def title_key(title: str) -> str:
    """Нормализованный заголовок: нижний регистр, без знаков препинания, схлопнутые пробелы."""
    return " ".join(re.sub(r"[^\w\s]|_", " ", (title or "").lower()).split())


def unique_documents(documents: Sequence[dict]) -> tuple[list[dict], int]:
    """Склейка документов пула по заголовку. Возвращает уникальные документы и сколько склеено.

    У склеенного документа текст — все тексты группы; автор — первый известный. Документ
    с пустым заголовком ни с чем не склеивается.
    """
    merged: dict[str, dict] = {}
    for number, doc in enumerate(documents):
        key = title_key(doc.get("title")) or f"\0{number}"
        authors = doc.get("authors") or []
        if key not in merged:
            merged[key] = {"title": doc.get("title") or "", "text": doc.get("text") or "",
                           "author": authors[0] if authors else None, "number": number}
        else:
            group = merged[key]
            group["text"] += " " + (doc.get("text") or "")
            group["author"] = group["author"] or (authors[0] if authors else None)
    return list(merged.values()), len(documents) - len(merged)


def phrase_pattern(term: str) -> re.Pattern | None:
    """Regex фразы для Ф4: границы слов, последнее слово допускает s/es. None для пустого."""
    words = plain_words(term)
    if not words:
        return None
    body = r"\s+".join(re.escape(word) for word in words) + r"(?:s|es)?"
    return re.compile(rf"(?<!\w){body}(?!\w)")


def repeat_counts(term: str, documents: Sequence[dict]) -> tuple[int, int]:
    """Ф4: d_pool — документов пула с фразой; d_authors — разных первых авторов среди них.

    documents — уже склеенные unique_documents. Документ без автора считается отдельным
    автором: без авторов d_authors = d_pool.
    """
    pattern = phrase_pattern(term)
    if pattern is None:
        return 0, 0
    found = [doc for doc in documents
             if pattern.search(" ".join(plain_words(f"{doc['title']} {doc['text']}")))]
    authors = {doc["author"] if doc["author"] else ("\0", doc["number"]) for doc in found}
    return len(found), len(authors)
