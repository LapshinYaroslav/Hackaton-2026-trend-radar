"""Склейка дублей в выдаче (задачи К, К5): после ранжирования, до отбора ТОП-15 и перевода.

Ключ термина — множество основ Snowball после нормализации: нижний регистр; «-», «/», «_» — пробел; прочая
пунктуация убирается; убираются фраза «artificial intelligence» и стоп-слова STOP_WORDS.
Дубль: ключи равны и не пусты — без вызова; иначе при общей основе ключей решает YandexGPT (yandexgpt-5-pro,
температура 0, промпт PROMPT дословно, кэш по паре). Ошибка или таймаут YandexGPT — пара «не дубль».
Проход: по убыванию score; дубль уже принятого кандидата не занимает место и попадает в его variants.
Правило проверки на разметке не выполнено; склейка включена решением команды — evidence/dedup_check.md.
"""
from __future__ import annotations

import hashlib
import json
import re
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Callable, Sequence

import snowballstemmer

ROOT = Path(__file__).resolve().parents[1]
STOP_WORDS = frozenset({"ai", "driven", "powered", "enhanced", "based", "assisted", "enabled", "using", "for", "of",
                        "the", "a", "an", "with", "and", "in", "on", "to", "system", "systems", "approach",
                        "approaches", "method", "methods", "framework", "frameworks", "tool", "tools"})
LLM_MODEL, WORKERS = "yandexgpt-5-pro", 4
CACHE_DIR = ROOT / "data" / "interim" / "cache" / "dedup_pairs"
PROMPT = ("Термин 1: «{a}». Термин 2: «{b}». Это одна и та же технология, названная по-разному (синонимы, другой "
          "порядок слов, общие слова вроде AI-driven, system, based)? Ответ «нет», если один термин — частный случай, "
          "разновидность, применение или более широкий класс другого, или если это разные технологии. Ответь одним "
          "словом: да или нет.")
LLM_FAILED = "склейка дублей: YandexGPT не ответил по {} парам — они не склеены"
_STEMMER = snowballstemmer.stemmer("english")


def term_key(term_en: str) -> frozenset[str]:
    """Множество основ термина без стоп-слов и фразы «artificial intelligence»."""
    text = re.sub(r"[-/_]", " ", (term_en or "").lower())
    text = re.sub(r"[^\w\s]", "", text).replace("artificial intelligence", " ")
    return frozenset(_STEMMER.stemWords([w for w in text.split() if w not in STOP_WORDS]))


def pair_path(a: str, b: str) -> Path:
    """Файл кэша пары: порядок терминов не важен."""
    first, second = sorted((a.strip(), b.strip()))
    digest = hashlib.sha256(json.dumps([first, second], ensure_ascii=False).encode("utf-8")).hexdigest()
    return CACHE_DIR / f"{digest}.json"


def ask_pair(a: str, b: str, llm: Callable) -> bool | None:
    """Ответ YandexGPT по паре (кэш): True — «да», False — «нет» или любой другой ответ, None — ошибка API
    (не кэшируется). Промпт дословно — системное сообщение (пустое API отклоняет), пара — пользовательское."""
    path = pair_path(a, b)
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))["same"]
    first, second = sorted((a.strip(), b.strip()))
    try:
        answer = llm(PROMPT.format(a=first, b=second), f"«{first}» — «{second}»", purpose="dedup-pair",
                     temperature=0.0, json_object=False, max_tokens=5, model=LLM_MODEL)
    except Exception:  # таймаут или сбой клиента — как ошибка API
        return None
    if answer["error"]:
        return None
    same = re.sub(r"[^а-яё]", "", (answer["text"] or "").lower()) == "да"
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"a": first, "b": second, "answer": answer["text"], "same": same}, ensure_ascii=False),
                    encoding="utf-8")
    return same


def decider(terms: Sequence[str], llm: Callable | None = None) -> tuple[Callable[[str, str], bool], list[str]]:
    """Решение «дубль» для терминов выдачи и предупреждения. Пары с общей основой, но разными ключами —
    YandexGPT заранее (до WORKERS параллельно); равные ключи — без вызова."""
    keys = {t: term_key(t) for t in dict.fromkeys(terms)}
    names = list(keys)
    pairs = [(a, b) for i, a in enumerate(names) for b in names[i + 1:]
             if keys[a] != keys[b] and keys[a] & keys[b]]
    if pairs and llm is None:
        from search.llm_yandex_gpt import ask_llm as llm
    with ThreadPoolExecutor(max_workers=WORKERS) as pool:
        answers = dict(zip(pairs, pool.map(lambda p: ask_pair(p[0], p[1], llm), pairs)))
    failed = sum(answer is None for answer in answers.values())

    def decide(a: str, b: str) -> bool:
        if keys[a] == keys[b]:
            return bool(keys[a])
        return bool(answers.get((a, b)) or answers.get((b, a)))
    return decide, [LLM_FAILED.format(failed)] if failed else []


def dedup(ranked: Sequence[dict], decide: Callable[[str, str], bool], name: str = "name_en") -> dict[int, int]:
    """{индекс дубля: индекс принятого} в ranked (по убыванию score); кандидат без score не участвует."""
    kept: list[int] = []
    duplicate_of: dict[int, int] = {}
    for index, item in enumerate(ranked):
        if item.get("score") is None:
            continue
        home = next((i for i in kept if decide(item[name], ranked[i][name])), None)
        if home is None:
            kept.append(index)
        else:
            duplicate_of[index] = home
    return duplicate_of
