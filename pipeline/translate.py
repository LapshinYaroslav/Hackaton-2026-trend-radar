"""Перевод названий на русский (задача З): после отбора ТОП-15, для кандидатов ТОП-15 и оценённых исключённых.

ТЗ: аналитическая выдача на русском; у зарубежных источников — оригинальное название и русское с пометкой
об автоматическом переводе. На кандидата — один вызов YandexGPT (yandexgpt-5-pro, температура 0;
промпт дословно — системное сообщение, термин — пользовательское), кэш по term_en.
Проверки: есть кириллица; не больше 8 слов (слово латиницей — одно слово; 8 вместо 6 — решение Ярослава);
не совпадает с term_en; нет признаков отказа (REFUSAL_RE); каждое латинское слово — из term_en или аббревиатура
из первых букв его слов (latin_ok); нет кириллических аббревиатур кроме ИИ, ИТ, ЦОД, БПЛА, ДНК, РНК и смешения алфавитов
в части слова (alphabet_ok, задача К3). Первая буква — заглавная.
Не прошло — один повтор при 0.3; снова нет — name_ru = term_ru шага 4, если он проходит проверки, иначе None.
term_en (name_en кандидата) не меняется: по нему идут источники и признаки.
"""
from __future__ import annotations

import hashlib
import json
import re
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Callable, Sequence

ROOT = Path(__file__).resolve().parents[1]
CACHE_DIR = ROOT / "data" / "interim" / "cache" / "translate_names"
LLM_MODEL = "yandexgpt-5-pro"
TEMPERATURES = (0.0, 0.3)
MAX_WORDS, WORKERS = 8, 4  # 8 слов — решение Ярослава 26.09 (в ТЗ предела нет)
CYRILLIC_RE = re.compile(r"[а-яё]", re.IGNORECASE)
LATIN_RE = re.compile(r"[A-Za-z0-9]*[A-Za-z][A-Za-z0-9]*")  # слова через дефис — отдельно
MAX_ABBREVIATION = 5
# Признаки отказа или обёртки вместо названия (без учёта регистра): такой ответ отклоняется.
# Задача К3: части слов (через пробел, дефис или тире) — без кириллических аббревиатур кроме белого списка
# и без смешения кириллицы с латиницей или цифрами внутри одной части («А2А», «ДНТ»; «6G-сети» — можно).
PART_SPLIT_RE = re.compile(r"[\s\-‐‑–—]+")
PART_STRIP = "«»\"'“”„()[],.;!"
CYR_ABBR_RE = re.compile(r"^(?=(?:.*[А-ЯЁ]){2})[А-ЯЁ0-9]+$")
ABBR_WHITELIST = {"ИИ", "ИТ", "ЦОД", "БПЛА", "ДНК", "РНК"}  # ДНК, РНК — решение Ярослава (задача К)
REFUSAL_RE = re.compile(r"не могу|извините|к сожалению|не удалось|языковая модель|перевод:|название:|\?|:", re.IGNORECASE)
WRAPS = {"«": "»", '"': '"', "“": "”", "„": "“", "'": "'"}
PROMPT = ("Переведи на русский название класса технологий для аналитического отчёта: «{term_en}» (контекст: "
          "«{quote}»). Правила: 1) если есть устоявшийся русский термин — используй его (например: federated "
          "learning → федеративное обучение); 2) если устоявшегося русского термина нет — оставь английское ядро "
          "латиницей и поясни по-русски (например: vision-language-action model → VLA-модели «зрение–язык–действие»; "
          "MCP server → MCP-серверы); 3) это название класса, а не пересказ фразы; не больше 8 слов; без кавычек и "
          "точки. Ответ — только название.")


def clean(text: str | None) -> str:
    """Ответ модели без пробелов и точки в конце; кавычки снимаются, только если обрамляют всё название
    (закрывающая кавычка внутреннего пояснения «…» остаётся)."""
    text = (text or "").strip().rstrip(".").strip()
    return text[1:-1].strip() if wrapped(text) else text


def wrapped(text: str) -> bool:
    """Первая кавычка закрывается последним символом, а не раньше: «A» и «B» — не обрамление."""
    opening, closing = text[:1], WRAPS.get(text[:1])
    if len(text) < 2 or closing != text[-1]:
        return False
    depth = 0
    for position, char in enumerate(text):
        if char == opening and (opening != closing or position == 0):
            depth += 1
        elif char == closing:
            depth -= 1
            if depth == 0:
                return position == len(text) - 1
    return False


def passes(name_ru: str | None, term_en: str) -> bool:
    """Проверки названия: кириллица, ≤ 8 слов, не совпадает с term_en, нет признаков отказа, латиница только из term_en."""
    if not name_ru or not CYRILLIC_RE.search(name_ru) or REFUSAL_RE.search(name_ru):
        return False
    return (len(name_ru.split()) <= MAX_WORDS and name_ru.strip().lower() != term_en.strip().lower()
            and all(latin_ok(word, term_en) for word in LATIN_RE.findall(name_ru)) and alphabet_ok(name_ru))


def alphabet_ok(name_ru: str) -> bool:
    """Нет кириллических аббревиатур вне ABBR_WHITELIST и нет смешения кириллицы с латиницей или цифрами в части."""
    for part in (p.strip(PART_STRIP) for p in PART_SPLIT_RE.split(name_ru)):
        cyrillic = bool(CYRILLIC_RE.search(part))
        if cyrillic and re.search(r"[A-Za-z0-9]", part):
            return False
        if CYR_ABBR_RE.match(part) and part not in ABBR_WHITELIST:
            return False
    return True


def latin_ok(word: str, term_en: str) -> bool:
    """Латинское слово name_ru есть в term_en (без регистра) или это аббревиатура до 5 заглавных букв из первых
    букв подряд идущих слов term_en, начиная с первого: VLA из vision-language-action, но не OF из order fairness."""
    words = [w.lower() for w in LATIN_RE.findall(term_en)]
    if word.lower() in words:
        return True
    initials = "".join(w[0] for w in words).upper()
    return word.isupper() and word.isalpha() and len(word) <= MAX_ABBREVIATION and initials.startswith(word)


def capitalized(name: str | None) -> str | None:
    """Первая буква заглавная; латинское слово в начале (аббревиатура, термин) не трогается."""
    if not name or not name[0].islower() or LATIN_RE.match(name):
        return name
    return name[0].upper() + name[1:]


def cache_path(term_en: str) -> Path:
    """Файл кэша перевода по term_en."""
    return CACHE_DIR / f"{hashlib.sha256(term_en.strip().encode('utf-8')).hexdigest()}.json"


def translate_one(term_en: str, quote: str | None, term_ru: str | None, llm: Callable,
                  use_cache: bool = True) -> dict:
    """name_ru, name_ru_source (translate | extract | None), name_ru_auto и число попыток (0 — из кэша)."""
    path = cache_path(term_en)
    if use_cache and path.exists():
        cached = json.loads(path.read_text(encoding="utf-8"))["answer"]
        if passes(cached["name_ru"], term_en):  # кэш прежних проверок: не прошедшее новые переводится заново
            return {**cached, "name_ru": capitalized(cached["name_ru"]), "attempts": 0}
    prompt = PROMPT.format(term_en=term_en, quote=(quote or "").strip())
    for attempt, temperature in enumerate(TEMPERATURES, start=1):
        answer = llm(prompt, term_en, purpose="translate-name", temperature=temperature, json_object=False,
                     max_tokens=60, model=LLM_MODEL)
        name = clean(answer["text"]) if not answer["error"] else ""
        if passes(name, term_en):
            result = {"name_ru": capitalized(name), "name_ru_source": "translate", "name_ru_auto": True}
            CACHE_DIR.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps({"term_en": term_en, "answer": result}, ensure_ascii=False), encoding="utf-8")
            return {**result, "attempts": attempt}
    fallback = capitalized(clean(term_ru)) if passes(clean(term_ru), term_en) else None
    return {"name_ru": fallback, "name_ru_source": "extract" if fallback else None, "name_ru_auto": True,
            "attempts": len(TEMPERATURES)}


def translate_entries(entries: Sequence[dict], llm: Callable | None = None, use_cache: bool = True,
                      on_done: Callable[[int], None] | None = None) -> dict:
    """Переводит name_ru записей выдачи на месте (до WORKERS параллельно); числа — для stats.

    on_done(done) — после каждого названия (прогресс, задача И1)."""
    if llm is None:
        from search.llm_yandex_gpt import ask_llm as llm
    done, lock = [0], threading.Lock()

    def one(entry: dict) -> dict:
        result = translate_one(entry["name_en"], entry.get("quote"), entry.get("name_ru"), llm, use_cache)
        with lock:
            done[0] += 1
            if on_done:
                on_done(done[0])
        return result
    with ThreadPoolExecutor(max_workers=WORKERS) as pool:
        results = list(pool.map(one, entries))
    for entry, result in zip(entries, results):
        entry.update({k: result[k] for k in ("name_ru", "name_ru_source", "name_ru_auto")})
    return {"переведено": len(results), "из кэша": sum(r["attempts"] == 0 for r in results),
            "с первого раза": sum(r["attempts"] == 1 and r["name_ru_source"] == "translate" for r in results),
            "с повтором": sum(r["attempts"] == 2 and r["name_ru_source"] == "translate" for r in results),
            "откат на term_ru": sum(r["name_ru_source"] == "extract" for r in results),
            "без названия": sum(r["name_ru_source"] is None for r in results)}
