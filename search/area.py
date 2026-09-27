"""Тема запроса → одна из шести областей модели, либо «Другое».

Сначала спрашиваем YandexGPT. Если вызов не удался или ответ не из списка,
смотрим ключевые слова. «Другое» возвращается как None: так API уже отдаёт
незнакомую область.
"""

from __future__ import annotations

from search.llm_yandex_gpt import ask_llm

AREAS = (
    "Edge",
    "Защита ИИ",
    "Индустриальный ИИ",
    "Инфраструктура ИИ",
    "Роботы",
    "Финтех",
)

_FAILED = object()

# Более узкие отрасли раньше общих: «промышленные роботы» — это Роботы.
_KEYWORDS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("Роботы", ("робот", "robot", "манипулятор", "cobot", "кобот", "гуманоид")),
    ("Финтех", ("финтех", "fintech", "платёж", "платеж", "банк", "страхов", "трейдинг")),
    ("Edge", ("edge", "периферийн", "краевые вычислен", "on-device")),
    (
        "Защита ИИ",
        ("защита ии", "защита моделей", "adversarial", "prompt injection", "отравлен"),
    ),
    (
        "Инфраструктура ИИ",
        ("инфраструктур", "дата-центр", "датацентр", "ускорител", "gpu"),
    ),
    ("Индустриальный ИИ", ("индустриальн", "станк", "предиктивн")),
)

_PROMPT = """Отнеси тему к одной области. Ответь строго одним значением из списка, без пояснений:
Edge
Защита ИИ
Индустриальный ИИ
Инфраструктура ИИ
Роботы
Финтех
Другое

Edge — вычисления и ИИ на устройстве, у края сети.
Защита ИИ — безопасность самих моделей и атаки на них.
Индустриальный ИИ — ИИ на производстве, в цехе, на станках, без роботов как главного предмета.
Инфраструктура ИИ — дата-центры, ускорители, платформы обучения и инференса.
Роботы — робототехника, манипуляторы, мобильные роботы.
Финтех — финансы, платежи, банки, страхование.
Другое — медицина, энергетика и любая тема не из списка."""


def parse_area(text: str) -> str | None | object:
    """Строка модели → область, None для «Другое» или _FAILED, если это не ответ."""
    line = (text or "").strip().splitlines()[0].strip().strip(".").strip("\"'«»") if text else ""
    folded = line.casefold()
    if folded in {"другое", "other"}:
        return None
    for area in AREAS:
        if folded == area.casefold():
            return area
    return _FAILED


def area_from_keywords(topic: str) -> str | None:
    """Первое подошедшее правило. Нет совпадений — None, то есть «Другое»."""
    folded = (topic or "").casefold()
    for area, words in _KEYWORDS:
        if any(word in folded for word in words):
            return area
    return None


def detect_area(topic: str, *, use_llm: bool = True) -> str | None:
    """Область для нормировки модели. None значит «Другое»."""
    if use_llm:
        found = _from_llm(topic)
        if found is not _FAILED:
            return found
    return area_from_keywords(topic)


def _from_llm(topic: str):
    try:
        answer = ask_llm(
            _PROMPT,
            f"Тема: {topic.strip()}",
            purpose="area",
            temperature=0.0,
            json_object=False,
            max_tokens=20,
        )
    except (ValueError, OSError):
        return _FAILED
    if answer.get("error"):
        return _FAILED
    return parse_area(answer.get("text") or "")
