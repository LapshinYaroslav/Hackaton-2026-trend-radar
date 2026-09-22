"""Сборка булева запроса из терминов. Модель кавычек не пишет — их ставит код.

Один и тот же текст запроса принимают OpenAlex и arXiv: проверено живыми вызовами
20.09.2026, ("a" OR "b") AND ("c" OR "d") и (all:"a" OR all:"b") AND (...) дают
у arXiv одинаковые 437 документов — фразу без префикса он трактует как all:.
TechCrunch булеву логику не понимает, там запрос не собирается вовсе.

Так модель не может ни сломать JSON экранированными кавычками (итерация 2: она теряла
закрывающую кавычку значения query), ни сложить два разных понятия через OR (итерация 2:
из-за этого v3 дала 89% запросов выше коридора).
"""
import re

MIN_TERMS = 2
MAX_TERMS = 4
QUOTES = "\"'«»“”„"
BRACKETS = "()[]{}"
OPERATOR_RE = re.compile(r"\b(?:AND|OR|NOT)\b")


def clean_terms(terms, field: str) -> tuple[list[str], list[dict]]:
    """Отбраковывает негодные термины. Возвращает годные (не больше MAX_TERMS) и нарушения.

    Термин с кавычкой отбраковывается, а не чинится: в синтаксисе OpenAlex нет
    документированного способа экранировать кавычку внутри фразы.
    """
    if not isinstance(terms, list):
        return [], [_violation("no_field", f"{field}: не список")]
    good: list[str] = []
    found: list[dict] = []
    for term in terms:
        text = str(term).strip() if isinstance(term, str) else ""
        if not text:
            found.append(_violation("empty_term", f"{field}: пустой термин"))
        elif any(char in text for char in QUOTES):
            found.append(_violation("quote_in_term", f"{field}: {text}"))
        elif any(char in text for char in BRACKETS) or OPERATOR_RE.search(text):
            found.append(_violation("operator_in_term", f"{field}: {text}"))
        elif text.casefold() not in {item.casefold() for item in good}:
            good.append(text)
    if len(good) > MAX_TERMS:
        found.append(_violation("too_many_terms", f"{field}: {len(good)} > {MAX_TERMS}"))
        good = good[:MAX_TERMS]
    return good, found


def build_query(terms, context_terms) -> tuple[str, list[dict]]:
    """Собирает ("t1" OR "t2") AND ("c1" OR "c2"). Если собрать нельзя — пустая строка.

    Пустой список контекста — это не отсутствие поля, а осознанный отказ от второго
    блока: запрос становится одним OR-блоком синонимов. Так работает поиск №2 после
    задачи 3, где контекст брался от колонки area и работал жёстким фильтром: у QKD
    из области «Финтех» он требовал от статьи ещё и слов про финансы и срезал объём
    в семь раз. Отсутствующее поле (None) по-прежнему ошибка: это сломанный ответ,
    а не решение.
    """
    main, found = clean_terms(terms, "terms")
    if not main:
        if isinstance(context_terms, list) and not context_terms:
            found.append(_violation("no_field", "terms: не осталось годных терминов"))
            return "", found
        context, context_found = clean_terms(context_terms, "context_terms")
        found += context_found
        found.append(_violation("no_field", "terms: не осталось годных терминов"))
        return "", found
    if isinstance(context_terms, list) and not context_terms:
        # Непригодны термины, среди которых нет ни одного непустого — это уже проверено
        # выше. Одного годного термина достаточно: после задачи 3В в запрос идёт ровно
        # одно каноническое название технологии, и второго блока у него нет по замыслу.
        return _block(main), found

    context, context_found = clean_terms(context_terms, "context_terms")
    found += context_found
    if len(main) < MIN_TERMS:
        found.append(_violation("too_few_terms", f"terms: {len(main)} < {MIN_TERMS}"))
    if len(context) < MIN_TERMS:
        # Второй блок запрошен, но не собрался: запрос описывал бы область, а не технологию.
        found.append(_violation("no_context_terms", f"context_terms: {len(context)} < {MIN_TERMS}"))
        return "", found
    return f"{_block(main)} AND {_block(context)}", found


def _block(terms: list[str]) -> str:
    """Один OR-блок из синонимов одного понятия."""
    return "(" + " OR ".join(f'"{term}"' for term in terms) + ")"


def _violation(rule: str, detail: str) -> dict:
    return {"rule": rule, "detail": detail, "subtopic": ""}
