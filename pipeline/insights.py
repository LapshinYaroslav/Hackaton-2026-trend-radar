"""Инсайт по клику: описание, преимущества и кейсы только из документов поиска."""

from __future__ import annotations

import json
import logging
import re
from typing import Any, Mapping, Sequence

from pipeline.weak_sources import is_weak_only, weak_source_note

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = (
    "Ты аналитик слабых технологических сигналов. Отвечай только по приложенным "
    "документам. Не выдумывай факты, числа модели и источники, которых нет в списке. "
    "Если данных мало — напиши об этом прямо. Ответ — JSON с полями description_ru "
    "(строка), advantages_ru (массив строк), cases_ru (массив объектов title, text, "
    "source_url), summaries (массив объектов url, summary_ru)."
)

GENERATED_NOTE = "сгенерировано"
TRANSLATED_NOTE = "перевод / сгенерировано"
NO_LLM_NOTE = "резюме по тексту документа, без языковой модели"
NO_LLM_TRANSLATE_NOTE = "перевод недоступен: нет ключа языковой модели, показано исходное резюме"


def documents_for(item: Mapping, documents: Sequence[Mapping] | None = None) -> list[dict]:
    """Документы карточки: сначала приложенные источники, затем поиск по URL/названию."""
    attached = [dict(source) for source in (item.get("sources") or [])]
    seen = {source.get("url") for source in attached if source.get("url")}
    name_en = (item.get("name_en") or "").casefold()
    for doc in documents or []:
        url = doc.get("url")
        if url and url in seen:
            for source in attached:
                if source.get("url") == url and not source.get("text"):
                    source["text"] = doc.get("text") or doc.get("body") or ""
            continue
        blob = " ".join(
            str(doc.get(key) or "") for key in ("title", "text", "body")
        ).casefold()
        if name_en and name_en in blob:
            attached.append(dict(doc))
            if url:
                seen.add(url)
    return attached


def _clip(text: str, limit: int = 400) -> str:
    text = re.sub(r"\s+", " ", (text or "").strip())
    if len(text) <= limit:
        return text
    return text[: limit - 1].rsplit(" ", 1)[0] + "…"


def _source_note(language: str, generated: bool, translated: bool) -> str:
    lang = (language or "").split("-")[0].lower()
    if translated:
        return TRANSLATED_NOTE
    if lang and lang != "ru":
        return NO_LLM_TRANSLATE_NOTE if generated else TRANSLATED_NOTE
    return GENERATED_NOTE if generated else NO_LLM_NOTE


def _fallback_summaries(docs: Sequence[Mapping], generated: bool = False) -> list[dict]:
    rows = []
    for doc in docs:
        language = str(doc.get("language") or "en")
        text = doc.get("text") or doc.get("body") or doc.get("title") or ""
        translated = False
        rows.append(
            {
                "title": doc.get("title"),
                "url": doc.get("url"),
                "published_at": doc.get("published_at"),
                "source": doc.get("source"),
                "source_type": doc.get("source_type"),
                "language": language,
                "trust_level": doc.get("trust_level"),
                "summary_ru": _clip(str(text)),
                "summary_note": _source_note(language, generated, translated),
                "translated": translated,
                "generated": generated,
            }
        )
    return rows


def extractive_insight(item: Mapping, documents: Sequence[Mapping] | None = None) -> dict[str, Any]:
    """Инсайт без LLM: короткие выдержки из заголовков и текстов документов."""
    docs = documents_for(item, documents)
    name = item.get("name_ru") or item.get("name_en") or "технология"
    titles = [str(doc.get("title") or "").strip() for doc in docs if doc.get("title")]
    snippets = [
        _clip(str(doc.get("text") or doc.get("body") or doc.get("title") or ""), 280)
        for doc in docs
        if doc.get("text") or doc.get("body") or doc.get("title")
    ]
    description = snippets[0] if snippets else f"По открытым документам описания «{name}» пока недостаточно."
    if item.get("explanation_ru"):
        description = (
            f"{description} Модель отмечает: " + "; ".join(item.get("explanation_ru") or [])
        )
    advantages = list(item.get("explanation_ru") or [])[:3]
    if not advantages and titles:
        advantages = [f"Есть публикации: {titles[0]}"]
    cases = []
    for doc, snippet in zip(docs, snippets):
        cases.append(
            {
                "title": doc.get("title") or "Документ",
                "text": snippet,
                "source_url": doc.get("url"),
            }
        )
    sources = _fallback_summaries(docs, generated=False)
    return {
        "description_ru": description,
        "advantages_ru": advantages,
        "cases_ru": cases[:5],
        "sources": sources,
        "generated": False,
        "model": None,
    }


def _parse_llm(text: str) -> dict[str, Any] | None:
    if not text:
        return None
    blob = text.strip()
    if blob.startswith("```"):
        blob = re.sub(r"^```(?:json)?\s*|\s*```$", "", blob, flags=re.I)
    try:
        data = json.loads(blob)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", blob, flags=re.S)
        if not match:
            return None
        try:
            data = json.loads(match.group(0))
        except json.JSONDecodeError:
            return None
    return data if isinstance(data, dict) else None


def _llm_insight(item: Mapping, docs: Sequence[Mapping]) -> dict[str, Any] | None:
    try:
        from search.llm_yandex_gpt import ask_llm
    except Exception as exc:  # модуль или ключ
        logger.info("insight llm unavailable: %s", exc)
        return None
    payload = []
    for doc in docs[:8]:
        payload.append(
            {
                "title": doc.get("title"),
                "url": doc.get("url"),
                "language": doc.get("language"),
                "source_type": doc.get("source_type"),
                "published_at": doc.get("published_at"),
                "text": _clip(str(doc.get("text") or doc.get("body") or ""), 1200),
            }
        )
    user = (
        f"Технология: {item.get('name_ru') or ''} / {item.get('name_en') or ''}\n"
        f"Документы:\n{json.dumps(payload, ensure_ascii=False)}"
    )
    try:
        answer = ask_llm(SYSTEM_PROMPT, user, purpose="insight", temperature=0.2, json_object=True)
    except Exception as exc:
        logger.info("insight llm failed: %s", exc)
        return None
    if answer.get("error") or not answer.get("text"):
        return None
    parsed = _parse_llm(answer["text"])
    if not parsed:
        return None
    summaries_by_url = {
        row.get("url"): row.get("summary_ru")
        for row in (parsed.get("summaries") or [])
        if isinstance(row, dict) and row.get("url")
    }
    sources = []
    for doc in docs:
        language = str(doc.get("language") or "en")
        translated = language.split("-")[0].lower() not in {"", "ru"}
        summary = summaries_by_url.get(doc.get("url")) or _clip(
            str(doc.get("text") or doc.get("body") or doc.get("title") or "")
        )
        sources.append(
            {
                "title": doc.get("title"),
                "url": doc.get("url"),
                "published_at": doc.get("published_at"),
                "source": doc.get("source"),
                "source_type": doc.get("source_type"),
                "language": language,
                "trust_level": doc.get("trust_level"),
                "summary_ru": summary,
                "summary_note": TRANSLATED_NOTE if translated else GENERATED_NOTE,
                "translated": translated,
                "generated": True,
            }
        )
    cases = []
    for row in parsed.get("cases_ru") or []:
        if isinstance(row, str):
            cases.append({"title": row, "text": row, "source_url": None})
        elif isinstance(row, dict):
            cases.append(
                {
                    "title": row.get("title") or "Кейс",
                    "text": row.get("text") or "",
                    "source_url": row.get("source_url") or row.get("url"),
                }
            )
    advantages = [str(item) for item in (parsed.get("advantages_ru") or []) if str(item).strip()]
    return {
        "description_ru": str(parsed.get("description_ru") or "").strip(),
        "advantages_ru": advantages,
        "cases_ru": cases,
        "sources": sources,
        "generated": True,
        "model": answer.get("model_version") or answer.get("model_uri"),
    }


def build_insight(
    item: Mapping,
    documents: Sequence[Mapping] | None = None,
    *,
    use_llm: bool = True,
) -> dict[str, Any]:
    """Полный инсайт карточки. LLM — если есть ключ, иначе выдержки из документов."""
    docs = documents_for(item, documents)
    payload = _llm_insight(item, docs) if use_llm else None
    if payload is None:
        payload = extractive_insight(item, docs)
    note = weak_source_note(item, docs)
    payload.update(
        {
            "name_ru": item.get("name_ru"),
            "name_en": item.get("name_en"),
            "rank": item.get("rank"),
            "weak_source_only": bool(note) or is_weak_only({**item, "sources": docs}),
            "weak_source_note": note,
            "status": "done",
        }
    )
    return payload
