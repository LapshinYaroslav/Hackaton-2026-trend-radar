"""Данные экрана из ответа пайплайна: проверка происхождения и строки для ТОП, исключённых, кандидатов, инсайта.

Без Streamlit, чтобы тесты сверяли экран с JSON run_query напрямую.
"""
from __future__ import annotations

REQUIRED = ("query_id", "topic", "model_version", "threshold", "cutoff_date", "subqueries", "stats", "top",
            "excluded", "timings", "extract_version", "candidates_version")
REQUIRED_STATS = ("documents_total", "candidates_found", "above_075")
TOP_N = 15


def check_result(data, query_id: str) -> dict:
    """Ответ run_query или ValueError со списком расхождений: данные не из пайплайна на экран не идут."""
    if not isinstance(data, dict):
        raise ValueError("ответ не из пайплайна: не JSON-объект")
    problems = [f"нет поля {key}" for key in REQUIRED if key not in data]
    problems += [f"нет stats.{key}" for key in REQUIRED_STATS if key not in (data.get("stats") or {})]
    if data.get("query_id") != query_id:
        problems.append(f"query_id {data.get('query_id')!r} вместо {query_id!r}")
    top = data.get("top") or []
    if len(top) > TOP_N:
        problems.append(f"в ТОП {len(top)} позиций, больше {TOP_N}")
    if [item.get("rank") for item in top] != list(range(1, len(top) + 1)):
        problems.append("ранги ТОП не идут подряд с 1")
    threshold = data.get("threshold")
    if isinstance(threshold, (int, float)):
        low = [item.get("name_en") for item in top
               if not isinstance(item.get("score"), (int, float)) or item["score"] < threshold]
        problems += [f"в ТОП score ниже порога: {name}" for name in low]
    if problems:
        raise ValueError("ответ не из пайплайна: " + "; ".join(problems))
    return data


def top_rows(data: dict) -> list[dict]:
    """ТОП как в терминале: ранг, названия, score, балл ранжирования (в старых прогонах = score), патенты."""
    return [{"rank": item["rank"], "name_ru": item.get("name_ru"), "name_en": item.get("name_en"),
             "score": item["score"], "rank_score": item.get("rank_score", item["score"]),
             "n_pat": item.get("n_pat")} for item in data["top"]]


def excluded_rows(data: dict) -> list[dict]:
    """Исключённые с кодом и текстом причины."""
    return [{"name_ru": item.get("name_ru"), "name_en": item.get("name_en"), "score": item.get("score"),
             "skipped_reason": item.get("skipped_reason"), "reason_ru": item.get("reason_ru")}
            for item in data["excluded"]]


def candidate_rows(data: dict) -> list[dict] | None:
    """Полный список имён из поля candidates; None — в ответе его нет (прогоны до каталога)."""
    if "candidates" not in data:
        return None
    return [{"stage": item.get("stage"), "name_ru": item.get("name_ru"), "name_en": item.get("name_en"),
             "score": item.get("score"), "reason_ru": item.get("reason_ru")} for item in data["candidates"]]


def stats_view(data: dict) -> dict:
    """Числа плашек и строки CLI «ТОП: N, исключено: M».

    documents_analyzed — поиск №1 + счётчики + патенты (run_query); в прогонах до этого поля — documents_total.
    """
    stats = data["stats"]
    return {"documents_analyzed": stats.get("documents_analyzed", stats["documents_total"]),
            "documents_total": stats["documents_total"], "candidates_found": stats["candidates_found"],
            "above_075": stats["above_075"], "top": len(data["top"]), "excluded": len(data["excluded"])}


def score_text(score) -> str:
    """Score тремя знаками, как в отчёте по прогону; нет оценки — прочерк."""
    return f"{score:.3f}" if isinstance(score, (int, float)) else "—"


def insight_view(payload: dict) -> dict | None:
    """Ответ /insights в формате pipeline.insights.build_insight; None — ещё готовится, ValueError — ошибка."""
    status = payload.get("status")
    if status == "error":
        raise ValueError(payload.get("error") or "ошибка генерации описания")
    if status != "done":
        return None
    return {"description": payload.get("description_ru") or "", "advantages": list(payload.get("advantages_ru") or []),
            "cases": list(payload.get("cases_ru") or []), "sources": list(payload.get("sources") or []),
            "weak_source_note": payload.get("weak_source_note")}


def progress_view(payload: dict) -> dict:
    """Полоса и подпись прогресса из GET /queries/{id}: доля 0..1 и «Этап · 37 % · осталось ~9 мин»."""
    total = max(int(payload.get("progress_total") or 100), 1)
    share = min(max(int(payload.get("progress_done") or 0) / total, 0.0), 1.0)
    text = f"{payload.get('progress_stage') or 'Запуск'} · {round(share * 100)} %"
    eta = payload.get("eta_s")
    if isinstance(eta, (int, float)):
        text += f" · осталось ~{-(-int(eta) // 60)} мин" if eta >= 60 else " · осталось меньше минуты"
    return {"share": share, "text": text}
