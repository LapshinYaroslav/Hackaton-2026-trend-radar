"""Полный список найденных имён для UI: ТОП, исключённые и остальные кандидаты."""

from __future__ import annotations

from typing import Iterable, Mapping


def _key(name_en: str | None, name_ru: str | None = None) -> str:
    return " ".join((name_en or name_ru or "").split()).casefold()


def _row(
    *,
    name_ru: str | None,
    name_en: str | None,
    stage: str,
    score: float | None = None,
    skipped_reason: str | None = None,
    reason_ru: str | None = None,
) -> dict:
    return {
        "name_ru": name_ru,
        "name_en": name_en,
        "stage": stage,
        "score": score,
        "skipped_reason": skipped_reason,
        "reason_ru": reason_ru,
    }


def catalog_candidates(
    found: Iterable[Mapping],
    top: Iterable[Mapping],
    excluded: Iterable[Mapping],
) -> list[dict]:
    """Все имена поиска: сначала ТОП и исключённые, затем оставшиеся найденные."""
    seen: set[str] = set()
    rows: list[dict] = []

    def add(item: Mapping, stage: str) -> None:
        name_en = item.get("name_en") or item.get("name")
        name_ru = item.get("name_ru")
        key = _key(name_en, name_ru)
        if not key or key in seen:
            return
        seen.add(key)
        rows.append(
            _row(
                name_ru=name_ru,
                name_en=name_en,
                stage=stage,
                score=item.get("score"),
                skipped_reason=item.get("skipped_reason"),
                reason_ru=item.get("reason_ru"),
            )
        )

    for item in top:
        add(item, "top")
    for item in excluded:
        add(item, "excluded")
    for item in found:
        add(item, "found")
    return rows
