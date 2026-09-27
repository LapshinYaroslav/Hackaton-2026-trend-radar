"""Правило ТЗ: блог и пресс-релиз не могут быть единственным основанием сигнала."""

from __future__ import annotations

from typing import Iterable, Mapping

from collector.constants import SOLE_SOURCE_WEAK_TYPES

WEAK_ONLY_REASON = "weak_sources_only"
WEAK_ONLY_TEXT = (
    "Единственные источники — блог или пресс-релиз. "
    "По ТЗ они не могут быть единственным основанием слабого сигнала"
)


def source_types_of(item: Mapping) -> set[str]:
    """Типы документов, которые уже приложены к кандидату."""
    return {
        str(source.get("source_type") or "").strip()
        for source in (item.get("sources") or [])
        if source.get("source_type")
    }


def is_weak_only(item: Mapping) -> bool:
    """Все известные типы источников входят в слабые (блог / пресс-релиз)."""
    types = source_types_of(item)
    return bool(types) and types <= set(SOLE_SOURCE_WEAK_TYPES)


def move_weak_only(top: list[dict], excluded: list[dict]) -> tuple[list[dict], list[dict]]:
    """ТОП с одними блогами/пресс-релизами уходит в исключённые, ранги пересчитываются."""
    kept: list[dict] = []
    extra: list[dict] = []
    for item in top:
        if is_weak_only(item):
            moved = {
                **item,
                "skipped_reason": WEAK_ONLY_REASON,
                "reason_ru": WEAK_ONLY_TEXT,
                "weak_source_only": True,
            }
            moved.pop("rank", None)
            extra.append(moved)
        else:
            kept.append({**item, "weak_source_only": False})
    for index, item in enumerate(kept, start=1):
        item["rank"] = index
    return kept, extra + list(excluded)


def weak_source_note(item: Mapping, sources: Iterable[Mapping] | None = None) -> str | None:
    """Текст для карточки, если после догрузки источников правило всё ещё срабатывает."""
    probe = dict(item)
    if sources is not None:
        probe["sources"] = list(sources)
    if is_weak_only(probe):
        return WEAK_ONLY_TEXT
    return None
