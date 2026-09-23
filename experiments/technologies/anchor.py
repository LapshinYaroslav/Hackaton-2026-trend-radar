"""Шаг 1Б: существует ли фраза вообще. Один поисковый вызов OpenAlex на технологию.

Порог ровно нуль и никакой другой. Ноль — механический признак несуществующего
словосочетания: у любой технологии, про которую методолог написал строку, есть хотя бы
одна публикация за шесть лет. Любой ненулевой порог стал бы фильтром по объёму
документов, а объём коррелирует с меткой класса — под него первыми попали бы самые
ранние сигналы, то есть лучшие представители положительного класса.

Объём якоря записывается в отчёт справочно и критерием отбраковки по величине не служит.
Проверка делается до сбора счётчиков: поймать плохое название за один вызов дешевле,
чем за семь.
"""
import json

from collector.adapters.openalex import OpenAlexAdapter
from collector.constants import COLLECTION_START, CUTOFF_DATE
from model.dataset import ROOT

CACHE_FILE = ROOT / "data" / "interim" / "collector" / "cache" / "anchor_volumes.json"


def load_cache() -> dict[str, int]:
    if not CACHE_FILE.exists():
        return {}
    try:
        return json.loads(CACHE_FILE.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def save_cache(cache: dict[str, int]) -> None:
    CACHE_FILE.parent.mkdir(parents=True, exist_ok=True)
    CACHE_FILE.write_text(json.dumps(cache, ensure_ascii=False, indent=2), encoding="utf-8")


def anchor_volume(name_en: str, adapter: OpenAlexAdapter, cache: dict[str, int]) -> int:
    """Сколько статей OpenAlex содержит эту фразу за весь период сбора."""
    key = name_en.strip().casefold()
    if key in cache:
        return cache[key]
    payload = adapter._get_works(search=f'"{name_en}"', date_from=COLLECTION_START,
                                 date_to_exclusive=CUTOFF_DATE, per_page=1, cursor=None,
                                 type_filter=adapter.type_filter)
    cache[key] = int((payload.get("meta") or {}).get("count") or 0)
    return cache[key]


def retry_note(names: list[str]) -> str:
    """Что сказать нормализатору, если ни одна из фраз не встречается в литературе."""
    listed = ", ".join(names)
    return (f"ни одна из фраз {listed} не встречается в научной литературе, "
            f"дай более употребительные названия той же технологии")
