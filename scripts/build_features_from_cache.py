"""Таблица признаков из файлового кэша счётчиков, без единого сетевого вызова.

Сеть исключена не сроком годности кэша, а тем, что DocumentCollector здесь не
создаётся вовсе: probe_source_totals вызывать некому. Счётчики читаются прямо из
JSON-файлов кэша, корпусные итоги — из копии в data/experiments/, чтобы истечение
суточного TTL оригинала ни на что не влияло.

Формулы признаков не дублируются: используются model.counters.aggregate_windows и
model.features.compute_features, те же самые, что в режиме запроса.

В таблицу попадают только строки technologies.csv, у которых в кэше есть счётчики
ровно под их набор терминов (совпадение tech_key И terms_hash).

Строки с одинаковой парой (tech_key, terms_hash) — это одна технология, записанная
в датасете в разных областях. Счётчики у них не похожие, а буквально одни и те же,
поэтому вектор признаков совпадает до последнего знака. В таблице остаётся первая
такая строка: вторая копия не добавила бы информации, зато одна и та же точка попала
бы и в обучение, и в проверку.

Запуск из корня проекта: python -m scripts.build_features_from_cache
"""
import json
from datetime import date
from pathlib import Path

import pandas as pd

from collector.api import terms_hash
from collector.models import SearchTerms
from model.corpus import load_training_totals
from model.counters import aggregate_windows
from model.features import FEATURE_NAMES, compute_features
from training.dataset import load_groups

ROOT = Path(__file__).resolve().parents[1]
TECHNOLOGIES = ROOT / "data" / "interim" / "technologies.csv"
COUNTERS_DIR = ROOT / "data" / "interim" / "collector" / "cache" / "counters"
OUTPUT_DIR = ROOT / "data" / "experiments"

# Источники, которые обязаны ответить по всем рабочим окнам: неполное покрытие —
# отказ считать признаки, а не тихий пропуск (model.features.check_coverage).
EXPECTED_SOURCES = {"openalex", "arxiv", "techcrunch"}
COLUMNS = ["tech_id", "name_en", "area", "label", "negative_type", "group", *FEATURE_NAMES]


def load_counter_cache() -> dict[tuple[str, str], pd.DataFrame]:
    """Счётчики из всех файлов кэша: ключ (tech_key, terms_hash), значение — таблица.

    Файл заведён на пару (технология, источник), поэтому на один ключ приходится
    несколько файлов, и строки складываются, а не замещают друг друга.
    """
    rows: dict[tuple[str, str], list[dict[str, object]]] = {}
    for path in sorted(COUNTERS_DIR.glob("*.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        key = (payload["tech_key"], payload["terms_hash"])
        rows.setdefault(key, []).extend(
            {"source": row["source"], "window": row["window"], "n": row["n"]}
            for row in payload["counters"])
    return {key: pd.DataFrame(items, columns=["source", "window", "n"])
            for key, items in rows.items()}


def group_of(tech_id: str, groups: dict[int, str]) -> str:
    """Группа дубликатов: у сигнала из signal_groups.csv — её имя, иначе сам tech_id.

    Негатив всегда сам себе группа: пары pair_with в группы не сводятся, это открытый
    вопрос plan_160.md, и решать его молча здесь нельзя.
    """
    if tech_id.startswith("s") and tech_id[1:].isdigit():
        return groups.get(int(tech_id[1:]), tech_id)
    return tech_id


def build_table() -> tuple[pd.DataFrame, list[str], list[str]]:
    """Таблица признаков, список tech_id без кэша и список отброшенных копий."""
    table = pd.read_csv(TECHNOLOGIES, encoding="utf-8-sig")
    cache = load_counter_cache()
    totals = aggregate_windows(load_training_totals(), value_column="n_total")
    groups = load_groups()

    rows: list[dict[str, object]] = []
    missing: list[str] = []
    duplicates: list[str] = []
    seen: dict[tuple[str, str], str] = {}
    for _, row in table.iterrows():
        search = SearchTerms(terms=json.loads(row["terms"]),
                             context_terms=json.loads(row["context_terms"]), query="")
        key = (row["tech_key"], terms_hash(search))
        counters = cache.get(key)
        if counters is None:
            missing.append(str(row["tech_id"]))
            continue
        if key in seen:
            duplicates.append(f"{row['tech_id']} = {seen[key]}")
            continue
        seen[key] = str(row["tech_id"])
        features = compute_features(aggregate_windows(counters), totals,
                                    expected_sources=EXPECTED_SOURCES)
        rows.append({"tech_id": row["tech_id"], "name_en": row["name_en"],
                     "area": row["area"], "label": int(row["label"]),
                     "negative_type": row["negative_type"],
                     "group": group_of(str(row["tech_id"]), groups), **features})
    return pd.DataFrame(rows, columns=COLUMNS), missing, duplicates


def main() -> None:
    frame, missing, duplicates = build_table()
    output = OUTPUT_DIR / f"features_{date.today():%Y%m%d}.csv"
    frame.to_csv(output, index=False, encoding="utf-8-sig")

    print(f"строк с кэшем: {len(frame)}  без кэша: {len(missing)}")
    print(f"отброшено копий с теми же счётчиками: {duplicates or 'нет'}")
    print(f"сигналов {int(frame['label'].sum())}, негативов {int((frame['label'] == 0).sum())}")
    print("области: " + ", ".join(f"{k} {v}" for k, v in frame["area"].value_counts().items()))
    print("\nNaN по признакам:")
    for name in FEATURE_NAMES:
        print(f"  {name:16} {int(frame[name].isna().sum())}")
    print("\nгруппы больше одной строки: "
          f"{[g for g, n in frame['group'].value_counts().items() if n > 1] or 'нет'}")
    print(f"\nзаписано: {output.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
