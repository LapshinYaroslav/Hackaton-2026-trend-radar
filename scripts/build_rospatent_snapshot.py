"""Собирает evidence/rospatent_training_counts.json — n_pat 160 обучающих технологий из кэша П0.2.

Запускается один раз, вместе с переобучением s2a2-v1: Роспатент индексирует непрерывно, и
признак share_patent обучения читается только из этого файла (model.corpus.load_training_patents).
Фраза — термин технологии из data/interim/technologies.csv, запрос и датасеты — как в П0.2.

Запуск: python -m scripts.build_rospatent_snapshot
"""
from __future__ import annotations

import json

import pandas as pd

from collector import rospatent
from model.config import ROSPATENT_DATASETS
from model.corpus import TRAINING_PATENTS
from model.dataset import ROOT

TECHNOLOGIES = ROOT / "data" / "interim" / "technologies.csv"


def items() -> list[dict]:
    """Строка на технологию: tech_id, фраза, n_pat и дата ответа из кэша."""
    table = pd.read_csv(TECHNOLOGIES, encoding="utf-8-sig")
    rows = []
    for _, row in table.iterrows():
        phrase = json.loads(row["terms"])[0]
        path = rospatent.cache_path(rospatent.request_key(phrase, ROSPATENT_DATASETS))
        record = json.loads(path.read_text(encoding="utf-8"))
        rows.append({"tech_id": str(row["tech_id"]), "phrase": phrase,
                     "n_pat": int(record["response"]["total"]),
                     "fetched_at": record["fetched_at"]})
    return rows


def main() -> None:
    """Пишет снимок с описанием запроса."""
    rows = items()
    payload = {"source": "rospatent", "url": rospatent.URL, "field": "total",
               "query": "фраза в кавычках", "window": rospatent.WINDOW,
               "datasets": list(ROSPATENT_DATASETS),
               "collected_from": min(row["fetched_at"] for row in rows),
               "collected_to": max(row["fetched_at"] for row in rows), "items": rows}
    TRAINING_PATENTS.write_text(json.dumps(payload, ensure_ascii=False, indent=1) + "\n",
                                encoding="utf-8", newline="\n")
    print(f"{TRAINING_PATENTS.relative_to(ROOT)}: {len(rows)} технологий, "
          f"{len({row['phrase'] for row in rows})} фраз")


if __name__ == "__main__":
    main()
