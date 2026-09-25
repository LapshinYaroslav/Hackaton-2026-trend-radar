"""Задача Н-диагностика, без сети: слепой лист остального пула и метаданные первого документа.

1. «Н2 пул слепо» — оценённые кандидаты трёх прогонов Н, которых нет в n_blind_key.csv; формат
   «Н слепо», id с N065, перемешано с seed 42; ключ — data/interim/n2_blind_key.csv.
2. data/interim/n_pool_meta.csv — для всех кандидатов с n_id: источник первого документа,
   подзапрос поиска №1 и его язык. В отчёт не выводится.

Первый документ — тот же, что в n_report.first_documents. Подзапрос: у финтеха — subquery_ids
документа; у кибербезопасности и ИИ — подзапросы прогона, в файле кэша поиска №1 которых лежит
этот документ (по url). Несколько подзапросов — через «; ».

Запуск: python -m scripts.n_pool
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

from collector.api import tech_key
from scripts import n_report
from scripts.t_report import append_t

N2_BLIND_KEY = n_report.ROOT / "data" / "interim" / "n2_blind_key.csv"
POOL_META = n_report.ROOT / "data" / "interim" / "n_pool_meta.csv"
FIRST_ID = 65


def pool(theme: str) -> pd.DataFrame:
    """Оценённые кандидаты темы без score: термины, цитата, tech_key; дубли tech_key — первый."""
    items = n_report.scored(n_report.run(n_report.THEMES[theme]))
    frame = pd.DataFrame([{"name_en": item["name_en"], "name_ru": item["name_ru"],
                           "quote": item["quote"], "tech_key": tech_key(item["name_en"])} for item in items])
    return frame.drop_duplicates("tech_key").reset_index(drop=True)


def subqueries(theme: str, doc: dict, cache_files: list[dict]) -> list[dict]:
    """Подзапросы прогона, из которых пришёл документ: по subquery_ids или по файлам кэша."""
    listed = n_report.run(n_report.THEMES[theme])["subqueries"]
    if doc["subquery_ids"] is not None:
        return [item for item in listed if item["subquery_id"] in doc["subquery_ids"]]
    texts = {c["query"] for c in cache_files
             if any(found.get("url") == doc["url"] for found in c["documents"])}
    return [item for item in listed if item["text"] in texts]


def meta_rows(theme: str, frame: pd.DataFrame, ids: dict, cache: list[dict],
              cache_files: list[dict]) -> list[dict]:
    """Строки n_pool_meta.csv для кандидатов темы, у которых есть n_id."""
    docs, _ = n_report.first_documents(theme, frame, cache)
    rows = []
    for doc, key in zip(docs, frame["tech_key"]):
        found = subqueries(theme, doc, cache_files)
        rows.append({"n_id": ids[(theme, key)], "тема": theme, "источник": doc["source"],
                     "подзапрос": "; ".join(item["text"] for item in found),
                     "язык подзапроса": "; ".join(sorted({item["language"] for item in found}))})
    return rows


def blind_sheet(new: list[tuple[str, str]], info: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    """«Н2 пул слепо»: новые (тема, tech_key), перемешаны с seed 42, id с N065."""
    order = [new[i] for i in np.random.default_rng(42).permutation(len(new))]
    sheet = pd.DataFrame([{"n_id": f"N{FIRST_ID + number:03d}", "тема": key[0], **info[key]}
                          for number, key in enumerate(order)],
                         columns=["n_id", "тема", "term_en", "term_ru", "quote", "заголовок", "url"])
    return sheet, sheet[["n_id", "тема", "term_en"]]


def main() -> None:
    """Слепой лист остального пула, ключ, метаданные первого документа; в печать — только числа."""
    key = pd.read_csv(n_report.N_BLIND_KEY, encoding="utf-8-sig")
    ids = {(theme, tech_key(term)): n_id for n_id, theme, term in key[["n_id", "тема", "term_en"]].itertuples(index=False)}
    cache = n_report.search_documents()
    cache_files = [json.loads(path.read_text(encoding="utf-8")) for path in sorted(n_report.SEARCH.glob("*.json"))]
    frames = {theme: pool(theme) for theme in n_report.THEMES}
    info, new = {}, []
    for theme, frame in frames.items():
        docs, _ = n_report.first_documents(theme, frame, cache)
        for doc, row in zip(docs, frame.to_dict("records")):
            info[(theme, row["tech_key"])] = {"term_en": row["name_en"], "term_ru": row["name_ru"],
                                              "quote": row["quote"], "заголовок": doc["заголовок"], "url": doc["url"]}
        new += sorted((theme, k) for k in frame["tech_key"] if (theme, k) not in ids)
    sheet, new_key = blind_sheet(new, info)
    ids.update({(theme, tech_key(term)): n_id for n_id, theme, term in new_key.itertuples(index=False)})
    meta = pd.DataFrame([row for theme, frame in frames.items()
                         for row in meta_rows(theme, frame, ids, cache, cache_files)])
    new_key.to_csv(N2_BLIND_KEY, index=False, encoding="utf-8-sig")
    meta.sort_values("n_id").to_csv(POOL_META, index=False, encoding="utf-8-sig")
    append_t({"Н2 пул слепо": sheet}, prefix="Н")
    print(f"размечено ранее (в n_blind_key): {len(key)}; новых в «Н2 пул слепо»: {len(sheet)} "
          f"{sheet['тема'].value_counts().to_dict()}; id {sheet['n_id'].min()}…{sheet['n_id'].max()}")
    print(f"n_pool_meta.csv: {len(meta)} строк; без подзапроса: {int((meta['подзапрос'] == '').sum())}; "
          f"с несколькими подзапросами: {int(meta['подзапрос'].str.contains('; ').sum())}")


if __name__ == "__main__":
    main()
