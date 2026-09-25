"""Задача Г3, без сети: слепой лист всех оценённых кандидатов четырёх прогонов и числа для отчёта.

Слепота: в печать и в листы «Г», кроме «Г слепо», — только числа и подзапросы. ТОП-15 — в
data/interim/g_tops.csv, ключ — data/interim/g_blind_key.csv.

Первый документ кандидата — первый по doc_ids пула из g_extra_<query_id>.json (документы шага 4
в том порядке, в каком их получил шаг 4). Дубли — по tech_key внутри темы.

Запуск: python -m scripts.g_blind
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

from collector.api import tech_key
from pipeline.run_query import document_stats

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "data" / "interim" / "pipeline_runs"
REGISTRY = ROOT / "data" / "interim" / "g_runs.json"
G_TOPS = ROOT / "data" / "interim" / "g_tops.csv"
G_BLIND_KEY = ROOT / "data" / "interim" / "g_blind_key.csv"
REPORT = ROOT / "report_tables.xlsx"
SOURCES = ("arxiv", "techcrunch", "openalex", "openalex_ru")
REJECTED_RE = re.compile(r"^отброшен \((ru|en)\): «.*» — (.+)$")


def load(entry: dict) -> tuple[dict, dict]:
    """Файл прогона и g_extra."""
    qid = entry["query_id"]
    return (json.loads((RUNS / f"{qid}.json").read_text(encoding="utf-8")),
            json.loads((RUNS / f"g_extra_{qid}.json").read_text(encoding="utf-8")))


def scored_rows(entry: dict) -> list[dict]:
    """Оценённые кандидаты прогона с первым документом шага 4, его источником и подзапросами."""
    data, extra = load(entry)
    ids = {tech_key(item["name_en"]): item["doc_ids"] for item in extra["pool"]}
    texts = {item["subquery_id"]: item["text"] for item in data["subqueries"]}
    rows = []
    for item in data["top"] + [x for x in data["excluded"] if x.get("score") is not None]:
        key = tech_key(item["name_en"])
        doc = extra["documents"][ids[key][0] - 1] if ids.get(key) else {}
        rows.append({"тема": entry["тема"], "вариант": entry["вариант"], "query_id": entry["query_id"],
                     "tech_key": key, "term_en": item["name_en"], "term_ru": item["name_ru"],
                     "quote": item.get("quote"), "заголовок": doc.get("title"), "url": doc.get("url"),
                     "источник": doc.get("source"), "подзапрос": "; ".join(
                         texts[i] for i in doc.get("subquery_ids") or [] if i in texts),
                     "место": item.get("rank"), "score": item["score"]})
    return rows


def run_numbers(entry: dict) -> tuple[dict, dict]:
    """Строка «Г прогоны» и строка «Г шаг 4 источники»: только числа."""
    data, extra = load(entry)
    stats = data["stats"]
    step4 = document_stats(extra["documents"], data["subqueries"])
    total = sum(step4.values())
    run_row = {**{k: entry[k] for k in ("тема", "вариант", "query_id", "секунд", "кредитов OpenAlex",
                                         "запросов OpenAlex")},
               "секунд (timings.total)": data["timings"]["total"], "документов в пуле": stats["documents_total"],
               "документов на шаг 4": stats["documents_for_candidates"], "кандидатов найдено": stats["candidates_found"],
               "названо": stats["candidates_named"], "оценено": stats["candidates_scored"],
               "выше порога": stats["above_threshold"], "мест в ТОП": len(data["top"])}
    share = {"тема": entry["тема"], "вариант": entry["вариант"], "документов на шаг 4": total,
             **{f"{s}": step4.get(s, 0) for s in SOURCES},
             **{f"доля {s}": round(step4.get(s, 0) / total, 3) if total else None for s in SOURCES},
             **{f"пул {s}": stats["documents_by_source"].get(s, 0) for s in SOURCES}}
    return run_row, share


def subquery_rows(entry: dict) -> tuple[list[dict], dict]:
    """Подзапросы прогона и счёт отброшенных по причинам (из warnings)."""
    data, _ = load(entry)
    rows = [{"тема": entry["тема"], "вариант": entry["вариант"], "язык": item["language"], "подзапрос": item["text"]}
            for item in data["subqueries"]]
    reasons: dict[str, int] = {}
    for warning in data["warnings"]:
        found = REJECTED_RE.match(warning)
        if found:
            reason = re.sub(r"(дубликат по основам|слово зрелой области|мета-слово): .*", r"\1", found.group(2))
            reason = re.sub(r"(слово зрелой области|мета-слово) .*", r"\1", reason)
            reasons[f"{found.group(1)}: {reason}"] = reasons.get(f"{found.group(1)}: {reason}", 0) + 1
    return rows, {"тема": entry["тема"], "вариант": entry["вариант"], **reasons}


def blind(rows: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """«Г слепо» (перемешано, seed 42) и ключ: варианты, источник и подзапрос первого документа по варианту."""
    keys = sorted(set(zip(rows["тема"], rows["tech_key"])))
    order = [keys[i] for i in np.random.default_rng(42).permutation(len(keys))]
    ids = {key: f"G{number:03d}" for number, key in enumerate(order, start=1)}
    first = rows.sort_values(["вариант"]).drop_duplicates(["тема", "tech_key"]).set_index(["тема", "tech_key"])
    sheet = pd.DataFrame([{"g_id": ids[key], "тема": key[0],
                           **first.loc[key, ["term_en", "term_ru", "quote", "заголовок", "url"]].to_dict()}
                          for key in order])
    key_rows = []
    for key in order:
        part = rows.loc[(rows["тема"] == key[0]) & (rows["tech_key"] == key[1])].drop_duplicates("вариант")
        by = part.set_index("вариант")
        key_rows.append({"g_id": ids[key], "тема": key[0], "term_en": first.loc[key, "term_en"],
                         "варианты": "; ".join(sorted(part["вариант"])),
                         **{f"{col} {v}": by.loc[v, col] if v in by.index else None
                            for v in ("v2", "v3") for col in ("источник", "подзапрос")}})
    return sheet, pd.DataFrame(key_rows)


def write_report(sheets: dict[str, pd.DataFrame]) -> None:
    """Перезаписывает report_tables.xlsx целиком: в книге только листы этого отчёта."""
    with pd.ExcelWriter(REPORT, engine="openpyxl", mode="w") as writer:
        for title, table in sheets.items():
            table.to_excel(writer, sheet_name=title[:31], index=False)


def main() -> None:
    """Г3: слепой лист, ключ, g_tops.csv; числа — в печать и в листы «Г»."""
    registry = json.loads(REGISTRY.read_text(encoding="utf-8"))
    rows = pd.DataFrame([row for entry in registry for row in scored_rows(entry)])
    numbers = [run_numbers(entry) for entry in registry]
    subqueries = [subquery_rows(entry) for entry in registry]
    sheet, key = blind(rows)
    pool = rows.drop_duplicates(["тема", "вариант", "tech_key"])
    overlap = [{"тема": theme, **{f"уникальных {v}": int((part["вариант"] == v).sum()) for v in ("v2", "v3")},
                "в обоих": int(part.groupby("tech_key")["вариант"].nunique().eq(2).sum()),
                "всего уникальных": int(part["tech_key"].nunique()),
                "без первого документа": int(rows.loc[rows["тема"] == theme, "url"].isna().sum())}
               for theme, part in pool.groupby("тема")]
    sheets = {"Г прогоны": pd.DataFrame([n[0] for n in numbers]),
              "Г шаг 4 источники": pd.DataFrame([n[1] for n in numbers]),
              "Г подзапросы": pd.DataFrame([r for s in subqueries for r in s[0]]),
              "Г подзапросы отброшены": pd.DataFrame([s[1] for s in subqueries]).fillna(0),
              "Г пул кандидатов": pd.DataFrame(overlap)}
    for title, table in sheets.items():
        print(f"\n{title}\n{table.to_string(index=False)}")
    tops = rows.loc[rows["место"].notna(), ["тема", "вариант", "query_id", "место", "term_en", "term_ru",
                                            "tech_key", "score"]].sort_values(["тема", "вариант", "место"])
    tops.to_csv(G_TOPS, index=False, encoding="utf-8-sig")
    key.to_csv(G_BLIND_KEY, index=False, encoding="utf-8-sig")
    write_report({**sheets, "Г слепо": sheet})
    print(f"\n«Г слепо»: {len(sheet)} строк; g_tops.csv: {len(tops)} строк; ключ: {len(key)} строк; "
          f"{REPORT.name} перезаписан: {len(sheets) + 1} листов")


if __name__ == "__main__":
    main()
