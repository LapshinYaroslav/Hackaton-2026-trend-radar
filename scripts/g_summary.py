"""Задача Г, сведение после разметки: пул и ТОП-15 v2/v3 по темам, источники, правило 1–4.

Метки — labels/g_blind_labels.csv по ключу data/interim/g_blind_key.csv; ТОП-15 — g_tops.csv; время —
g_runs.json. report_tables.xlsx перезаписывается целиком: в нём только листы этого сведения.

Запуск: python -m scripts.g_summary
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from collector.api import tech_key

ROOT = Path(__file__).resolve().parents[1]
INTERIM = ROOT / "data" / "interim"
REPORT = ROOT / "report_tables.xlsx"
GARBAGE = ("noise", "mature", "umbrella")
VERSIONS = ("v2", "v3")
TIME_LIMIT_S = 1200


def labelled() -> pd.DataFrame:
    """Ключ с метками: одна строка на (g_id, вариант), источник первого документа — свой у варианта."""
    key = pd.read_csv(INTERIM / "g_blind_key.csv", encoding="utf-8-sig")
    marks = pd.read_csv(ROOT / "labels" / "g_blind_labels.csv", encoding="utf-8-sig")
    frame = key.merge(marks, on="g_id", how="inner", validate="1:1")
    frame = frame.assign(tech_key=frame["term_en"].map(tech_key),
                         concept=frame["canonical_concept"].str.strip().str.lower())
    rows = [row | {"вариант": v, "источник": row[f"источник {v}"]}
            for row in frame.to_dict("records") for v in VERSIONS if v in row["варианты"].split("; ")]
    return pd.DataFrame(rows)


def concepts(rows: pd.DataFrame) -> int:
    """Разные plausible_signal-понятия с on_topic = yes."""
    good = rows.loc[(rows["label"] == "plausible_signal") & (rows["on_topic"] == "yes"), "concept"]
    return int(good.nunique())


def pool_row(rows: pd.DataFrame) -> dict:
    """Пул варианта: кандидаты по меткам и разные понятия."""
    return {"кандидатов": len(rows), **{label: int((rows["label"] == label).sum())
                                         for label in ("noise", "mature", "umbrella", "uncertain", "plausible_signal")},
            "разных plausible-понятий (on_topic=yes)": concepts(rows)}


def top_row(rows: pd.DataFrame) -> dict:
    """ТОП-15 варианта: места, мусор и его доля, uncertain, разные понятия."""
    garbage = int((rows["label"].isin(GARBAGE) | (rows["on_topic"] == "no")).sum())
    return {"мест": len(rows), "мусор": garbage, "доля мусора": round(garbage / len(rows), 4) if len(rows) else None,
            "uncertain": int((rows["label"] == "uncertain").sum()),
            "разных plausible-понятий (on_topic=yes)": concepts(rows)}


def by_theme(frame: pd.DataFrame, build) -> pd.DataFrame:
    """Строки по теме и варианту плюс сумма по двум темам (понятия — сумма по темам)."""
    out = []
    for version in VERSIONS:
        parts = [{"вариант": version, "тема": theme, **build(frame.loc[(frame["вариант"] == version)
                                                                      & (frame["тема"] == theme)])}
                 for theme in sorted(frame["тема"].unique())]
        total = {k: sum(p[k] for p in parts) for k in parts[0] if k not in ("вариант", "тема", "доля мусора")}
        if "мусор" in total:
            total["доля мусора"] = round(total["мусор"] / total["мест"], 4)
        out += parts + [{"вариант": version, "тема": "сумма по двум темам", **total}]
    return pd.DataFrame(out)


def sources(pool: pd.DataFrame) -> pd.DataFrame:
    """По источнику первого документа, суммарно по темам: кандидаты, plausible, uncertain и их доли."""
    rows = []
    for version in VERSIONS:
        part = pool.loc[pool["вариант"] == version]
        for source in ("arxiv", "openalex", "techcrunch"):
            mine = part.loc[part["источник"] == source]
            plausible, uncertain = int((mine["label"] == "plausible_signal").sum()), int((mine["label"] == "uncertain").sum())
            rows.append({"вариант": version, "источник": source, "кандидатов": len(mine),
                         "доля пула": round(len(mine) / len(part), 4), "plausible": plausible,
                         "доля plausible в источнике": round(plausible / len(mine), 4) if len(mine) else None,
                         "uncertain": uncertain,
                         "доля uncertain в источнике": round(uncertain / len(mine), 4) if len(mine) else None})
    return pd.DataFrame(rows)


def rule(pool: pd.DataFrame, top: pd.DataFrame, runs: list[dict]) -> pd.DataFrame:
    """Правило задачи Г по суммам двух тем: левая и правая части каждого условия."""
    total = lambda frame: frame.loc[frame["тема"] == "сумма по двум темам"].set_index("вариант")
    p, t = total(pool), total(top)
    name = "разных plausible-понятий (on_topic=yes)"
    slow = [run for run in runs if run["вариант"] == "v3"]
    rows = [("1: понятий в пуле v3 ≥ 2 × v2", p.loc["v3", name], f"≥ {2 * p.loc['v2', name]}",
             p.loc["v3", name] >= 2 * p.loc["v2", name]),
            ("2: понятий в ТОП-15 v3 ≥ v2 + 2", t.loc["v3", name], f"≥ {t.loc['v2', name] + 2}",
             t.loc["v3", name] >= t.loc["v2", name] + 2),
            ("3: доля мусора в ТОП-15 v3 ≤ v2", t.loc["v3", "доля мусора"], f"≤ {t.loc['v2', 'доля мусора']}",
             t.loc["v3", "доля мусора"] <= t.loc["v2", "доля мусора"]),
            ("4: время каждого прогона v3 ≤ 1200 с", max(run["секунд"] for run in slow), f"≤ {TIME_LIMIT_S}",
             all(run["секунд"] <= TIME_LIMIT_S for run in slow))]
    return pd.DataFrame([{"условие": a, "v3": float(b), "граница": c, "выполнено": bool(d)} for a, b, c, d in rows])


def main() -> None:
    """Сведение: печать и перезапись report_tables.xlsx только листами этого сведения."""
    pool = labelled()
    tops = pd.read_csv(INTERIM / "g_tops.csv", encoding="utf-8-sig")
    top = tops[["тема", "вариант", "tech_key", "место"]].merge(
        pool, on=["тема", "вариант", "tech_key"], how="left", validate="1:1")
    if top["label"].isna().any():
        raise ValueError("в ТОП-15 есть термины без метки")
    runs = json.loads((INTERIM / "g_runs.json").read_text(encoding="utf-8"))
    pool_table, top_table = by_theme(pool, pool_row), by_theme(top, top_row)
    sheets = {"Г пул": pool_table, "Г ТОП-15": top_table, "Г источники": sources(pool),
              "Г время": pd.DataFrame([{k: run[k] for k in ("тема", "вариант", "query_id", "секунд",
                                                            "кредитов OpenAlex")} for run in runs]),
              "Г правило": rule(pool_table, top_table, runs)}
    for title, frame in sheets.items():
        print(f"\n{title}\n{frame.to_string(index=False)}")
    with pd.ExcelWriter(REPORT, engine="openpyxl", mode="w") as writer:
        for title, frame in sheets.items():
            frame.to_excel(writer, sheet_name=title, index=False)
    print(f"\n{REPORT.name}: перезаписан, листов {len(sheets)}")


if __name__ == "__main__":
    main()
