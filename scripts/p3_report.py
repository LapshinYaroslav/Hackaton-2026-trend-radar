"""Задача П3: сверка s2a2-v1 с листами П1 и проверка прогона пайплайна на теме роботов.

Прогон — последний файл data/interim/pipeline_runs с model_version s2a2-v1 и темой
«роботы для промышленности» (тема размечена в задаче К, ТОП-15 можно показывать).
Прогоны задачи Н и Р3 не читаются: отбор по теме и версии модели.

Запуск: python -m scripts.p3_report
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from model.config import FEATURES_S2A2
from model.train import artifact_paths
from scripts.ru_patents_p0 import append_sheets

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "data" / "interim" / "pipeline_runs"
TOPIC = "роботы для промышленности"
VERSION = "s2a2-v1"


def latest_run() -> tuple[str, dict]:
    """Самый свежий прогон s2a2-v1 по теме роботов."""
    for path in sorted(RUNS.glob("q*.json"), key=lambda item: item.stat().st_mtime, reverse=True):
        data = json.loads(path.read_text(encoding="utf-8"))
        if data.get("topic") == TOPIC and data.get("model_version") == VERSION:
            return path.name, data
    raise FileNotFoundError(f"нет прогона {VERSION} по теме «{TOPIC}»")


def weights_sheet() -> pd.DataFrame:
    """Веса артефакта s2a2-v1 против листа «П1 веса» (лист округлён до шести знаков)."""
    meta = json.loads(artifact_paths(VERSION)[1].read_text(encoding="utf-8"))
    sheet = pd.read_excel(ROOT / "report_tables.xlsx", sheet_name="П1 веса").set_index("набор").loc["PB"]
    got = {**meta["coefficients"], "intercept": meta["intercept"], "порог": meta["threshold"]}
    return pd.DataFrame([{"параметр": name, "артефакт": round(float(got[name]), 6),
                          "лист П1": float(sheet[name]),
                          "совпало": abs(round(float(got[name]), 6) - float(sheet[name])) < 1e-9}
                         for name in FEATURES_S2A2 + ["intercept", "порог"]])


def metrics_sheet() -> pd.DataFrame:
    """20 метрик эталона гейта s2a2-v1 против листа «П1 сводка» (лист — три знака)."""
    from scripts.verify_reproduction import REFERENCE_S2A2
    sheet = pd.read_excel(ROOT / "report_tables.xlsx", sheet_name="П1 сводка")
    sheet = sheet.loc[sheet["набор"] == "PB"]
    rows = []
    for count in (160, 149):
        reference = REFERENCE_S2A2[count]
        for mode in ("cv", "loao"):
            line = sheet.loc[(sheet["режим"] == mode.upper()) & (sheet["строк"] == count)].iloc[0]
            values = {**reference[mode], "auc": reference[f"auc_{mode}"]}
            for key, value in values.items():
                shown = float(str(line[key]).split(" ")[0])
                rows.append({"строк": count, "режим": mode.upper(), "метрика": key,
                             "эталон гейта": round(value, 6), "лист П1": shown,
                             "совпало": abs(round(value, 3) - shown) < 1e-9})
    return pd.DataFrame(rows)


def run_sheets(name: str, data: dict) -> dict[str, pd.DataFrame]:
    """Версия и время по очередям, покрытие патентного признака, ТОП-15, примеры объяснений."""
    scored = [item for item in data["top"] + data["excluded"] if item["score"] is not None]
    covered = sum(item.get("share_patent") is not None or item.get("rospatent_failed") is True
                  for item in scored)
    summary = pd.DataFrame([{
        "прогон": name, "model_version": data["model_version"],
        "версия у всех кандидатов": all(item.get("model_version") == VERSION
                                        for item in data["top"] + data["excluded"]),
        "оценено кандидатов": len(scored),
        "share_patent или пометка сбоя": covered,
        "сбоев Роспатента": data["stats"]["rospatent_failures"],
        "rospatent_enabled": data["stats"]["rospatent_enabled"],
        "этап counters, с": data["timings"]["counters"], "весь прогон, с": data["timings"]["total"]}])
    queues = pd.DataFrame(data["timings"]["queues"])
    top = pd.DataFrame([{"место": item["rank"], "term_en": item["name_en"],
                         "score": round(item["score"], 4), "n_pat": item.get("n_pat"),
                         "share_patent": item.get("share_patent"),
                         "rospatent_failed": item.get("rospatent_failed")} for item in data["top"]])
    texts = [{"term_en": item["name_en"], "где": "ТОП-15", "текст": text}
             for item in data["top"] for text in item["explanation_ru"] if "атент" in text]
    texts += [{"term_en": item["name_en"], "где": item["skipped_reason"], "текст": item["reason_ru"]}
              for item in data["excluded"] if "атент" in (item.get("reason_ru") or "")]
    return {"П3 прогон": summary, "П3 очереди": queues, "П3 ТОП-15 роботы": top,
            "П3 патентные объяснения": pd.DataFrame(texts, columns=["term_en", "где", "текст"])}


def main() -> None:
    """Печатает и дописывает листы П3."""
    name, data = latest_run()
    sheets = {"П3 веса s2a2-v1": weights_sheet(), "П3 метрики s2a2-v1": metrics_sheet(),
              **run_sheets(name, data)}
    for title, table in sheets.items():
        print(f"\n{title}\n{table.to_string(index=False)}")
    append_sheets(sheets)


if __name__ == "__main__":
    main()
