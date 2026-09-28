"""Задача Е1, без сети: эталон организаторов, подмешанный в сохранённые пулы открытого режима.

Технологии области (сигналы и мейнстримы) и кандидаты пула оцениваются одной LOAO-моделью без этой
области (s2a2-v1, C=1, balanced, порог по accuracy внутри обучающих областей — как в гейте).
Входы — evidence/e1_inputs/pools.json (в git): по кандидату сохранённый score, годовые счётчики и n_pat.
Файл собран один раз из прогонов data/interim/pipeline_runs и кэшей счётчиков и Роспатента
(python -m scripts.e_mix export). Признаки пересчитываются из счётчиков, сохранённые score пула
воспроизводятся моделью его прогона. Термины кандидатов не печатаются.

Нормировка отложенной области — два варианта: «А» как в гейте (общие центр и масштаб пяти
обучающих областей), «Б» — центр и масштаб по обучающим технологиям самой области, без меток.

Запуск: python -m scripts.e_mix   (сборка входов: python -m scripts.e_mix export)
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

from collector import rospatent
from collector.api import tech_key
from model.candidate import features_from_counters
from model.config import FEATURES_S2A2, ROSPATENT_DATASETS
from model.predict import load, predict
from pipeline.fetch import COUNTERS_CACHE_DIR, one_call

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "data" / "interim" / "pipeline_runs"
INPUTS = ROOT / "evidence" / "e1_inputs" / "pools.json"
POOLS = {"К R1": "q20260923155830", "К R2": "q20260923161459", "Н финтех": "q20260923170352",
         "Н кибербезопасность": "q20260923192323", "Н ИИ": "q20260924224939",
         "Г энергетика v3": "q20260925220056", "Г медицина v3": "q20260925221415",
         "Д агро A": "q20260925235405", "Д транспорт A": "q20260926000912"}
SOURCES = ("openalex", "arxiv", "techcrunch")


def cached_counters(key: str, source: str) -> list[dict]:
    """Счётчики пайплайна из кэша: сначала текущий режим опроса, затем другой (старые прогоны)."""
    modes = ["one_call", "per_window"] if one_call(source) else ["per_window", "one_call"]
    for mode in modes:
        raw = json.dumps([key, source, mode], ensure_ascii=False)
        path = COUNTERS_CACHE_DIR / f"{hashlib.sha256(raw.encode('utf-8')).hexdigest()}.json"
        if path.exists():
            return json.loads(path.read_text(encoding="utf-8"))
    raise FileNotFoundError(f"нет счётчиков {source} в кэше")


def patents(item: dict) -> int | None:
    """n_pat кандидата: из прогона, иначе из кэша Роспатента (по tech_key, затем по name_en)."""
    if item.get("n_pat") is not None:
        return int(item["n_pat"])
    for phrase in (tech_key(item["name_en"]), item["name_en"]):
        path = rospatent.cache_path(rospatent.request_key(phrase, ROSPATENT_DATASETS))
        if path.exists():
            return int(json.loads(path.read_text(encoding="utf-8"))["response"]["total"])
    return None


def export() -> None:
    """Входы Е1 из прогонов и кэшей в INPUTS: по пулу — область, модель прогона, кандидаты со счётчиками."""
    out = {}
    for name, query_id in POOLS.items():
        run = json.loads((RUNS / f"{query_id}.json").read_text(encoding="utf-8"))
        items = run["top"] + [x for x in run["excluded"] if x.get("score") is not None]
        out[name] = {"query_id": query_id, "area": run["area"], "model_version": run["model_version"], "candidates": [
            {"tech_key": tech_key(i["name_en"]), "score": i["score"], "n_pat": patents(i),
             "counters": [[r["source"], r["window"], r["n"]] for s_ in SOURCES
                          for r in cached_counters(tech_key(i["name_en"]), s_)]} for i in items]}
    INPUTS.parent.mkdir(parents=True, exist_ok=True)
    INPUTS.write_text(json.dumps(out, ensure_ascii=False), encoding="utf-8")


def pool(name: str) -> tuple[pd.DataFrame, dict]:
    """Кандидаты пула из INPUTS с признаками s2a2-v1 и проверкой сохранённого score моделью прогона."""
    data = json.loads(INPUTS.read_text(encoding="utf-8"))[name]
    own = load(version=data["model_version"])
    rows, gaps = [], []
    for item in data["candidates"]:
        counters = pd.DataFrame(item["counters"], columns=["source", "window", "n"])
        again, _ = predict(features_from_counters(counters, n_pat=item["n_pat"], version=data["model_version"]),
                           data["area"], own)
        gaps.append(abs(again - item["score"]))
        rows.append({"tech_key": item["tech_key"], "area": data["area"], "label": np.nan, "kind": "пул",
                     **features_from_counters(counters, n_pat=item["n_pat"], version="s2a2-v1")})
    check = {"кандидатов": len(data["candidates"]), "модель прогона": data["model_version"],
             "макс. |Δscore|": float(max(gaps)) if gaps else None}
    return pd.DataFrame(rows), check


def training_rows() -> pd.DataFrame:
    """Обучающая таблица s2a2-v1 с tech_key технологии."""
    from model.train import training_table
    keys = pd.read_csv(ROOT / "data" / "interim" / "technologies.csv", encoding="utf-8-sig")[["tech_id", "tech_key"]]
    return training_table().reset_index(drop=True).merge(keys, on="tech_id", how="left", validate="1:1")


def fitted(frame: pd.DataFrame, rows: np.ndarray, own_stats: bool):
    """Пайплайн s2a2-v1 (C=1, balanced) на строках rows; own_stats — отложенная область по своей статистике."""
    from scripts.tuning import CURRENT, pipeline_for
    model = pipeline_for(*CURRENT, columns=FEATURES_S2A2)
    model.set_params(area_scaler__transductive=own_stats)
    return model.fit(frame.iloc[rows][FEATURES_S2A2 + ["area"]], frame["label"].iloc[rows])


def inner_threshold(frame: pd.DataFrame, train: np.ndarray, own_stats: bool) -> float:
    """Порог по accuracy: каждая обучающая область по очереди отложена (_area_splits гейта)."""
    from model.train import choose_threshold
    areas = frame["area"].iloc[train].to_numpy()
    probability = np.full(len(train), np.nan)
    for area in sorted(set(areas)):
        hold = areas == area
        model = fitted(frame, train[~hold], own_stats)
        probability[hold] = model.predict_proba(frame.iloc[train[hold]][FEATURES_S2A2 + ["area"]])[:, 1]
    return choose_threshold(frame["label"].iloc[train].to_numpy(), probability)


def loao(frame: pd.DataFrame, area: str, own_stats: bool) -> tuple:
    """LOAO-модель без области и её порог. Вариант Б: центр и масштаб области — по её обучающим технологиям."""
    from model.area_scaler import centre_scale
    train = np.flatnonzero((frame["area"] != area).to_numpy())
    model = fitted(frame, train, own_stats)
    if own_stats:
        block = frame.loc[frame["area"] == area, FEATURES_S2A2]
        model.named_steps["area_scaler"].by_area_[area] = centre_scale(block, "area_z")
    return model, inner_threshold(frame, train, own_stats)


def mixed(frame: pd.DataFrame, candidates: pd.DataFrame, area: str, own_stats: bool) -> tuple[pd.DataFrame, float, int]:
    """Технологии области + кандидаты пула одной LOAO-моделью: score, ранг, место в ТОП-15, порог, дублей."""
    model, threshold = loao(frame, area, own_stats)
    known = frame.loc[frame["area"] == area].assign(kind=lambda x: np.where(x["label"] == 1, "сигнал", "мейнстрим"))
    duplicates = int(candidates["tech_key"].isin(known["tech_key"]).sum())
    rows = pd.concat([known, candidates.loc[~candidates["tech_key"].isin(known["tech_key"])]], ignore_index=True)
    rows["score"] = model.predict_proba(rows[FEATURES_S2A2 + ["area"]])[:, 1]
    rows = rows.sort_values("score", ascending=False, kind="stable").reset_index(drop=True)
    rows["ранг"] = np.arange(1, len(rows) + 1)
    rows["в ТОП"] = (rows["score"] >= threshold) & (rows["ранг"] <= 15)
    return rows, threshold, duplicates


def numbers(rows: pd.DataFrame, threshold: float) -> dict:
    """Строка отчёта Е1 по одному пулу (или по объединению пулов)."""
    kind = rows["kind"]
    above = rows["score"] >= threshold
    return {"сигналов подмешано": int((kind == "сигнал").sum()), "мейнстримов подмешано": int((kind == "мейнстрим").sum()),
            "кандидатов пула": int((kind == "пул").sum()), "мест в ТОП": int(rows["в ТОП"].sum()),
            **{f"{k} в ТОП": int((rows["в ТОП"] & (kind == k)).sum()) for k in ("сигнал", "мейнстрим", "пул")},
            **{f"медианный ранг: {k}": float(rows.loc[kind == k, "ранг"].median()) for k in ("сигнал", "мейнстрим", "пул")},
            "доля сигналов выше порога": round(float(above[kind == "сигнал"].mean()), 3),
            "доля мейнстримов выше порога": round(float(above[kind == "мейнстрим"].mean()), 3),
            "доля пула выше порога": round(float(above[kind == "пул"].mean()), 3)}


def report(own_stats: bool) -> pd.DataFrame:
    """Таблица Е1 по пулам и «итого» (суммы; медианы и доли — по объединению строк всех пулов)."""
    frame, out, everything = training_rows(), [], []
    for name, query_id in POOLS.items():
        candidates, check = pool(name)
        area = candidates["area"].iloc[0]
        rows, threshold, duplicates = mixed(frame, candidates, area, own_stats)
        rows["above"] = rows["score"] >= threshold
        everything.append(rows)
        out.append({"пул": name, "область": area, "порог LOAO": threshold, "дублей с эталоном": duplicates,
                    **numbers(rows, threshold), "макс. |Δscore| пересчёта": check["макс. |Δscore|"]})
    union = pd.concat(everything, ignore_index=True)
    total = numbers(union.assign(score=union["above"].astype(float)), 0.5)
    return pd.DataFrame(out + [{"пул": "итого", "дублей с эталоном": sum(r["дублей с эталоном"] for r in out), **total}])


def main() -> None:
    """Е1 в двух вариантах нормировки: печать и report_tables.xlsx (только таблицы Е, перезапись)."""
    sheets = {"Е1 вариант А (как гейт)": report(False), "Е1 вариант Б (своя статистика)": report(True)}
    for title, table in sheets.items():
        print(f"\n{title}\n{table.to_string(index=False)}")
    with pd.ExcelWriter(ROOT / "report_tables.xlsx", engine="openpyxl", mode="w") as writer:
        for title, table in sheets.items():
            table.to_excel(writer, sheet_name=title[:31], index=False)


if __name__ == "__main__":
    import sys
    export() if sys.argv[1:] == ["export"] else main()
