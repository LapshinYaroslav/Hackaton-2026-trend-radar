"""Задача П2: открытый режим для PB на пулах R1 и R2 задачи К (роботы).

П2.1 (сеть, только Роспатент): n_pat для всех кандидатов пулов, которые оценивала модель
(ТОП плюс исключённые со score), тем же запросом и ключом кэша, что в П0.2.
П2.2 (без сети): V0 и PB обучаются на 160 строках, ТОП-15 по каждому пулу, лист новичков PB.

Счётчики кандидатов — из кэша пайплайна в режимах опроса, которыми пользовались прогоны К:
TechCrunch тогда ещё опрашивался по окнам, одним вызовом — только arXiv. n_research —
OpenAlex + arXiv по шести годовым окнам 2020–2025, без prev6.

Запуск: python -m scripts.ru_patents_p2
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
from dotenv import load_dotenv

from collector.api import tech_key
from collector.constants import AGGREGATE_WINDOWS
from model.candidate import features_from_counters
from model.config import MODEL_VERSION
from model.features import RESEARCH_SOURCES, share_patent
from model.train import build_pipeline, choose_threshold, out_of_fold
from scripts.ru_patents_collect import cache_path, fetch, request_key
from scripts.ru_patents_p0 import REPORT, append_sheets, feature_table
from scripts.ru_patents_p1 import ALL_SETS

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "data" / "interim" / "pipeline_runs"
COUNTERS = ROOT / "data" / "interim" / "cache" / "counters_pipeline"
BLIND_KEY = ROOT / "data" / "interim" / "k_blind_key.csv"
BLIND_LABELS = ROOT / "labels" / "k_blind_labels.csv"
P2_BLIND_KEY = ROOT / "data" / "interim" / "p2_blind_key.csv"
# Пулы задачи К: прогон и файл с документами. Других прогонов скрипт не читает.
POOLS = {"R1": ("q20260923155830", "k_R1_extra_20260923T155830"),
         "R2": ("q20260923161459", "k_R2_extra_20260923T161459")}
K_MODES = {"openalex": "per_window", "arxiv": "one_call", "techcrunch": "per_window"}
SETS = {"V0": ALL_SETS["V0"], "PB": ALL_SETS["PB"]}
EXPECTED_THRESHOLD = {"V0": 0.325, "PB": 0.400}
TOP_N = 15


def run(pool: str) -> tuple[dict, dict]:
    """Файл прогона и файл документов пула."""
    run_id, extra_id = POOLS[pool]
    return (json.loads((RUNS / f"{run_id}.json").read_text(encoding="utf-8")),
            json.loads((RUNS / f"{extra_id}.json").read_text(encoding="utf-8")))


def scored(pool: str) -> list[dict]:
    """Кандидаты пула, которые оценивала модель: ТОП и исключённые со score."""
    data, _ = run(pool)
    return data["top"] + [item for item in data["excluded"] if item["score"] is not None]


def cached_counters(name_en: str) -> pd.DataFrame:
    """Годовые счётчики кандидата из кэша пайплайна, три источника в режимах К."""
    rows = []
    for source, mode in K_MODES.items():
        raw = json.dumps([tech_key(name_en), source, mode], ensure_ascii=False)
        path = COUNTERS / f"{hashlib.sha256(raw.encode('utf-8')).hexdigest()}.json"
        rows += [{"source": row["source"], "window": row["window"], "n": row["n"]}
                 for row in json.loads(path.read_text(encoding="utf-8"))]
    return pd.DataFrame(rows, columns=["source", "window", "n"])


def collect() -> pd.DataFrame:
    """П2.1: n_pat по уникальным фразам обоих пулов; замер времени и кодов ответа."""
    load_dotenv(ROOT / ".env")
    secrets = {"rospatent": os.environ["ROSPATENT"]}
    phrases = sorted({tech_key(item["name_en"]) for pool in POOLS for item in scored(pool)})
    cached = sum(cache_path(request_key("rospatent", phrase)).exists() for phrase in phrases)
    spent = {"openalex": 0, "rospatent": 0, "openalex_credits": 0.0}
    for phrase in phrases:
        fetch("rospatent", phrase, secrets, spent)
    seconds = np.array(spent.get("seconds", []), dtype=float)
    statuses = spent.get("statuses", [])
    stat = (lambda f: round(float(f(seconds)), 2) if len(seconds) else None)
    return pd.DataFrame([{"уникальных фраз": len(phrases), "уже в кэше": cached,
                          "запросов": spent["rospatent"], "попыток всего": len(statuses),
                          "ответов 429": statuses.count(429),
                          "ответов не 200": sum(code != 200 for code in statuses),
                          "с/запрос среднее": stat(np.mean), "медиана": stat(np.median),
                          "максимум": stat(np.max)}])


def estimate(collected: pd.DataFrame) -> pd.DataFrame:
    """Сколько длится очередь Роспатента на один прогон: кандидатов × (среднее время + пауза)."""
    mean = collected["с/запрос среднее"].iloc[0]
    rows = []
    for pool in POOLS:
        data, _ = run(pool)
        count = len(scored(pool))
        rows.append({"пул": pool, "кандидатов со score": count,
                     "очередь Роспатента, с (пауза 0.5)":
                         None if mean is None else round(count * (mean + 0.5), 1),
                     "очередь без паузы, с": None if mean is None else round(count * mean, 1),
                     "этап counters прогона К, с": data["timings"]["counters"],
                     "весь прогон К, с": data["timings"]["total"]})
    return pd.DataFrame(rows)


def pool_frame(pool: str) -> pd.DataFrame:
    """Кандидаты пула: признаки модели, n_research по шести окнам, n_pat и share_patent."""
    data, _ = run(pool)
    rows = []
    for item in scored(pool):
        counters = cached_counters(item["name_en"])
        six = counters.loc[counters["window"].isin(AGGREGATE_WINDOWS["all"])
                           & counters["source"].isin(RESEARCH_SOURCES), "n"]
        record = json.loads(cache_path(request_key("rospatent", tech_key(item["name_en"])))
                            .read_text(encoding="utf-8"))
        n_pat, n_research = int(record["response"]["total"]), int(six.sum())
        rows.append({"name_en": item["name_en"], "name_ru": item["name_ru"],
                     "quote": item.get("quote"), "area": data["area"],
                     **features_from_counters(counters, version=MODEL_VERSION), "n_pat": n_pat,
                     "n_research": n_research, "share_patent": share_patent(n_pat, n_research)})
    return pd.DataFrame(rows)


def fit_sets(frame: pd.DataFrame) -> dict[str, tuple]:
    """V0 и PB на всех 160 строках; порог — как в model.train, сверяется с ожидаемым."""
    fitted = {}
    for name, columns in SETS.items():
        threshold = choose_threshold(frame["label"].to_numpy(),
                                     np.nanmean(out_of_fold(frame, columns), axis=1))
        if threshold != EXPECTED_THRESHOLD[name]:
            raise ValueError(f"{name}: порог {threshold}, ожидался {EXPECTED_THRESHOLD[name]}")
        model = build_pipeline(features=columns).fit(frame[columns + ["area"]], frame["label"])
        fitted[name] = (model, threshold)
    return fitted


def weights_check(fitted: dict) -> pd.DataFrame:
    """Веса PB против листа «П1 веса»: допуск 1e-6 (лист округлён до шести знаков)."""
    sheet = pd.read_excel(REPORT, sheet_name="П1 веса").set_index("набор").loc["PB"]
    model, threshold = fitted["PB"]
    logistic = model.named_steps["logistic"]
    got = {**dict(zip(SETS["PB"], logistic.coef_[0])), "intercept": logistic.intercept_[0],
           "порог": threshold}
    return pd.DataFrame([{"параметр": key, "получено": round(float(value), 6),
                          "лист П1": float(sheet[key]),
                          "совпало": abs(float(value) - float(sheet[key])) <= 1e-6}
                         for key, value in got.items()])


def top15(pool: pd.DataFrame, model, threshold: float, columns: list[str]) -> pd.DataFrame:
    """ТОП-15 выше порога по убыванию score, как в pipeline.run_query.split_ranked."""
    scored_pool = pool.assign(score=model.predict_proba(pool[columns + ["area"]])[:, 1])
    above = scored_pool.loc[scored_pool["score"] >= threshold]
    top = above.sort_values("score", ascending=False, kind="stable").head(TOP_N)
    top = top.assign(место=range(1, len(top) + 1), tech_key=top["name_en"].map(tech_key))
    key = pd.read_csv(BLIND_KEY, encoding="utf-8-sig").drop_duplicates("tech_key")
    labels = pd.read_csv(BLIND_LABELS, encoding="utf-8-sig")
    labels = labels.assign(tech_key=labels["term_en"].map(tech_key))
    top = top.merge(key[["tech_key", "blind_id"]], on="tech_key", how="left")
    return top.merge(labels[["tech_key", "label"]], on="tech_key", how="left")


def control(tops: dict) -> pd.DataFrame:
    """ТОП-15 V0 против ключа К: термин и score на каждой позиции."""
    key = pd.read_csv(BLIND_KEY, encoding="utf-8-sig")
    rows = []
    for pool in POOLS:
        mine = tops[("V0", pool)]
        theirs = key.loc[key["рука"] == pool].sort_values("место").reset_index(drop=True)
        for place in range(max(len(mine), len(theirs))):
            got = mine.iloc[place] if place < len(mine) else None
            want = theirs.iloc[place] if place < len(theirs) else None
            same = (got is not None and want is not None and got["tech_key"] == want["tech_key"]
                    and abs(got["score"] - want["score"]) < 5e-5)
            rows.append({"пул": pool, "место": place + 1,
                         "V0": None if got is None else got["tech_key"],
                         "К": None if want is None else want["tech_key"],
                         "V0 score": None if got is None else round(float(got["score"]), 4),
                         "К score": None if want is None else want["score"], "совпало": same})
    return pd.DataFrame(rows)


def first_document(pool: str, name_en: str) -> dict:
    """Первый документ кандидата; doc_ids нумеруют документы с единицы."""
    _, extra = run(pool)
    ids = extra["doc_ids"].get(name_en) or []
    doc = extra["documents"][ids[0] - 1] if ids else {}
    return {"заголовок": doc.get("title"), "url": doc.get("url")}


def newcomers(tops: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Термины ТОП-15 PB без метки К: лист для слепой разметки и ключ соответствия."""
    seen, key_rows = {}, []
    for pool in POOLS:
        top = tops[("PB", pool)]
        for _, row in top.loc[top["label"].isna()].iterrows():
            seen.setdefault(row["tech_key"], {"term_en": row["name_en"], "term_ru": row["name_ru"],
                                              "quote": row["quote"],
                                              **first_document(pool, row["name_en"])})
            key_rows.append({"tech_key": row["tech_key"], "набор": "PB", "пул": pool,
                             "место": int(row["место"]), "score": round(float(row["score"]), 4)})
    order = np.random.default_rng(42).permutation(sorted(seen))
    ids = {term: f"P{number:02d}" for number, term in enumerate(order, start=1)}
    sheet = pd.DataFrame([{"new_blind_id": ids[term], **seen[term]} for term in order],
                         columns=["new_blind_id", "term_en", "term_ru", "quote", "заголовок", "url"])
    key = pd.DataFrame(key_rows, columns=["tech_key", "набор", "пул", "место", "score"])
    key.insert(0, "new_blind_id", key["tech_key"].map(ids))
    return sheet, key.sort_values(["new_blind_id", "пул"]).reset_index(drop=True)


def recount(collected: pd.DataFrame) -> tuple[dict[str, pd.DataFrame], pd.DataFrame]:
    """П2.2 без сети: контроль V0 и весов PB, ТОП-15, новички PB."""
    fitted = fit_sets(feature_table())
    pools = {pool: pool_frame(pool) for pool in POOLS}
    tops = {(name, pool): top15(frame, *fitted[name], SETS[name])
            for name in SETS for pool, frame in pools.items()}
    sheets = {"П2.1 сбор Роспатента": collected, "П2.1 оценка времени": estimate(collected),
              "П2 контроль V0 = К": control(tops), "П2 контроль весов PB": weights_check(fitted)}
    for (name, pool), top in tops.items():
        table = top[["место", "name_en", "score", "share_patent", "n_pat", "blind_id", "label"]]
        sheets[f"П2 ТОП {name} {pool}"] = table.rename(
            columns={"name_en": "term_en", "label": "метка"}).assign(
            score=table["score"].round(4), share_patent=table["share_patent"].round(4))
    sheet, key = newcomers(tops)
    sheets["П2 новички PB слепо"] = sheet
    return sheets, key


def main() -> None:
    """П2.1 (сеть к Роспатенту), затем П2.2; печать, дозапись листов П2, ключ новичков."""
    sheets, key = recount(collect())
    for title, table in sheets.items():
        print(f"\n{title}\n{table.to_string(index=False)}")
    key.to_csv(P2_BLIND_KEY, index=False, encoding="utf-8-sig")
    names = append_sheets(sheets)
    print(f"\nлистов в {REPORT.name}: {len(names)}; ключ: {P2_BLIND_KEY.relative_to(ROOT)} "
          f"({len(key)} строк)")


if __name__ == "__main__":
    main()
