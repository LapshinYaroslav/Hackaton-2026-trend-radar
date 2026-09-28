"""Гейт воспроизводимости: числа обеих версий модели пересчитываются с нуля и сверяются с эталоном.

s2a1-v1 — по эталонам evidence/model_release_20260922.md; s2a2-v1 (боевая, задача П3) —
по листам «П1 веса» и «П1 сводка», на 160 и на 149 строках, плюс сверка патентного признака
со снимком evidence/rospatent_training_counts.json.

Проверки s2a1-v1:
  1. Признаки. Пересчитываются из кэша счётчиков теми же функциями, что работают в
     режиме запроса, и сверяются с таблицей, на которой обучалась модель. Допуск 1e-9:
     это арифметика над одними и теми же числами, расходиться ей не на чем.
  2. Метрики. Precision, Recall, F1, Accuracy по кросс-валидации и leave-one-area-out
     плюс AUC пересчитываются и сверяются с эталонными числами ниже. Допуск 0.001.

Эталон вписан константами намеренно: гейт должен падать, если числа поехали, а не
подстраиваться под новый результат. Числа взяты из evidence/model_release_20260922.md.

Запуск: python -m scripts.verify_reproduction
Код возврата 0 — всё сошлось, 1 — есть расхождение.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold

from collector.api import terms_hash
from collector.models import SearchTerms
from model.corpus import load_training_totals
from model.counters import aggregate_windows
from model.config import (FEATURES, FEATURES_S2A2, MODEL_VERSION, N_REPEATS, N_SPLITS,
                          ROSPATENT_DATASETS, SEED, canonical)
from model.corpus import load_training_patents
from model.features import RESEARCH_SOURCES, growth_by_sources, share_patent
from model.first_mention import build as build_first_mention
from model.train import build_pipeline
from model.training_table import training_table
from scripts.build_features_from_cache import TECHNOLOGIES, load_counter_cache
from scripts.build_features_from_cache import build_table
from scripts.report_tools import auc_by_area, auc_by_cross_validation
from scripts.tuning import CURRENT, nested_cv, nested_loao

TOLERANCE_FEATURES = 1e-9
TOLERANCE_METRICS = 1e-3

# Эталонные числа. Меняются только вместе с осознанным пересмотром модели.
REFERENCE = {
    "rows": 160,
    "signals": 100,
    "negatives": 60,
    "groups": 150,
    # Пересчитать из кэша можно 156 строк: четыре технологии делят с другими и фразу
    # поиска, и набор терминов, поэтому в кэше у них одна общая строка счётчиков.
    "recomputable_rows": 156,
    "auc_cv": 0.804750,
    "auc_loao": 0.811244,
    "cv": {"precision": 0.764026, "recall": 0.787000,
           "f1": 0.774928, "accuracy": 0.715000},
    "loao": {"precision": 0.816660, "recall": 0.763480,
             "f1": 0.779435, "accuracy": 0.738367},
    "coefficients": {"volume": -0.023622925933802,
                     "growth_research": -0.021796980996005,
                     "recency": 1.095964339108083,
                     "share_news_wordmatch": 1.448515127986352,
                     "share_prev6": -0.246112443170204,
                     "age_first_arxiv": -0.517362891011743},
    "intercept": 0.408409261011003,
    "threshold": 0.325,
}


S2A2 = "s2a2-v1"
# Эталон s2a2-v1: прогон П3 от 24.09.2026, совпал с листами «П1 веса» и «П1 сводка»
# (набор PB задачи П) до округления листа.
REFERENCE_S2A2 = {
    "coefficients": {"share_patent": -0.6614673800834331,
                     "growth_research": 0.0027032769355555875,
                     "recency": 1.0494593322685128,
                     "share_news_wordmatch": 1.3346690281497156,
                     "share_prev6": -0.09976107431855738,
                     "age_first_arxiv": -0.41752806697862366},
    "intercept": 0.3812488866445876,
    "threshold": 0.4,
    160: {"auc_cv": 0.8275, "auc_loao": 0.8369485294117648,
          "cv": {"precision": 0.7761846222910098, "recall": 0.843,
                 "f1": 0.8076801844293244, "accuracy": 0.749375},
          "loao": {"precision": 0.8322896745497365, "recall": 0.7830882352941176,
                   "f1": 0.7971849284349285, "accuracy": 0.7573599240265908}},
    149: {"auc_cv": 0.8124689881268828, "auc_loao": 0.8118487993487994,
          "cv": {"precision": 0.7625712281281418, "recall": 0.8195652173913043,
                 "f1": 0.7897090517602465, "accuracy": 0.7308724832214766},
          "loao": {"precision": 0.7961182336182336, "recall": 0.7975853803059686,
                   "f1": 0.7861261946057784, "accuracy": 0.7400351808685143}},
}


def recomputed_features() -> pd.DataFrame:
    """Признаки, пересчитанные из кэша счётчиков, включая growth_research и возраст.

    Счётчики складываются в рабочие окна тем же aggregate_windows, что и в
    build_features_from_cache: growth считается по окнам now и before, а в кэше
    лежат годовые.
    """
    frame = canonical(build_table()[0])
    cache = load_counter_cache()
    totals = aggregate_windows(load_training_totals(), value_column="n_total")
    table = pd.read_csv(TECHNOLOGIES, encoding="utf-8-sig")
    growth = {}
    for _, row in table.iterrows():
        search = SearchTerms(terms=json.loads(row["terms"]),
                             context_terms=json.loads(row["context_terms"]), query="")
        counters = cache.get((row["tech_key"], terms_hash(search)))
        if counters is not None:
            growth[str(row["tech_id"])] = growth_by_sources(
                aggregate_windows(counters), totals, RESEARCH_SOURCES)
    frame.loc[:, "growth_research"] = frame["tech_id"].astype(str).map(growth)
    ages = build_first_mention(table)[["tech_id", "age_first_arxiv"]]
    return frame.merge(ages, on="tech_id", how="left")


def check_features() -> list[str]:
    """Сверка пересчитанных признаков с таблицей обучения. Допуск 1e-9."""
    stored, _ = training_table()
    fresh = recomputed_features()
    problems = []
    shared = sorted(set(stored["tech_id"]) & set(fresh["tech_id"]))
    if len(shared) != REFERENCE["recomputable_rows"]:
        problems.append(f"строк для сверки {len(shared)}, "
                        f"эталон {REFERENCE['recomputable_rows']}")
    print(f"  строк пересчитано {len(shared)} из {len(stored)}: "
          f"{len(stored) - len(shared)} технологий делят фразу поиска и набор "
          f"терминов с другими, у них те же счётчики и та же строка кэша")
    left = stored.set_index("tech_id").loc[shared]
    right = fresh.set_index("tech_id").loc[shared]
    for name in FEATURES:
        if name not in right.columns:
            problems.append(f"{name}: пересчитать не удалось, колонки нет")
            continue
        a, b = left[name].to_numpy(dtype=float), right[name].to_numpy(dtype=float)
        gap_nan = int((np.isnan(a) != np.isnan(b)).sum())
        both = ~np.isnan(a) & ~np.isnan(b)
        worst = float(np.abs(a[both] - b[both]).max()) if both.any() else 0.0
        if gap_nan or worst > TOLERANCE_FEATURES:
            problems.append(f"{name}: макс. расхождение {worst:.2e}, "
                            f"несовпадений пропусков {gap_nan}")
        else:
            print(f"  признак {name:22s} совпал, макс. расхождение {worst:.2e}")
    return problems


def compare(title: str, got: float, expected: float, tolerance: float) -> str | None:
    """Печатает сверку одного числа и возвращает текст ошибки, если не сошлось."""
    gap = abs(got - expected)
    mark = "совпало" if gap <= tolerance else "РАСХОЖДЕНИЕ"
    print(f"  {title:34s} получено {got:.6f}  эталон {expected:.6f}  "
          f"разница {gap:.2e}  {mark}")
    return None if gap <= tolerance else f"{title}: {got:.6f} против {expected:.6f}"


def check_shape(frame: pd.DataFrame) -> list[str]:
    """Состав выборки: число строк, классов и групп."""
    actual = {"rows": len(frame), "signals": int(frame["label"].sum()),
              "negatives": int((frame["label"] == 0).sum()),
              "groups": int(frame["group"].nunique())}
    problems = []
    for key, value in actual.items():
        same = value == REFERENCE[key]
        print(f"  {key:34s} получено {value:6d}  эталон {REFERENCE[key]:6d}  "
              f"{'совпало' if same else 'РАСХОЖДЕНИЕ'}")
        if not same:
            problems.append(f"{key}: {value} против {REFERENCE[key]}")
    return problems


def check_metrics(frame: pd.DataFrame, reference: dict = REFERENCE,
                  columns: list[str] = FEATURES) -> list[str]:
    """Пересчёт метрик набора признаков и сверка с эталоном. Допуск 0.001."""
    problems = []
    problems.append(compare("AUC, кросс-валидация", auc_by_cross_validation(frame, columns=columns),
                            reference["auc_cv"], TOLERANCE_METRICS))
    problems.append(compare("AUC, leave-one-area-out", auc_by_area(frame, columns=columns),
                            reference["auc_loao"], TOLERANCE_METRICS))
    cv_table, _ = nested_cv(frame, "accuracy", combos=[CURRENT], columns=columns)
    for key, expected in reference["cv"].items():
        problems.append(compare(f"CV {key}", float(cv_table[key].mean()), expected,
                                TOLERANCE_METRICS))
    loao_table = nested_loao(frame, "accuracy", combos=[CURRENT], columns=columns)
    for key, expected in reference["loao"].items():
        problems.append(compare(f"LOAO {key}", float(loao_table[key].mean()), expected,
                                TOLERANCE_METRICS))
    return [item for item in problems if item]


def check_coefficients(frame: pd.DataFrame, reference: dict = REFERENCE,
                       version: str = MODEL_VERSION) -> list[str]:
    """Обучение версии на всех строках должно давать те же веса, что в эталоне."""
    from model.train import train
    _, meta = train(frame, version)
    problems = []
    for name, expected in reference["coefficients"].items():
        problems.append(compare(f"вес {name}", meta["coefficients"][name], expected, 1e-12))
    problems.append(compare("intercept", meta["intercept"], reference["intercept"], 1e-12))
    problems.append(compare("порог", meta["threshold"], reference["threshold"], 1e-12))
    return [item for item in problems if item]


def check_patents(frame: pd.DataFrame) -> list[str]:
    """share_patent таблицы = n_pat снимка / (n_pat + n_research из кэша по шести окнам)."""
    snapshot = load_training_patents(ROSPATENT_DATASETS).set_index("tech_id")["n_pat"]
    cache = load_counter_cache()
    table = pd.read_csv(TECHNOLOGIES, encoding="utf-8-sig").set_index("tech_id")
    stored = frame.set_index("tech_id")["share_patent"]
    problems, checked = [], 0
    for tech_id, value in stored.items():
        row = table.loc[tech_id]
        search = SearchTerms(terms=json.loads(row["terms"]),
                             context_terms=json.loads(row["context_terms"]), query="")
        counters = cache.get((row["tech_key"], terms_hash(search)))
        if counters is None:
            continue
        windows = aggregate_windows(counters)
        research = int(windows.loc[(windows["window"] == "all")
                                   & windows["source"].isin(RESEARCH_SOURCES), "n"].sum())
        fresh = share_patent(int(snapshot.loc[tech_id]), research)
        checked += 1
        if np.isnan(fresh) != np.isnan(value) or (not np.isnan(fresh) and abs(fresh - value) > 1e-12):
            problems.append(f"share_patent {tech_id}: {fresh} против {value}")
    print(f"  share_patent сверен на {checked} строках из {len(stored)}, расхождений {len(problems)}")
    if checked < REFERENCE["recomputable_rows"]:
        problems.append(f"share_patent сверен только на {checked} строках")
    return problems


def check_s2a2(frame: pd.DataFrame) -> list[str]:
    """s2a2-v1: патентный признак, веса и метрики на 160 и на 149 строках."""
    problems = []
    print(f"\n=== {S2A2} ===\nПАТЕНТНЫЙ ПРИЗНАК: снимок и пересчёт, допуск 1e-12")
    problems += check_patents(frame)
    print("\nКОЭФФИЦИЕНТЫ: обучение на всех строках, допуск 1e-12")
    problems += check_coefficients(frame, REFERENCE_S2A2, S2A2)
    blind = frame.loc[frame["blind_choice"].astype(bool)].reset_index(drop=True)
    for rows, block in ((160, frame), (149, blind)):
        print(f"\nМЕТРИКИ на {len(block)} строках: пересчёт, допуск 0.001")
        if len(block) != rows:
            problems.append(f"выборка {rows} строк: получено {len(block)}")
        problems += check_metrics(block, REFERENCE_S2A2[rows], FEATURES_S2A2)
    return problems


def main() -> int:
    """Прогоняет все проверки обеих версий и возвращает код: 0 — сошлось, 1 — нет."""
    frame, source = training_table()
    frame = frame.reset_index(drop=True)
    print(f"Таблица обучения: {source.name}\n\n=== {MODEL_VERSION} ===")
    problems: list[str] = []
    print("СОСТАВ ВЫБОРКИ")
    problems += check_shape(frame)
    print("\nПРИЗНАКИ: пересчёт из кэша счётчиков, допуск 1e-9")
    problems += check_features()
    print("\nКОЭФФИЦИЕНТЫ: обучение на всех строках, допуск 1e-12")
    problems += check_coefficients(frame)
    print("\nМЕТРИКИ: пересчёт, допуск 0.001")
    problems += check_metrics(frame)
    problems += check_s2a2(frame)
    print()
    if problems:
        print(f"ГЕЙТ КРАСНЫЙ: расхождений {len(problems)}")
        for item in problems:
            print(f"  - {item}")
        return 1
    print("ГЕЙТ ЗЕЛЁНЫЙ: все числа воспроизвелись")
    return 0


if __name__ == "__main__":
    sys.exit(main())
