"""Задача Т: фильтры выдачи на пулах R1 и R2 задачи К, без сети. Модель — s2a2-v1 (PB), порог 0.400.

Оценка кандидатов — как в П2.2 (scripts.ru_patents_p2): те же счётчики, n_pat, веса и порог,
сверка весов с листом «П1 веса». Читаются только два пула К, контрольные прогоны Н и Р3 — нет.

Запуск: python -m scripts.t_report
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from collector.api import tech_key
from collector.constants import AGGREGATE_WINDOWS
from scripts import t_filters as tf
from scripts.ru_patents_p0 import REPORT, feature_table, sheet_names
from scripts.ru_patents_p2 import (POOLS, SETS, TOP_N, first_document, fit_sets, pool_frame, run,
                                   weights_check)

ROOT = Path(__file__).resolve().parents[1]
LABEL_FILES = [ROOT / "labels" / "k_blind_labels.csv", ROOT / "labels" / "p2_blind_labels.csv"]
T_BLIND_KEY = ROOT / "data" / "interim" / "t_blind_key.csv"
TECHNOLOGIES = ROOT / "data" / "interim" / "technologies.csv"
COLUMNS = SETS["PB"]
SINGLE = ["Ф1", "Ф2", "Ф3", "Ф4a", "Ф4b"]
BASE = {"шум": 8, "зрелые + зонтичные": 10, "разных plausible-понятий": 4}
NO_AUTHORS = "Ф4b ≡ Ф4a: в сохранённых документах нет авторов, каждый документ — отдельный автор"


def labels() -> pd.DataFrame:
    """Метки К и П2 по tech_key: blind_id, label, понятие (canonical_concept или сам термин)."""
    frame = pd.concat([pd.read_csv(path, encoding="utf-8-sig") for path in LABEL_FILES])
    frame = frame.assign(tech_key=frame["term_en"].map(tech_key))
    concept = frame["canonical_concept"].fillna(frame["tech_key"]).astype(str).str.lower()
    return frame.assign(concept=concept)[["tech_key", "blind_id", "label", "concept"]]


def pool_table(pool: str, model, threshold: float, marks: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """Кандидаты пула: score PB, n_research, метка, d_pool, d_authors. Второе — склеено документов."""
    frame = pool_frame(pool)
    frame = frame.assign(score=model.predict_proba(frame[COLUMNS + ["area"]])[:, 1],
                         tech_key=frame["name_en"].map(tech_key), пул=pool)
    frame = frame.merge(marks, on="tech_key", how="left")
    _, extra = run(pool)
    documents, glued = tf.unique_documents(extra["documents"])
    counts = [tf.repeat_counts(term, documents) for term in frame["name_en"]]
    frame.loc[:, "d_pool"] = [pair[0] for pair in counts]
    frame.loc[:, "d_authors"] = [pair[1] for pair in counts]
    frame = frame.assign(doc_ids=[extra["doc_ids"].get(name) or [] for name in frame["name_en"]],
                         above=frame["score"] >= threshold)
    return frame.reset_index(drop=True), glued


def reasons(frame: pd.DataFrame, name: str, t_mature: float) -> tuple[pd.Series, list[list[int]]]:
    """Причина исключения по одному фильтру (None — остаётся) и группы дублей для Ф2."""
    if name == "Ф2":
        groups = tf.duplicate_groups(list(frame["name_en"]), list(frame["score"]))
        reason = pd.Series([None] * len(frame), index=frame.index, dtype=object)
        for group in groups:
            for member in group[1:]:
                reason.iloc[member] = tf.REASON_DUPLICATE.format(kept=frame["name_en"].iloc[group[0]])
        return reason, groups
    if name == "Ф1":
        keep = [not tf.is_mature(n, t_mature) for n in frame["n_research"]]
        text = [tf.REASON_MATURE.format(n=int(n)) for n in frame["n_research"]]
    elif name == "Ф3":
        keep = [not tf.is_template(term) for term in frame["name_en"]]
        text = [tf.REASON_TEMPLATE] * len(frame)
    else:
        column = "d_pool" if name == "Ф4a" else "d_authors"
        keep = list(frame[column] >= 2)
        text = [tf.REASON_SINGLE] * len(frame)
    return pd.Series([None if k else r for k, r in zip(keep, text)], index=frame.index, dtype=object), []


def apply(frame: pd.DataFrame, names: list[str], t_mature: float) -> tuple[pd.DataFrame, list]:
    """Фильтры по порядку на оставшихся; колонки «фильтр» и «причина». Второе — группы Ф2."""
    out = frame.assign(фильтр=None, причина=None)
    groups = []
    for name in names:
        alive = out.loc[out["причина"].isna()]
        reason, found = reasons(alive.reset_index(drop=True), name, t_mature)
        reason.index = alive.index
        hit = reason.notna()
        out.loc[hit[hit].index, "фильтр"] = name
        out.loc[hit[hit].index, "причина"] = reason[hit]
        groups += [[alive.index[i] for i in group] for group in found if len(group) > 1]
    return out, groups


def top(frame: pd.DataFrame) -> pd.DataFrame:
    """ТОП-15: оставшиеся выше порога по убыванию score, без добивки ниже порога."""
    left = frame.loc[frame["причина"].isna() & frame["above"]]
    chosen = left.sort_values("score", ascending=False, kind="stable").head(TOP_N)
    return chosen.assign(место=range(1, len(chosen) + 1))


def show(chosen: pd.DataFrame, with_repeat: bool) -> pd.DataFrame:
    """Лист ТОП-15: место, термин, score, метка; для Ф4 — ещё d_pool и d_authors."""
    columns = ["место", "name_en", "score", "blind_id", "label"]
    columns += ["d_pool", "d_authors"] if with_repeat else []
    table = chosen[columns].rename(columns={"name_en": "term_en", "label": "метка"})
    return table.assign(score=table["score"].round(4), метка=table["метка"].fillna("—"),
                        blind_id=table["blind_id"].fillna("—"))


def dropped(base_top: pd.DataFrame, filtered: pd.DataFrame) -> pd.DataFrame:
    """Кандидаты исходного ТОП-15, которых исключил фильтр: метка и причина."""
    rows = filtered.loc[base_top.index]
    rows = rows.loc[rows["причина"].notna()].assign(место_без_фильтра=base_top["место"])
    table = rows[["место_без_фильтра", "name_en", "score", "blind_id", "label", "фильтр", "причина"]]
    return table.rename(columns={"name_en": "term_en", "label": "метка"}).assign(
        score=table["score"].round(4), метка=table["label"].fillna("—"),
        blind_id=table["blind_id"].fillna("—"))


def counts(tops: list[pd.DataFrame], frames: list[pd.DataFrame], groups: list[list]) -> dict:
    """Числа для правила принятия, суммарно по ТОП-15 двух пулов."""
    both = pd.concat(tops)
    pool = pd.concat(frames)
    kept_groups = sum(group[0] in chosen.index for chosen, found in zip(tops, groups)
                      for group in found)
    plausible = both.loc[both["label"] == "plausible_signal", "concept"]
    return {"шум": int((both["label"] == "noise").sum()),
            "зрелые + зонтичные": int(both["label"].isin(["mature", "umbrella"]).sum()),
            "разных plausible-понятий": int(plausible.nunique()),
            "plausible исключено в пуле": int(((pool["label"] == "plausible_signal")
                                               & pool["причина"].notna()).sum()),
            "склеенных групп в ТОП": int(kept_groups),
            "uncertain в ТОП": int((both["label"] == "uncertain").sum()),
            "без метки в ТОП": int(both["label"].isna().sum()), "мест в ТОП": len(both)}


def passes(name: str, row: dict, signals_cut: int) -> bool:
    """Правило принятия, условия 1–4, механически по числам (для выбора состава «ВСЕ»)."""
    if name == "Ф2":
        first, effect = row["разных plausible-понятий"] >= BASE["разных plausible-понятий"], \
            row["склеенных групп в ТОП"] >= 2
    else:
        first = row["plausible исключено в пуле"] == 0
        effect = (row["зрелые + зонтичные"] <= BASE["зрелые + зонтичные"] - 2 if name == "Ф1"
                  else row["шум"] <= BASE["шум"] - 2)
    rest = (row["шум"] <= BASE["шум"] and row["зрелые + зонтичные"] <= BASE["зрелые + зонтичные"]
            and row["разных plausible-понятий"] >= BASE["разных plausible-понятий"])
    return first and effect and rest and (name != "Ф3" or signals_cut == 0)


def training_check(train: pd.DataFrame, t_mature: float) -> dict[str, pd.DataFrame]:
    """Т1: Ф1 и Ф3 на 100 сигналах и 60 мейнстримах обучающей таблицы, со списком названий."""
    classes = {1: "сигнал", 0: "мейнстрим"}
    names = pd.read_csv(TECHNOLOGIES, encoding="utf-8-sig").set_index("tech_id")["name_ru"]
    cut = {"Ф1": train["n_research6"] > t_mature,
           "Ф3": train["name_en"].map(tf.is_template)}
    sheets, summary = {}, []
    for name, mask in cut.items():
        hit = train.loc[mask]
        sheets[f"Т1 {name} обучение"] = pd.DataFrame(
            {"tech_id": hit["tech_id"], "класс": hit["label"].map(classes),
             "name_en": hit["name_en"], "name_ru": hit["tech_id"].map(names),
             "n_research": hit["n_research6"]})
        summary.append({"фильтр": name, **{f"{classes[label]}ов исключено": int(
            (hit["label"] == label).sum()) for label in classes},
            "сигналов всего": int((train["label"] == 1).sum()),
            "мейнстримов всего": int((train["label"] == 0).sum())})
    return {"Т1 сводка": pd.DataFrame(summary), **sheets}


def variant(pools: dict, names: list[str], label: str, t_mature: float) -> tuple[dict, dict, dict]:
    """Один вариант на обоих пулах: листы ТОП и «выбыли», строки счёта, числа правила и ТОПы."""
    sheets, rows, tops, frames, groups = {}, [], {}, [], []
    for pool, frame in pools.items():
        filtered, found = apply(frame, names, t_mature)
        chosen, base_top = top(filtered), top(frame.assign(причина=None))
        sheets[f"Т2 {label} {pool} ТОП"] = show(chosen, any(n.startswith("Ф4") for n in names))
        sheets[f"Т2 {label} {pool} выбыли"] = dropped(base_top, filtered)
        passed = filtered["причина"].isna()
        rows.append({"вариант": label, "пул": pool, "кандидатов со score": len(filtered),
                     "прошло фильтр": int(passed.sum()),
                     "из них выше порога": int((passed & filtered["above"]).sum()),
                     "в ТОП-15": len(chosen)})
        tops[pool], groups = chosen, groups + [found]
        frames.append(filtered)
    return sheets, {"строки": rows, **counts(list(tops.values()), frames, groups)}, tops


def repeat_sheets(pools: dict, glued: dict) -> dict[str, pd.DataFrame]:
    """Ф4: все кандидаты обоих пулов, медианы по меткам, склейка документов по заголовку."""
    both = pd.concat(pools.values())
    table = pd.DataFrame({"пул": both["пул"], "term_en": both["name_en"],
                          "метка": both["label"].fillna("—"), "d_pool": both["d_pool"],
                          "d_authors": both["d_authors"]})
    medians = table.groupby("метка").agg(кандидатов=("term_en", "size"),
                                         d_pool_медиана=("d_pool", "median"),
                                         d_authors_медиана=("d_authors", "median")).reset_index()
    documents = pd.DataFrame([{"пул": pool, "документов": len(run(pool)[1]["documents"]),
                               "склеено по заголовку": glued[pool],
                               "уникальных": len(run(pool)[1]["documents"]) - glued[pool],
                               "Ф4b": NO_AUTHORS} for pool in pools])
    return {"Т2 Ф4 кандидаты": table, "Т2 Ф4 медианы": medians.assign(Ф4b=NO_AUTHORS),
            "Т2 Ф4 склейка документов": documents}


def mixed_groups(pools: dict) -> pd.DataFrame:
    """Ф2 отдельно: группы дублей с разными метками — метка каждого члена и кто остался."""
    rows = []
    for pool, frame in pools.items():
        groups = tf.duplicate_groups(list(frame["name_en"]), list(frame["score"]))
        for number, group in enumerate([g for g in groups if len(g) > 1], start=1):
            marks = frame["label"].iloc[group].fillna("—")
            if marks.nunique() < 2:
                continue
            union = sorted({i for member in group for i in frame["doc_ids"].iloc[member]})
            for member in group:
                rows.append({"пул": pool, "группа": number, "term_en": frame["name_en"].iloc[member],
                             "score": round(float(frame["score"].iloc[member]), 4),
                             "метка": marks.loc[frame.index[member]],
                             "остался": member == group[0],
                             "doc_ids после склейки": len(union) if member == group[0] else None})
    return pd.DataFrame(rows, columns=["пул", "группа", "term_en", "score", "метка", "остался",
                                       "doc_ids после склейки"])


def blind(all_tops: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Слепой лист: термины из любого ТОП-15 задачи без метки, перемешаны с seed 42."""
    seen, key_rows = {}, []
    for (label, pool), chosen in all_tops.items():
        for _, row in chosen.loc[chosen["label"].isna()].iterrows():
            seen.setdefault(row["tech_key"], {"term_en": row["name_en"], "term_ru": row["name_ru"],
                                              "quote": row["quote"],
                                              **first_document(pool, row["name_en"])})
            key_rows.append({"tech_key": row["tech_key"], "вариант": label, "пул": pool,
                             "место": int(row["место"]), "score": round(float(row["score"]), 4)})
    order = np.random.default_rng(42).permutation(sorted(seen))
    ids = {term: f"T{number:02d}" for number, term in enumerate(order, start=1)}
    sheet = pd.DataFrame([{"new_blind_id": ids[term], **seen[term]} for term in order],
                         columns=["new_blind_id", "term_en", "term_ru", "quote", "заголовок", "url"])
    key = pd.DataFrame(key_rows, columns=["tech_key", "вариант", "пул", "место", "score"])
    key.insert(0, "new_blind_id", key["tech_key"].map(ids))
    return sheet, key.sort_values(["new_blind_id", "вариант", "пул"]).reset_index(drop=True)


def append_t(sheets: dict[str, pd.DataFrame], prefix: str = "Т") -> list[str]:
    """Дописывает листы задачи (prefix) в report_tables.xlsx; чужие не трогает и проверяет, что целы."""
    before = sheet_names()
    clash = [name for name in sheets if name[:31] in before and not name.startswith(prefix)]
    if clash or len({name[:31] for name in sheets}) != len(sheets):
        raise ValueError(f"конфликт имён листов: {clash}")
    with pd.ExcelWriter(REPORT, engine="openpyxl", mode="a", if_sheet_exists="replace") as writer:
        for title, table in sheets.items():
            table.to_excel(writer, sheet_name=title[:31], index=False)
    after = sheet_names()
    lost = [name for name in before if name not in after]
    if lost:
        raise RuntimeError(f"пропали прежние листы: {lost}")
    return after


def control(pools: dict) -> pd.DataFrame:
    """ТОП-15 без фильтров против листов «П2 ТОП PB»: термин и score на каждом месте."""
    rows = []
    for pool, frame in pools.items():
        mine = top(frame.assign(причина=None)).reset_index(drop=True)
        theirs = pd.read_excel(REPORT, sheet_name=f"П2 ТОП PB {pool}")
        for place in range(max(len(mine), len(theirs))):
            got = mine.iloc[place] if place < len(mine) else None
            want = theirs.iloc[place] if place < len(theirs) else None
            same = (got is not None and want is not None and got["name_en"] == want["term_en"]
                    and abs(round(float(got["score"]), 4) - float(want["score"])) < 1e-9)
            rows.append({"пул": pool, "место": place + 1,
                         "Т": None if got is None else got["name_en"],
                         "П2": None if want is None else want["term_en"], "совпало": same})
    return pd.DataFrame(rows)


def rule_table(results: dict, signals_cut: int) -> pd.DataFrame:
    """Числа правила по вариантам против базы; решаемость — нет ли в ТОП неразмеченных."""
    rows = [{"вариант": "база s2a2-v1", **BASE}]
    for label, result in results.items():
        numbers = {key: value for key, value in result.items() if key != "строки"}
        name = "Ф4" if label.startswith("Ф4") else label
        decided = ("нельзя решить: в ТОП есть новички" if numbers["без метки в ТОП"]
                   else ("условия 1–4 выполнены" if passes(name, numbers, signals_cut)
                         else "условия 1–4 не выполнены"))
        rows.append({"вариант": label, **numbers,
                     "по числам": decided if label != "ВСЕ" else "—"})
    return pd.DataFrame(rows)


def all_names(results: dict, signals_cut: int) -> tuple[list[str], str]:
    """Состав «ВСЕ»: прошедшие правило в порядке Ф2 → Ф1 → Ф3 → Ф4, если решаемо; иначе все."""
    if any(results[label]["без метки в ТОП"] for label in SINGLE):
        return ["Ф2", "Ф1", "Ф3", "Ф4a"], "все фильтры: в ТОП есть неразмеченные новички"
    ok = {label for label in SINGLE
          if passes("Ф4" if label.startswith("Ф4") else label, results[label], signals_cut)}
    four = "Ф4a" if "Ф4a" in ok else ("Ф4b" if "Ф4b" in ok else None)
    names = [n for n in ["Ф2", "Ф1", "Ф3"] if n in ok] + ([four] if four else [])
    return names, "только прошедшие правило по числам К"


def prepare() -> tuple[pd.DataFrame, dict, dict, float, float]:
    """Обучение PB со сверкой весов и порога, T_mature, пулы с метками и Ф4-счётчиками."""
    train = feature_table()
    fitted = fit_sets(train)
    check = weights_check(fitted)
    if not check["совпало"].all():
        raise RuntimeError(f"веса/порог PB не совпали с листом «П1 веса»:\n{check}")
    model, threshold = fitted["PB"]
    t_mature = tf.mature_threshold(train.loc[train["label"] == 1, "n_research6"])
    print(f"T_mature = {t_mature:.4f}: 95-й процентиль n_research 100 обучающих сигналов; "
          f"окно порога и окно кандидатов одно — OpenAlex + arXiv, годы "
          f"{'…'.join(AGGREGATE_WINDOWS['all'][::5])} (2020-09…2026-08), без prev6")
    marks = labels()
    built = {pool: pool_table(pool, model, threshold, marks) for pool in POOLS}
    pools = {pool: pair[0] for pool, pair in built.items()}
    glued = {pool: pair[1] for pool, pair in built.items()}
    return train, pools, glued, t_mature, threshold


def main() -> None:
    """Т1 и Т2 целиком: печать таблиц, дозапись листов «Т», ключ слепой разметки."""
    train, pools, glued, t_mature, threshold = prepare()
    sheets = {"Т порог Ф1": pd.DataFrame([{
        "T_mature": round(t_mature, 4), "сигналов": int((train["label"] == 1).sum()),
        "окно порога": "OpenAlex + arXiv, 6 годовых окон 2020-09…2026-08, без prev6",
        "окно кандидатов": "OpenAlex + arXiv, 6 годовых окон 2020-09…2026-08, без prev6",
        "порог модели": threshold}]), "Т контроль базы = П2": control(pools)}
    if not sheets["Т контроль базы = П2"]["совпало"].all():
        raise RuntimeError("ТОП-15 без фильтров разошёлся с листами П2")
    sheets.update(training_check(train, t_mature))
    signals_cut = int(sheets["Т1 сводка"].set_index("фильтр").loc["Ф3", "сигналов исключено"])
    results, all_tops, rows = {}, {}, []
    for label in SINGLE:
        part, results[label], tops = variant(pools, [label], label, t_mature)
        sheets.update(part)
        all_tops.update({(label, pool): chosen for pool, chosen in tops.items()})
        rows += results[label].pop("строки")
    names, why = all_names(results, signals_cut)
    part, results["ВСЕ"], tops = variant(pools, names, "ВСЕ", t_mature)
    sheets.update(part)
    all_tops.update({("ВСЕ", pool): chosen for pool, chosen in tops.items()})
    rows += results["ВСЕ"].pop("строки")
    sheets["Т2 счёт кандидатов"] = pd.DataFrame(rows)
    sheets["Т2 сводка правила"] = rule_table(results, signals_cut)
    sheets["Т2 ВСЕ состав"] = pd.DataFrame([{"порядок": " → ".join(names), "почему": why}])
    sheets["Т2 Ф2 смешанные группы"] = mixed_groups(pools)
    sheets.update(repeat_sheets(pools, glued))
    sheet, key = blind(all_tops)
    sheets["Т новички слепо"] = sheet
    for title, table in sheets.items():
        print(f"\n{title}\n{table.to_string(index=False)}")
    key.to_csv(T_BLIND_KEY, index=False, encoding="utf-8-sig")
    after = append_t(sheets)
    print(f"\nлистов в {REPORT.name}: {len(after)}; ключ: {T_BLIND_KEY.relative_to(ROOT)} ({len(key)} строк)")


if __name__ == "__main__":
    main()
