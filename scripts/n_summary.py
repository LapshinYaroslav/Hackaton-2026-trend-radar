"""Задача Н, сведение после разметки: счёт по конфигурациям, правила 1–3, исключённые, матрица Н2.

Метки — labels/n_blind_labels.csv по ключу data/interim/n_blind_key.csv, термины — (тема, tech_key).
ТОП-15 — data/interim/n_tops.csv. Причины исключения в n_tops.csv не хранятся, поэтому пулы B и C
пересчитываются функциями scripts.n_report; ответы YandexGPT — только из кэша, без сети.

Запуск: python -m scripts.n_summary
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from collector.api import tech_key
from scripts import n_report
from scripts.t_report import append_t, top

ROOT = Path(__file__).resolve().parents[1]
LABELS = ROOT / "labels" / "n_blind_labels.csv"
GARBAGE = ("noise", "mature", "umbrella")
CONFIGS = ("A0", "A", "B", "C")


def labels() -> pd.DataFrame:
    """Метка, понятие и on_topic по (тема, tech_key)."""
    key = pd.read_csv(n_report.N_BLIND_KEY, encoding="utf-8-sig")
    marks = pd.read_csv(LABELS, encoding="utf-8-sig").merge(key, on="n_id", how="inner", validate="1:1")
    if len(marks) != len(key):
        raise ValueError("разметка не покрывает ключ")
    marks = marks.assign(tech_key=marks["term_en"].map(tech_key),
                         concept=marks["canonical_concept"].str.strip().str.lower())
    return marks[["тема", "tech_key", "n_id", "label", "concept", "on_topic", "note"]]


def tally(rows: pd.DataFrame) -> dict:
    """Счёт одного ТОП-15: категории, мусор, разные plausible-понятия."""
    off = rows["on_topic"] == "no"
    other_off = off & ~rows["label"].isin(GARBAGE)
    plausible = rows.loc[rows["label"] == "plausible_signal", "concept"]
    return {"мест": len(rows), **{label: int((rows["label"] == label).sum()) for label in GARBAGE},
            "прочих on_topic=no": int(other_off.sum()),
            "мусор": int(rows["label"].isin(GARBAGE).sum() + other_off.sum()),
            "uncertain": int((rows["label"] == "uncertain").sum()),
            "plausible_signal мест": int(plausible.size),
            "plausible on_topic=no": int(((rows["label"] == "plausible_signal") & off).sum()),
            "разных plausible-понятий": int(plausible.nunique())}


def table(tops: pd.DataFrame) -> pd.DataFrame:
    """Счёт по конфигурации и теме плюс сумма по трём темам (понятия — сумма по темам)."""
    rows = []
    for config in CONFIGS:
        parts = [{"конфигурация": config, "тема": theme,
                  **tally(tops.loc[(tops["конфигурация"] == config) & (tops["тема"] == theme)])}
                 for theme in n_report.THEMES]
        total = {name: sum(part[name] for part in parts) for name in parts[0] if name not in ("конфигурация", "тема")}
        rows += parts + [{"конфигурация": config, "тема": "сумма по трём темам", **total}]
    return pd.DataFrame(rows)


def rules(summary: pd.DataFrame) -> pd.DataFrame:
    """Правила 1–3 по суммам: левые и правые части неравенств."""
    total = summary.loc[summary["тема"] == "сумма по трём темам"].set_index("конфигурация")
    garbage, concepts = total["мусор"], total["разных plausible-понятий"]
    spec = [("1: A против A0", "A", garbage["A0"], "<=", "A0"),
            ("2: B против A", "B", 0.75 * garbage["A"], "<=", "A"),
            ("3: C против B", "C", garbage["B"], "<", "B")]
    rows = []
    for name, left, bound, sign, right in spec:
        holds = garbage[left] <= bound if sign == "<=" else garbage[left] < bound
        rows.append({"правило": name, "проверяется": left, "мусор": int(garbage[left]),
                     "граница мусора": f"{sign} {bound:g}", "мусор: неравенство выполнено": bool(holds),
                     "понятий": int(concepts[left]), "эталон": right, "понятий у эталона": int(concepts[right]),
                     "понятия: не меньше": bool(concepts[left] >= concepts[right])})
    return pd.DataFrame(rows)


def cache_only(prompt: str, temperature: float, spent: dict) -> str | None:
    """Замена n_report.ask при сведении: ответ YandexGPT только из кэша, иначе отказ."""
    path = n_report.relevance_path(prompt, temperature)
    if not path.exists():
        raise FileNotFoundError("ответа YandexGPT нет в кэше — сведение без сети невозможно")
    spent["из кэша"] += 1
    return json.loads(path.read_text(encoding="utf-8"))["text"]


def pools() -> dict[str, dict[str, pd.DataFrame]]:
    """Пулы A, B, C по темам, как в n_report.main, без сети."""
    n_report.ask = cache_only
    data, frames, meta = n_report.prepare()
    cache = n_report.search_documents()
    out = {}
    for theme, frame in frames.items():
        docs, _ = n_report.first_documents(theme, frame, cache)
        spent = {"вызовов": 0, "из кэша": 0, "ошибок API": 0, "повторов": 0}
        out[theme] = n_report.configurations(data[theme]["topic"], frame, docs, meta["limits"], spent)
    return out


def same_tops(pooled: dict, tops: pd.DataFrame) -> None:
    """Пересчитанные ТОП-15 обязаны совпасть с n_tops.csv: место и tech_key."""
    for theme, by_config in pooled.items():
        for config, pool in by_config.items():
            got = list(top(pool)["tech_key"])
            saved = tops.loc[(tops["тема"] == theme) & (tops["конфигурация"] == config)].sort_values("место")
            if got != list(saved["tech_key"]):
                raise RuntimeError(f"ТОП {theme} {config} разошёлся с n_tops.csv")


def dropped(pooled: dict, marks: pd.DataFrame) -> pd.DataFrame:
    """Для B и C: термины ТОП-15 A, исключённые фильтрами, с меткой и причиной."""
    rows = []
    for theme, by_config in pooled.items():
        top_a = top(by_config["A"])
        for config in ("B", "C"):
            pool = by_config[config].loc[top_a.index]
            for index, row in pool.loc[pool["причина"].notna()].iterrows():
                rows.append({"тема": theme, "конфигурация": config, "место в A": int(top_a.loc[index, "место"]),
                             "term_en": row["name_en"], "tech_key": row["tech_key"], "score A": round(float(row["score"]), 4),
                             "фильтр": row["фильтр"], "причина": row["причина"]})
    frame = pd.DataFrame(rows)
    return frame.merge(marks, on=["тема", "tech_key"], how="left").drop(columns=["tech_key", "note"])


def matrix(pooled: dict, marks: pd.DataFrame) -> pd.DataFrame:
    """Н2: ответ YandexGPT × on_topic по проверенным кандидатам с меткой; по темам и всего."""
    checked = pd.concat([pool["C"].loc[pool["C"]["relevant"].notna()].assign(тема=theme)
                         for theme, pool in pooled.items()])
    checked = checked.merge(marks, on=["тема", "tech_key"], how="left")
    rows = []
    for theme in [*n_report.THEMES, "все темы"]:
        part = checked if theme == "все темы" else checked.loc[checked["тема"] == theme]
        labelled = part.loc[part["on_topic"].notna()]
        row = {"тема": theme, "проверено": len(part), "с меткой": len(labelled)}
        for answer in (True, False):
            for topic in ("yes", "no"):
                row[f"relevant={str(answer).lower()} × on_topic={topic}"] = int(
                    ((labelled["relevant"] == answer) & (labelled["on_topic"] == topic)).sum())
        rows.append(row)
    return pd.DataFrame(rows)


def main() -> None:
    """Сведение: печать и дозапись листов «Н»."""
    marks = labels()
    tops = pd.read_csv(n_report.N_TOPS, encoding="utf-8-sig")
    labelled = tops.merge(marks, on=["тема", "tech_key"], how="left")
    if labelled["label"].isna().any():
        raise ValueError("в ТОП-15 есть термины без метки")
    pooled = pools()
    same_tops(pooled, tops)
    summary = table(labelled)
    sheets = {"Н итог по темам": summary, "Н правила": rules(summary),
              "Н исключено из ТОП A": dropped(pooled, marks), "Н2 матрица": matrix(pooled, marks)}
    for title, frame in sheets.items():
        print(f"\n{title}\n{frame.to_string(index=False)}")
    after = append_t(sheets, prefix="Н")
    print(f"\nлистов в книге: {len(after)}")


if __name__ == "__main__":
    main()
