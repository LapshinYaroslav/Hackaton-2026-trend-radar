"""Задача Н: слепые темы — конфигурации A0, A, B, C, фильтр релевантности, слепой лист.

Слепота: в печать и в листы «Н», кроме «Н слепо», идут только числа. Полные ТОП-15 по
конфигурациям — в data/interim/n_tops.csv. Сеть — только к YandexGPT (Н2), ответы
кэшируются в data/interim/cache/relevance. Счётчики и n_pat — только из кэша; нет в кэше — отказ.

Первый документ кандидата: у финтеха — первый по doc_ids из файла документов прогона; у
кибербезопасности и ИИ файла документов нет, и берётся первый документ кэша поиска №1,
содержащий цитату кандидата (на финтехе совпадает с doc_ids у 60 из 60).

Запуск: python -m scripts.n_report
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
from model.features import RESEARCH_SOURCES
from model.predict import load, predict
from scripts import t_filters as tf
from scripts.ru_patents_collect import cache_path, request_key
from scripts.ru_patents_p0 import feature_table
from scripts.t_report import append_t, apply, top
from search import llm_yandex_gpt as llm

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "data" / "interim" / "pipeline_runs"
COUNTERS = ROOT / "data" / "interim" / "cache" / "counters_pipeline"
SEARCH = ROOT / "data" / "interim" / "cache" / "search_recent"
RELEVANCE = ROOT / "data" / "interim" / "cache" / "relevance"
N_TOPS = ROOT / "data" / "interim" / "n_tops.csv"
N_BLIND_KEY = ROOT / "data" / "interim" / "n_blind_key.csv"
THEMES = {"финтех": "q20260923170352", "кибербезопасность": "q20260923192323",
          "технологии в ИИ": "q20260924224939"}
CYBER_SECOND = "q20260923200541"
S2A2_SAVED_RUN = "q20260924234025"  # роботы, s2a2-v1: единственный прогон с сохранёнными оценками s2a2
FINTECH_EXTRA = "l3_extra_20260923T170351"
MODES = {"openalex": "per_window", "arxiv": "one_call", "techcrunch": "one_call"}
T_MATURE = 4870.45
MODEL, RETRY_TEMPERATURE = "yandexgpt-5-pro", 0.3
REASON_OFF_TOPIC = "Не относится к теме запроса"
PROMPT = ("Тема запроса пользователя: «{topic}».\n"
          "Кандидат: «{term_en}» ({term_ru}).\n"
          "Фрагмент источника: «{quote}» — из «{title}».\n"
          "Относится ли эта технология к теме запроса? Ответь только JSON: "
          "{{\"relevant\": true}} или {{\"relevant\": false}}.")


def run(run_id: str) -> dict:
    """Файл прогона."""
    return json.loads((RUNS / f"{run_id}.json").read_text(encoding="utf-8"))


def scored(data: dict) -> list[dict]:
    """Кандидаты прогона, которые оценивала модель: ТОП и исключённые со score."""
    return data["top"] + [item for item in data["excluded"] if item["score"] is not None]


def counters(name_en: str) -> pd.DataFrame:
    """Годовые счётчики из кэша пайплайна в режимах прогонов Н. Нет файла — отказ, не сеть."""
    rows = []
    for source, mode in MODES.items():
        raw = json.dumps([tech_key(name_en), source, mode], ensure_ascii=False)
        path = COUNTERS / f"{hashlib.sha256(raw.encode('utf-8')).hexdigest()}.json"
        if not path.exists():
            raise FileNotFoundError(f"нет счётчика в кэше: {source}/{mode} — задача останавливается")
        rows += [{"source": row["source"], "window": row["window"], "n": row["n"]}
                 for row in json.loads(path.read_text(encoding="utf-8"))]
    return pd.DataFrame(rows, columns=["source", "window", "n"])


def n_pat(name_en: str) -> int:
    """n_pat из кэша Роспатента (П, Р4). Нет файла — отказ."""
    path = cache_path(request_key("rospatent", tech_key(name_en)))
    if not path.exists():
        raise FileNotFoundError("нет n_pat в кэше Роспатента — задача останавливается")
    return int(json.loads(path.read_text(encoding="utf-8"))["response"]["total"])


def candidates(data: dict) -> pd.DataFrame:
    """Оценённые кандидаты прогона: оценки A0 (s2a1-v1) и A (s2a2-v1), n_research, n_pat."""
    models = {version: load(version=version) for version in ("s2a1-v1", "s2a2-v1")}
    rows = []
    for item in scored(data):
        table, patents = counters(item["name_en"]), n_pat(item["name_en"])
        six = table.loc[table["window"].isin(AGGREGATE_WINDOWS["all"])
                        & table["source"].isin(RESEARCH_SOURCES), "n"]
        score = {version: predict(features_from_counters(table, n_pat=patents, version=version),
                                  data["area"], artifact)[0] for version, artifact in models.items()}
        rows.append({"name_en": item["name_en"], "name_ru": item["name_ru"], "quote": item["quote"],
                     "tech_key": tech_key(item["name_en"]), "n_research": int(six.sum()),
                     "n_pat": patents, "n_pat_saved": item.get("n_pat"),
                     "score_saved": item["score"], "score_A0": score["s2a1-v1"],
                     "score_A": score["s2a2-v1"]})
    return pd.DataFrame(rows)


def thresholds() -> dict[str, float]:
    """Пороги из артефактов: A0 — s2a1-v1, A — s2a2-v1."""
    return {"A0": load(version="s2a1-v1")["meta"]["threshold"],
            "A": load(version="s2a2-v1")["meta"]["threshold"]}


def check_row(what: str, got: pd.Series, saved: pd.Series) -> dict:
    """Строка сверки: сколько совпало с допуском 1e-9 и максимальное расхождение."""
    known = saved.notna()
    diff = (got[known].astype(float) - saved[known].astype(float)).abs()
    return {"сверка": what, "кандидатов": int(known.sum()), "совпало": int((diff <= 1e-9).sum()),
            "макс. расхождение": float(diff.max()) if len(diff) else None}


def verify(frames: dict[str, pd.DataFrame], saved_versions: dict[str, str]) -> pd.DataFrame:
    """A0 против оценок прогонов s2a1-v1; A против прогона роботов s2a2-v1; n_pat против прогонов."""
    rows = []
    for theme, frame in frames.items():
        column = "score_A0" if saved_versions[theme] == "s2a1-v1" else "score_A"
        rows.append(check_row(f"{theme}: {column} = score в прогоне ({saved_versions[theme]})",
                              frame[column], frame["score_saved"]))
        if frame["n_pat_saved"].notna().any():
            rows.append(check_row(f"{theme}: n_pat кэша = n_pat в прогоне", frame["n_pat"],
                                  frame["n_pat_saved"]))
    robots = candidates(run(S2A2_SAVED_RUN))
    rows.append(check_row(f"роботы {S2A2_SAVED_RUN}: score_A = score в прогоне (s2a2-v1)",
                          robots["score_A"], robots["score_saved"]))
    rows.append(check_row(f"роботы {S2A2_SAVED_RUN}: n_pat кэша = n_pat в прогоне",
                          robots["n_pat"], robots["n_pat_saved"]))
    return pd.DataFrame(rows)


def norm(text: str | None) -> str:
    """Нижний регистр и схлопнутые пробелы: для поиска цитаты в документе."""
    return " ".join((text or "").lower().split())


def search_documents() -> list[dict]:
    """Все документы кэша поиска №1 в порядке файлов и документов в файле."""
    return [doc for path in sorted(SEARCH.glob("*.json"))
            for doc in json.loads(path.read_text(encoding="utf-8"))["documents"]]


def first_documents(theme: str, frame: pd.DataFrame, cache: list[dict]) -> tuple[list[dict], dict]:
    """Заголовок, url и источник первого документа кандидата; счёт способов и неоднозначностей.

    subquery_ids есть только у финтеха (из файла документов прогона); у остальных — None.
    """
    if theme == "финтех":
        extra = json.loads((RUNS / f"{FINTECH_EXTRA}.json").read_text(encoding="utf-8"))
        found = [extra["documents"][extra["doc_ids"][name][0] - 1] for name in frame["name_en"]]
        return ([{"заголовок": doc["title"], "url": doc["url"], "source": doc["source"],
                  "subquery_ids": doc["subquery_ids"]} for doc in found],
                {"тема": theme, "способ": "doc_ids прогона", "кандидатов": len(found)})
    texts = [norm(doc.get("title")) + " \n " + norm(doc.get("text")) for doc in cache]
    out, ambiguous, missing = [], 0, 0
    for quote in frame["quote"]:
        hits = [i for i, text in enumerate(texts) if norm(quote) and norm(quote) in text]
        ambiguous += len({cache[i]["url"] for i in hits}) > 1
        missing += not hits
        doc = cache[hits[0]] if hits else {}
        out.append({"заголовок": doc.get("title"), "url": doc.get("url"), "source": doc.get("source"),
                    "subquery_ids": None})
    return out, {"тема": theme, "способ": "цитата в кэше поиска №1", "кандидатов": len(out),
                 "не найдено": missing, "несколько url, взят первый": ambiguous}


def parse_relevance(text: str | None) -> bool | None:
    """{"relevant": bool} → bool; всё остальное — невалидный ответ (None)."""
    try:
        answer = json.loads((text or "").strip())
    except json.JSONDecodeError:
        return None
    value = answer.get("relevant") if isinstance(answer, dict) else None
    return value if isinstance(value, bool) else None


def relevance_path(prompt: str, temperature: float) -> Path:
    """Файл кэша ответа YandexGPT: ключ — модель, температура и промпт."""
    key = json.dumps([MODEL, temperature, prompt], ensure_ascii=False)
    return RELEVANCE / f"{hashlib.sha256(key.encode('utf-8')).hexdigest()}.json"


def ask(prompt: str, temperature: float, spent: dict) -> str | None:
    """Один вызов YandexGPT: промпт — единственное сообщение user. Ответы с текстом кэшируются."""
    path = relevance_path(prompt, temperature)
    if path.exists():
        spent["из кэша"] += 1
        return json.loads(path.read_text(encoding="utf-8"))["text"]
    body = {"modelUri": llm.build_model_uri(MODEL), "json_object": True,
            "completionOptions": {"stream": False, "temperature": temperature, "maxTokens": "50"},
            "messages": [{"role": "user", "text": prompt}]}
    headers = {"Authorization": f"Api-Key {os.environ['YANDEX_API_KEY'].strip()}",
               "x-folder-id": os.environ["YANDEX_FOLDER_ID"].strip(), "Content-Type": "application/json"}
    spent["вызовов"] += 1
    try:
        text, version, _ = llm._read_answer(llm._post_with_retry(body, headers), False)
    except Exception as exc:  # сбой сети или API = невалидный ответ, в кэш не пишется
        spent["ошибок API"] += 1
        llm._log_call({"purpose": "n2-relevance", "user_prompt": prompt, "error": repr(exc)})
        return None
    llm._log_call({"purpose": "n2-relevance", "model_version": version, "temperature": temperature,
                   "user_prompt": prompt, "response_text": text})
    RELEVANCE.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"prompt": prompt, "temperature": temperature, "text": text,
                                "model_version": version}, ensure_ascii=False), encoding="utf-8")
    return text


def relevance(prompt: str, spent: dict) -> tuple[bool | None, bool]:
    """Н2: температура 0, при невалидном ответе один повтор с 0.3. (relevant, relevance_failed)."""
    answer = parse_relevance(ask(prompt, 0.0, spent))
    if answer is None:
        spent["повторов"] += 1
        answer = parse_relevance(ask(prompt, RETRY_TEMPERATURE, spent))
    return answer, answer is None


def relevance_pass(topic: str, out_b: pd.DataFrame, docs: list[dict], spent: dict) -> pd.DataFrame:
    """C: Н2 для прошедших B и выше порога; исключение только при relevant = false."""
    out = out_b.assign(relevant=None, relevance_failed=False)
    for index in out.index[out["причина"].isna() & out["above"]]:
        row = out.loc[index]
        prompt = PROMPT.format(topic=topic, term_en=row["name_en"], term_ru=row["name_ru"],
                               quote=row["quote"], title=docs[index]["заголовок"])
        answer, failed = relevance(prompt, spent)
        out.loc[index, "relevant"], out.loc[index, "relevance_failed"] = answer, failed
        if answer is False:
            out.loc[index, "фильтр"], out.loc[index, "причина"] = "Н2", REASON_OFF_TOPIC
    return out


def configurations(topic: str, frame: pd.DataFrame, docs: list[dict], limits: dict,
                   spent: dict) -> dict[str, pd.DataFrame]:
    """Пул после каждой конфигурации: колонки score, above, фильтр, причина."""
    base = {name: frame.assign(score=frame[f"score_{name}"], above=frame[f"score_{name}"] >= limits[name],
                               фильтр=None, причина=None) for name in ("A0", "A")}
    out_b, _ = apply(base["A"], ["Ф2", "Ф1", "Ф3"], T_MATURE)
    return {**base, "B": out_b, "C": relevance_pass(topic, out_b, docs, spent)}


def place_counts(theme: str, pools: dict[str, pd.DataFrame]) -> list[dict]:
    """Числа без терминов: кандидаты, выше порога, исключения по фильтрам, заполненные места."""
    rows = []
    for name, pool in pools.items():
        cut = pool["причина"].notna()
        rows.append({"тема": theme, "конфигурация": name, "кандидатов со score": len(pool),
                     "выше порога до фильтров": int(pool["above"].sum()),
                     **{f"исключил {f}": int((pool["фильтр"] == f).sum()) for f in ("Ф2", "Ф1", "Ф3", "Н2")},
                     "исключено выше порога": int((cut & pool["above"]).sum()),
                     "мест заполнено": len(top(pool))})
    return rows


def top_rows(theme: str, run_id: str, pools: dict[str, pd.DataFrame]) -> list[dict]:
    """Строки n_tops.csv: полные ТОП-15 по конфигурациям (только в файл, не в отчёт)."""
    rows = []
    for name, pool in pools.items():
        for _, row in top(pool).iterrows():
            rows.append({"тема": theme, "прогон": run_id, "конфигурация": name, "место": int(row["место"]),
                         "term_en": row["name_en"], "term_ru": row["name_ru"], "tech_key": row["tech_key"],
                         "score": float(row["score"]), "relevant": row.get("relevant"),
                         "relevance_failed": row.get("relevance_failed")})
    return rows


def blind(tops: pd.DataFrame, info: dict[tuple[str, str], dict]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """«Н слепо»: объединение ТОП-15, дубли по tech_key внутри темы, перемешано с seed 42."""
    keys = sorted(set(zip(tops["тема"], tops["tech_key"])))
    order = [keys[i] for i in np.random.default_rng(42).permutation(len(keys))]
    sheet = pd.DataFrame([{"n_id": f"N{number:03d}", "тема": key[0], **info[key]}
                          for number, key in enumerate(order, start=1)],
                         columns=["n_id", "тема", "term_en", "term_ru", "quote", "заголовок", "url"])
    return sheet, sheet[["n_id", "тема", "term_en"]]


def jaccard() -> pd.DataFrame:
    """Жаккар множеств фраз кандидатов двух кибер-прогонов: все кандидаты и оценённые."""
    first, second = run(THEMES["кибербезопасность"]), run(CYBER_SECOND)
    rows = []
    for what, pick in (("все кандидаты прогона", lambda d: d["top"] + d["excluded"]), ("оценённые", scored)):
        a, b = ({tech_key(item["name_en"]) for item in pick(data)} for data in (first, second))
        rows.append({"множество": what, THEMES["кибербезопасность"]: len(a), CYBER_SECOND: len(b),
                     "пересечение": len(a & b), "объединение": len(a | b),
                     "Жаккар": round(len(a & b) / len(a | b), 4) if a | b else None})
    return pd.DataFrame(rows)


def prepare() -> tuple[dict, dict, dict]:
    """Кандидаты трёх тем без сети, проверка порогов и T_mature, сверка оценок. Сбой — отказ."""
    limits = thresholds()
    train = feature_table()
    t_mature = round(tf.mature_threshold(train.loc[train["label"] == 1, "n_research6"]), 2)
    if limits != {"A0": 0.325, "A": 0.4} or t_mature != T_MATURE:
        raise RuntimeError(f"пороги {limits} или T_mature {t_mature} не совпали с заданием")
    data = {theme: run(run_id) for theme, run_id in THEMES.items()}
    frames = {theme: candidates(item) for theme, item in data.items()}
    checks = verify(frames, {theme: item["model_version"] for theme, item in data.items()})
    print(f"\nН сверка\n{checks.to_string(index=False)}")
    if (checks["совпало"] != checks["кандидатов"]).any():
        raise RuntimeError("сверка оценок или n_pat не сошлась — задача останавливается")
    return data, frames, {"limits": limits, "checks": checks}


def main() -> None:
    """Н1–Н3: числа в печать и в листы «Н», ТОП-15 в n_tops.csv, слепой лист и ключ."""
    load_dotenv(ROOT / ".env")
    data, frames, meta = prepare()
    cache = search_documents()
    places, llm_rows, doc_rows, tops, info = [], [], [], [], {}
    for theme, frame in frames.items():
        docs, stat = first_documents(theme, frame, cache)
        spent = {"вызовов": 0, "из кэша": 0, "ошибок API": 0, "повторов": 0}
        pools = configurations(data[theme]["topic"], frame, docs, meta["limits"], spent)
        checked = pools["C"]["relevant"].notna() | pools["C"]["relevance_failed"]
        llm_rows.append({"тема": theme, "кандидатов на проверке": int(checked.sum()), **spent,
                         "исключено (false)": int((pools["C"]["фильтр"] == "Н2").sum()),
                         "relevance_failed": int(pools["C"]["relevance_failed"].sum())})
        places += place_counts(theme, pools)
        tops += top_rows(theme, THEMES[theme], pools)
        doc_rows.append(stat)
        info.update({(theme, key): {"term_en": row["name_en"], "term_ru": row["name_ru"],
                                    "quote": row["quote"], **docs[i]}
                     for i, (key, row) in enumerate(zip(frame["tech_key"], frame.to_dict("records")))})
    tops = pd.DataFrame(tops)
    sheet, key = blind(tops, info)
    numbers = {"Н сверка": meta["checks"], "Н документы": pd.DataFrame(doc_rows),
               "Н1 места": pd.DataFrame(places), "Н2 релевантность": pd.DataFrame(llm_rows),
               "Н Жаккар кибер": jaccard()}
    for title, table in numbers.items():
        print(f"\n{title}\n{table.to_string(index=False)}")
    tops.to_csv(N_TOPS, index=False, encoding="utf-8-sig")
    key.to_csv(N_BLIND_KEY, index=False, encoding="utf-8-sig")
    after = append_t({**numbers, "Н слепо": sheet}, prefix="Н")
    print(f"\nслепой лист: {len(sheet)} строк; ТОП-15 всех конфигураций: {N_TOPS.relative_to(ROOT)} "
          f"({len(tops)} строк); ключ: {N_BLIND_KEY.relative_to(ROOT)}; листов в книге: {len(after)}")


if __name__ == "__main__":
    main()
