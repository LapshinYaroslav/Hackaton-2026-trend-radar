"""Названия кандидатов через нормализатор обучения: name_ru -> три варианта -> выбор по следу.

Нормализатор перенесён из feature/axis-screening без правок (experiments/technologies/terms.py,
промпт tech_norm_v4, модель yandexgpt-5-pro): тем же путём строились 160 обучающих названий.
Здесь только цикл повторов, повторяющий experiments/technologies/build.py::build_one: до
MAX_RETRIES*2+1 попыток; нарушение формата — повтор со строкой о нарушении; все три варианта
без работ — повтор с anchor.retry_note; первая попытка при 0.3, повторы при 0.8 (terms.ask_name).

Отличия от обучения — осознанные, перечислены в DEVIATIONS и уходят в выход:
- company_stoplist_off: стоп-лист компаний в валидации выключен. В нём лежат термины самих
  сигналов (sovereign ai, physical intelligence), а при обучении он отклонил 3 варианта из 549.
Выбор варианта — по максимуму работ (при обучении выбирал человек вслепую; правило совпало
с его выбором в 122 из 145 строк). Объём — в окне обучения anchor_volume: 2020-09 … 2026-08,
фраза, type:article, дешёвый режим group_by (meta.count тот же).
"""
from __future__ import annotations

import re
import threading
from concurrent.futures import ThreadPoolExecutor
from typing import Callable, Sequence

from collector.constants import COLLECTION_START, CUTOFF_DATE
from collector.exceptions import AdapterError
from collector.models import build_search_terms
from experiments.technologies import anchor
from experiments.technologies import terms as T

CHOICE_RULE = "max_trace"
DEVIATIONS = ["company_stoplist_off"]
ATTEMPTS = T.MAX_RETRIES * 2 + 1
MAX_WORKERS = 10
# Шаблон, который не совпадает ни с чем: стоп-лист компаний выключен без правки terms.py.
NO_STOPLIST = re.compile(r"(?!)")


def trace(name: str, openalex) -> dict:
    """Работы и организации по фразе в окне обучения. None — источник не ответил."""
    try:
        answer = openalex.count_institutions(build_search_terms([name], []), COLLECTION_START, CUTOFF_DATE)
    except AdapterError:
        answer = None
    answer = answer or {}
    return {"name": name, "n_works": answer.get("works"), "n_institutions": answer.get("institutions"),
            "n_institutions_capped": answer.get("capped")}


def choose(variants: Sequence[dict]) -> dict:
    """Вариант с наибольшим числом работ; при ничьей — первый по порядку нормализатора."""
    best = max(range(len(variants)), key=lambda i: (variants[i]["n_works"] or 0, -i))
    return variants[best]


def normalize_one(name_ru: str, area: str, openalex) -> dict:
    """Три варианта с объёмами и выбранный. chosen=None — названия нет, причина в outcome."""
    note, problems, variants = "", [], []
    for attempt in range(ATTEMPTS):
        raw, _ = T.ask_name(name_ru, area, repeat=attempt, note=note)
        names, problems = T.check_candidates(raw, NO_STOPLIST)
        if problems:
            note = "; ".join(problems)
            continue
        variants = [trace(name, openalex) for name in names]
        if any(v["n_works"] is None for v in variants) and not any(v["n_works"] for v in variants):
            return {"variants": variants, "chosen": None, "outcome": "trace_unknown", "attempts": attempt + 1}
        if any(v["n_works"] for v in variants):
            return {"variants": variants, "chosen": choose(variants), "outcome": "ok", "attempts": attempt + 1}
        problems = [f"все три кандидата не встречаются в литературе: {names}"]
        note = anchor.retry_note(names)
    outcome = "no_trace" if variants and all(v["n_works"] == 0 for v in variants) else "bad_format"
    return {"variants": variants, "chosen": None, "outcome": outcome, "attempts": ATTEMPTS, "problems": problems}


def normalize_all(names_ru: Sequence[str], area: str, openalex,
                  on_done: Callable[[int, int], None] | None = None) -> list[dict]:
    """Нормализатор по каждому name_ru параллельно, не больше MAX_WORKERS вызовов одновременно."""
    done, lock = [0], threading.Lock()

    def one(name_ru: str) -> dict:
        result = normalize_one(name_ru, area, openalex)
        with lock:
            done[0] += 1
            if on_done:
                on_done(done[0], len(names_ru))
        return result

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
        return list(pool.map(one, names_ru))
