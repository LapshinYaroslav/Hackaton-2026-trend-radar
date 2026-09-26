"""Прогресс прогона в процентах (задача И1): события для CLI, API и UI.

Вызовы этапов (stage, done, total) превращаются в события {"pct", "stage", "stage_ru", "done", "total",
"elapsed_s", "eta_s"}. Веса этапов — по живому прогону 26.09.2026 (877 с), внутри этапа — доля done/total.
pct не убывает, внутри этапа не чаще раза в секунду (на границе этапа — всегда), до конца не выше 99;
последнее событие — ровно 100 и stage = "done". Ошибка в колбэке не роняет прогон: одно предупреждение,
дальше без прогресса.
"""
from __future__ import annotations

import time
from typing import Callable

# Этапы в порядке прогона и их веса, % (сумма 100).
# Склейка дублей (задача К) — 1 %, взят у ранжирования (оно быстрее секунды).
STAGE_WEIGHTS = {"subqueries": 1, "search": 3, "candidates": 6, "naming": 2, "counters": 82, "ranking": 0,
                 "dedup": 1, "translate": 5}
STAGE_RU = {"subqueries": "Подзапросы", "search": "Поиск документов", "candidates": "Извлечение технологий",
            "naming": "Проверка названий",
            "counters": "Сбор счётчиков (arXiv, OpenAlex, TechCrunch, Роспатент)", "ranking": "Оценка моделью", "dedup": "Склейка дублей",
            "translate": "Перевод названий", "done": "Готово"}
MIN_INTERVAL_S = 1.0
ETA_FROM_PCT = 10
CALLBACK_FAILED = "прогресс отключён: ошибка в on_progress ({})"
Event = dict
Report = Callable[[str, int, int], None]


def stage_start(stage: str) -> float:
    """Процент, с которого начинается этап: сумма весов предыдущих."""
    order = list(STAGE_WEIGHTS)
    return float(sum(STAGE_WEIGHTS[s] for s in order[:order.index(stage)]))


def tracker(on_progress: Callable[[Event], None], warn: Callable[[str], None],
            clock: Callable[[], float] = time.monotonic) -> tuple[Report, Callable[[], None]]:
    """(report(stage, done, total), finish()): report считает pct и шлёт события, finish — последнее, 100 %."""
    state = {"started": clock(), "last": None, "pct": 0, "stage": None, "active": True}

    def send(stage: str, done: int, total: int, pct: int) -> None:
        elapsed = clock() - state["started"]
        eta = round(elapsed * (100 - pct) / pct, 1) if pct >= ETA_FROM_PCT else None
        event = {"pct": pct, "stage": stage, "stage_ru": STAGE_RU.get(stage, stage), "done": done, "total": total,
                 "elapsed_s": round(elapsed, 1), "eta_s": eta}
        try:
            on_progress(event)
        except Exception as exc:  # колбэк интерфейса не должен ронять прогон
            state["active"] = False
            warn(CALLBACK_FAILED.format(f"{type(exc).__name__}: {exc}"))
        state["last"] = clock()

    def report(stage: str, done: int, total: int) -> None:
        if not state["active"] or stage not in STAGE_WEIGHTS:
            return
        share = min(1.0, done / total) if total else 1.0
        pct = min(99, max(state["pct"], int(stage_start(stage) + STAGE_WEIGHTS[stage] * share)))
        boundary = stage != state["stage"] or done >= total
        if not boundary and state["last"] is not None and clock() - state["last"] < MIN_INTERVAL_S:
            return
        state["pct"], state["stage"] = pct, stage
        send(stage, done, total, pct)

    def finish() -> None:
        if state["active"]:
            state["pct"], state["stage"] = 100, "done"
            send("done", 1, 1, 100)

    return report, finish
