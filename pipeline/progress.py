"""Прогресс прогона в процентах (задача И1, с задачи Х4 — от бюджета времени): события для CLI, API и UI.

Вызовы этапов (stage, done, total) превращаются в события {"pct", "stage", "stage_ru", "done", "total",
"elapsed_s", "eta_s"}. pct = прошло / (прошло + оценка остатка). Оценка остатка (remaining_s): до счётчиков —
остаток бюджета; на счётчиках — темп этапа × оставшиеся запросы + хвост, но не больше остатка бюджета;
после счётчиков — хвост минус прошедшее с их конца. Этап, кончившийся раньше бюджета, сразу поднимает pct.
eta_s — та же оценка остатка. pct не убывает, внутри этапа не чаще раза в секунду (на границе этапа — всегда),
до конца не выше 99; последнее событие — ровно 100 и stage = "done". Ошибка в колбэке не роняет прогон:
одно предупреждение, дальше без прогресса.
"""
from __future__ import annotations

import time
from typing import Callable

# Этапы в порядке прогона; после счётчиков — хвост (ранжирование, склейка, перевод).
STAGES = ("subqueries", "search", "candidates", "naming", "counters", "ranking", "dedup", "translate")
AFTER_COUNTERS = frozenset({"ranking", "dedup", "translate"})
STAGE_RU = {"subqueries": "Подзапросы", "search": "Поиск документов", "candidates": "Извлечение технологий",
            "naming": "Проверка названий",
            "counters": "Сбор счётчиков (arXiv, OpenAlex, TechCrunch, Роспатент)", "ranking": "Оценка моделью", "dedup": "Склейка дублей",
            "translate": "Перевод названий", "done": "Готово"}
MIN_INTERVAL_S = 1.0
MIN_LEFT_S = 5.0  # оценка остатка не меньше: до события done pct не доходит до 100
CALLBACK_FAILED = "прогресс отключён: ошибка в on_progress ({})"
Event = dict
Report = Callable[[str, int, int], None]


def remaining_s(stage: str, done: int, total: int, elapsed: float, budget: float, tail: float,
                marks: dict) -> float:
    """Оценка остатка прогона в секундах. marks — начало и конец счётчиков (секунды от старта)."""
    if stage in AFTER_COUNTERS:
        return max(tail - (elapsed - marks["counters_end"]), MIN_LEFT_S)
    if stage == "counters" and done:
        rate = (elapsed - marks["counters_start"]) / done
        return max(min(budget - elapsed, rate * (total - done) + tail), MIN_LEFT_S)
    return max(budget - elapsed, tail)


def tracker(on_progress: Callable[[Event], None], warn: Callable[[str], None],
            clock: Callable[[], float] = time.monotonic, budget: float = 900.0,
            tail: float = 90.0) -> tuple[Report, Callable[[], None]]:
    """(report(stage, done, total), finish()): report считает pct и шлёт события, finish — последнее, 100 %.

    budget — бюджет прогона, tail — оценка времени после счётчиков (резерв бюджета), оба в секундах.
    """
    state = {"started": clock(), "last": None, "pct": 0, "stage": None, "active": True}
    marks: dict[str, float] = {}

    def send(stage: str, done: int, total: int, pct: int, eta: float) -> None:
        event = {"pct": pct, "stage": stage, "stage_ru": STAGE_RU.get(stage, stage), "done": done, "total": total,
                 "elapsed_s": round(clock() - state["started"], 1), "eta_s": round(eta, 1)}
        try:
            on_progress(event)
        except Exception as exc:  # колбэк интерфейса не должен ронять прогон
            state["active"] = False
            warn(CALLBACK_FAILED.format(f"{type(exc).__name__}: {exc}"))
        state["last"] = clock()

    def report(stage: str, done: int, total: int) -> None:
        if not state["active"] or stage not in STAGE_RU or stage == "done":
            return
        elapsed = clock() - state["started"]
        if stage == "counters":
            marks.setdefault("counters_start", elapsed)
            marks["counters_last"] = elapsed
        elif stage in AFTER_COUNTERS:  # конец счётчиков — их последнее событие (или начало хвоста, если их не было)
            marks.setdefault("counters_end", marks.get("counters_last", elapsed))
        left = remaining_s(stage, done, total, elapsed, budget, tail, marks)
        pct = min(99, max(state["pct"], int(100 * elapsed / (elapsed + left))))
        boundary = stage != state["stage"] or done >= total
        if not boundary and state["last"] is not None and clock() - state["last"] < MIN_INTERVAL_S:
            return
        state["pct"], state["stage"] = pct, stage
        send(stage, done, total, pct, left)

    def finish() -> None:
        if state["active"]:
            state["pct"], state["stage"] = 100, "done"
            send("done", 1, 1, 100, 0.0)

    return report, finish
