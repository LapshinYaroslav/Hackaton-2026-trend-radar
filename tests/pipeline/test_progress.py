"""Прогресс в процентах (И1): трекер, прогон на заглушках, строка CLI."""
import io

from pipeline import progress as pg
from pipeline.__main__ import printer, progress_line
from tests.pipeline import test_run_query as base


class Clock:
    """Часы, которые двигает тест."""
    def __init__(self):
        self.now = 0.0

    def __call__(self):
        return self.now


def make(clock=None):
    events, warnings = [], []
    report, finish = pg.tracker(events.append, warnings.append, clock or Clock())
    return events, warnings, report, finish


def test_every_stage_has_russian_name() -> None:
    assert set(pg.STAGES) < set(pg.STAGE_RU) and pg.AFTER_COUNTERS < set(pg.STAGES)


def test_before_counters_pct_and_eta_from_budget() -> None:
    """До счётчиков: pct = прошло / бюджет, осталось = бюджет − прошло."""
    clock = Clock()
    events, _, report, _ = make(clock)
    clock.now = 90.0
    report("naming", 1, 2)
    assert (events[-1]["pct"], events[-1]["eta_s"]) == (10, 810.0)


def test_counters_ending_early_raise_pct_then_tail() -> None:
    """Все счётчики из кэша за 1 с: остаток — хвост 90 с, а не остаток бюджета; хвост тает после счётчиков."""
    clock = Clock()
    events, _, report, finish = make(clock)
    clock.now = 100.0
    report("counters", 0, 10)
    assert events[-1]["pct"] == 11     # 100 / 900
    clock.now = 101.0
    report("counters", 10, 10)
    assert (events[-1]["pct"], events[-1]["eta_s"]) == (52, 90.0)   # 101 / (101 + 90)
    clock.now = 150.0
    report("translate", 1, 2)
    assert (events[-1]["pct"], events[-1]["eta_s"]) == (78, 41.0)   # хвост 90 − 49 с после счётчиков
    finish()
    assert events[-1] == {**events[-1], "pct": 100, "stage": "done", "stage_ru": "Готово", "eta_s": 0.0}


def test_counters_eta_by_rate_but_not_beyond_budget() -> None:
    """Темп 1 с на запрос: 50 оставшихся + хвост 90 = 140 с; при медленном темпе — не больше остатка бюджета."""
    clock = Clock()
    events, _, report, _ = make(clock)
    clock.now = 100.0
    report("counters", 0, 100)
    clock.now = 150.0
    report("counters", 50, 100)
    assert events[-1]["eta_s"] == 140.0
    clock.now = 700.0
    report("counters", 60, 100)        # темп 10 с: 400 + 90 > 200 — остаток бюджета
    assert events[-1]["eta_s"] == 200.0 and events[-1]["pct"] == 77


def test_budget_exceeded_stays_below_100_until_done() -> None:
    """Бюджет 0: остаток не меньше 5 с, pct до события done ниже 100."""
    clock, events = Clock(), []
    report, finish = pg.tracker(events.append, [].append, clock, budget=0)
    report("subqueries", 0, 1)
    clock.now = 50.0
    report("counters", 5, 10)
    assert all(e["pct"] < 100 for e in events) and events[-1]["eta_s"] == pg.MIN_LEFT_S
    finish()
    assert events[-1]["pct"] == 100


def test_rate_limit_inside_stage_but_boundaries_always_sent() -> None:
    clock = Clock()
    events, _, report, _ = make(clock)
    report("counters", 0, 100)
    for done in range(1, 100):         # 99 вызовов за 0.5 с — внутри этапа не шлются
        clock.now += 0.005
        report("counters", done, 100)
    assert len(events) == 1
    clock.now += 1.0
    report("counters", 50, 100)
    report("counters", 100, 100)       # граница этапа — отправляется даже через 0 с
    assert [e["done"] for e in events] == [0, 50, 100]


def test_pct_never_decreases_and_stays_below_100_until_done() -> None:
    clock = Clock()
    events, _, report, finish = make(clock)
    for stage in pg.STAGES:
        clock.now += 60.0
        report(stage, 0, 2)
        report(stage, 2, 2)
    report("naming", 0, 5)             # поздний вызов раннего этапа не откатывает проценты
    pcts = [e["pct"] for e in events]
    assert pcts == sorted(pcts) and 90 <= max(pcts) <= 99
    finish()
    assert events[-1]["pct"] == 100


def test_callback_error_is_one_warning_and_run_completes(tmp_path) -> None:
    calls = []

    def broken(event):
        calls.append(event)
        raise RuntimeError("UI упал")
    out, _, _ = base.run(tmp_path, on_progress=broken)
    assert len(calls) == 1 and len(out["top"]) > 0
    assert [w for w in out["warnings"] if "прогресс отключён" in w] == \
        ["прогресс отключён: ошибка в on_progress (RuntimeError: UI упал)"]


def test_run_events_are_monotonic_and_end_with_done(tmp_path) -> None:
    events = []
    base.run(tmp_path, on_progress=events.append)
    pcts = [e["pct"] for e in events]
    assert pcts == sorted(pcts) and events[-1]["pct"] == 100 and events[-1]["stage"] == "done"
    assert {e["stage"] for e in events} == set(pg.STAGES) | {"done"}
    assert all(e["eta_s"] is not None for e in events)
    assert all(set(e) == {"pct", "stage", "stage_ru", "done", "total", "elapsed_s", "eta_s"} for e in events)


def test_cli_line_and_pipe_interval() -> None:
    event = {"pct": 37, "stage": "counters", "stage_ru": "Сбор счётчиков", "done": 812, "total": 2160,
             "elapsed_s": 300.0, "eta_s": 530.0}
    assert progress_line(event) == "[ 37%] Сбор счётчиков 812/2160 · осталось ~9 мин"
    assert progress_line({**event, "eta_s": None}) == "[ 37%] Сбор счётчиков 812/2160"
    stream, clock = io.StringIO(), Clock()
    show = printer(stream, clock)
    for second in range(25):           # в трубу — строка раз в 10 с, плюс последняя
        clock.now = float(second)
        show(event)
    show({**event, "pct": 100, "stage": "done", "stage_ru": "Готово", "done": 1, "total": 1, "eta_s": 0.0})
    assert len(stream.getvalue().splitlines()) == 4


def test_cli_tty_rewrites_one_line() -> None:
    class Tty(io.StringIO):
        def isatty(self):
            return True
    stream = Tty()
    show = printer(stream)
    show({"pct": 5, "stage": "search", "stage_ru": "Поиск", "done": 0, "total": 1, "elapsed_s": 1, "eta_s": None})
    show({"pct": 100, "stage": "done", "stage_ru": "Готово", "done": 1, "total": 1, "elapsed_s": 2, "eta_s": 0})
    assert stream.getvalue().count("\r") == 2 and stream.getvalue().endswith("\n")
