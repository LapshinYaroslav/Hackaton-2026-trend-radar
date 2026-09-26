"""Прогресс в процентах (И1): трекер, прогон на заглушках, строка CLI."""
import io
import json
import re
from pathlib import Path

from pipeline import progress as pg
from pipeline.__main__ import printer, progress_line
from tests.pipeline import test_run_query as base

SNAPSHOT = Path(__file__).resolve().parents[1] / "fixtures" / "run_query_snapshot.json"


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


def test_weights_sum_to_100_and_every_stage_has_russian_name() -> None:
    assert sum(pg.STAGE_WEIGHTS.values()) == 100 and set(pg.STAGE_WEIGHTS) < set(pg.STAGE_RU)


def test_skipped_stages_jump_to_start_and_cached_stage_to_its_end() -> None:
    events, _, report, finish = make()
    report("subqueries", 0, 1)
    report("counters", 0, 10)          # поиск, шаг 4 и названия пропущены — прыжок на начало счётчиков
    assert events[-1]["pct"] == 12
    report("counters", 10, 10)         # всё из кэша: сразу конец этапа
    assert events[-1]["pct"] == 94
    finish()
    assert events[-1] == {**events[-1], "pct": 100, "stage": "done", "stage_ru": "Готово"}


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
    events, _, report, finish = make()
    for stage in pg.STAGE_WEIGHTS:
        report(stage, 0, 2)
        report(stage, 2, 2)
    report("naming", 0, 5)             # поздний вызов раннего этапа не откатывает проценты
    pcts = [e["pct"] for e in events]
    assert pcts == sorted(pcts) and max(pcts) == 99
    finish()
    assert events[-1]["pct"] == 100


def test_eta_only_from_10_percent() -> None:
    clock = Clock()
    events, _, report, _ = make(clock)
    clock.now = 5.0
    report("search", 1, 1)             # 4 %
    assert events[-1]["eta_s"] is None
    clock.now = 20.0
    report("counters", 41, 82)         # 12 + 41 = 53 %
    assert events[-1]["pct"] == 53 and events[-1]["eta_s"] == round(20.0 * 47 / 53, 1)


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
    assert {e["stage"] for e in events} == set(pg.STAGE_WEIGHTS) | {"done"}
    assert all(set(e) == {"pct", "stage", "stage_ru", "done", "total", "elapsed_s", "eta_s"} for e in events)


def _strip(value):
    if isinstance(value, dict):
        return {k: _strip(v) for k, v in value.items() if k not in ("query_id", "timings", "published_at")}
    return [_strip(v) for v in value] if isinstance(value, list) else value


def test_without_callback_output_is_byte_identical_to_snapshot(tmp_path) -> None:
    """Снимок снят до задачи И1 на тех же заглушках (без изменчивых query_id, времени и дат).

    Переснят в задаче К (решение Ярослава): склейка дублей включена, у оценённых кандидатов поле variants.
    """
    out, _, _ = base.run(tmp_path)
    text = json.dumps(_strip(out), ensure_ascii=False, sort_keys=True, indent=1) + "\n"
    assert re.sub(r"q\d{14}", "Q", text) == SNAPSHOT.read_text(encoding="utf-8")  # query_id внутри subquery_ids


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
