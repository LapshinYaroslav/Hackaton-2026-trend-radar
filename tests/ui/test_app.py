"""Экран Streamlit целиком: UI → HTTP → FastAPI → run_query (подменён прогоном examples/robots.json). На экране — ровно JSON пайплайна."""
import copy
import os
import time

import pytest

pytest.importorskip("streamlit")
from streamlit.testing.v1 import AppTest  # noqa: E402

from tests.conftest import ROBOTS  # noqa: E402

BASE = "http://127.0.0.1:8000"


@pytest.fixture
def screen(client, monkeypatch):
    """AppTest, у которого requests ходит в TestClient вместо сети."""
    monkeypatch.setenv("API_URL", BASE)
    monkeypatch.setattr("requests.get", lambda url, params=None, timeout=None: client.get(url.removeprefix(BASE),
                                                                                          params=params))
    monkeypatch.setattr("requests.post", lambda url, json=None, timeout=None: client.post(url.removeprefix(BASE),
                                                                                           json=json))
    return AppTest.from_file(os.path.join("ui", "app.py"), default_timeout=60)


def search(at: AppTest, topic: str) -> AppTest:
    """Ввод темы и «Найти сигналы»; AppTest не крутит таймер фрагмента, поэтому перезапуск до конца расчёта."""
    at.run()
    at.text_input[0].input(topic)
    next(b for b in at.button if b.label == "Найти сигналы").click().run()
    for _ in range(100):
        if not at.session_state["polling"]:
            break
        time.sleep(0.05)
        at.run()
    return at


def test_polling_redraws_only_progress_fragment(screen, monkeypatch) -> None:
    """Пока идёт расчёт: одна полоса с процентом во фрагменте, без sleep+rerun всей страницы."""
    import threading

    release = threading.Event()

    def slow(topic, **kwargs):
        kwargs["on_progress"]({"pct": 41, "stage_ru": "Сбор счётчиков", "eta_s": 480.0})
        release.wait(5)
        return copy.deepcopy(ROBOTS)

    monkeypatch.setattr("pipeline.run_query.run_query", slow)
    at = screen.run()
    at.text_input[0].input("роботы для промышленности")
    next(b for b in at.button if b.label == "Найти сигналы").click().run()
    time.sleep(0.2)
    at.run()
    assert at.session_state["polling"] and [p.value for p in at.get("progress")] == [41]
    release.set()


def test_header_has_topic_without_area(screen) -> None:
    at = search(screen, "роботы для промышленности")
    assert [s.value for s in at.success] == ["Запрос: «роботы для промышленности»"]


def names_on_screen(at: AppTest) -> list[str]:
    return [m.value for m in at.markdown if "class='signal-name'" in m.value]


def test_screen_shows_pipeline_top(screen) -> None:
    at = search(screen, "роботы для промышленности")
    assert not at.exception and not at.error
    assert names_on_screen(at) == [f"<div class='signal-name'>{t['rank']}. {t['name_ru']}</div>" for t in ROBOTS["top"]]
    assert [m.value for m in at.metric] == [str(ROBOTS["stats"][key])
                                           for key in ("documents_total", "candidates_found", "above_075")]
    # в robots.json нет documents_analyzed — плашка берёт documents_total
    scores = [m.value for m in at.markdown if "балл" in m.value]
    assert [s.rsplit(" ", 1)[-1].removesuffix("</span>") for s in scores] == \
        [f"{t['score']:.3f}" for t in ROBOTS["top"]]


def test_screen_rejects_answer_not_from_pipeline(screen, monkeypatch) -> None:
    spoiled = copy.deepcopy(ROBOTS)
    spoiled["top"][0]["rank"] = 7
    monkeypatch.setattr("pipeline.run_query.run_query", lambda *args, **kwargs: copy.deepcopy(spoiled))
    at = search(screen, "роботы для промышленности")
    assert any("ответ не из пайплайна" in e.value for e in at.error)
    assert names_on_screen(at) == []


def test_search_without_api_is_explicit_error(monkeypatch) -> None:
    monkeypatch.delenv("API_URL", raising=False)
    at = search(AppTest.from_file(os.path.join("ui", "app.py"), default_timeout=60), "квантовые технологии")
    assert [e.value for e in at.error] == ["Нет связи с пайплайном: не задан API_URL (например, http://127.0.0.1:8000)."]
    assert names_on_screen(at) == []



def test_no_ready_run_picker(monkeypatch) -> None:
    monkeypatch.delenv("API_URL", raising=False)
    at = AppTest.from_file(os.path.join("ui", "app.py"), default_timeout=60).run()
    assert not at.exception and len(at.selectbox) == 0
    assert "Открыть готовый прогон" not in [b.label for b in at.button]


def test_screen_shows_documents_analyzed(screen, monkeypatch) -> None:
    answer = copy.deepcopy(ROBOTS)
    answer["stats"]["documents_analyzed"] = 580376
    monkeypatch.setattr("pipeline.run_query.run_query", lambda *args, **kwargs: copy.deepcopy(answer))
    at = search(screen, "роботы для промышленности")
    assert at.metric[0].label == "Обработано документов" and at.metric[0].value == "580 376"


def test_card_explains_choice_in_words(screen, monkeypatch) -> None:
    answer = copy.deepcopy(ROBOTS)
    answer["top"][0]["why_ru"] = "Почти всё, что о ней написано, появилось за последние два года."
    monkeypatch.setattr("pipeline.run_query.run_query", lambda *args, **kwargs: copy.deepcopy(answer))
    at = search(screen, "роботы для промышленности")
    next(b for b in at.button if b.label == "Смотреть").click().run()
    assert "Почему это слабый сигнал" in [h.value for h in at.subheader]
    assert answer["top"][0]["why_ru"] in [m.value for m in at.markdown]


def test_screen_shows_rank_score_as_ball(screen, monkeypatch) -> None:
    answer = copy.deepcopy(ROBOTS)
    answer["top"][0]["rank_score"] = round(answer["top"][0]["score"], 3) + 0.004
    monkeypatch.setattr("pipeline.run_query.run_query", lambda *args, **kwargs: copy.deepcopy(answer))
    at = search(screen, "роботы для промышленности")
    first = next(m.value for m in at.markdown if "балл" in m.value)
    assert first.endswith(f"балл {answer['top'][0]['rank_score']:.3f}</span>")
