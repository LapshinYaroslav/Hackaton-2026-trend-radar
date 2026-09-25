"""Тесты чистых функций задачи Н: разбор ответа LLM, промпт, повтор при невалидном ответе."""
import pytest

from scripts import n_report
from scripts.n_report import PROMPT, norm, parse_relevance


@pytest.mark.parametrize("text, expected", [
    ('{"relevant": true}', True), (' {"relevant": false}\n', False),
    ('{"relevant": "true"}', None), ('{"relevant": 1}', None), ('{"other": true}', None),
    ('[true]', None), ('да', None), ('', None), (None, None),
    ('```json\n{"relevant": true}\n```', None)])
def test_parse_relevance(text, expected):
    assert parse_relevance(text) is expected


def test_prompt_keeps_literal_json_and_fills_fields():
    text = PROMPT.format(topic="т", term_en="e", term_ru="р", quote="ц", title="з")
    assert text == ("Тема запроса пользователя: «т».\nКандидат: «e» (р).\n"
                    "Фрагмент источника: «ц» — из «з».\nОтносится ли эта технология к теме запроса? "
                    'Ответь только JSON: {"relevant": true} или {"relevant": false}.')


def test_norm():
    assert norm("  A\n b  ") == "a b" and norm(None) == ""


def test_retry_once_then_failed(monkeypatch):
    answers = iter(["не json", "снова не json"])
    temperatures = []

    def fake_ask(prompt, temperature, spent):
        temperatures.append(temperature)
        return next(answers)

    monkeypatch.setattr(n_report, "ask", fake_ask)
    spent = {"повторов": 0}
    assert n_report.relevance("p", spent) == (None, True)
    assert temperatures == [0.0, 0.3] and spent["повторов"] == 1


def test_valid_answer_needs_no_retry(monkeypatch):
    monkeypatch.setattr(n_report, "ask", lambda prompt, temperature, spent: '{"relevant": false}')
    spent = {"повторов": 0}
    assert n_report.relevance("p", spent) == (False, False) and spent["повторов"] == 0
