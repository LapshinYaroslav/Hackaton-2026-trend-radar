"""Перевод названий без сети: снимок промпта, проверки, повтор, откат на term_ru, null, кэш."""
import hashlib

import pytest

from pipeline import translate as tr

# Снимок промпта задачи З (sha256 шаблона): промпт дословный, меняется только сознательно.
PROMPT_SHA = "20b1692c05cdd928da84fbac6ac5399c8d2b53a685e272f9c2e5213c2d90bee8"


@pytest.fixture(autouse=True)
def _cache(tmp_path, monkeypatch):
    monkeypatch.setattr(tr, "CACHE_DIR", tmp_path)


def fake(answers):
    """LLM с ответами по очереди; строка «ERR…» — ошибка API."""
    calls = []

    def llm(system, user, **kwargs):
        calls.append({"system": system, "user": user, **kwargs})
        text = answers[min(len(calls), len(answers)) - 1]
        return {"text": None, "error": text} if text.startswith("ERR") else {"text": text, "error": None}
    return llm, calls


def test_prompt_snapshot_and_substitution() -> None:
    assert hashlib.sha256(tr.PROMPT.encode("utf-8")).hexdigest() == PROMPT_SHA
    assert tr.PROMPT.startswith("Переведи на русский название класса технологий для аналитического отчёта: «{term_en}»")
    assert tr.PROMPT.endswith("не больше 8 слов; без кавычек и точки. Ответ — только название.")
    llm, calls = fake(["федеративное обучение"])
    tr.translate_one("federated learning", "We use federated learning.", None, llm)
    assert "«federated learning» (контекст: «We use federated learning.»)" in calls[0]["system"]
    assert calls[0]["user"] == "federated learning" and calls[0]["temperature"] == 0.0
    assert calls[0]["model"] == "yandexgpt-5-pro"


def test_checks() -> None:
    assert tr.passes("федеративное обучение", "federated learning")
    assert tr.passes("VLA-модели «зрение–язык–действие»", "vision-language-action model")
    assert not tr.passes("federated learning", "federated learning")        # нет кириллицы / совпадает
    assert not tr.passes("очень длинное название из девяти разных слов тут и там", "x y")  # 9 слов
    assert tr.passes("Обнаружение угроз в реальном времени на основе ИИ", "ai-powered real-time threat detection")  # 8
    assert not tr.passes("", "x") and not tr.passes(None, "x")


def test_first_try_ok_and_cleaned() -> None:
    llm, calls = fake(["«Федеративное обучение»."])
    got = tr.translate_one("federated learning", "q", "фед. обучение", llm)
    assert got == {"name_ru": "Федеративное обучение", "name_ru_source": "translate", "name_ru_auto": True,
                   "attempts": 1}
    assert len(calls) == 1


def test_retry_at_03_then_ok() -> None:
    llm, calls = fake(["federated learning", "федеративное обучение"])
    got = tr.translate_one("federated learning", "q", None, llm)
    assert got["name_ru_source"] == "translate" and got["attempts"] == 2
    assert [c["temperature"] for c in calls] == [0.0, 0.3]


def test_fallback_to_term_ru_then_null() -> None:
    llm, _ = fake(["ERR 403", "ERR 403"])
    got = tr.translate_one("federated learning", "q", "федеративное обучение", llm)
    assert got["name_ru"] == "Федеративное обучение" and got["name_ru_source"] == "extract"
    got = tr.translate_one("federated learning", "q", "federated learning", fake(["no", "no"])[0])
    assert got == {"name_ru": None, "name_ru_source": None, "name_ru_auto": True, "attempts": 2}


def test_cache_by_term_en_and_failures_not_cached() -> None:
    tr.translate_one("federated learning", "q", None, fake(["федеративное обучение"])[0])
    llm, calls = fake(["другое"])
    got = tr.translate_one("federated learning", "other quote", None, llm)
    assert got["name_ru"] == "Федеративное обучение" and got["attempts"] == 0 and calls == []
    tr.translate_one("edge ai", "q", None, fake(["ERR", "ERR"])[0])
    assert not tr.cache_path("edge ai").exists()


def test_translate_entries_updates_in_place_and_counts() -> None:
    entries = [{"name_en": "federated learning", "name_ru": "фед", "quote": "q"},
               {"name_en": "edge ai", "name_ru": "периферийный ИИ", "quote": "q"},
               {"name_en": "x y", "name_ru": "x y", "quote": "q"}]
    answers = {"federated learning": "федеративное обучение", "edge ai": "edge ai", "x y": "x y"}

    def llm(system, user, **kwargs):
        return {"text": answers[user], "error": None}
    stats = tr.translate_entries(entries, llm)
    assert [e["name_ru_source"] for e in entries] == ["translate", "extract", None]
    assert stats == {"переведено": 3, "из кэша": 0, "с первого раза": 1, "с повтором": 0,
                     "откат на term_ru": 1, "без названия": 1}


def test_clean_keeps_inner_quotes_and_strips_wrapping() -> None:
    assert tr.clean("VLA-модели «зрение–язык–действие»") == "VLA-модели «зрение–язык–действие»"
    assert tr.clean("Протокол «агент-клиент».") == "Протокол «агент-клиент»"
    assert tr.clean("«Федеративное обучение».") == "Федеративное обучение"
    assert tr.clean("«A» и «B»") == "«A» и «B»"  # кавычки не обрамляют всё название
    assert tr.clean(None) == "" and tr.clean("  ") == ""


def test_latin_words_must_come_from_term_or_its_initials() -> None:
    assert not tr.passes("Batch-OF справедливость очередности пакетов", "batch order fairness")
    assert tr.passes("VLA-модели «зрение–язык–действие»", "vision-language-action models")
    assert tr.passes("MCP — протокол контекста модели", "Model Context Protocol")
    assert tr.passes("MCP-серверы", "MCP server") and tr.passes("6G-сети на базе ИИ", "AI-native 6G networks")
    assert tr.passes("Сети IoT на базе MQTT", "MQTT based IoT networks")
    assert not tr.passes("LoRA-адаптация", "low rank adaptation")   # не заглавные: не аббревиатура
    assert tr.passes("GPU-кластеры", "graphics processing units")      # G, P, U — первые буквы подряд


def test_capitalized_first_letter_but_not_latin_start() -> None:
    assert tr.capitalized("эталон агентского управления") == "Эталон агентского управления"
    assert tr.capitalized("fog-вычисления") == "fog-вычисления" and tr.capitalized("VLA-модели") == "VLA-модели"
    assert tr.capitalized(None) is None and tr.capitalized("") == ""


def test_latin_failure_retried_then_fallback() -> None:
    llm, calls = fake(["Batch-OF справедливость", "Batch-OF справедливость"])
    got = tr.translate_one("batch order fairness", "q", "справедливость порядка пакетов", llm)
    assert [c["temperature"] for c in calls] == [0.0, 0.3]
    assert got["name_ru"] == "Справедливость порядка пакетов" and got["name_ru_source"] == "extract"



@pytest.mark.parametrize("answer", ["Извините, не могу помочь", "Не могу перевести этот термин",
                                    "К сожалению, перевод не найден", "Не удалось подобрать термин",
                                    "Я языковая модель", "Перевод: федеративное обучение",
                                    "Название: федеративное обучение", "Федеративное обучение?", "Термин: обучение"])
def test_refusal_is_rejected(answer) -> None:
    assert not tr.passes(answer, "federated learning")


def test_refusal_goes_to_retry_then_fallback() -> None:
    llm, calls = fake(["Извините, не могу помочь", "К сожалению, не могу"])
    got = tr.translate_one("federated learning", "q", "федеративное обучение", llm)
    assert [c["temperature"] for c in calls] == [0.0, 0.3]
    assert got["name_ru"] == "Федеративное обучение" and got["name_ru_source"] == "extract"


@pytest.mark.parametrize("name, term", [("ДНТ — глубокие нейронные трансформаторы", "deep neural transformer"),
                                        ("БСД — бисимметрическое взвешенное расстояние", "bi-symmetrical weighted distance"),
                                        ("А2А-финансы (агент-агенту)", "agent-to-agent finance"),
                                        ("Сетиlora модели", "lora networks")])
def test_cyrillic_abbreviations_and_mixed_alphabets_rejected(name, term) -> None:
    assert not tr.passes(name, term)


@pytest.mark.parametrize("name, term", [("ИИ-агенты", "AI agents"), ("6G-сети на базе ИИ", "AI-native 6G networks"),
                                        ("VLA-модели «зрение–язык–действие»", "vision-language-action models"),
                                        ("Сети IoT на базе MQTT", "MQTT based IoT networks"),
                                        ("ЦОД на базе ИИ", "AI data centers"),
                                        ("3D-моделирование молекул", "3D molecular generation"),
                                        ("ДНК-кодируемая библиотека", "DNA-encoded library"),
                                        ("РНК-интерференция на базе ИИ", "AI RNA interference")])
def test_whitelist_and_homogeneous_parts_pass(name, term) -> None:
    assert tr.passes(name, term)


def test_abbreviation_goes_to_retry_then_fallback() -> None:
    llm, calls = fake(["ДНТ — глубокие нейронные трансформаторы", "ДНТ-модели"])
    got = tr.translate_one("deep neural transformer", "q", "глубокие нейронные трансформеры", llm)
    assert [c["temperature"] for c in calls] == [0.0, 0.3]
    assert got["name_ru"] == "Глубокие нейронные трансформеры" and got["name_ru_source"] == "extract"


def test_cached_name_failing_new_checks_is_translated_again() -> None:
    """Кэш, записанный до новой проверки (К3), не отдаётся, если название её не проходит."""
    import json as _json
    tr.CACHE_DIR.mkdir(parents=True, exist_ok=True)
    tr.cache_path("deep neural transformer").write_text(_json.dumps({"term_en": "deep neural transformer", "answer": {
        "name_ru": "ДНТ — глубокие нейронные трансформаторы", "name_ru_source": "translate", "name_ru_auto": True}},
        ensure_ascii=False), encoding="utf-8")
    llm, calls = fake(["Глубокие нейронные трансформеры"])
    got = tr.translate_one("deep neural transformer", "q", None, llm)
    assert got["name_ru"] == "Глубокие нейронные трансформеры" and got["attempts"] == 1 and len(calls) == 1
