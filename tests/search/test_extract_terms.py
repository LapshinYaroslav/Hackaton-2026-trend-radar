"""Шаг 4 extract-v4 без сети: ask_llm подменяется."""
import json
from unittest.mock import patch

import pytest

from search import extract_candidates as ec
from search import extract_terms as et

SNIPPETS = {1: "SPROUT: The Open-Source Soft Growing Robot for Search and Rescue\nWe present a robot",
            2: "On the Efficiency of LoRA Fine-Tuning for Vision-Language-Action Models"}
GOOD = {"term_en": "soft growing robot", "term_ru": "мягкие растущие роботы",
        "quote": "open-source  soft growing robot", "doc": 1}


def reply(items) -> dict:
    return {"text": json.dumps(items, ensure_ascii=False), "model_uri": "gpt://t/m", "model_version": "t",
            "usage": {}, "elapsed_s": 0, "error": None}


@pytest.fixture(autouse=True)
def isolated(tmp_path):
    with patch.object(ec, "CACHE_DIR", tmp_path), patch.object(et, "_cache_get", ec._cache_get), \
         patch.object(et, "build_model_uri", lambda model=None: f"gpt://t/{model or 'yandexgpt-5-pro'}"):
        yield


@pytest.mark.parametrize("item, fragment", [
    ({**GOOD, "term_en": "robot"}, "слов 1"),
    ({**GOOD, "term_en": "soft growing robot arm system"}, "слов 5"),
    ({**GOOD, "term_en": "мягкий робот"}, "не латиница"),
    ({**GOOD, "term_en": "soft (growing) robot"}, "запрещённые символы"),
    ({**GOOD, "term_en": "robot swarm optimization"}, "последнее слово"),
    ({**GOOD, "doc": 7}, "не из этой пачки"),
    ({**GOOD, "quote": "a phrase that is not there"}, "quote не найден"),
])
def test_each_validation_rule(item, fragment) -> None:
    assert fragment in et.violation(item, SNIPPETS)


def test_quote_matches_after_case_and_space_normalization() -> None:
    assert et.violation(GOOD, SNIPPETS) is None


def test_bad_batch_retried_once_at_08_with_violations() -> None:
    bad = {**GOOD, "term_en": "robot"}
    with patch.object(et, "ask_llm", side_effect=[reply([bad]), reply([GOOD])]) as model:
        good, warnings = et.extract_batch("роботы", list(SNIPPETS.items()), "qwen3-235b-a22b-fp8", True)
    assert [c["name_en"] for c in good] == ["soft growing robot"]
    assert [c.kwargs["temperature"] for c in model.call_args_list] == [0.2, 0.8]
    assert "нарушил требования" in model.call_args_list[1].args[0] and "слов 1" in model.call_args_list[1].args[0]
    assert model.call_args_list[0].kwargs["model"] == "qwen3-235b-a22b-fp8"
    assert any("отброшен" in w for w in warnings)


def test_clean_batch_not_retried_and_cached() -> None:
    with patch.object(et, "ask_llm", return_value=reply([GOOD])) as model:
        et.extract_batch("роботы", list(SNIPPETS.items()), None, True)
        et.extract_batch("роботы", list(SNIPPETS.items()), None, True)
    assert model.call_count == 1


def test_extract_terms_keeps_quote_and_doc_ids() -> None:
    docs = [{"title": "SPROUT: The Open-Source Soft Growing Robot for Search and Rescue", "text": "We present a robot"}]
    with patch.object(et, "ask_llm", return_value=reply([GOOD])):
        out = et.extract_terms(docs, "роботы")
    [candidate] = out["candidates"]
    assert candidate["name_en"] == "soft growing robot" and candidate["doc_ids"] == [1]
    assert candidate["quote"] == "open-source  soft growing robot"


def test_parallel_and_sequential_give_same_candidates() -> None:
    """Пачки со случайными задержками: параллельный и последовательный прогоны дают одно и то же."""
    import random
    import time

    docs = [{"title": f"Soft Growing Robot number {i}", "text": "body"} for i in range(1, 121)]

    def fake(system, user, **kwargs):
        numbers = [int(n) for n in __import__("re").findall(r"^\[(\d+)\]", user, flags=__import__("re").M)]
        time.sleep(random.Random(numbers[0]).uniform(0, 0.05) if numbers[0] % 2 else 0.06)
        items = [{"term_en": f"growing robot type {n % 7 + 2}", "term_ru": "р", "quote": f"robot number {n}", "doc": n}
                 for n in numbers if n % 3 == 0]
        return reply(items)

    with patch.object(et, "ask_llm", side_effect=fake):
        parallel = et.extract_terms(docs, "роботы", batch_size=40, use_cache=False, parallel=True)
        sequential = et.extract_terms(docs, "роботы", batch_size=40, use_cache=False, parallel=False)
    strip = lambda out: [(c["name_en"], c["doc_ids"]) for c in out["candidates"]]
    assert strip(parallel) == strip(sequential) and len(parallel["candidates"]) > 0
    assert parallel["warnings"] == sequential["warnings"]


def test_system_prompt_is_v4_with_topic() -> None:
    prompt = et.system_prompt("  ИИ в медицине ")
    assert et.PROMPT_VERSION == "extract-v4" and et.PROMPT_PATH.name == "extract_candidates_v4.txt"
    assert prompt.startswith("Тема запроса пользователя: «ИИ в медицине».")
    assert "{topic}" not in prompt and "Не добавляй отрасль применения" not in prompt
    assert "clinical triage AI agents" in prompt and "не относящиеся к теме запроса" in prompt
    assert not et.PROMPT_PATH.with_name("extract_candidates_v2.txt").exists()


def test_topic_is_in_prompt_purpose_and_cache_key() -> None:
    docs = [{"title": SNIPPETS[1]}]
    with patch.object(et, "ask_llm", return_value=reply([GOOD])) as model:
        first = et.extract_terms(docs, "роботы")
        et.extract_terms(docs, "роботы")
        et.extract_terms(docs, "медицина")
    assert model.call_count == 2  # та же тема — из кэша, другая тема — новый вызов
    assert {c.kwargs["purpose"] for c in model.call_args_list} == {"extract-v4"}
    assert model.call_args_list[1].args[0].startswith("Тема запроса пользователя: «медицина».")
    assert first["stats"]["prompt_version"] == "extract-v4"


def test_without_documents_calls_nothing() -> None:
    with patch.object(et, "ask_llm") as model:
        result = et.extract_terms([], "роботы")
    assert model.call_count == 0 and result["candidates"] == []
