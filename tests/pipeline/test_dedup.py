"""Склейка дублей (задача К): ключ, равные ключи без вызова, YandexGPT при общей основе, отказоустойчивость, место в выдаче."""
import pytest

from pipeline import dedup as dd
from pipeline.run_query import TOP_N, split_ranked

CLIMATE = ["hybrid AI-climate model", "AI for climate modeling", "AI-driven climate modelling",
           "AI-enhanced climate modelling"]


@pytest.fixture(autouse=True)
def _cache(tmp_path, monkeypatch):
    monkeypatch.setattr(dd, "CACHE_DIR", tmp_path / "pairs")


def fake_llm(yes: set[frozenset[str]], calls: list, error: str | None = None, raises: bool = False):
    """«да» для пар из yes, «нет» для прочих; error — ошибка API, raises — исключение (таймаут)."""
    def llm(system, user, **kwargs):
        calls.append({"system": system, "user": user, **kwargs})
        if raises:
            raise TimeoutError("slow")
        if error:
            return {"text": None, "error": error}
        pair = frozenset(p.strip("«»") for p in user.split(" — "))
        return {"text": "Да." if pair in yes else "нет", "error": None}
    return llm


def test_key_normalizes_stopwords_phrase_and_stems() -> None:
    assert dd.term_key("AI-driven climate modelling") == dd.term_key("climate model")
    assert dd.term_key("artificial intelligence tokens") == frozenset({"token"})
    assert dd.term_key("AI-powered system") == frozenset()


def test_equal_keys_merge_without_call_and_empty_keys_do_not() -> None:
    calls = []
    decide, notes = dd.decider(["AI for climate modeling", "AI-driven climate modelling", "AI system", "AI tools"],
                               fake_llm(set(), calls))
    assert decide("AI for climate modeling", "AI-driven climate modelling") and not decide("AI system", "AI tools")
    assert calls == [] and notes == []


def test_common_stem_asks_llm_with_verbatim_prompt_other_pairs_are_not_asked() -> None:
    calls = []
    yes = {frozenset({"hybrid AI-climate model", "AI for climate modeling"})}
    decide, _ = dd.decider(["hybrid AI-climate model", "AI for climate modeling", "quantum sensing"], fake_llm(yes, calls))
    assert decide("AI for climate modeling", "hybrid AI-climate model")
    assert not decide("quantum sensing", "hybrid AI-climate model") and len(calls) == 1
    call = calls[0]
    assert call["system"].startswith("Термин 1: «") and call["system"].endswith("Ответь одним словом: да или нет.")
    assert call["temperature"] == 0.0 and call["model"] == "yandexgpt-5-pro"


def test_cache_is_order_free_and_non_yes_is_no() -> None:
    calls = []
    assert dd.ask_pair("agentic AI", "agentic coding", fake_llm(set(), calls)) is False
    assert dd.ask_pair("agentic coding", "agentic AI", fake_llm(set(), calls)) is False and len(calls) == 1


@pytest.mark.parametrize("kwargs", [{"error": "HTTPError: 500"}, {"raises": True}])
def test_llm_failure_is_not_duplicate_with_one_warning_and_not_cached(kwargs, tmp_path) -> None:
    calls = []
    decide, notes = dd.decider(["agentic AI", "agentic coding", "agentic AI agents"], fake_llm(set(), calls, **kwargs))
    assert not decide("agentic AI", "agentic coding")
    assert notes == ["склейка дублей: YandexGPT не ответил по 2 парам — они не склеены"]  # agentic AI agents = agentic AI по ключу
    assert not (tmp_path / "pairs").exists()


def test_pass_merges_task_examples_and_keeps_distinct_ones() -> None:
    yes = {frozenset({"hybrid AI-climate model", other}) for other in CLIMATE[1:]} |           {frozenset({"ai-powered real-time threat detection", "AI-driven threat detection"})}
    names = CLIMATE + ["ai-powered real-time threat detection", "AI-driven threat detection", "agentic AI", "agentic coding"]
    ranked = [{"name_en": n, "score": 1 - i / 100} for i, n in enumerate(names)]
    decide, _ = dd.decider(names, fake_llm(yes, []))
    assert dd.dedup(ranked, decide) == {1: 0, 2: 0, 3: 0, 5: 4}


def _ranked(names_scores):
    ranked = [{"name": n, "score": s, "features": {}, "contributions": {}, "counters": {}} for n, s in names_scores]
    by_key = {n: {"name_en": n, "name_ru": n, "doc_ids": []} for n, _ in names_scores}
    return ranked, by_key


def test_split_ranked_moves_duplicate_to_excluded_and_refills_top() -> None:
    names = [(f"distinct topic number {i} alpha{i}", 0.99 - i / 100) for i in range(TOP_N)]
    ranked, by_key = _ranked([("climate model", 0.999), ("AI climate modelling", 0.998)] + names)
    top, dropped = split_ranked(ranked, by_key, [], 0.4, duplicate_of={1: 0})
    assert len(top) == TOP_N and top[0]["variants"] == [{"term_en": "AI climate modelling", "score": 0.998}]
    assert top[-1]["name_en"] == names[TOP_N - 2][0]  # добор следующим по score
    duplicate = next(d for d in dropped if d["skipped_reason"] == "duplicate_of")
    assert duplicate["duplicate_of"] == "climate model" and "«climate model»" in duplicate["reason_ru"]


def test_run_query_survives_yandexgpt_failure(tmp_path, monkeypatch) -> None:
    """Сбой YandexGPT в склейке: прогон не падает, пары не склеены, одно предупреждение."""
    from tests.pipeline import test_run_query as base
    monkeypatch.setattr(dd, "ask_pair", lambda a, b, llm: None)
    out, _, _ = base.run(tmp_path)
    notes = [w for w in out["warnings"] if w.startswith("склейка дублей")]
    assert len(out["top"]) > 0 and len(notes) <= 1
    assert all(e["skipped_reason"] != "duplicate_of" for e in out["excluded"]) or notes == []
