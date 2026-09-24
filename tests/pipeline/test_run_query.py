"""Оркестратор без сети: заглушки LLM, нормализатора и трёх источников, настоящий артефакт модели."""
import json
from datetime import date, timedelta
from unittest.mock import patch

import pytest

import search.extract_candidates as ec
import search.extract_terms as et
import search.subqueries as sq
from collector.api import tech_key
from collector.constants import COLLECTION_START, COUNTER_WINDOWS, CUTOFF_DATE
from collector.settings import Settings
from pipeline import naming
from pipeline import run_query as rq
from pipeline import fetch as fetch_module
from pipeline import search_cache
from tests.collector.fakes import FakeAdapter, doc

SIGNAL, MAINSTREAM = "robotic teleoperation data", "humanoid robot"
NO_TRACE, PARTIAL = "phantom gripper lattice", "partial counter device"
RU = ["фотонные вычисления", "оптический интерконнект", "мемристорные матрицы",
      "нейроморфные ускорители", "квантовые сенсоры"]
EN = ["photonic computing", "optical interconnect", "memristor crossbar arrays",
      "neuromorphic accelerators", "quantum sensing", "spiking neural networks",
      "silicon photonic modulators", "analog inference chips"]

# Нормализатор: name_ru шага 4 -> три варианта. Выбирается вариант с наибольшим следом.
VARIANTS = {
    "данные телеуправления роботами": ["teleoperation data", SIGNAL, "robot teleoperation dataset"],
    "данные телеуправления": [SIGNAL, "teleoperation logs", "teleoperation data"],
    "гуманоидный робот": [MAINSTREAM, "humanoid robots", "bipedal humanoid"],
    "фантомная решётка захвата": [NO_TRACE, "phantom lattice gripper", "lattice gripper phantom"],
    "частичное устройство счёта": [PARTIAL, "partial counters", "counter device"],
    "робот сварщик": ["welding", "welds", "welder"],
}
# След за 2020-09 … 2026-08: у сигнала и дубликата максимум — одна и та же фраза.
TRACE = {SIGNAL: 5, "teleoperation data": 2, MAINSTREAM: 900, "humanoid robots": 40,
         PARTIAL: 3, "welding": 50}


def window_counts(per_window: dict[str, int]) -> dict:
    """Счётчики по всем семи окнам: окно без значения — ноль."""
    return {bounds: per_window.get(name, 0) for name, bounds in COUNTER_WINDOWS.items()}


# Числа подобраны под профили: у сигнала мало науки и свежие новости, у мейнстрима — горы статей.
COUNTS = {
    "openalex": {SIGNAL: window_counts({"2024": 3, "2025": 5}),
                 MAINSTREAM: window_counts({"prev6": 20000, **{y: 5000 for y in map(str, range(2020, 2026))}}),
                 PARTIAL: window_counts({"2025": 4})},
    "arxiv": {SIGNAL: window_counts({"2025": 2}),
              MAINSTREAM: window_counts({"prev6": 3000, **{y: 500 for y in map(str, range(2020, 2026))}}),
              PARTIAL: {bounds: 1 for name, bounds in COUNTER_WINDOWS.items() if name != "2025"}},
    "techcrunch": {SIGNAL: window_counts({"2024": 40, "2025": 60}),
                   MAINSTREAM: window_counts({"2025": 5}), PARTIAL: window_counts({"2025": 1})},
}


class QueryFake(FakeAdapter):
    """Заглушка источника, у которой счётчик зависит от фразы запроса."""

    def count_matching(self, search, date_from, date_to_exclusive):
        self.match_calls.append((search.query, date_from, date_to_exclusive))
        return COUNTS[self.source].get(search.terms[0], {}).get((date_from, date_to_exclusive))

    def count_institutions(self, search, date_from, date_to_exclusive):
        """След в окне обучения и организации одним вызовом."""
        assert (date_from, date_to_exclusive) == (COLLECTION_START, CUTOFF_DATE)
        works = TRACE.get(search.terms[0], 0)
        return {"works": works, "institutions": 3 if works else 0, "capped": False}

    def count_windows_one_call(self, search, windows):
        """arXiv одним вызовом: окно без счётчика -> None, как неполный ответ, и откат на семь."""
        self.one_calls = getattr(self, "one_calls", 0) + 1
        known = COUNTS[self.source].get(search.terms[0], {})
        counts = {name: known.get(bounds) for name, bounds in windows.items()}
        return None if None in counts.values() else counts


def adapters() -> list[QueryFake]:
    # Даты внутри окна поиска №1 (180 дней до сегодня), чем больше i, тем свежее документ.
    fresh = lambda i: (date.today() - timedelta(days=60 - 10 * i)).isoformat()
    docs = lambda s, t: [doc(source=s, source_type=t, published_at=fresh(i), url=f"https://{s}/{i}",
                             title=f"{s} doc {i}") for i in range(3, 6)]
    return [QueryFake("openalex", "paper", documents=docs("openalex", "paper")),
            QueryFake("arxiv", "preprint", documents=docs("arxiv", "preprint")),
            QueryFake("techcrunch", "news", documents=docs("techcrunch", "news"))]


def fake_subqueries(system_prompt, user_prompt, **kwargs):
    lang = "ru" if '"ru"' in system_prompt else "en"
    return {"text": json.dumps({lang: RU if lang == "ru" else EN}, ensure_ascii=False),
            "model_uri": "gpt://t/yandexgpt-5-pro", "model_version": "t", "usage": {}, "elapsed_s": 0, "error": None}


def fake_extract(system_prompt, user_prompt, **kwargs):
    """Шаг 4 (extract-v2) без сети: термин, русское название, цитата из документа и его номер."""
    raw = ["robotic teleoperation dataset", "Robotic  Teleoperation Data", "humanoid robot", "phantom thing",
           "partial counter device", "robotic welding"]
    items = [{"doc": d, "term_ru": ru, "term_en": en, "quote": "doc"}
             for d, (ru, en) in enumerate(zip(VARIANTS, raw), start=1)]
    return {"text": json.dumps(items, ensure_ascii=False), "model_uri": "m",
            "model_version": "t", "usage": {}, "elapsed_s": 0, "error": None}


def fake_ask_name(name_ru, area, repeat=0, note=""):
    """Нормализатор без сети: три строки VARIANT, как отвечает модель."""
    return "\n".join(f"VARIANT: {name}" for name in VARIANTS[name_ru]), True


def run(tmp_path, extract=fake_extract, patents=None, **options):
    """run_query на заглушках; по умолчанию с нормализатором, как в Д5 и И2.

    Роспатент всегда заглушка (patents — ответы по фразам; по умолчанию все сбои): сеть не нужна.
    """
    from collector import rospatent as rp
    options = {"naming_mode": "normalizer", **options}
    stages, extract_prompts = [], []

    def ask(system_prompt, user_prompt, **kwargs):
        extract_prompts.append(user_prompt)
        return extract(system_prompt, user_prompt, **kwargs)

    with patch.object(sq, "ask_llm", side_effect=fake_subqueries), \
         patch.object(sq, "build_model_uri", lambda: "gpt://t/yandexgpt-5-pro"), \
         patch.object(sq, "CACHE_DIR", tmp_path / "subq"), \
         patch.object(et, "ask_llm", side_effect=ask), \
         patch.object(et, "build_model_uri", lambda model=None: "gpt://t/yandexgpt-5-pro"), \
         patch.object(ec, "ask_llm", side_effect=ask), \
         patch.object(ec, "build_model_uri", lambda model=None: "gpt://t/yandexgpt-5-pro"), \
         patch.object(ec, "CACHE_DIR", tmp_path / "extract"), \
         patch.object(search_cache, "CACHE_DIR", tmp_path / "search"), \
         patch.object(fetch_module, "COUNTERS_CACHE_DIR", tmp_path / "counters"), \
         patch.object(rp, "count_all", patents or fake_patents({})), \
         patch.object(naming.T, "ask_name", side_effect=fake_ask_name), \
         patch.object(et, "dedupe_candidates", lambda items, threshold=None: [
             {**item, "doc_ids": [item["doc"]], "doc_count": 1} for item in items]):
        out = rq.run_query("роботы для промышленности", "Роботы", adapters=adapters(), settings=Settings(),
                           progress=lambda stage, done, total: stages.append(stage), **options)
    return out, stages, extract_prompts


@pytest.fixture
def result(tmp_path):
    out, stages, _ = run(tmp_path)
    return out, stages


def test_candidate_sources_limit_step4_documents(tmp_path) -> None:
    """candidate_sources={arxiv}: шаг 4 видит только документы arXiv; поиск и stats — по всем."""
    out, _, prompts = run(tmp_path, candidate_sources={"arxiv"})
    text = "\n".join(prompts)
    assert "arxiv doc" in text and "openalex doc" not in text and "techcrunch doc" not in text
    assert out["candidate_sources"] == ["arxiv"]
    assert out["stats"]["documents_for_candidates"] == 3 and out["stats"]["documents_total"] == 9


def test_candidate_sources_default_is_all(tmp_path) -> None:
    out, _, prompts = run(tmp_path)
    text = "\n".join(prompts)
    assert all(f"{s} doc" in text for s in ("openalex", "arxiv", "techcrunch"))
    assert out["candidate_sources"] is None and out["stats"]["documents_for_candidates"] == 9


DETAIL_KEYS = {"name_raw", "name_variants", "name_choice_rule", "n_works", "n_institutions",
               "n_institutions_capped", "n_docs", "n_sources", "known_training_label", "quote"}


def test_output_matches_schema(result) -> None:
    out, stages = result
    assert set(out) == {"query_id", "topic", "area", "model_version", "threshold", "cutoff_date", "subqueries",
                        "candidate_sources", "extract_version", "extract_model", "naming_mode", "stats", "normalizer_deviations", "top", "excluded",
                        "timings", "warnings"}
    assert out["normalizer_deviations"] == ["company_stoplist_off"]
    assert set(out["stats"]) == {"documents_by_source", "documents_total", "documents_for_candidates",
                                 "candidates_found", "candidates_named",
                                 "candidates_scored", "above_threshold", "above_075",
                                 "rospatent_enabled", "rospatent_failures"}
    assert out["model_version"] == "s2a2-v1"
    assert set(out["timings"]) == {"subqueries", "search", "candidates", "naming", "counters", "ranking", "total",
                                   "queues"}
    assert {"subqueries", "search", "candidates", "naming", "counters", "ranking"} <= set(stages)
    patent_keys = {"n_pat", "share_patent", "rospatent_failed", "note_ru"}
    for item in out["top"]:
        assert set(item) == {"rank", "name_ru", "name_en", "score", "explanation_ru", "contributions",
                             "counters", "sources", "model_version"} | DETAIL_KEYS | patent_keys
        assert item["name_choice_rule"] == "max_trace" and len(item["name_variants"]) == 3
        assert len(item["sources"]) <= 5
    for item in out["excluded"]:
        base = {"name_ru", "name_en", "score", "skipped_reason", "reason_ru", "model_version"} | DETAIL_KEYS
        assert set(item) == (base | patent_keys if item["score"] is not None else base)
        assert item["model_version"] == "s2a2-v1"


def test_each_reason_in_its_case(result) -> None:
    out, _ = result
    reasons = {item["name_en"]: item["skipped_reason"] for item in out["excluded"]}
    no_trace = next(item for item in out["excluded"] if item["name_raw"] == "phantom thing")
    assert no_trace["skipped_reason"] == "no_trace" and no_trace["name_en"] == "phantom thing"
    assert [v["n_works"] for v in no_trace["name_variants"]] == [0, 0, 0]
    assert reasons[PARTIAL] == "no_counters"
    assert reasons["welding"] == "bad_name"
    assert reasons[MAINSTREAM] == "below_threshold"
    assert [tech_key(item["name_en"]) for item in out["top"]] == [SIGNAL]


def test_normalized_names_merged_with_doc_ids(result) -> None:
    """Два кандидата шага 4 с разными формулировками дали одно название: одна запись, doc_ids вместе."""
    out, _ = result
    assert out["stats"]["candidates_found"] == 6 and out["stats"]["candidates_named"] == 4
    top = out["top"][0]
    assert top["name_raw"] in {"robotic teleoperation dataset", "Robotic Teleoperation Data"}
    assert tech_key(top["name_en"]) == SIGNAL and top["n_works"] == 5
    assert max(v["n_works"] for v in top["name_variants"]) == 5
    assert top["n_docs"] == 2 and top["n_sources"] == 1
    assert {s["url"] for s in top["sources"]} == {"https://openalex/3", "https://openalex/4"}


def test_humanoid_robot_is_known_training_mainstream(result) -> None:
    out, _ = result
    humanoid = next(item for item in out["excluded"] if item["name_en"] == MAINSTREAM)
    assert humanoid["known_training_label"]["label"] == "mainstream"
    assert out["top"][0]["known_training_label"] == {"id": "s14", "label": "signal"}


def test_merge_candidates_unites_doc_ids() -> None:
    got = rq.merge_candidates([{"name_ru": "а", "name_en": "Edge AI Chips", "doc_ids": [3, 1]},
                               {"name_ru": "б", "name_en": "edge  ai chips", "doc_ids": [2, 3]}])
    assert got == [{"name_ru": "а", "name_en": "Edge AI Chips", "doc_ids": [1, 2, 3]}]


@pytest.mark.parametrize("name, ok", [("edge ai chips", True), ("chips", False),
                                      ("one two three four five six", False), ("чипы для edge", False),
                                      ("edge (ai) chips", False), ("edge ai/ml chips", False), ("1 2", False)])
def test_good_name(name, ok) -> None:
    assert rq.good_name(name) is ok


def test_cap_keeps_most_documented() -> None:
    items = [{"name_ru": n, "name_en": n, "doc_ids": list(range(k))} for n, k in [("a b", 1), ("c d", 3), ("e f", 2)]]
    kept, capped = rq.apply_cap(items, [], limit=2)
    assert [item["name_en"] for item in kept] == ["c d", "e f"]
    assert [(item["name_en"], item["skipped_reason"]) for item in capped] == [("a b", "cap")]


def test_top_is_not_padded_below_threshold() -> None:
    """Выше порога один кандидат — в ТОП один, остальные ниже порога идут в исключённые."""
    names = {f"tech {i}": {"name_ru": f"т {i}", "name_en": f"tech {i}", "doc_ids": []} for i in range(3)}
    ranked = [{"name": "tech 0", "score": 0.9, "features": {"recency": 0.5}, "contributions": {"recency": 1.0},
               "counters": {}},
              *[{"name": f"tech {i}", "score": 0.1, "features": {"recency": 0.1},
                 "contributions": {"recency": -1.0}, "counters": {}} for i in (1, 2)]]
    top, dropped = rq.split_ranked(ranked, names, [], threshold=0.325)
    assert [item["name_en"] for item in top] == ["tech 0"]
    assert [item["skipped_reason"] for item in dropped] == ["below_threshold", "below_threshold"]


def test_sources_fresh_first_and_limited() -> None:
    docs = [{"title": str(i), "url": f"u{i}", "published_at": f"2026-0{i}-01", "source": "arxiv",
             "source_type": "preprint", "language": None, "trust_level": None} for i in range(1, 8)]
    got = rq.sources_of([1, 2, 3, 4, 5, 6, 7], docs)
    assert [item["title"] for item in got] == ["7", "6", "5", "4", "3"]
    assert got[0]["language"] is None and got[0]["trust_level"] is None


def test_sixteenth_above_threshold_is_beyond_top() -> None:
    """Выше порога 16 кандидатов: в ТОП 15, шестнадцатый — beyond_top, а не below_threshold."""
    names = {f"tech {i}": {"name_ru": f"т {i}", "name_en": f"tech {i}", "doc_ids": []} for i in range(16)}
    ranked = [{"name": f"tech {i}", "score": 0.9 - i / 100, "features": {}, "contributions": {}, "counters": {}}
              for i in range(16)]
    top, dropped = rq.split_ranked(ranked, names, [], threshold=0.325)
    assert len(top) == 15
    assert [(item["name_en"], item["skipped_reason"]) for item in dropped] == [("tech 15", "beyond_top")]


def test_known_training_label() -> None:
    """Совпадение tech_key с обучающей технологией даёт её id и класс, иначе null."""
    assert rq.training_labels()["edge model compression"] == {"id": "s1", "label": "signal"}
    assert rq.details({"name_ru": "а", "name_en": "No Such Technology", "doc_ids": []}, [])[
        "known_training_label"] is None


def test_arxiv_one_call_and_fallback() -> None:
    """Флаг включён: полный ответ — один вызов без count_matching; None — откат на семь окон."""
    from pipeline.fetch import count_source
    from collector.api import DocumentCollector
    from collector.models import Candidate

    arxiv = [a for a in adapters() if a.source == "arxiv"][0]
    collector = DocumentCollector(adapters=[arxiv])
    full = count_source(collector, Candidate(candidate_id="c", name_en=SIGNAL, terms=[SIGNAL]))
    assert arxiv.one_calls == 1 and arxiv.match_calls == []
    assert {row.window for row in full.counters} == set(COUNTER_WINDOWS)
    partial = count_source(collector, Candidate(candidate_id="c", name_en=PARTIAL, terms=[PARTIAL]))
    assert arxiv.one_calls == 2 and len(arxiv.match_calls) == len(COUNTER_WINDOWS)
    assert {row.window for row in partial.counters} == set(COUNTER_WINDOWS) - {"2025"}


@pytest.mark.parametrize("a, b, merged", [
    ("vision language action model optimization", "vision language action model", True),
    ("uwb robot localization", "localization method in robotics", False),
    ("mobile ai robots", "ai for robotics", False),
])
def test_subtopic_merge_only_equal_stems(a, b, merged) -> None:
    """Сливаются только одинаковые множества основ; надмножество — нет."""
    items = [{"name_ru": "а", "name_en": a, "n_works": 15, "doc_ids": [1]},
             {"name_ru": "б", "name_en": b, "n_works": 532, "doc_ids": [2]}]
    got = rq.merge_subtopics(items)
    if merged:
        assert got == [{"name_ru": "б", "name_en": b, "n_works": 532, "doc_ids": [1, 2]}]
    else:
        assert [item["name_en"] for item in got] == [a, b]


def test_direct_naming_with_extract_v2(tmp_path) -> None:
    """extract-v2 + direct: нормализатор не вызывается, name_en = термин шага 4, quote в выходе."""
    items = [{"term_en": SIGNAL, "term_ru": "данные телеуправления", "quote": "doc 3", "doc": 1},
             {"term_en": MAINSTREAM, "term_ru": "гуманоидный робот", "quote": "doc 4", "doc": 2}]
    answer = {"text": json.dumps(items, ensure_ascii=False), "model_uri": "m", "model_version": "t",
              "usage": {}, "elapsed_s": 0, "error": None}
    with patch.object(naming, "normalize_all") as normalizer:
        out, _, _ = run(tmp_path, extract=lambda *args, **kwargs: answer, naming_mode="direct")
    assert normalizer.call_count == 0
    assert out["naming_mode"] == "direct" and out["extract_version"] == "v2"
    assert [item["name_en"] for item in out["top"]] == [SIGNAL]
    assert out["top"][0]["quote"] == "doc 3" and out["top"][0]["name_choice_rule"] == "direct"


def test_counters_file_cache_shared_between_runs(tmp_path) -> None:
    """Второй запрос той же фразы берёт счётчики из файла, источник не опрашивается."""
    from collector.api import DocumentCollector
    from collector.models import Candidate

    with patch.object(fetch_module, "COUNTERS_CACHE_DIR", tmp_path):
        openalex = adapters()[0]
        collector = DocumentCollector(adapters=[openalex])
        first = fetch_module.cached_count_source(collector, Candidate(candidate_id="c", name_en=SIGNAL, terms=[SIGNAL]))
        calls = len(openalex.match_calls)
        again = fetch_module.cached_count_source(collector, Candidate(candidate_id="c", name_en=SIGNAL, terms=[SIGNAL]))
    assert len(openalex.match_calls) == calls
    assert [(r.window, r.n) for r in again.counters] == [(r.window, r.n) for r in first.counters]


def test_default_mode_is_r1() -> None:
    """По умолчанию — рука R1: extract-v2, yandexgpt-5-pro, без нормализатора."""
    import inspect
    defaults = {name: p.default for name, p in inspect.signature(rq.run_query).parameters.items()}
    assert (defaults["extract_model"], defaults["naming_mode"]) == ("yandexgpt-5-pro", "direct")
    assert defaults["extract_version"] == "v2"


def fake_extract_v1(system_prompt, user_prompt, **kwargs):
    """Прежний шаг 4 (v1) без сети: те же шесть кандидатов в формате {"candidates": [...]}."""
    raw = ["robotic teleoperation dataset", "Robotic  Teleoperation Data", "humanoid robot", "phantom thing",
           "partial counter device", "robotic welding"]
    items = [{"doc": d, "name_ru": ru, "name_en": en, "terms": [en], "context_terms": []}
             for d, (ru, en) in enumerate(zip(VARIANTS, raw), start=1)]
    return {"text": json.dumps({"candidates": items}, ensure_ascii=False), "model_uri": "m",
            "model_version": "t", "usage": {}, "elapsed_s": 0, "error": None}


def test_extract_v1_with_normalizer_end_to_end(tmp_path) -> None:
    """Старый путь: шаг 4 v1 + нормализатор даёт те же причины и ТОП, что v2 + нормализатор."""
    with patch.object(ec, "dedupe_candidates", lambda items, threshold=None: [
            {**item, "doc_ids": [item["doc"]], "doc_count": 1} for item in items]), \
         patch.object(rq, "extract_terms", side_effect=AssertionError("v2 не должен вызываться")):
        out, _, prompts = run(tmp_path, extract=fake_extract_v1, extract_version="v1")
    assert (out["extract_version"], out["extract_model"], out["naming_mode"]) == ("v1", None, "normalizer")
    assert prompts and out["stats"]["candidates_found"] == 6 and out["stats"]["candidates_named"] == 4
    reasons = {item["name_en"]: item["skipped_reason"] for item in out["excluded"]}
    assert reasons[PARTIAL] == "no_counters" and reasons["welding"] == "bad_name"
    assert reasons[MAINSTREAM] == "below_threshold" and reasons["phantom thing"] == "no_trace"
    assert [tech_key(item["name_en"]) for item in out["top"]] == [SIGNAL]


def test_unknown_extract_version_rejected() -> None:
    with pytest.raises(ValueError, match="extract_version"):
        rq.run_query("тема", extract_version="v3", adapters=[], settings=Settings())


def test_techcrunch_one_call_in_pipeline() -> None:
    """Флаг задачи Л включён: TechCrunch в оркестраторе — одним вызовом, без count_matching."""
    from pipeline.fetch import count_source, one_call
    from collector.api import DocumentCollector
    from collector.models import Candidate

    assert one_call("techcrunch") and one_call("arxiv") and not one_call("openalex")
    techcrunch = [a for a in adapters() if a.source == "techcrunch"][0]
    got = count_source(DocumentCollector(adapters=[techcrunch]), Candidate(candidate_id="c", name_en=SIGNAL, terms=[SIGNAL]))
    assert techcrunch.one_calls == 1 and techcrunch.match_calls == []
    assert {row.source for row in got.counters} == {"techcrunch"}


def test_source_queues_match_per_candidate_mode(tmp_path) -> None:
    """Очереди по источникам (Л5.2) дают те же счётчики и предупреждения, что покандидатный режим."""
    import threading
    from collector.models import SearchTerms

    phrases = [SIGNAL, MAINSTREAM, PARTIAL]
    results = {}
    for mode in ("queues", "per_candidate"):
        open_now, peak, lock = {}, {}, threading.Lock()
        sources = adapters()
        for adapter in sources:
            original = adapter.count_windows_one_call

            def wrapped(search, windows, _a=adapter, _orig=original):
                with lock:
                    open_now[_a.source] = open_now.get(_a.source, 0) + 1
                    peak[_a.source] = max(peak.get(_a.source, 0), open_now[_a.source])
                try:
                    return _orig(search, windows)
                finally:
                    with lock:
                        open_now[_a.source] -= 1
            adapter.count_windows_one_call = wrapped
        warnings = []
        with patch.object(fetch_module, "COUNTERS_CACHE_DIR", tmp_path / mode):
            fetch = fetch_module.parallel_fetch(sources, Settings(), warnings)
            if mode == "queues":
                fetch.prefetch(phrases)
            frames = [fetch(SearchTerms(terms=[p], context_terms=[], query="")) for p in phrases]
        results[mode] = ([f.sort_values(["source", "window"]).to_dict("records") for f in frames], warnings)
        assert max(peak.values()) <= 1
    assert results["queues"] == results["per_candidate"]


def fake_patents(results: dict):
    """Заглушка очереди Роспатента: фиксированные ответы по фразам, вызовы записываются."""
    calls = []

    def count_all(phrases, datasets, token, *, parallel=1, use_cache=True, **kwargs):
        calls.append(list(phrases))
        found = {phrase: {"phrase": phrase, "failed": True, "n_pat": None, **results.get(phrase, {})}
                 for phrase in dict.fromkeys(phrases)}
        return found, {"source": "rospatent", "started": 0.0, "finished": 0.0,
                       "candidates": len(found), "parallel": parallel, "requests": 0,
                       "cache_hits": 0, "retries": 0, "n429": 0,
                       "failures": sum(item["failed"] for item in found.values())}

    count_all.calls = calls
    return count_all


def test_rospatent_disabled_makes_no_request_and_warns(tmp_path) -> None:
    """--no-rospatent: очередь не запускается, у кандидатов нет патентных полей, прогон помечен."""
    fake = fake_patents({})
    out, _, _ = run(tmp_path, patents=fake, rospatent=False)
    assert fake.calls == []
    assert {queue["source"] for queue in out["timings"]["queues"]} == {"openalex", "arxiv", "techcrunch"}
    assert all("n_pat" not in item for item in out["top"] + out["excluded"])
    assert out["stats"]["rospatent_enabled"] is False and rq.ROSPATENT_OFF_WARNING in out["warnings"]


def test_missing_rospatent_key_is_a_run_warning(tmp_path, monkeypatch) -> None:
    """Роспатент включён, ключа нет: прогон не падает, в warnings — запись об этом."""
    monkeypatch.delenv("ROSPATENT", raising=False)
    out, _, _ = run(tmp_path)
    assert rq.ROSPATENT_NO_KEY_WARNING in out["warnings"]


def test_rospatent_enabled_by_default_feeds_the_model(tmp_path) -> None:
    """По умолчанию Роспатент включён: n_pat идёт в share_patent и меняет score; сбой — NaN и пометка."""
    fake = fake_patents({SIGNAL: {"n_pat": 0, "failed": False}})
    plain, _, _ = run(tmp_path / "plain", patents=fake_patents({}), rospatent=False)
    out, _, _ = run(tmp_path / "patents", patents=fake)
    assert len(fake.calls) == 1 and len(fake.calls[0]) == len(set(fake.calls[0]))
    signal = out["top"][0]
    assert (signal["n_pat"], signal["share_patent"], signal["rospatent_failed"]) == (0, 0.0, False)
    assert "note_ru" not in signal
    mainstream = next(item for item in out["excluded"] if item["name_en"] == MAINSTREAM)
    assert (mainstream["n_pat"], mainstream["share_patent"], mainstream["rospatent_failed"]) == (None, None, True)
    assert mainstream["note_ru"] == rq.PATENT_FAILED_NOTE
    assert out["stats"]["rospatent_failures"] >= 1
    assert "rospatent" in {queue["source"] for queue in out["timings"]["queues"]}
    scores = lambda result: {item["name_en"]: item["score"] for item in result["top"] + result["excluded"]}
    # Сбой = NaN = медиана обучения, как при выключенном Роспатенте; ноль патентов — другое значение.
    assert scores(out)[MAINSTREAM] == scores(plain)[MAINSTREAM]
    assert scores(out)[signal["name_en"]] != scores(plain)[signal["name_en"]]
    json.dumps(out, allow_nan=False)


def test_patent_fields_zero_documents_is_null_not_failure() -> None:
    """Успешный ответ total = 0 и ноль научных работ: доля не определена (null), это не сбой."""
    fields = rq.patent_fields({"2024": {"openalex": 0, "arxiv": 0}}, {"n_pat": 0, "failed": False})
    assert fields == {"n_pat": 0, "share_patent": None, "rospatent_failed": False}


def test_patent_fields_use_six_windows_without_prev6() -> None:
    counters = {"prev6": {"openalex": 1000, "arxiv": 1000}, "2020": {"openalex": 6, "arxiv": 2,
                                                                      "techcrunch": 50}}
    fields = rq.patent_fields(counters, {"n_pat": 2, "failed": False})
    assert fields["share_patent"] == 0.2
