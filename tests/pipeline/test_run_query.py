"""Оркестратор без сети: заглушки LLM и трёх источников, настоящий артефакт модели."""
import json
import sys
from datetime import date, timedelta
from unittest.mock import patch

import pytest

import search.extract_terms as et
import search.subqueries as sq
from collector.api import tech_key
from collector.constants import COLLECTION_START, COUNTER_WINDOWS, CUTOFF_DATE
from collector.settings import Settings
from pipeline import dedup as dedup_module, translate
from search import llm_yandex_gpt as llm_module
from pipeline import run_query as rq
from pipeline import fetch as fetch_module
from pipeline import search_cache
from tests.collector.fakes import FakeAdapter, doc
from tests.data_required import require_model

SIGNAL, MAINSTREAM = "robotic teleoperation data", "humanoid robot"
PARTIAL = "partial counter device"
RU = ["фотонные вычисления", "оптический интерконнект", "мемристорные матрицы",
      "нейроморфные ускорители", "квантовые сенсоры"]
EN = ["photonic computing", "optical interconnect", "memristor crossbar arrays",
      "neuromorphic accelerators", "quantum sensing", "spiking neural networks",
      "silicon photonic modulators", "analog inference chips"]

# Русские термины шага 4 по порядку документов.
TERMS_RU = ["данные телеуправления роботами", "данные телеуправления", "гуманоидный робот",
            "фантомная решётка захвата", "частичное устройство счёта", "робот сварщик"]
# След за 2020-09 … 2026-08 по термину шага 4 (он же name_en); нет в словаре — нулевой след.
TRACE = {SIGNAL: 5, MAINSTREAM: 900, PARTIAL: 3}


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


# Подзапросы раунда добора (Х2): на промпт с «Уже использованы» заглушка отвечает этими списками.
RU_R1 = ["голографическая память", "плазмонные волноводы", "фотонные нейросети"]
EN_R1 = ["holographic storage media", "plasmonic waveguides", "optical tensor cores",
         "thin film lithium niobate", "microring resonator arrays", "photonic neural networks"]


def fake_subqueries(system_prompt, user_prompt, **kwargs):
    lang = "ru" if '"ru"' in system_prompt else "en"
    extra = "Уже использованы подзапросы" in user_prompt
    items = (RU_R1 if extra else RU) if lang == "ru" else (EN_R1 if extra else EN)
    return {"text": json.dumps({lang: items}, ensure_ascii=False),
            "model_uri": "gpt://t/yandexgpt-5-pro", "model_version": "t", "usage": {}, "elapsed_s": 0, "error": None}


def fake_extract(system_prompt, user_prompt, **kwargs):
    """Шаг 4 (extract-v4) без сети: термин, русское название, цитата из документа и его номер."""
    raw = ["robotic teleoperation data", "Robotic  Teleoperation Data", "humanoid robot", "phantom thing",
           "partial counter device", "robotic welding"]
    items = [{"doc": d, "term_ru": ru, "term_en": en, "quote": "doc"}
             for d, (ru, en) in enumerate(zip(TERMS_RU, raw), start=1)]
    return {"text": json.dumps(items, ensure_ascii=False), "model_uri": "m",
            "model_version": "t", "usage": {}, "elapsed_s": 0, "error": None}


def fake_translate(system, user, **kwargs):
    """Перевод названия без сети: «русское <термин>»."""
    return {"text": f"русское {user}", "error": None}


def run(tmp_path, extract=fake_extract, patents=None, **options):
    """run_query на заглушках в боевом режиме: extract-v4, name_en = термин шага 4.

    Роспатент всегда заглушка (patents — ответы по фразам; по умолчанию все сбои): сеть не нужна.
    Нужен настоящий артефакт модели: без data/model тест пропускается.
    """
    require_model()
    from collector import rospatent as rp
    stages, extract_prompts = [], []

    def ask(system_prompt, user_prompt, **kwargs):
        extract_prompts.append(user_prompt)
        return extract(system_prompt, user_prompt, **kwargs)

    with patch.object(sq, "ask_llm", side_effect=fake_subqueries), \
         patch.object(sq, "build_model_uri", lambda: "gpt://t/yandexgpt-5-pro"), \
         patch.object(sq, "CACHE_DIR", tmp_path / "subq"), \
         patch.object(et, "ask_llm", side_effect=ask), \
         patch.object(et, "build_model_uri", lambda model=None: "gpt://t/yandexgpt-5-pro"), \
         patch.object(et, "CACHE_DIR", tmp_path / "extract"), \
         patch.object(search_cache, "CACHE_DIR", tmp_path / "search"), \
         patch.object(fetch_module, "COUNTERS_CACHE_DIR", tmp_path / "counters"), \
         patch.object(fetch_module, "openalex_quota", lambda: None), \
         patch.object(rp, "count_all", patents or fake_patents({})), \
         patch.object(translate, "CACHE_DIR", tmp_path / "translate"), \
         patch.object(dedup_module, "CACHE_DIR", tmp_path / "dedup"), \
         patch.object(llm_module, "ask_llm", side_effect=fake_translate), \
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
SCORED_KEYS = {"tech_key", "features", "is_signal", "threshold", "weak_source_only"}


def test_output_matches_schema(result) -> None:
    out, stages = result
    assert set(out) == {"query_id", "topic", "area", "model_version", "threshold", "cutoff_date", "subqueries",
                        "candidate_sources", "extract_version", "extract_model", "naming_mode", "candidates_version",
                        "stats", "normalizer_deviations", "top", "excluded", "candidates",
                        "enrichment", "_documents",
                        "timings", "warnings"}
    assert out["normalizer_deviations"] == ["company_stoplist_off"]
    assert set(out["stats"]) == {"documents_by_source", "documents_total", "documents_for_candidates",
                                 "documents_analyzed",
                                 "documents_for_candidates_by_source",
                                 "candidates_found", "candidates_named",
                                 "candidates_scored", "above_threshold", "above_075",
                                 "rospatent_enabled", "rospatent_failures", "translation",
                                 "time_budget_s", "elapsed_s", "stopped_at", "counters_schedule",
                                 "rounds", "documents_topped_up"}
    assert out["stats"]["time_budget_s"] == rq.TIME_BUDGET_S and out["stats"]["stopped_at"] is None
    assert out["model_version"] == "s2a2-v1"
    assert set(out["timings"]) == {"subqueries", "search", "candidates", "naming", "counters", "ranking", "dedup",
                                   "translate", "enrich", "total", "queues"}
    assert out["enrichment"] == "done"
    assert {"subqueries", "search", "candidates", "naming", "counters", "ranking", "dedup", "translate",
            "enrich"} <= set(stages)
    patent_keys = {"n_pat", "share_patent", "rospatent_failed", "note_ru"}
    for item in out["top"]:
        assert set(item) == {"rank", "name_ru", "name_en", "score", "rank_score", "explanation_ru", "why_ru", "contributions",
                             "counters", "sources", "model_version", "name_ru_source", "name_ru_auto",
                             "variants"} | DETAIL_KEYS | patent_keys | SCORED_KEYS
        assert item["name_ru"] == f"Русское {item['name_en']}" and item["name_ru_source"] == "translate"
        assert item["name_choice_rule"] == "direct" and len(item["name_variants"]) == 1
        assert len(item["sources"]) <= 5
    for item in out["excluded"]:
        base = {"name_ru", "name_en", "score", "skipped_reason", "reason_ru", "model_version",
                "name_ru_source", "name_ru_auto", "tech_key"} | DETAIL_KEYS
        scored_extra = patent_keys | {"variants", "features", "contributions", "counters",
                                      "is_signal", "threshold"}
        assert set(item) == (base | scored_extra if item["score"] is not None else base)
        assert item["model_version"] == "s2a2-v1" and item["name_ru_auto"] is True
        scored = item["skipped_reason"] in ("below_threshold", "beyond_top")
        assert item["name_ru_source"] == ("translate" if scored else "extract" if item["name_ru"] else None)


def test_each_reason_in_its_case(result) -> None:
    out, _ = result
    reasons = {item["name_en"]: item["skipped_reason"] for item in out["excluded"]}
    no_trace = next(item for item in out["excluded"] if item["name_raw"] == "phantom thing")
    assert no_trace["skipped_reason"] == "no_trace" and no_trace["name_en"] == "phantom thing"
    assert [v["n_works"] for v in no_trace["name_variants"]] == [0]
    assert reasons[PARTIAL] == "no_counters" and reasons["robotic welding"] == "no_trace"
    assert reasons[MAINSTREAM] == "below_threshold"
    assert [tech_key(item["name_en"]) for item in out["top"]] == [SIGNAL]


def test_same_term_merged_with_doc_ids(result) -> None:
    """Два кандидата шага 4 с одним tech_key термина: одна запись, doc_ids вместе."""
    out, _ = result
    assert out["stats"]["candidates_found"] == 6 and out["stats"]["candidates_named"] == 3
    top = out["top"][0]
    assert tech_key(top["name_raw"]) == SIGNAL
    assert tech_key(top["name_en"]) == SIGNAL and top["n_works"] == 5
    assert max(v["n_works"] for v in top["name_variants"]) == 5
    assert top["n_docs"] == 2 and top["n_sources"] == 1
    # v3 (по умолчанию) ставит arXiv первым на шаге 4: документы 1 и 2 — arXiv
    assert {s["url"] for s in top["sources"][:2]} == {"https://arxiv/3", "https://arxiv/4"}  # дальше — догрузка


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


def test_direct_naming_with_extract_v4(tmp_path) -> None:
    """extract-v4 + direct: name_en = термин шага 4, quote в выходе."""
    items = [{"term_en": SIGNAL, "term_ru": "данные телеуправления", "quote": "doc 3", "doc": 1},
             {"term_en": MAINSTREAM, "term_ru": "гуманоидный робот", "quote": "doc 4", "doc": 2}]
    answer = {"text": json.dumps(items, ensure_ascii=False), "model_uri": "m", "model_version": "t",
              "usage": {}, "elapsed_s": 0, "error": None}
    out, _, _ = run(tmp_path, extract=lambda *args, **kwargs: answer)
    assert out["naming_mode"] == "direct" and out["extract_version"] == "v4"
    assert [item["name_en"] for item in out["top"]] == [SIGNAL]
    assert out["top"][0]["quote"] == "doc 3" and out["top"][0]["name_choice_rule"] == "direct"


def test_extract_v4_gets_topic_in_system_prompt(tmp_path) -> None:
    """По умолчанию шаг 4 идёт с промптом extract-v4: тема запроса в первой строке системного промпта."""
    items = [{"term_en": SIGNAL, "term_ru": "данные телеуправления", "quote": "doc 3", "doc": 1}]
    answer = {"text": json.dumps(items, ensure_ascii=False), "model_uri": "m", "model_version": "t",
              "usage": {}, "elapsed_s": 0, "error": None}
    systems = []
    out, _, _ = run(tmp_path, extract=lambda system, user, **kwargs: systems.append(system) or answer)
    assert systems and all(s.startswith("Тема запроса пользователя: «") for s in systems)
    assert (out["extract_version"], out["extract_model"]) == ("v4", "yandexgpt-5-pro")
    assert [item["name_en"] for item in out["top"]] == [SIGNAL]


def test_cli_has_no_extract_version_flag(tmp_path) -> None:
    """Флаг --extract-version удалён вместе с extract-v2: CLI его не принимает и run_query не передаёт."""
    from pipeline import __main__ as cli
    fake = {"query_id": "q1", "top": [], "excluded": [], "timings": {"total": 0}}
    with patch("pipeline.run_query.run_query", return_value=fake) as runner:
        cli.main(["тема", "--quiet", "--out", str(tmp_path / "a.json")])
        with pytest.raises(SystemExit):
            cli.main(["тема", "--quiet", "--extract-version", "v4"])
    assert runner.call_count == 1 and "extract_version" not in runner.call_args.kwargs


@pytest.mark.parametrize("arg, env, expected", [(None, None, 900.0), (None, "300", 300.0), (120, "300", 120.0),
                                                (0, None, 0.0), (-5, None, 0.0)])
def test_budget_seconds_argument_then_env_then_default(arg, env, expected, monkeypatch) -> None:
    """Бюджет: аргумент важнее env TIME_BUDGET_S, без обоих — 900; отрицательный — ноль."""
    if env is None:
        monkeypatch.delenv("TIME_BUDGET_S", raising=False)
    else:
        monkeypatch.setenv("TIME_BUDGET_S", env)
    assert rq.budget_seconds(arg) == expected


def test_cli_passes_time_budget(tmp_path) -> None:
    """--time-budget доходит до run_query; без флага — None (решает env или значение по умолчанию)."""
    from pipeline import __main__ as cli
    fake = {"query_id": "q1", "top": [], "excluded": [], "timings": {"total": 0}}
    with patch("pipeline.run_query.run_query", return_value=fake) as runner:
        cli.main(["тема", "--quiet", "--out", str(tmp_path / "a.json"), "--time-budget", "0"])
        cli.main(["тема", "--quiet", "--out", str(tmp_path / "b.json")])
    assert [call.kwargs["time_budget"] for call in runner.call_args_list] == [0.0, None]


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
    """Боевой режим — рука R1: extract-v4, yandexgpt-5-pro, без нормализатора."""
    import inspect
    defaults = {name: p.default for name, p in inspect.signature(rq.run_query).parameters.items()}
    assert defaults["extract_model"] == "yandexgpt-5-pro"
    assert (rq.EXTRACT_VERSION, rq.NAMING_MODE) == ("v4", "direct")


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


def test_queue_stats_show_transport_retries() -> None:
    """И3: у arXiv и TechCrunch retries и n429 — из транспорта; у остальных источников — как раньше."""
    from pipeline.fetch import retry_counts

    class Adapter:
        def __init__(self, source, transport):
            self.source, self._transport = source, transport

    class Transport:
        retry_stats = {"export.arxiv.org": {"retries": 2, "n429": 1, "wait_s": 20.0, "disabled": False}}

    class Collector:
        def __init__(self, source):
            self.adapters = [Adapter(source, Transport())]
    assert retry_counts(Collector("arxiv")) == {"retries": 2, "n429": 1}
    assert retry_counts(Collector("techcrunch")) == {} and retry_counts(Collector("openalex")) == {}


class ClockFetch:
    """Заглушка fetch для расписания: кандидат без кэша сдвигает часы на seconds (с номера slow_from — на slow).

    timeout — дедлайн пачки: кандидаты, не успевшие к нему, не возвращаются, часы встают на дедлайн.
    """

    def __init__(self, start: float = 100.0, seconds: float = 10.0, cached: set | None = None,
                 slow_from: int = 10**6, slow: float = 50.0):
        self.now, self.seconds, self.cached, self.batches = start, seconds, cached or set(), []
        self.slow_from, self.slow = slow_from, slow

    def is_cached(self, phrase: str) -> bool:
        return phrase in self.cached

    def prefetch(self, batch, timeout=None) -> list[str]:
        self.batches.append(list(batch))
        deadline, done = (self.now + timeout if timeout is not None else float("inf")), []
        for phrase in batch:
            cost = 0 if phrase in self.cached else self.slow if int(phrase[1:]) >= self.slow_from else self.seconds
            if self.now + cost > deadline:
                self.now = deadline
                break
            self.now += cost
            done.append(phrase)
        return done


def schedule(fetch: ClockFetch, n: int, budget: float, quota=lambda: None):
    warnings = []
    got = rq.schedule_counters(fetch, [f"p{i}" for i in range(n)], lambda: fetch.now, budget, warnings, quota)
    return got, warnings


def test_schedule_stops_when_forecast_exceeds_budget_without_reserve() -> None:
    """10 с на кандидата с 300-й секунды: пачки до 800-й, следующая (800 + 100 > 900 − 90) не берётся."""
    fetch = ClockFetch(start=300.0)
    (scored, stop, info), _ = schedule(fetch, 60, 900)
    assert (len(scored), stop, fetch.now) == (50, "time_budget", 800.0)
    assert scored == [f"p{i}" for i in range(50)] and info["per_candidate_s"] == 10.0


def test_schedule_ceiling_is_60() -> None:
    """Быстро и времени много: оценивается не больше 60, остальные — cap."""
    (scored, stop, _), _ = schedule(ClockFetch(seconds=1.0), 150, 900)
    assert (len(scored), stop) == (rq.MAX_SCORED_HARD, "cap") and rq.MAX_SCORED_HARD == 60


def test_schedule_hard_deadline_inside_batch() -> None:
    """С p30 источник замедлился до 50 с: пачка p30–p39 обрывается на 810-й секунде, успели 8 из 10."""
    fetch = ClockFetch(slow_from=30)
    (scored, stop, _), _ = schedule(fetch, 60, 900)
    assert (len(scored), stop, fetch.now) == (38, "time_budget", 810.0)
    assert len(fetch.batches) == 4


def test_schedule_zero_budget_scores_nothing() -> None:
    fetch = ClockFetch(start=0.0)
    (scored, stop, _), _ = schedule(fetch, 30, 0)
    assert (scored, stop, fetch.batches) == ([], "time_budget", [])


def test_schedule_cache_hits_cost_nothing_up_to_hard_limit() -> None:
    """Все счётчики в кэше: время не растёт, берутся все до 60, остальные — cap."""
    fetch = ClockFetch(cached={f"p{i}" for i in range(250)})
    (scored, stop, _), _ = schedule(fetch, 250, 900)
    assert (len(scored), stop) == (rq.MAX_SCORED_HARD, "cap")


def test_schedule_first_estimate_is_twelve_seconds() -> None:
    """До первых 5 оценённых прогноз 12 с: пачка 10 × 12 = 120 с при остатке 20 до 810 не берётся."""
    (scored, stop, _), _ = schedule(ClockFetch(start=790.0), 20, 900)
    assert (scored, stop) == ([], "time_budget")


def test_schedule_quota_limits_with_margin() -> None:
    """Первая пачка потратила 300 запросов квоты на 10 кандидатов: остаток 700 / (30 × 1.2) = 19 ещё — всего 29."""
    answers = iter([1000, 700])
    (scored, stop, info), warnings = schedule(ClockFetch(seconds=0.1), 150, 900, quota=lambda: next(answers))
    assert (len(scored), stop, info["quota_limit"]) == (29, "cap", 29)
    assert len(warnings) == 1 and "29" in warnings[0]

def test_zero_budget_run_does_not_fail(tmp_path) -> None:
    """--time-budget 0: прогон не падает, оценённых нет, названные — time_budget с текстом причины."""
    out, _, _ = run(tmp_path, time_budget=0)
    assert out["top"] == [] and out["stats"]["candidates_scored"] == 0
    assert out["stats"]["stopped_at"] == "counters" and out["stats"]["time_budget_s"] == 0
    budget = [item for item in out["excluded"] if item["skipped_reason"] == "time_budget"]
    assert budget and all(item["reason_ru"] == "Не хватило времени на проверку" for item in budget)


def test_merge_queues_sums_batches() -> None:
    """Две пачки: суммы запросов и кэша, конец — последней; повторы arXiv копятся в транспорте — последнее."""
    first = [{"source": "arxiv", "candidates": 10, "requests": 8, "cache_hits": 2, "failures": 0, "retries": 1,
              "n429": 0, "started_s": 0.0, "finished_s": 5.0, "duration_s": 5.0},
             {"source": "rospatent", "candidates": 10, "requests": 10, "cache_hits": 0, "failures": 1, "retries": 2,
              "n429": 0, "started_s": 0.0, "finished_s": 4.0, "duration_s": 4.0}]
    second = [{**first[0], "requests": 10, "cache_hits": 0, "retries": 3, "started_s": 5.0, "finished_s": 9.0,
               "duration_s": 4.0}, {**first[1], "started_s": 5.0, "finished_s": 8.0, "duration_s": 3.0}]
    merged = {q["source"]: q for q in fetch_module.merge_queues(fetch_module.merge_queues([], first), second)}
    assert (merged["arxiv"]["requests"], merged["arxiv"]["cache_hits"], merged["arxiv"]["retries"]) == (18, 2, 3)
    assert (merged["arxiv"]["started_s"], merged["arxiv"]["finished_s"], merged["arxiv"]["duration_s"]) == (0.0, 9.0, 9.0)
    assert (merged["rospatent"]["retries"], merged["rospatent"]["failures"]) == (4, 2)


def test_prefetch_deadline_returns_ready_and_does_not_wait(tmp_path) -> None:
    """arXiv завис на второй фразе: prefetch(timeout=0.3) отдаёт первую и не ждёт зависший запрос;
    поздний ответ в сводку очереди не попадает."""
    import threading
    import time as clock
    from collector.models import SearchTerms

    release = threading.Event()
    sources = adapters()
    original = sources[1].count_windows_one_call

    def stuck(search, windows):
        if search.terms[0] == MAINSTREAM:
            release.wait(5)
        return original(search, windows)
    sources[1].count_windows_one_call = stuck
    with patch.object(fetch_module, "COUNTERS_CACHE_DIR", tmp_path):
        fetch = fetch_module.parallel_fetch(sources, Settings(), [])
        begin = clock.monotonic()
        ready = fetch.prefetch([SIGNAL, MAINSTREAM], timeout=0.3)
        waited = clock.monotonic() - begin
        release.set()
        frame = fetch(SearchTerms(terms=[SIGNAL], context_terms=[], query=""))
    assert ready == [SIGNAL] and waited < 2
    assert not frame.empty
    arxiv = next(q for q in fetch.queues if q["source"] == "arxiv")
    assert arxiv["candidates"] == 1


HOLO = "holographic memory chips"


class RoundArxiv(QueryFake):
    """arXiv, у которого подзапрос раунда добора находит три новых документа."""

    def search(self, query, date_from, date_to_exclusive, *, limit, **options):
        if "holographic" not in query:
            return super().search(query, date_from, date_to_exclusive, limit=limit, **options)
        fresh = (date.today() - timedelta(days=5)).isoformat()
        return [doc(source="arxiv", source_type="preprint", published_at=fresh, url=f"https://arxiv/r{i}",
                    title=f"arxiv round doc {i}") for i in range(1, 4)]


def round_extract(system_prompt, user_prompt, **kwargs):
    """Шаг 4: на документы раунда — новый термин (док. 1) и уже известный сигнал (док. 2)."""
    if "round doc" not in user_prompt:
        return fake_extract(system_prompt, user_prompt, **kwargs)
    items = [{"doc": 1, "term_ru": "голографические чипы", "term_en": HOLO, "quote": "doc"},
             {"doc": 2, "term_ru": "данные телеуправления", "term_en": SIGNAL, "quote": "doc"}]
    return {"text": json.dumps(items, ensure_ascii=False), "model_uri": "m", "model_version": "t", "usage": {},
            "elapsed_s": 0, "error": None}


PLAIN_ADAPTERS = adapters  # до подмены в тесте раунда: иначе рекурсия


def round_adapters():
    plain = PLAIN_ADAPTERS()
    return [plain[0], RoundArxiv("arxiv", "preprint", documents=plain[1].documents), plain[2]]


def test_one_round_adds_new_term_and_shifts_doc_ids(tmp_path, monkeypatch) -> None:
    """Раунд 1: новые подзапросы, только новые документы в шаг 4, doc_ids сдвинуты на 9 прежних документов;
    названных по-прежнему меньше 60, но второго раунда нет (MAX_ROUNDS = 1)."""
    monkeypatch.setitem(TRACE, HOLO, 7)
    monkeypatch.setattr(sys.modules[__name__], "adapters", round_adapters)
    out, _, prompts = run(tmp_path, extract=round_extract)
    rounds = out["stats"]["rounds"]
    assert rq.MAX_ROUNDS == 1 and rq.TARGET_NAMED == 60
    assert [r["round"] for r in rounds] == [0, 1] and rounds[1]["subqueries"][:3] == RU_R1
    assert (rounds[1]["documents_new"], rounds[1]["documents_for_candidates"], rounds[1]["named_gain"]) == (3, 3, 1)
    assert sum("round doc" in prompt and "openalex doc" in prompt for prompt in prompts) == 0
    assert out["stats"]["documents_total"] == 12 and out["stats"]["candidates_named"] == 4
    assert any(s["subquery_id"].startswith(f"{out['query_id']}-r1-en-") for s in out["subqueries"])
    holo = next(item for item in out["top"] + out["excluded"] if item["name_en"] == HOLO)
    assert holo["n_docs"] == 1 and holo["n_works"] == 7
    signal = next(item for item in out["top"] if tech_key(item["name_en"]) == SIGNAL)
    assert "https://arxiv/r2" in {s["url"] for s in signal["sources"]}  # док. 2 раунда -> №11


def test_no_round_when_enough_named_or_no_time(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(rq, "TARGET_NAMED", 3)
    out, _, _ = run(tmp_path / "enough")
    assert len(out["stats"]["rounds"]) == 1
    monkeypatch.setattr(rq, "TARGET_NAMED", 60)
    out, _, _ = run(tmp_path / "late", time_budget=0)
    assert len(out["stats"]["rounds"]) == 1


def test_round_without_new_subqueries_is_a_warning(tmp_path, monkeypatch) -> None:
    """Два раунда разрешены: раунд 2 повторяет подзапросы раунда 1 — все отброшены, добор кончается."""
    monkeypatch.setattr(rq, "MAX_ROUNDS", 2)
    out, _, _ = run(tmp_path)
    assert len(out["stats"]["rounds"]) == 2
    assert any(w.startswith("раунд добора 2 не выполнен") for w in out["warnings"])


def test_documents_analyzed_sums_search_counters_and_patents() -> None:
    ranked = [{"name": "a", "score": 0.9, "counters": {"prev6": {"arxiv": 5, "openalex": 10}, "2025": {"techcrunch": 2}}},
              {"name": "b", "score": 0.1, "counters": {"2024": {"openalex": 100}}},
              {"name": "c", "score": None, "counters": {"2024": {"openalex": 1000}}},  # не оценён — не считается
              {"name": "d", "score": 0.5}]
    assert rq.documents_analyzed([{"url": "u1"}, {"url": "u2"}], ranked, {"a": 3, "b": 0}) == 2 + 17 + 100 + 3
    assert rq.documents_analyzed([], [], {}) == 0


def test_documents_analyzed_in_run_covers_search_documents(result) -> None:
    out, _ = result
    assert out["stats"]["documents_analyzed"] >= out["stats"]["documents_total"] > 0


def scored(name: str, score: float | None, docs: int = 0) -> dict:
    return {"name": name, "score": score, "counters": {"2025": {"openalex": docs}} if score is not None else {}}


def test_rank_score_splits_equal_scores_by_evidence() -> None:
    ranked = [scored("a", 0.870737, 1), scored("b", 0.870877, 2), scored("c", 0.870737, 1), scored("d", 0.95, 5),
              scored("e", None)]
    out = rq.with_rank_scores(ranked, {"a": 3})
    by = {item["name"]: item.get("rank_score") for item in out}
    # a: 1 публикация + 3 патента = 4 > b: 2 > c: 1; группа 0.871 из трёх -> +0.010, +0.005, +0.000
    assert (by["a"], by["b"], by["c"], by["d"]) == (0.881, 0.876, 0.871, 0.95)
    assert [item["name"] for item in out] == ["d", "a", "b", "c", "e"]
    assert [item["score"] for item in out[:4]] == [0.95, 0.870737, 0.870877, 0.870737]  # score модели не тронут


def test_rank_score_values_differ_on_screen_up_to_eleven() -> None:
    out = rq.with_rank_scores([scored(f"t{i:02d}", 0.5, i) for i in range(11)], {})
    shown = [f"{item['rank_score']:.3f}" for item in out]
    assert len(set(shown)) == 11 and shown[0] == "0.510" and shown[-1] == "0.500"


def test_rank_score_large_group_still_distinct_values() -> None:
    out = rq.with_rank_scores([scored(f"t{i:02d}", 0.5, i) for i in range(15)], {})
    values = [item["rank_score"] for item in out]
    assert len(set(values)) == 15 and max(values) == 0.51 and min(values) == 0.5


def test_rank_score_tie_in_evidence_breaks_by_name_and_empty_input() -> None:
    out = rq.with_rank_scores([scored("b", 0.7, 1), scored("a", 0.7, 1)], {})
    assert [(item["name"], item["rank_score"]) for item in out] == [("a", 0.71), ("b", 0.7)]
    assert rq.with_rank_scores([], {}) == []


def test_failed_translation_keeps_english_name_and_warns(tmp_path, monkeypatch) -> None:
    """Перевод отклонён на всех попытках, term_ru не годится: в ТОП английское название и предупреждение."""
    monkeypatch.setattr(sys.modules[__name__], "fake_translate",
                        lambda system, user, **kwargs: {"text": "Не могу перевести", "error": None})
    monkeypatch.setattr(translate, "passes", lambda name, term_en: False)
    out, _, _ = run(tmp_path)
    assert out["top"] and all(item["name_ru"] == item["name_en"] for item in out["top"])
    assert out["stats"]["translation"]["на английском"] == len(out["top"]) + sum(
        e["skipped_reason"] in ("below_threshold", "beyond_top") for e in out["excluded"])
    warning = next(w for w in out["warnings"] if w.startswith("Перевод не прошёл проверки"))
    assert all(item["name_en"] in warning for item in out["top"])
