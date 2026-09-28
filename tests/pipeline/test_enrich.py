"""Догрузка источников ТОП: лимит, бюджет времени, прогресс; след поиска №1 (n_docs, n_sources) не меняется."""
from pipeline.enrich import enrich_top


class Found:
    def __init__(self, documents):
        self.documents = documents

    def to_dict(self):
        return {"documents": self.documents}


class FakeCollector:
    """search_recent по имени: три документа на имя; имя "broken" — ошибка источника."""

    def __init__(self):
        self.calls = []

    def search_recent(self, subqueries, limit):
        name = subqueries[0]["text"]
        self.calls.append(name)
        if name == "broken":
            raise RuntimeError("источник не ответил")
        return Found([{"title": f"{name} {i}", "url": f"https://x/{name}/{i}", "source": "arxiv"} for i in range(3)])


def result(*names):
    top = [{"rank": i + 1, "name_en": name, "n_docs": 2, "n_sources": 1,
            "sources": [{"url": f"https://x/{name}/0", "source": "openalex"}]} for i, name in enumerate(names)]
    return {"query_id": "q1", "top": top, "_documents": [{"url": "https://x/old"}]}


def test_adds_sources_up_to_limit_without_duplicates() -> None:
    data = enrich_top(result("a", "b"), FakeCollector(), source_limit=3)
    first = data["top"][0]
    assert [s["url"] for s in first["sources"]] == ["https://x/a/0", "https://x/a/1", "https://x/a/2"]
    assert (first["n_docs"], first["n_sources"]) == (2, 1)
    assert len(data["_documents"]) == 1 + 6 and data["enrichment"] == "done"


def test_no_time_left_is_partial_without_search() -> None:
    collector = FakeCollector()
    data = enrich_top(result("a", "b"), collector, time_left=lambda: 0.0)
    assert collector.calls == [] and data["enrichment"] == "partial"


def test_budget_runs_out_after_first_item() -> None:
    collector, left = FakeCollector(), iter([5.0, -1.0])
    data = enrich_top(result("a", "b"), collector, time_left=lambda: next(left))
    assert collector.calls == ["a"] and data["enrichment"] == "partial"


def test_progress_events() -> None:
    events = []
    enrich_top(result("a", "b"), FakeCollector(), progress=lambda *event: events.append(event))
    assert events == [("enrich", 0, 2), ("enrich", 1, 2), ("enrich", 2, 2)]


def test_empty_top_is_done() -> None:
    collector = FakeCollector()
    data = enrich_top({"query_id": "q1", "top": [], "_documents": []}, collector)
    assert collector.calls == [] and data["enrichment"] == "done" and data["_documents"] == []


def test_source_error_skips_item() -> None:
    data = enrich_top(result("broken", "b"), FakeCollector())
    assert [s["url"] for s in data["top"][0]["sources"]] == ["https://x/broken/0"]
    assert len(data["top"][1]["sources"]) == 3 and data["enrichment"] == "done"
