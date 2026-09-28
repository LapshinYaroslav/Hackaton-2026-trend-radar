"""Догрузка источников ТОП: очереди по источникам параллельно, лимит, бюджет, прогресс; n_docs и n_sources не меняются."""
import threading
from datetime import date

from collector.api import DocumentCollector
from collector.models import Document
from pipeline.enrich import enrich_top

SOURCE_TYPE = {"openalex": "paper", "arxiv": "preprint", "techcrunch": "news"}


class Source:
    """Источник поиска №1: три документа на имя; имя "broken" — ошибка; barrier — ждать другой источник."""

    def __init__(self, source: str, barrier: threading.Barrier | None = None):
        self.source, self.barrier, self.calls = source, barrier, []
        self.source_type = SOURCE_TYPE[source]

    def search(self, query, date_from, date_to_exclusive, *, limit, **options):
        self.calls.append(query)
        if self.barrier is not None:
            self.barrier.wait(timeout=2)  # второй источник должен быть в поиске одновременно
        if query == "broken":
            raise RuntimeError("источник не ответил")
        return [Document(published_at=date(2026, 8, 1), source=self.source, source_type=self.source_type,
                         title=f"{query} {i}", url=f"https://{self.source}/{query}/{i}") for i in range(3)][:limit]


def collector(*sources) -> DocumentCollector:
    return DocumentCollector(adapters=list(sources))


def result(*names):
    top = [{"rank": i + 1, "name_en": name, "n_docs": 2, "n_sources": 1,
            "sources": [{"url": f"https://openalex/{name}/0", "source": "openalex"}]} for i, name in enumerate(names)]
    return {"query_id": "q1", "top": top, "_documents": [{"url": "https://old"}]}


def test_adds_sources_up_to_limit_without_duplicates() -> None:
    data = enrich_top(result("a", "b"), collector(Source("openalex"), Source("arxiv")), source_limit=4)
    first = data["top"][0]
    assert [s["url"] for s in first["sources"]] == ["https://openalex/a/0", "https://openalex/a/1",
                                                    "https://openalex/a/2", "https://arxiv/a/0"]
    assert (first["n_docs"], first["n_sources"]) == (2, 1)
    assert len(data["_documents"]) == 1 + 2 * 2 * 3 and data["enrichment"] == "done"


def test_sources_run_in_parallel() -> None:
    barrier = threading.Barrier(2)
    openalex, arxiv = Source("openalex", barrier), Source("arxiv", barrier)
    data = enrich_top(result("a"), collector(openalex, arxiv))
    assert not barrier.broken  # последовательный обход сломал бы барьер по таймауту
    assert {s["url"] for s in data["top"][0]["sources"]} >= {"https://openalex/a/1", "https://arxiv/a/0"}


def test_each_source_walks_items_in_order() -> None:
    openalex, arxiv = Source("openalex"), Source("arxiv")
    enrich_top(result("a", "b", "c"), collector(openalex, arxiv))
    assert openalex.calls == arxiv.calls == ["a", "b", "c"]


def test_no_time_left_is_partial_without_search() -> None:
    openalex = Source("openalex")
    data = enrich_top(result("a", "b"), collector(openalex), time_left=lambda: 0.0)
    assert openalex.calls == [] and data["enrichment"] == "partial"


def test_budget_runs_out_after_first_item() -> None:
    openalex, left = Source("openalex"), iter([5.0, -1.0])
    data = enrich_top(result("a", "b"), collector(openalex), time_left=lambda: next(left))
    assert openalex.calls == ["a"] and data["enrichment"] == "partial"


def test_progress_counts_item_source_pairs() -> None:
    events = []
    enrich_top(result("a", "b"), collector(Source("openalex"), Source("arxiv")),
               progress=lambda *event: events.append(event))
    assert events[0] == ("enrich", 0, 4) and events[-1] == ("enrich", 4, 4)
    assert [done for _, done, _ in events] == [0, 1, 2, 3, 4]


def test_empty_top_is_done() -> None:
    openalex = Source("openalex")
    data = enrich_top({"query_id": "q1", "top": [], "_documents": []}, collector(openalex))
    assert openalex.calls == [] and data["enrichment"] == "done" and data["_documents"] == []


def test_source_error_skips_item_only() -> None:
    data = enrich_top(result("broken", "b"), collector(Source("openalex")))
    assert [s["url"] for s in data["top"][0]["sources"]] == ["https://openalex/broken/0"]
    assert len(data["top"][1]["sources"]) == 3 and data["enrichment"] == "done"
