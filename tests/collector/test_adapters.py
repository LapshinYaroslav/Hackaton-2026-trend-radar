import httpx

from collector.adapters.arxiv import ArxivAdapter
from collector.adapters.base import default_trust
from collector.adapters.openalex import OpenAlexAdapter
from collector.adapters.techcrunch import TechCrunchAdapter
from collector.constants import WINDOW_BEFORE_END, WINDOW_BEFORE_START


class ScriptedTransport:
    def __init__(self, responses: list[httpx.Response]) -> None:
        self._responses = list(responses)
        self.calls: list[tuple[str, str, dict]] = []

    def get(self, url, *, params=None, headers=None, timeout=None):
        self.calls.append(("GET", url, {"params": params, "headers": headers}))
        return self._next(url)

    def close(self) -> None:
        return None

    def _next(self, url: str) -> httpx.Response:
        response = self._responses.pop(0)
        if getattr(response, "_request", None) is None:
            response._request = httpx.Request("GET", url)
        return response


def test_openalex_maps_paper_dates_and_totals() -> None:
    work = {
        "id": "https://openalex.org/W1",
        "display_name": "Photonic inference",
        "publication_date": "2024-03-20",
        "type": "article",
        "doi": "https://doi.org/10.1/xyz",
        "language": "en",
        "abstract_inverted_index": {"hello": [0], "world": [1]},
        "authorships": [{"institutions": [{"display_name": "MIT"}]}],
        "primary_location": {"landing_page_url": "https://openalex.example/W1"},
    }
    search_resp = httpx.Response(
        200,
        json={"results": [work], "meta": {"count": 1, "next_cursor": None}},
    )
    count_resp = httpx.Response(200, json={"results": [], "meta": {"count": 1000}})
    transport = ScriptedTransport([count_resp, search_resp])
    adapter = OpenAlexAdapter(transport=transport, mailto="dev@example.com")

    total = adapter.count_total(WINDOW_BEFORE_START, WINDOW_BEFORE_END)
    docs = adapter.search("photonic", WINDOW_BEFORE_START, WINDOW_BEFORE_END, limit=10)

    assert total == 1000
    assert len(docs) == 1
    assert docs[0].source == "openalex"
    assert docs[0].source_type == "paper"
    assert docs[0].published_at.isoformat() == "2024-03-20"
    assert docs[0].organizations == ["MIT"]
    assert docs[0].text == "hello world"
    assert docs[0].trust_level == "high"


def test_arxiv_maps_preprint_and_uses_openalex_for_totals() -> None:
    atom = """<?xml version="1.0" encoding="UTF-8"?>
    <feed xmlns="http://www.w3.org/2005/Atom" xmlns:arxiv="http://arxiv.org/schemas/atom">
      <entry>
        <id>http://arxiv.org/abs/2401.00001v1</id>
        <published>2024-01-02T12:00:00Z</published>
        <title>Optical accelerators</title>
        <summary>A preprint.</summary>
        <arxiv:affiliation>MIT</arxiv:affiliation>
      </entry>
    </feed>
    """
    totals = httpx.Response(200, json={"meta": {"count": 500}})
    search = httpx.Response(200, text=atom)
    transport = ScriptedTransport([totals, search])
    adapter = ArxivAdapter(transport=transport)

    assert adapter.count_total(WINDOW_BEFORE_START, WINDOW_BEFORE_END) == 500
    docs = adapter.search("optical", WINDOW_BEFORE_START, WINDOW_BEFORE_END, limit=10)
    assert docs[0].source == "arxiv"
    assert docs[0].source_type == "preprint"
    assert docs[0].published_at.isoformat() == "2024-01-02"
    assert docs[0].organizations == ["MIT"]


def test_techcrunch_maps_news_and_totals_from_header() -> None:
    post = {
        "id": 1,
        "date_gmt": "2024-04-01T10:00:00",
        "link": "https://techcrunch.com/2024/04/01/photonic/",
        "title": {"rendered": "Photonic chip startup"},
        "content": {"rendered": "<p>News body</p>"},
        "class_list": ["category-news"],
    }
    totals = httpx.Response(200, json=[{}], headers={"X-WP-Total": "777"})
    search = httpx.Response(200, json=[post], headers={"X-WP-Total": "1"})
    transport = ScriptedTransport([totals, search])
    adapter = TechCrunchAdapter(transport=transport)

    assert adapter.count_total(WINDOW_BEFORE_START, WINDOW_BEFORE_END) == 777
    docs = adapter.search("photonic", WINDOW_BEFORE_START, WINDOW_BEFORE_END, limit=10)
    assert docs[0].source == "techcrunch"
    assert docs[0].source_type == "news"
    assert docs[0].published_at.isoformat() == "2024-04-01"
    assert docs[0].text == "News body"


def test_techcrunch_press_release_type() -> None:
    post = {
        "id": 2,
        "date_gmt": "2024-04-02T10:00:00",
        "link": "https://techcrunch.com/2024/04/02/pr/",
        "title": {"rendered": "Press release: shipping GPUs"},
        "content": {"rendered": "PR"},
        "class_list": ["press-release"],
    }
    transport = ScriptedTransport(
        [httpx.Response(200, json=[post], headers={"X-WP-Total": "1"})]
    )
    docs = TechCrunchAdapter(transport=transport).search(
        "gpu", WINDOW_BEFORE_START, WINDOW_BEFORE_END, limit=5
    )
    assert docs[0].source_type == "press_release"
    assert docs[0].trust_level == "low"


def test_techcrunch_totals_unavailable_without_header() -> None:
    transport = ScriptedTransport([httpx.Response(200, json=[])])
    adapter = TechCrunchAdapter(transport=transport)
    assert adapter.count_total(WINDOW_BEFORE_START, WINDOW_BEFORE_END) is None


def test_default_trust_levels() -> None:
    assert default_trust("paper") == "high"
    assert default_trust("preprint") == "medium"
    assert default_trust("blog") == "low"
