import httpx

from collector.adapters import arxiv, openalex, techcrunch
from collector.adapters.arxiv import ArxivAdapter
from collector.adapters.base import default_trust
from collector.adapters.openalex import OpenAlexAdapter
from collector.adapters.techcrunch import TechCrunchAdapter
from collector.api import DocumentCollector, default_adapters
from collector.constants import WINDOW_BEFORE_END, WINDOW_BEFORE_START
from collector.models import SearchTerms, build_search_terms
from collector.settings import Settings


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
    adapter = OpenAlexAdapter(transport=transport, mailto="dev@example.com", min_interval_s=0)

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


def test_arxiv_maps_preprint_and_counts_its_own_totals() -> None:
    """Бывший test_arxiv_maps_preprint_and_uses_openalex_for_totals.

    Разбор препринта проверяется теми же четырьмя утверждениями. Изменился только
    источник корпусных итогов: с 20.09.2026 arXiv считает их сам, через
    opensearch:totalResults, а не через OpenAlex (фильтр type:article отсёк бы препринты).
    """
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
    totals = httpx.Response(
        200,
        text=(
            '<?xml version="1.0" encoding="UTF-8"?>'
            '<feed xmlns="http://www.w3.org/2005/Atom" '
            'xmlns:opensearch="http://a9.com/-/spec/opensearch/1.1/">'
            "<opensearch:totalResults>500</opensearch:totalResults></feed>"
        ),
    )
    search = httpx.Response(200, text=atom)
    transport = ScriptedTransport([totals, search])
    adapter = ArxivAdapter(transport=transport, min_interval_s=0)

    assert adapter.count_total(WINDOW_BEFORE_START, WINDOW_BEFORE_END) == 500
    assert transport.calls[0][1] == "https://export.arxiv.org/api/query"
    assert "api.openalex.org" not in transport.calls[0][1]
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
    adapter = TechCrunchAdapter(transport=transport, min_interval_s=0)

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
    docs = TechCrunchAdapter(transport=transport, min_interval_s=0).search(
        "gpu", WINDOW_BEFORE_START, WINDOW_BEFORE_END, limit=5
    )
    assert docs[0].source_type == "press_release"
    assert docs[0].trust_level == "low"


def test_techcrunch_totals_unavailable_without_header() -> None:
    transport = ScriptedTransport([httpx.Response(200, json=[])])
    adapter = TechCrunchAdapter(transport=transport, min_interval_s=0)
    assert adapter.count_total(WINDOW_BEFORE_START, WINDOW_BEFORE_END) is None


def test_default_trust_levels() -> None:
    assert default_trust("paper") == "high"
    assert default_trust("preprint") == "medium"
    assert default_trust("blog") == "low"


def test_default_adapters_keep_rate_limits() -> None:
    """Боевой путь передаёт transport, но паузы между запросами остаются."""
    adapters = default_adapters(transport=ScriptedTransport([]))
    intervals = {item.source: item._limiter._min_interval_s for item in adapters}
    assert intervals == {
        "openalex": openalex.MIN_INTERVAL_S,
        "arxiv": arxiv.MIN_INTERVAL_S,
        "techcrunch": techcrunch.MIN_INTERVAL_S,
    }
    assert all(value > 0 for value in intervals.values())


class ErrorTransport:
    """Транспорт, который всегда отвечает ошибкой HTTP с заданным кодом."""

    def __init__(self, status: int) -> None:
        self._status = status
        self.calls = 0

    def get(self, url, *, params=None, headers=None, timeout=None):
        self.calls += 1
        request = httpx.Request("GET", url)
        response = httpx.Response(self._status, request=request)
        raise httpx.HTTPStatusError(f"HTTP {self._status}", request=request, response=response)

    def close(self) -> None:
        return None


def test_techcrunch_400_means_unknown_not_zero() -> None:
    """Ошибка 400 даёт None, а не ноль: ноль обнулил бы знаменатель growth."""
    adapter = TechCrunchAdapter(transport=ErrorTransport(400), min_interval_s=0)
    assert adapter.count_total(WINDOW_BEFORE_START, WINDOW_BEFORE_END) is None
    assert adapter.search("photonic", WINDOW_BEFORE_START, WINDOW_BEFORE_END, limit=5) == []


def test_techcrunch_400_excludes_source_from_growth() -> None:
    """Источник с неизвестным итогом помечается available=False и не попадает в контракт."""
    adapter = TechCrunchAdapter(transport=ErrorTransport(400), min_interval_s=0)
    totals = DocumentCollector(adapters=[adapter]).probe_source_totals()

    assert [row.n_total for row in totals] == [0, 0]
    assert all(not row.available for row in totals)


def test_openalex_sends_api_key_from_settings() -> None:
    """Ключ берётся из настроек и уходит в запрос: без него OpenAlex платный с 24.02.2026."""
    transport = ScriptedTransport(
        [
            httpx.Response(200, json={"results": [], "meta": {"count": 12}}),
            httpx.Response(200, json={"results": [], "meta": {"next_cursor": None}}),
        ]
    )
    adapter = OpenAlexAdapter(
        transport=transport,
        settings=Settings(openalex_api_key="k-123"),
        min_interval_s=0,
    )
    adapter.count_total(WINDOW_BEFORE_START, WINDOW_BEFORE_END)
    adapter.search("photonic", WINDOW_BEFORE_START, WINDOW_BEFORE_END, limit=5)

    assert all(call[2]["params"]["api_key"] == "k-123" for call in transport.calls)


def test_no_api_key_means_no_parameter() -> None:
    """Без ключа параметра в запросе нет: пустая строка сломала бы демо-режим."""
    transport = ScriptedTransport([httpx.Response(200, json={"results": [], "meta": {"count": 1}})])
    adapter = OpenAlexAdapter(transport=transport, settings=Settings(), min_interval_s=0)
    adapter.count_total(WINDOW_BEFORE_START, WINDOW_BEFORE_END)

    assert "api_key" not in transport.calls[0][2]["params"]


def test_arxiv_does_not_sort_by_date() -> None:
    """Потолок в 200 документов плюс сортировка по дате опустошали бы окно before."""
    transport = ScriptedTransport([httpx.Response(200, text="<feed/>")])
    adapter = ArxivAdapter(transport=transport, min_interval_s=0)
    adapter.search("photonic", WINDOW_BEFORE_START, WINDOW_BEFORE_END, limit=10)

    params = transport.calls[0][2]["params"]
    assert "sortBy" not in params
    assert "sortOrder" not in params


def test_page_size_matches_source_limits() -> None:
    """Страница берётся целиком: у OpenAlex каждый вызов платный, у TechCrunch — секунда паузы."""
    assert openalex.PAGE_SIZE == 100
    assert techcrunch.PAGE_SIZE == 100


def _terms() -> "SearchTerms":
    return build_search_terms(
        ["neuromorphic computing", "neuromorphic chip"],
        ["spiking neural network", "brain-inspired computing"],
    )


def test_openalex_counts_with_type_filter() -> None:
    """Счётчик — это meta.count с фильтром типа, документы не выгружаются."""
    transport = ScriptedTransport([httpx.Response(200, json={"meta": {"count": 4463}})])
    adapter = OpenAlexAdapter(transport=transport, min_interval_s=0)

    assert adapter.count_matching(_terms(), WINDOW_BEFORE_START, WINDOW_BEFORE_END) == 4463
    params = transport.calls[0][2]["params"]
    assert params["per_page"] == 1
    assert "type:article" in params["filter"]
    assert params["search"].startswith('("neuromorphic computing" OR')


def test_openalex_count_is_grouped_to_cost_one_credit() -> None:
    """Счётный вызов идёт с group_by: замер 21.09.2026 дал 1 кредит против 10 без него.

    meta.count при этом тот же самый, а работы счётчику не нужны — он их выбрасывает.
    Вместе с group_by уходит select: выбирать поля не из чего.
    """
    transport = ScriptedTransport([httpx.Response(200, json={"meta": {"count": 4463}})])
    adapter = OpenAlexAdapter(transport=transport, min_interval_s=0)

    assert adapter.count_matching(_terms(), WINDOW_BEFORE_START, WINDOW_BEFORE_END) == 4463
    params = transport.calls[0][2]["params"]
    assert params["group_by"] == openalex.COUNT_GROUP_BY
    assert "select" not in params


def test_openalex_document_search_stays_ungrouped() -> None:
    """Выгрузка документов группировку не получает: с ней работы не возвращаются."""
    transport = ScriptedTransport([httpx.Response(200, json={"meta": {}, "results": []})])
    adapter = OpenAlexAdapter(transport=transport, min_interval_s=0)

    adapter.search("neuromorphic chip", WINDOW_BEFORE_START, WINDOW_BEFORE_END, limit=5)
    params = transport.calls[0][2]["params"]
    assert "group_by" not in params
    assert "select" in params


def test_openalex_totals_request_unchanged_by_grouping() -> None:
    """Корпусный итог группировку не получает: иначе сменилась бы его подпись,
    и весь кэш итогов пришлось бы собирать заново."""
    adapter = OpenAlexAdapter(transport=ScriptedTransport([]), min_interval_s=0)
    _, params = adapter.totals_request(WINDOW_BEFORE_START, WINDOW_BEFORE_END)
    assert "group_by" not in params


def test_arxiv_counts_from_opensearch_total() -> None:
    """У arXiv число берётся из opensearch:totalResults, запрос тот же, что у OpenAlex."""
    atom = (
        '<?xml version="1.0" encoding="UTF-8"?>'
        '<feed xmlns="http://www.w3.org/2005/Atom" '
        'xmlns:opensearch="http://a9.com/-/spec/opensearch/1.1/">'
        "<opensearch:totalResults>379</opensearch:totalResults></feed>"
    )
    transport = ScriptedTransport([httpx.Response(200, text=atom)])
    adapter = ArxivAdapter(transport=transport, min_interval_s=0)

    assert adapter.count_matching(_terms(), WINDOW_BEFORE_START, WINDOW_BEFORE_END) == 379
    query = transport.calls[0][2]["params"]["search_query"]
    assert '("neuromorphic computing" OR "neuromorphic chip")' in query
    assert "submittedDate:[202309010000 TO 202408312359]" in query


def test_techcrunch_counts_max_over_terms() -> None:
    """Булев запрос TechCrunch не понимает: термины идут по одному, берётся максимум."""
    transport = ScriptedTransport(
        [
            httpx.Response(200, json=[], headers={"X-WP-Total": "7"}),
            httpx.Response(200, json=[], headers={"X-WP-Total": "3"}),
        ]
    )
    adapter = TechCrunchAdapter(transport=transport, min_interval_s=0)

    assert adapter.count_matching(_terms(), WINDOW_BEFORE_START, WINDOW_BEFORE_END) == 7
    searched = [call[2]["params"]["search"] for call in transport.calls]
    assert searched == ["neuromorphic computing", "neuromorphic chip"]


def test_techcrunch_count_unknown_without_header() -> None:
    """Нет заголовка — нет числа: источник по этому окну в признаки не входит."""
    transport = ScriptedTransport([httpx.Response(200, json=[]), httpx.Response(200, json=[])])
    adapter = TechCrunchAdapter(transport=transport, min_interval_s=0)
    assert adapter.count_matching(_terms(), WINDOW_BEFORE_START, WINDOW_BEFORE_END) is None


def test_unusable_terms_give_no_count() -> None:
    """Запрос не собрался — сети не касаемся вовсе.

    Непригодность переопределена в задаче 3Д: непригодны термины, среди которых нет
    ни одного непустого. Раньше здесь стоял один годный термин, потому что тогда
    второй блок был обязателен; теперь в запрос идёт ровно одно каноническое название
    технологии, и такой вход стал рабочим. Смысл проверки — адаптеры не ходят в сеть
    по несобравшемуся запросу — сохранён дословно.
    """
    broken = build_search_terms(["   ", ""], [])
    oa = ScriptedTransport([])
    ax = ScriptedTransport([])

    assert OpenAlexAdapter(transport=oa, min_interval_s=0).count_matching(
        broken, WINDOW_BEFORE_START, WINDOW_BEFORE_END
    ) is None
    assert ArxivAdapter(transport=ax, min_interval_s=0).count_matching(
        broken, WINDOW_BEFORE_START, WINDOW_BEFORE_END
    ) is None
    assert oa.calls == [] and ax.calls == []


def test_arxiv_recent_search_uses_words_counters_keep_phrase() -> None:
    """Поиск №1 ищет слова по отдельности, счётчик признаков — по-прежнему фразой."""
    atom = (
        '<?xml version="1.0" encoding="UTF-8"?>'
        '<feed xmlns="http://www.w3.org/2005/Atom" '
        'xmlns:opensearch="http://a9.com/-/spec/opensearch/1.1/">'
        "<opensearch:totalResults>5</opensearch:totalResults></feed>"
    )
    transport = ScriptedTransport([httpx.Response(200, text="<feed/>"), httpx.Response(200, text=atom)])
    adapter = ArxivAdapter(transport=transport, min_interval_s=0)

    adapter.search("edge model compression", WINDOW_BEFORE_START, WINDOW_BEFORE_END, limit=5, words=True)
    adapter.count_matching(build_search_terms(["edge model compression"], []),
                           WINDOW_BEFORE_START, WINDOW_BEFORE_END)
    recent = transport.calls[0][2]["params"]["search_query"]
    counter = transport.calls[1][2]["params"]["search_query"]
    assert recent.startswith("all:edge AND all:model AND all:compression AND submittedDate:")
    assert '"edge model compression"' in counter
    assert "all:edge AND" not in counter


def test_arxiv_search_default_is_phrase() -> None:
    """Без words поиск arXiv остаётся фразовым: выгрузка истории кандидата не меняется."""
    transport = ScriptedTransport([httpx.Response(200, text="<feed/>")])
    ArxivAdapter(transport=transport, min_interval_s=0).search(
        "edge model compression", WINDOW_BEFORE_START, WINDOW_BEFORE_END, limit=5)
    assert transport.calls[0][2]["params"]["search_query"].startswith('all:"edge model compression" AND ')


def test_openalex_language_filter_only_when_asked() -> None:
    """language:ru попадает в фильтр только по запросу; обычный поиск его не получает."""
    empty = {"results": [], "meta": {"count": 0, "next_cursor": None}}
    transport = ScriptedTransport([httpx.Response(200, json=empty), httpx.Response(200, json=empty)])
    adapter = OpenAlexAdapter(transport=transport, min_interval_s=0)

    adapter.search("квантовые сенсоры", WINDOW_BEFORE_START, WINDOW_BEFORE_END, limit=5, language="ru")
    adapter.search("quantum sensing", WINDOW_BEFORE_START, WINDOW_BEFORE_END, limit=5)
    assert "language:ru" in transport.calls[0][2]["params"]["filter"]
    assert "language" not in transport.calls[1][2]["params"]["filter"]


def _atom_feed(total: int, published: list[str]) -> str:
    """Лента Atom с totalResults и записями с заданными датами первой версии."""
    entries = "".join(
        f"<entry><id>http://arxiv.org/abs/{i}</id><title>t{i}</title>"
        f"<published>{day}</published><updated>2026-01-01T00:00:00Z</updated></entry>"
        for i, day in enumerate(published))
    return ('<?xml version="1.0" encoding="UTF-8"?>'
            '<feed xmlns="http://www.w3.org/2005/Atom" '
            'xmlns:opensearch="http://a9.com/-/spec/opensearch/1.1/">'
            f"<opensearch:totalResults>{total}</opensearch:totalResults>{entries}</feed>")


def test_arxiv_one_call_splits_by_first_version_date() -> None:
    """Записи раскладываются по окнам по <published>; граница окна полуоткрытая."""
    from collector.constants import COUNTER_WINDOWS

    days = ["2015-03-01T10:00:00Z", "2020-08-31T23:59:59Z", "2020-09-01T00:00:00Z", "2025-12-01T00:00:00Z"]
    transport = ScriptedTransport([httpx.Response(200, text=_atom_feed(4, days))])
    adapter = ArxivAdapter(transport=transport, min_interval_s=0)

    counts = adapter.count_windows_one_call(build_search_terms(["edge model compression"], []),
                                            COUNTER_WINDOWS)
    assert counts == {"prev6": 2, "2020": 1, "2021": 0, "2022": 0, "2023": 0, "2024": 0, "2025": 1}
    params = transport.calls[0][2]["params"]
    assert params["sortBy"] == "submittedDate" and params["sortOrder"] == "ascending"
    assert params["max_results"] == 2000
    assert '"edge model compression"' in params["search_query"]
    assert "submittedDate:[201409010000 TO 202608312359]" in params["search_query"]


def test_arxiv_one_call_falls_back_when_too_many_or_incomplete() -> None:
    """totalResults больше потолка или записей меньше обещанного — None, откат на семь счётчиков."""
    from collector.constants import COUNTER_WINDOWS

    terms = build_search_terms(["edge model compression"], [])
    too_many = ScriptedTransport([httpx.Response(200, text=_atom_feed(2001, ["2021-01-01T00:00:00Z"]))])
    short = ScriptedTransport([httpx.Response(200, text=_atom_feed(3, ["2021-01-01T00:00:00Z"]))])
    assert ArxivAdapter(transport=too_many, min_interval_s=0).count_windows_one_call(terms, COUNTER_WINDOWS) is None
    assert ArxivAdapter(transport=short, min_interval_s=0).count_windows_one_call(terms, COUNTER_WINDOWS) is None


def test_openalex_institutions_count_distinct_groups() -> None:
    """Число организаций — число групп authorships.institutions.id без unknown; работы — meta.count."""
    groups = [{"key": "https://openalex.org/I1", "count": 5}, {"key": "https://openalex.org/I2", "count": 1},
              {"key": "unknown", "count": 3}]
    transport = ScriptedTransport([httpx.Response(200, json={"meta": {"count": 7}, "group_by": groups})])
    adapter = OpenAlexAdapter(transport=transport, min_interval_s=0)

    got = adapter.count_institutions(build_search_terms(["edge model compression"], []),
                                     WINDOW_BEFORE_START, WINDOW_BEFORE_END)
    assert got == {"works": 7, "institutions": 2, "capped": False}
    params = transport.calls[0][2]["params"]
    assert params["group_by"] == "authorships.institutions.id" and params["per_page"] == 200
    assert "type:article" in params["filter"]


def test_openalex_institutions_capped_at_page() -> None:
    """200 групп на странице — число не точное, capped=True."""
    groups = [{"key": f"https://openalex.org/I{i}", "count": 1} for i in range(200)]
    transport = ScriptedTransport([httpx.Response(200, json={"meta": {"count": 999}, "group_by": groups})])
    got = OpenAlexAdapter(transport=transport, min_interval_s=0).count_institutions(
        build_search_terms(["humanoid robot"], []), WINDOW_BEFORE_START, WINDOW_BEFORE_END)
    assert got == {"works": 999, "institutions": 200, "capped": True}


def test_techcrunch_split_windows_strict_bounds_and_field() -> None:
    """Полночь первого дня окна не попадает никуда (after/before строгие); date и date_gmt различаются."""
    from datetime import date as d
    from collector.adapters.techcrunch import split_by_windows

    windows = {"2020": (d(2020, 9, 1), d(2021, 9, 1)), "2021": (d(2021, 9, 1), d(2022, 9, 1))}
    posts = [{"date": "2021-09-01T00:00:00", "date_gmt": "2021-09-01T07:00:00"},
             {"date": "2021-08-31T23:30:00", "date_gmt": "2021-09-01T06:30:00"},
             {"date": "2021-09-01T00:00:01", "date_gmt": "2021-09-01T07:00:01"}]
    assert split_by_windows(posts, windows, "date") == {"2020": 1, "2021": 1}
    assert split_by_windows(posts, windows, "date_gmt") == {"2020": 0, "2021": 3}


def test_techcrunch_one_call_pages_and_params() -> None:
    """Одна фраза, один диапазон, _fields только даты, страницы до X-WP-Total."""
    from collector.constants import COUNTER_WINDOWS

    page = lambda n: [{"date": "2023-05-01T10:00:00", "date_gmt": "2023-05-01T17:00:00"}] * n
    transport = ScriptedTransport([httpx.Response(200, json=page(100), headers={"X-WP-Total": "150"}),
                                   httpx.Response(200, json=page(50), headers={"X-WP-Total": "150"})])
    adapter = TechCrunchAdapter(transport=transport, min_interval_s=0)
    counts = adapter.count_windows_one_call(build_search_terms(["edge model compression"], []), COUNTER_WINDOWS)
    assert counts["2022"] == 150 and sum(counts.values()) == 150
    params = transport.calls[0][2]["params"]
    assert params["search"] == "edge model compression" and params["_fields"] == "date,date_gmt"
    assert params["after"] == "2014-09-01T00:00:00" and params["before"] == "2026-09-01T00:00:00"
    assert params["per_page"] == 100 and params["status"] == "publish" and len(transport.calls) == 2


def test_techcrunch_one_call_falls_back_over_600_or_short() -> None:
    from collector.constants import COUNTER_WINDOWS

    terms = build_search_terms(["edge model compression"], [])
    many = ScriptedTransport([httpx.Response(200, json=[], headers={"X-WP-Total": "601"})])
    short = ScriptedTransport([httpx.Response(200, json=[{"date": "2023-05-01T10:00:00"}] * 3,
                                              headers={"X-WP-Total": "5"})])
    assert TechCrunchAdapter(transport=many, min_interval_s=0).count_windows_one_call(terms, COUNTER_WINDOWS) is None
    assert TechCrunchAdapter(transport=short, min_interval_s=0).count_windows_one_call(terms, COUNTER_WINDOWS) is None


def test_arxiv_calls_https_directly() -> None:
    """Запросы arXiv сразу на https: http давал 301 на каждый вызов (задача Л4.1)."""
    transport = ScriptedTransport([httpx.Response(200, text="<feed/>")])
    ArxivAdapter(transport=transport, min_interval_s=0).search("photonic", WINDOW_BEFORE_START,
                                                                WINDOW_BEFORE_END, limit=5)
    assert transport.calls[0][1] == "https://export.arxiv.org/api/query"
