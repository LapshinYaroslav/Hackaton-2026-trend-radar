"""arXiv adapter. Preprints via Atom API.

Корпусные итоги берутся у самого arXiv: запрос submittedDate без терминов возвращает
opensearch:totalResults по всему корпусу за окно. Раньше итоги брались из OpenAlex
по primary_location.source.id, но с фильтром type:article это сломалось бы — препринты
arXiv под него не подходят, а фильтр обязан стоять на обеих сторонах формулы growth.
Замер 20.09.2026: свои итоги 233 029 и 333 181 против 41 653 и 49 963 через OpenAlex —
OpenAlex индексирует малую часть arXiv.
"""

from __future__ import annotations

import xml.etree.ElementTree as ET
from datetime import date
from typing import Any

import httpx

from collector.adapters.base import default_trust
from collector.constants import inclusive_end
from collector.exceptions import AdapterError
from collector.http import HttpTransport, HttpxTransport, RateLimiter
from collector.models import Document, SearchTerms, parse_utc_date, require_source_type
from collector.settings import Settings

ARXIV_API = "http://export.arxiv.org/api/query"
ATOM = "{http://www.w3.org/2005/Atom}"
OPENSEARCH_TOTAL = "{http://a9.com/-/spec/opensearch/1.1/}totalResults"
ARXIV_NS = "{http://arxiv.org/schemas/atom}"

# arXiv просит не чаще одного запроса в три секунды.
MIN_INTERVAL_S = 3.1

# Задача В: семь счётчиков одним запросом. Читает только оркестратор (pipeline/fetch.py);
# обучающая таблица и гейт считаются по кэшу и флаг не видят. Сверка 23.09.2026 на 156
# технологиях: 6 пар (технология, окно) из 1029 не совпали, |Δscore| не больше 1.5e-4.
ARXIV_ONE_CALL_COUNTS = True
# Потолок одного ответа Atom API. Больше записей — откат на семь счётчиков.
ONE_CALL_MAX_RESULTS = 2000


class ArxivAdapter:
    source = "arxiv"
    source_type = "preprint"
    type_filter = None
    query_variant = "phrase|all"

    def __init__(
        self,
        transport: HttpTransport | None = None,
        settings: Settings | None = None,
        min_interval_s: float = MIN_INTERVAL_S,
    ) -> None:
        self._settings = settings or Settings()
        self._transport = transport or HttpxTransport(self._settings)
        self._limiter = RateLimiter(min_interval_s)

    def search(
        self,
        query: str,
        date_from: date,
        date_to_exclusive: date,
        *,
        limit: int,
        words: bool = False,
    ) -> list[Document]:
        """Документы по запросу. words=True — каждое слово отдельно (поиск №1), иначе фраза."""
        collected: list[Document] = []
        start = 0
        page = min(50, max(1, limit))
        while len(collected) < limit:
            raw = self._query_atom(query, date_from, date_to_exclusive, start=start,
                                   max_results=page, words=words)
            entries = _parse_entries(raw)
            if not entries:
                break
            for entry in entries:
                doc = self._to_document(entry)
                if doc is None:
                    continue
                if not (date_from <= doc.published_at < date_to_exclusive):
                    continue
                collected.append(doc)
                if len(collected) >= limit:
                    break
            start += len(entries)
            if len(entries) < page:
                break
        return collected

    def count_matching(
        self,
        search: SearchTerms,
        date_from: date,
        date_to_exclusive: date,
    ) -> int | None:
        """totalResults по булеву запросу с фильтром submittedDate.

        Тот же текст запроса, что у OpenAlex: фразу без префикса arXiv трактует как all:.
        Проверено 20.09.2026 — обе записи дали одинаковые 437 документов.
        """
        if not search.usable:
            return None
        query = f"({search.query}) AND {_submitted_range(date_from, date_to_exclusive)}"
        try:
            return _parse_total(self._query_atom_raw(query, start=0, max_results=1))
        except AdapterError:
            return None

    def count_windows_one_call(
        self,
        search: SearchTerms,
        windows: dict[str, tuple[date, date]],
    ) -> dict[str, int] | None:
        """Счётчики по окнам одним запросом: записи сортируются по дате подачи и
        раскладываются по окнам локально, по дате первой версии (<published>).

        Не подключено к рабочему пути (флаг ARXIV_ONE_CALL_COUNTS): проверяется задачей В
        против count_matching по семи окнам. None — нужен откат на count_matching:
        запрос не собрался, источник не ответил, записей больше ONE_CALL_MAX_RESULTS или
        в ответе их меньше, чем обещает totalResults.
        """
        if not search.usable or not windows:
            return None
        start = min(bounds[0] for bounds in windows.values())
        end = max(bounds[1] for bounds in windows.values())
        query = f"({search.query}) AND {_submitted_range(start, end)}"
        params = {**_atom_params(query, start=0, max_results=ONE_CALL_MAX_RESULTS),
                  "sortBy": "submittedDate", "sortOrder": "ascending"}
        self._limiter.wait()
        try:
            raw = self._transport.get(ARXIV_API, params=params).text
            entries = _parse_entries(raw)
        except (httpx.HTTPError, AdapterError):
            return None
        total = _parse_total(raw)
        if total is None or total > ONE_CALL_MAX_RESULTS or len(entries) != total:
            return None
        counts = {name: 0 for name in windows}
        for entry in entries:
            day = parse_utc_date(entry["published"])
            for name, (lower, upper) in windows.items():
                if lower <= day < upper:
                    counts[name] += 1
        return counts

    def totals_request(self, date_from: date, date_to_exclusive: date) -> tuple[str, dict[str, Any]]:
        """Весь корпус arXiv за окно: запрос по одной дате подачи, без терминов."""
        return ARXIV_API, _atom_params(
            _submitted_range(date_from, date_to_exclusive), start=0, max_results=1
        )

    def count_total(self, date_from: date, date_to_exclusive: date) -> int | None:
        try:
            raw = self._query_atom_raw(
                _submitted_range(date_from, date_to_exclusive), start=0, max_results=1
            )
        except AdapterError:
            return None
        return _parse_total(raw)

    def _query_atom(
        self,
        query: str,
        date_from: date,
        date_to_exclusive: date,
        *,
        start: int,
        max_results: int,
        words: bool = False,
    ) -> str:
        match = _words_query(query) if words else f"all:{_quote_term(query)}"
        search = f"{match} AND {_submitted_range(date_from, date_to_exclusive)}"
        return self._query_atom_raw(search, start=start, max_results=max_results)

    def _query_atom_raw(self, search: str, *, start: int, max_results: int) -> str:
        """Один запрос к Atom API уже собранной строкой search_query."""
        params = _atom_params(search, start=start, max_results=max_results)
        self._limiter.wait()
        try:
            response = self._transport.get(ARXIV_API, params=params)
            return response.text
        except httpx.HTTPError as exc:
            raise AdapterError(self.source, str(exc)) from exc

    def _to_document(self, entry: dict[str, str]) -> Document | None:
        published = entry.get("published")
        if not published:
            return None
        title = entry.get("title", "").strip()
        url = entry.get("id", "").strip()
        if not title or not url:
            return None
        require_source_type(self.source_type, self.source)
        orgs = [org for org in entry.get("affiliations", "").split("|") if org]
        return Document(
            published_at=parse_utc_date(published),
            source=self.source,
            source_type=self.source_type,
            title=" ".join(title.split()),
            url=url,
            language="en",
            trust_level=default_trust(self.source_type),
            organizations=orgs,
            text=entry.get("summary", "").strip(),
        )


def _atom_params(search: str, *, start: int, max_results: int) -> dict[str, Any]:
    """Параметры запроса к Atom API. Отдельно от отправки: по ним считается подпись.

    Без sortBy: порядок по релевантности. Сортировка по дате вместе с потолком
    в 200 документов оставляла только свежие препринты и опустошала окно before.
    """
    return {"search_query": search, "start": start, "max_results": max_results}


def _submitted_range(date_from: date, date_to_exclusive: date) -> str:
    """Фильтр arXiv по дате подачи. Границы включающие, поэтому конец сдвигается на день назад."""
    date_to = inclusive_end(date_to_exclusive)
    return (
        f"submittedDate:[{date_from.strftime('%Y%m%d')}0000 TO {date_to.strftime('%Y%m%d')}2359]"
    )


def _parse_total(xml_text: str) -> int | None:
    """Число подходящих записей из opensearch:totalResults."""
    try:
        total = ET.fromstring(xml_text).findtext(OPENSEARCH_TOTAL)
    except ET.ParseError:
        return None
    return int(total) if total is not None and total.strip().isdigit() else None


def _quote_term(query: str) -> str:
    cleaned = " ".join(query.split())
    if " " in cleaned:
        return f'"{cleaned}"'
    return cleaned


def _words_query(query: str) -> str:
    """all:w1 AND all:w2: слова в любом месте записи, а не подряд.

    Только для поиска №1. Замер А3 от 23.09.2026: фраза дала 0 документов в 13 парах
    «подзапрос × arXiv» из 20, слова по отдельности — в 5 из 18 (два счётчика не получены,
    429). Счётчики признаков
    (count_matching) по-прежнему считают фразой: на ней обучена модель.
    """
    return " AND ".join(f"all:{word}" for word in query.split())


def _parse_entries(xml_text: str) -> list[dict[str, str]]:
    try:
        root = ET.fromstring(xml_text)
    except ET.ParseError as exc:
        raise AdapterError("arxiv", f"invalid atom xml: {exc}") from exc
    entries: list[dict[str, str]] = []
    for node in root.findall(f"{ATOM}entry"):
        affiliations = [
            (aff.text or "").strip()
            for aff in node.findall(f"{ARXIV_NS}affiliation")
            if (aff.text or "").strip()
        ]
        published = (node.findtext(f"{ATOM}published") or node.findtext(f"{ATOM}updated") or "").strip()
        entries.append(
            {
                "id": (node.findtext(f"{ATOM}id") or "").strip(),
                "title": (node.findtext(f"{ATOM}title") or "").strip(),
                "summary": (node.findtext(f"{ATOM}summary") or "").strip(),
                "published": published,
                "affiliations": "|".join(affiliations),
            }
        )
    return entries
