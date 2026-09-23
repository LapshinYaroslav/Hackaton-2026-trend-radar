"""OpenAlex works adapter. Papers; corpus totals via meta.count."""

from __future__ import annotations

from datetime import date
from typing import Any

import httpx

from collector.adapters.base import default_trust
from collector.constants import OPENALEX_TYPE_FILTER, inclusive_end
from collector.exceptions import AdapterError
from collector.http import HttpTransport, HttpxTransport, RateLimiter
from collector.models import Document, SearchTerms, require_source_type
from collector.settings import Settings

OPENALEX_WORKS = "https://api.openalex.org/works"

# Вежливый пул OpenAlex: около десяти запросов в секунду.
MIN_INTERVAL_S = 0.12

# Максимум, который отдаёт OpenAlex за один вызов. Каждый вызов с search= стоит денег,
# поэтому страница берётся целиком: 200 документов — это два вызова, а не четыре.
PAGE_SIZE = 100

# Группировка для счётного вызова. Замер 21.09.2026 по заголовку x-ratelimit-credits-used:
# поисковый вызов, отдающий сами работы, стоит 10 кредитов, тот же запрос с group_by —
# 1 кредит, а meta.count у них совпадает до единицы (13903 против 13903 на одном окне,
# 5866015 против 5866015 на другом). Счётчику нужен только meta.count, работы он
# выбрасывает, поэтому платить за их материализацию незачем.
#
# Поле группировки на число не влияет: группировка меняет только то, что лежит рядом с
# meta.count. Взят publication_year как самое дешёвое — в годовом окне групп одна-две.
COUNT_GROUP_BY = "publication_year"

# Разные организации авторов: группировка по точному id организации, страница 200 —
# потолок OpenAlex для group_by. Замер 23.09.2026: groups_count равен числу групп на
# странице (200 у «humanoid robot» при 26 007 работах), а не общему числу.
INSTITUTION_GROUP_BY = "authorships.institutions.id"
INSTITUTION_PAGE = 200

_TYPE_MAP = {
    "article": "paper",
    "journal-article": "paper",
    "review": "paper",
    "letter": "paper",
    "preprint": "preprint",
    "posted-content": "preprint",
    "report": "report",
    "book": "report",
    "book-chapter": "report",
    "standard": "standard",
    "dataset": "report",
    "dissertation": "paper",
}


def reconstruct_abstract(inverted: dict[str, list[int]] | None) -> str:
    if not inverted:
        return ""
    positions: list[tuple[int, str]] = []
    for word, idxs in inverted.items():
        for idx in idxs:
            positions.append((idx, word))
    if not positions:
        return ""
    positions.sort()
    return " ".join(word for _, word in positions)


class OpenAlexAdapter:
    source = "openalex"
    source_type = "paper"
    type_filter = OPENALEX_TYPE_FILTER
    query_variant = "phrase|article"

    def __init__(
        self,
        transport: HttpTransport | None = None,
        settings: Settings | None = None,
        mailto: str | None = None,
        min_interval_s: float = MIN_INTERVAL_S,
        api_key: str | None = None,
    ) -> None:
        self._settings = settings or Settings()
        self._transport = transport or HttpxTransport(self._settings)
        self._mailto = mailto or self._settings.openalex_mailto
        self._api_key = api_key or self._settings.openalex_api_key
        self._limiter = RateLimiter(min_interval_s)

    def search(
        self,
        query: str,
        date_from: date,
        date_to_exclusive: date,
        *,
        limit: int,
        language: str | None = None,
    ) -> list[Document]:
        """Документы по запросу. language — фильтр OpenAlex по языку работы (поиск №1, ru)."""
        collected: list[Document] = []
        cursor = "*"
        per_page = min(PAGE_SIZE, max(1, limit))
        while len(collected) < limit and cursor:
            payload = self._get_works(
                search=query,
                date_from=date_from,
                date_to_exclusive=date_to_exclusive,
                per_page=min(per_page, limit - len(collected)),
                cursor=cursor,
                language=language,
            )
            results = payload.get("results") or []
            for work in results:
                doc = self._to_document(work)
                if doc is not None:
                    collected.append(doc)
                if len(collected) >= limit:
                    break
            cursor = (payload.get("meta") or {}).get("next_cursor") or ""
            if not results:
                break
        return collected

    def count_matching(
        self,
        search: SearchTerms,
        date_from: date,
        date_to_exclusive: date,
    ) -> int | None:
        """meta.count по булеву запросу с фильтром типа. Документы не выгружаются."""
        if not search.usable:
            return None
        payload = self._get_works(
            search=search.query,
            date_from=date_from,
            date_to_exclusive=date_to_exclusive,
            per_page=1,
            cursor=None,
            type_filter=OPENALEX_TYPE_FILTER,
            group_by=COUNT_GROUP_BY,
        )
        count = (payload.get("meta") or {}).get("count")
        return int(count) if count is not None else None

    def count_institutions(
        self,
        search: SearchTerms,
        date_from: date,
        date_to_exclusive: date,
    ) -> dict[str, Any] | None:
        """Работы и разные организации авторов по фразе одним дешёвым вызовом (group_by, 1 кредит).

        Группа authorships.institutions.id — одна организация, её count — число работ с ней,
        поэтому число групп — число разных организаций, а не пар «работа–организация».
        Сверено 23.09.2026 полной выгрузкой: 32 группы = 32 разные организации на 14 работах.
        lineage не подходит: добавляет родительские организации (39 групп на тех же работах).
        Групп больше INSTITUTION_PAGE одним вызовом не узнать: тогда capped=True, число — нижняя граница.
        meta.count тот же, что у count_matching: фильтры одни, поле группировки на него не влияет.
        """
        if not search.usable:
            return None
        payload = self._get_works(search=search.query, date_from=date_from, date_to_exclusive=date_to_exclusive,
                                  per_page=INSTITUTION_PAGE, cursor=None, type_filter=OPENALEX_TYPE_FILTER,
                                  group_by=INSTITUTION_GROUP_BY)
        works = (payload.get("meta") or {}).get("count")
        groups = [g for g in payload.get("group_by") or [] if "unknown" not in str(g.get("key")).lower()]
        return {"works": int(works) if works is not None else None, "institutions": len(groups),
                "capped": len(payload.get("group_by") or []) >= INSTITUTION_PAGE}

    def totals_request(self, date_from: date, date_to_exclusive: date) -> tuple[str, dict[str, Any]]:
        """Запрос корпусного итога: тот же фильтр типа, что и у счётчика технологии.

        Фильтр обязан стоять на обеих сторонах формулы growth. Без него числитель
        и знаменатель считались бы по разным множествам (pipeline.md 0.2).
        """
        return OPENALEX_WORKS, self._works_params(
            search=None,
            date_from=date_from,
            date_to_exclusive=date_to_exclusive,
            per_page=1,
            cursor=None,
            type_filter=OPENALEX_TYPE_FILTER,
        )

    def count_total(self, date_from: date, date_to_exclusive: date) -> int | None:
        url, params = self.totals_request(date_from, date_to_exclusive)
        payload = self._get(url, params)
        count = (payload.get("meta") or {}).get("count")
        return int(count) if count is not None else None

    def _get_works(
        self,
        *,
        search: str | None,
        date_from: date,
        date_to_exclusive: date,
        per_page: int,
        cursor: str | None,
        type_filter: str | None = None,
        group_by: str | None = None,
        language: str | None = None,
    ) -> dict[str, Any]:
        params = self._works_params(
            search=search,
            date_from=date_from,
            date_to_exclusive=date_to_exclusive,
            per_page=per_page,
            cursor=cursor,
            type_filter=type_filter,
            group_by=group_by,
            language=language,
        )
        return self._get(OPENALEX_WORKS, params)

    def _works_params(
        self,
        *,
        search: str | None,
        date_from: date,
        date_to_exclusive: date,
        per_page: int,
        cursor: str | None,
        type_filter: str | None = None,
        group_by: str | None = None,
        language: str | None = None,
    ) -> dict[str, Any]:
        """Параметры запроса к works. Отдельно от отправки: по ним считается подпись.

        language добавляет фильтр только когда передан: у счётчиков и корпусных итогов
        его нет, поэтому их параметры и подписи не меняются.
        """
        date_to = inclusive_end(date_to_exclusive)
        filters = [
            f"from_publication_date:{date_from.isoformat()}",
            f"to_publication_date:{date_to.isoformat()}",
        ]
        if type_filter:
            filters.append(f"type:{type_filter}")
        if language:
            filters.append(f"language:{language}")
        params: dict[str, Any] = {
            "filter": ",".join(filters),
            "per_page": per_page,
        }
        if group_by:
            # Счётный вызов: работы не возвращаются, поэтому select не нужен, а вместе
            # с group_by он и не имеет смысла — выбирать поля не из чего.
            params["group_by"] = group_by
        else:
            params["select"] = (
                "id,doi,title,display_name,publication_date,type,primary_location,"
                "abstract_inverted_index,language,authorships"
            )
        if search:
            params["search"] = search
        if cursor:
            params["cursor"] = cursor
        if self._mailto:
            params["mailto"] = self._mailto
        if self._api_key:
            params["api_key"] = self._api_key
        return params

    def _get(self, url: str, params: dict[str, Any]) -> dict[str, Any]:
        self._limiter.wait()
        try:
            response = self._transport.get(url, params=params)
            return response.json()
        except (httpx.HTTPError, ValueError) as exc:
            raise AdapterError(self.source, str(exc)) from exc

    def _to_document(self, work: dict[str, Any]) -> Document | None:
        published = work.get("publication_date")
        if not published:
            return None
        title = (work.get("display_name") or work.get("title") or "").strip()
        if not title:
            return None
        try:
            url = _work_url(work)
        except ValueError:
            return None
        raw_type = str(work.get("type") or "article")
        source_type = _TYPE_MAP.get(raw_type, "paper")
        require_source_type(source_type, self.source)
        orgs: list[str] = []
        for authorship in work.get("authorships") or []:
            for inst in authorship.get("institutions") or []:
                name = (inst.get("display_name") or "").strip()
                if name and name not in orgs:
                    orgs.append(name)
        language = (work.get("language") or "en").split("-")[0] or "en"
        return Document(
            published_at=published,
            source=self.source,
            source_type=source_type,
            title=title,
            url=url,
            language=language,
            trust_level=default_trust(source_type),
            organizations=orgs,
            text=reconstruct_abstract(work.get("abstract_inverted_index")),
        )


def _work_url(work: dict[str, Any]) -> str:
    location = work.get("primary_location") or {}
    landing = (location.get("landing_page_url") or "").strip()
    if landing:
        return landing
    doi = (work.get("doi") or "").strip()
    if doi:
        return doi if doi.startswith("http") else f"https://doi.org/{doi.removeprefix('https://doi.org/')}"
    openalex_id = str(work.get("id") or "").strip()
    if openalex_id:
        return openalex_id
    raise ValueError("OpenAlex work has no url")
