"""TechCrunch adapter via WordPress REST API. News / occasional press releases.

Corpus totals: X-WP-Total on a dated /posts request without a search query.
If the header is missing, count_total returns None and TechCrunch is excluded from growth.
"""

from __future__ import annotations

import re
from datetime import date, datetime, timezone
from typing import Any

import httpx

from collector.adapters.base import default_trust
from collector.exceptions import AdapterError
from collector.http import HttpTransport, HttpxTransport, RateLimiter
from collector.models import Document, SearchTerms, require_source_type
from collector.settings import Settings

POSTS_URL = "https://techcrunch.com/wp-json/wp/v2/posts"
TAG_RE = re.compile(r"<[^>]+>")

# WordPress REST у TechCrunch без ключа: держим один запрос в секунду.
MIN_INTERVAL_S = 1.0

# Максимум per_page у WordPress REST. При 20 потолок в 200 документов стоил
# десять запросов по секунде на каждый термин.
PAGE_SIZE = 100


class TechCrunchAdapter:
    source = "techcrunch"
    source_type = "news"
    type_filter = None
    query_variant = "words|all"

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
    ) -> list[Document]:
        collected: list[Document] = []
        page = 1
        per_page = min(PAGE_SIZE, max(1, limit))
        while len(collected) < limit:
            posts, total = self._fetch_posts(
                search=query,
                date_from=date_from,
                date_to_exclusive=date_to_exclusive,
                page=page,
                per_page=per_page,
            )
            if not posts:
                break
            for post in posts:
                doc = self._to_document(post)
                if doc is not None:
                    collected.append(doc)
                if len(collected) >= limit:
                    break
            if total is not None and page * per_page >= total:
                break
            if total is None and len(posts) < per_page:
                break
            page += 1
        return collected

    def count_matching(
        self,
        search: SearchTerms,
        date_from: date,
        date_to_exclusive: date,
    ) -> int | None:
        """Максимум X-WP-Total по каждому термину в отдельности.

        WordPress ищет по словам с условием AND, кавычки игнорирует, а слово OR ищет
        буквально: булев запрос из query.build_query даёт здесь ноль (проверено 20.09.2026).
        Поэтому термины идут по одному. Берётся максимум, а не сумма: документ, попавший
        под два термина, посчитался бы дважды, а пересечение без выгрузки не проверить.
        Это честная нижняя оценка, одинаковая для обоих классов технологий.
        Контекстные термины не используются: сами по себе они описывают область, а не
        технологию, и добавили бы посторонние статьи.

        Про годовые окна. Оговорка о неаддитивности здесь больше не нужна: после
        задачи 3В у технологии ровно один термин, а максимум по одному термину равен
        ему самому. Сумма годовых значений поэтому точно равна значению за период,
        и рабочие окна складываются из годов без потерь.

        Код остался максимумом по списку, а не значением единственного термина:
        так поиск №1 и ручные прогоны с несколькими терминами продолжают работать.
        Если термины когда-нибудь вернутся, вернётся и оговорка — сумма годовых
        максимумов не меньше максимума за период и не больше настоящего объединения,
        то есть остаётся честной нижней оценкой, только более плотной.
        """
        best: int | None = None
        for term in search.terms:
            _, total = self._fetch_posts(
                search=term,
                date_from=date_from,
                date_to_exclusive=date_to_exclusive,
                page=1,
                per_page=1,
            )
            if total is not None:
                best = total if best is None else max(best, total)
        return best

    def totals_request(self, date_from: date, date_to_exclusive: date) -> tuple[str, dict[str, Any]]:
        """Запрос корпусного итога: та же выдача без поискового слова, нужен заголовок."""
        return POSTS_URL, _posts_params(
            search=None,
            date_from=date_from,
            date_to_exclusive=date_to_exclusive,
            page=1,
            per_page=1,
        )

    def count_total(self, date_from: date, date_to_exclusive: date) -> int | None:
        _, total = self._fetch_posts(
            search=None,
            date_from=date_from,
            date_to_exclusive=date_to_exclusive,
            page=1,
            per_page=1,
        )
        return total

    def _fetch_posts(
        self,
        *,
        search: str | None,
        date_from: date,
        date_to_exclusive: date,
        page: int,
        per_page: int,
    ) -> tuple[list[dict[str, Any]], int | None]:
        params = _posts_params(
            search=search,
            date_from=date_from,
            date_to_exclusive=date_to_exclusive,
            page=page,
            per_page=per_page,
        )
        self._limiter.wait()
        try:
            response = self._transport.get(POSTS_URL, params=params)
        except httpx.HTTPStatusError as exc:
            if exc.response is not None and exc.response.status_code == 400:
                # 400 значит «не знаю», а не «ноль документов»: ноль в корпусном итоге
                # обнулил бы знаменатель growth. Итог неизвестен -> None -> available=False.
                return [], None
            raise AdapterError(self.source, str(exc)) from exc
        except httpx.HTTPError as exc:
            raise AdapterError(self.source, str(exc)) from exc
        try:
            payload = response.json()
        except ValueError as exc:
            raise AdapterError(self.source, f"invalid json: {exc}") from exc
        if not isinstance(payload, list):
            payload = []
        total_header = response.headers.get("X-WP-Total")
        if total_header is None or not str(total_header).isdigit():
            return payload, None
        return payload, int(total_header)

    def _to_document(self, post: dict[str, Any]) -> Document | None:
        published = post.get("date_gmt") or post.get("date")
        if not published:
            return None
        title = _rendered(post.get("title"))
        url = (post.get("link") or "").strip()
        if not title or not url:
            return None
        source_type = _classify(post)
        require_source_type(source_type, self.source)
        text = _strip_html(_rendered(post.get("content")) or _rendered(post.get("excerpt")))
        return Document(
            published_at=published,
            source=self.source,
            source_type=source_type,
            title=title,
            url=url,
            language="en",
            trust_level=default_trust(source_type),
            organizations=[],
            text=text,
        )


def _posts_params(
    *,
    search: str | None,
    date_from: date,
    date_to_exclusive: date,
    page: int,
    per_page: int,
) -> dict[str, Any]:
    """Параметры запроса к WordPress REST. Отдельно от отправки: по ним считается подпись."""
    params: dict[str, Any] = {
        "after": _start_of_day(date_from),
        "before": _start_of_day(date_to_exclusive),
        "page": page,
        "per_page": per_page,
        "status": "publish",
        "_fields": "id,date_gmt,modified_gmt,link,title,content,excerpt,categories,class_list",
    }
    if search:
        params["search"] = search
    return params


def _start_of_day(value: date) -> str:
    return datetime(value.year, value.month, value.day, tzinfo=timezone.utc).strftime(
        "%Y-%m-%dT%H:%M:%S"
    )


def _rendered(block: Any) -> str:
    if isinstance(block, dict):
        return (block.get("rendered") or "").strip()
    if isinstance(block, str):
        return block.strip()
    return ""


def _strip_html(raw: str) -> str:
    return " ".join(TAG_RE.sub(" ", raw).split())


def _classify(post: dict[str, Any]) -> str:
    classes = " ".join(post.get("class_list") or []).casefold()
    title = _rendered(post.get("title")).casefold()
    if "press-release" in classes or title.startswith("press release"):
        return "press_release"
    return "news"
