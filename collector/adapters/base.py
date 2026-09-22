"""Source adapter contract.

Every source is probed for corpus totals on both growth windows *before* collection.
If count_total returns None, the source is excluded from growth and kept for other features.
"""

from __future__ import annotations

import hashlib
import json
from datetime import date
from typing import Any, Protocol, runtime_checkable

from collector.models import Document, SearchTerms

HIGH_TRUST_TYPES = {"paper", "patent", "standard"}
MEDIUM_TRUST_TYPES = {"preprint", "report", "news", "product"}


@runtime_checkable
class SourceAdapter(Protocol):
    source: str
    source_type: str
    # Фильтр типа записи, с которым считаются и счётчик технологии, и корпусный итог.
    # Одно объявление на источник: числитель и знаменатель growth обязаны совпадать.
    type_filter: str | None
    # Семантика запроса для ключа кэша: "phrase|article", "words|all" и т. п.
    # Объявляется рядом с type_filter, потому что фильтр типа в неё входит.
    query_variant: str

    def search(
        self,
        query: str,
        date_from: date,
        date_to_exclusive: date,
        *,
        limit: int,
    ) -> list[Document]:
        """Documents matching `query` with date_from <= published_at < date_to_exclusive."""
        ...

    def count_matching(
        self,
        search: SearchTerms,
        date_from: date,
        date_to_exclusive: date,
    ) -> int | None:
        """Сколько записей источника подходит под запрос за окно, без выгрузки документов.

        Это поиск №2 версии 0.2: признаки считаются по этим числам. None означает
        «источник не ответил» — такой источник в признаки по этому окну не входит.
        """
        ...

    def totals_request(self, date_from: date, date_to_exclusive: date) -> tuple[str, dict[str, Any]]:
        """URL и параметры запроса корпусного итога. По ним же считается подпись способа."""
        ...

    def count_total(self, date_from: date, date_to_exclusive: date) -> int | None:
        """Corpus size of the whole source in the window, independent of technology.

        Return None if the source cannot answer this. Callers must probe both windows
        before collection starts (pipeline.md).
        """
        ...


# Параметры, которые не должны попадать в подпись: секреты и вежливые пометки.
SIGNATURE_SKIP_PARAMS = frozenset({"api_key", "mailto"})
# Окно, на котором считается подпись. Фиксированное, иначе подпись зависела бы от дат.
SIGNATURE_WINDOW = (date(2023, 9, 1), date(2024, 9, 1))


def totals_signature(adapter: SourceAdapter) -> str:
    """Отпечаток способа подсчёта корпусного итога: URL и параметры запроса.

    Считается из того же запроса, который адаптер реально отправляет, поэтому меняется
    сам — от смены эндпоинта, фильтра, любого параметра. Помнить про версию вручную
    не нужно: закэшированный итог, посчитанный прежним способом, перестанет подходить
    в тот же момент, когда изменится запрос.
    """
    url, params = adapter.totals_request(*SIGNATURE_WINDOW)
    payload = json.dumps(
        {
            "url": url,
            "params": {
                str(key): str(value)
                for key, value in sorted(params.items())
                if key not in SIGNATURE_SKIP_PARAMS
            },
        },
        ensure_ascii=False,
        sort_keys=True,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:12]


def default_trust(source_type: str) -> str:
    if source_type in HIGH_TRUST_TYPES:
        return "high"
    if source_type in MEDIUM_TRUST_TYPES:
        return "medium"
    return "low"
