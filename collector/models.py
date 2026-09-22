"""Public data contracts. Changes only via PR agreed with Danya and Yaroslav (pipeline.md)."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from datetime import date, datetime, timezone
from typing import Any

from collector.constants import (
    ALLOWED_COUNTER_WINDOWS,
    ALLOWED_SOURCE_TYPES,
    ALLOWED_TRUST_LEVELS,
    ALLOWED_WINDOWS,
    COLLECTION_START,
    CUTOFF_DATE,
)
from collector.exceptions import InvalidSourceTypeError
from collector.query import build_query, clean_terms


def parse_utc_date(value: date | datetime | str) -> date:
    """Normalize to a UTC calendar day. Naive datetimes are treated as UTC."""
    if isinstance(value, datetime):
        if value.tzinfo is not None:
            return value.astimezone(timezone.utc).date()
        return value.date()
    if isinstance(value, date):
        return value
    text = str(value).strip()
    if not text:
        raise ValueError("empty date")
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    if "T" in text:
        parsed = datetime.fromisoformat(text)
        if parsed.tzinfo is not None:
            return parsed.astimezone(timezone.utc).date()
        return parsed.date()
    return date.fromisoformat(text[:10])


def require_source_type(value: str, source: str | None = None) -> str:
    if value not in ALLOWED_SOURCE_TYPES:
        raise InvalidSourceTypeError(value, source)
    return value


@dataclass(frozen=True)
class Document:
    """One collected item. This is the only document shape both modes emit."""

    published_at: date
    source: str
    source_type: str
    title: str
    url: str
    language: str = "en"
    trust_level: str = "medium"
    organizations: list[str] = field(default_factory=list)
    text: str = ""

    def __post_init__(self) -> None:
        object.__setattr__(self, "published_at", parse_utc_date(self.published_at))
        source = self.source.strip()
        title = self.title.strip()
        url = self.url.strip()
        if not source or not title or not url:
            raise ValueError("source, title and url must be non-empty")
        object.__setattr__(self, "source", source)
        object.__setattr__(self, "title", title)
        object.__setattr__(self, "url", url)
        object.__setattr__(self, "source_type", require_source_type(str(self.source_type), source))
        trust = str(self.trust_level).strip()
        if trust not in ALLOWED_TRUST_LEVELS:
            raise ValueError(f"invalid trust_level={trust!r}")
        object.__setattr__(self, "trust_level", trust)
        language = (self.language or "en").strip() or "en"
        object.__setattr__(self, "language", language.split("-")[0])
        orgs = [str(item).strip() for item in (self.organizations or []) if str(item).strip()]
        object.__setattr__(self, "organizations", orgs)
        object.__setattr__(self, "text", self.text or "")

    def in_collection_window(self) -> bool:
        return COLLECTION_START <= self.published_at < CUTOFF_DATE

    def to_dict(self) -> dict[str, Any]:
        return {
            "published_at": self.published_at.isoformat(),
            "source": self.source,
            "source_type": self.source_type,
            "title": self.title,
            "url": self.url,
            "language": self.language,
            "trust_level": self.trust_level,
            "organizations": list(self.organizations),
            "text": self.text,
        }


@dataclass(frozen=True)
class SourceTotal:
    """Corpus size of a source in a growth window, independent of technology."""

    source: str
    window: str
    n_total: int
    available: bool = True
    collected_at: datetime | None = None  # служебное, в to_dict() не попадает
    type_filter: str | None = None  # под каким фильтром посчитан корпус; в to_dict() не идёт
    totals_signature: str | None = None  # отпечаток запроса, которым посчитан; служебное

    def __post_init__(self) -> None:
        source = self.source.strip()
        window = str(self.window).strip()
        if not source:
            raise ValueError("source must be a non-empty string")
        if window not in ALLOWED_WINDOWS:
            raise ValueError(f"invalid window={window!r}")
        if self.n_total < 0:
            raise ValueError("n_total must be >= 0")
        object.__setattr__(self, "source", source)
        object.__setattr__(self, "window", window)

    def to_dict(self) -> dict[str, Any]:
        return {"source": self.source, "window": self.window, "n_total": self.n_total}


@dataclass(frozen=True)
class SearchTerms:
    """Термины кандидата и собранный из них булев запрос (pipeline.md 0.2, «Контракты»).

    query понимают OpenAlex и arXiv. TechCrunch булеву логику не читает и работает
    по списку terms — поэтому оба представления лежат рядом.
    """

    terms: list[str]
    context_terms: list[str]
    query: str
    violations: list[dict[str, Any]] = field(default_factory=list)

    @property
    def usable(self) -> bool:
        """Собрался ли запрос: без него считать счётчики нельзя."""
        return bool(self.query)


@dataclass(frozen=True)
class Counter:
    """Сколько записей источника подходит под запрос за окно. Документы не выгружаются."""

    tech_key: str
    source: str
    window: str
    date_from: date
    date_to: date
    n: int
    type_filter: str | None = None
    # Семантика запроса: фраза или слова, по каким полям, с каким фильтром типа.
    # Два источника могут искать одну строку по-разному, и без этого поля их числа
    # в кэше неразличимы. Часть ключа строки вместе с (tech_key, terms_hash, source, window).
    query_variant: str | None = None

    def __post_init__(self) -> None:
        window = str(self.window).strip()
        if window not in ALLOWED_COUNTER_WINDOWS:
            raise ValueError(f"invalid counter window={window!r}")
        if not self.tech_key.strip() or not self.source.strip():
            raise ValueError("tech_key and source must be non-empty")
        if self.n < 0:
            raise ValueError("n must be >= 0")
        if self.date_from >= self.date_to:
            raise ValueError("date_from must be earlier than date_to")
        object.__setattr__(self, "window", window)
        object.__setattr__(self, "tech_key", self.tech_key.strip())
        object.__setattr__(self, "source", self.source.strip())

    def to_dict(self) -> dict[str, Any]:
        return {
            "tech_key": self.tech_key,
            "source": self.source,
            "window": self.window,
            "date_from": self.date_from.isoformat(),
            "date_to": self.date_to.isoformat(),
            "type_filter": self.type_filter,
            "query_variant": self.query_variant,
            "n": self.n,
        }


@dataclass
class CounterResult:
    """Выход поиска №2 версии 0.2: только числа.

    Поля documents здесь нет намеренно. Признаки принимают эту структуру, а поиск №1
    возвращает RecentSearchResult без счётчиков — перепутать два поиска не даст система типов.
    """

    candidate_id: str
    tech_key: str
    terms_hash: str
    counters: list[Counter] = field(default_factory=list)
    source_totals: list[SourceTotal] = field(default_factory=list)
    query_id: str | None = None
    cache_hit: bool = False
    skipped_as_mainstream: bool = False

    def to_contract_dict(self) -> dict[str, Any]:
        """Проводной формат для Ярослава. Недоступные источники в итоги не идут."""
        return {
            "candidate_id": self.candidate_id,
            "tech_key": self.tech_key,
            "counters": [row.to_dict() for row in self.counters],
            "source_totals": [row.to_dict() for row in self.source_totals if row.available],
        }


@dataclass
class Candidate:
    """Input from Danya (query mode, Search #2)."""

    candidate_id: str
    name_en: str
    query_id: str | None = None
    name_ru: str | None = None
    aliases: list[str] = field(default_factory=list)
    # Два списка терминов для счётчиков (pipeline.md 0.2). aliases остаётся для
    # collect_history: он по-прежнему выгружает документы поиску №1 и инсайтам.
    terms: list[str] = field(default_factory=list)
    context_terms: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        if not self.candidate_id.strip() or not self.name_en.strip():
            raise ValueError("candidate_id and name_en are required")
        self.aliases = _clean_list(self.aliases)
        self.terms = _clean_list(self.terms)
        self.context_terms = _clean_list(self.context_terms)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> Candidate:
        return cls(
            candidate_id=str(payload["candidate_id"]),
            query_id=_optional_str(payload.get("query_id")),
            name_ru=_optional_str(payload.get("name_ru")),
            name_en=str(payload["name_en"]),
            aliases=list(payload.get("aliases") or []),
            terms=list(payload.get("terms") or []),
            context_terms=list(payload.get("context_terms") or []),
        )


@dataclass
class Technology:
    """Input for training mode: one labelled technology (signal or negative)."""

    tech_id: str
    name_en: str
    aliases: list[str] = field(default_factory=list)
    label: str | None = None
    queries: list[str] | None = None
    terms: list[str] = field(default_factory=list)
    context_terms: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        if not self.tech_id.strip() or not self.name_en.strip():
            raise ValueError("tech_id and name_en are required")
        self.aliases = _clean_list(self.aliases)
        self.terms = _clean_list(self.terms)
        self.context_terms = _clean_list(self.context_terms)

    def as_candidate(self) -> Candidate:
        return Candidate(
            candidate_id=self.tech_id,
            query_id="training",
            name_en=self.name_en,
            aliases=self.aliases,
            terms=self.terms,
            context_terms=self.context_terms,
        )

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> Technology:
        return cls(
            tech_id=str(payload.get("tech_id") or payload["candidate_id"]),
            name_en=str(payload["name_en"]),
            aliases=list(payload.get("aliases") or []),
            label=_optional_str(payload.get("label")),
            queries=list(payload["queries"]) if payload.get("queries") else None,
            terms=list(payload.get("terms") or []),
            context_terms=list(payload.get("context_terms") or []),
        )


@dataclass
class CollectionResult:
    """Output for Yaroslav: documents + source_totals. Same shape in both modes."""

    candidate_id: str
    documents: list[Document] = field(default_factory=list)
    source_totals: list[SourceTotal] = field(default_factory=list)
    query_id: str | None = None
    independent_confirmation: bool = True
    cache_hit: bool = False
    skipped_as_mainstream: bool = False

    def to_contract_dict(self) -> dict[str, Any]:
        """Wire format agreed with Yaroslav. Extra collector fields are omitted."""
        return {
            "candidate_id": self.candidate_id,
            "documents": [doc.to_dict() for doc in self.documents],
            "source_totals": [row.to_dict() for row in self.source_totals if row.available],
        }


@dataclass
class RecentSearchResult:
    """Search #1 output: fresh documents for candidate extraction. Not for features."""

    documents: list[Document] = field(default_factory=list)
    subqueries: list[str] = field(default_factory=list)
    date_from: date | None = None
    date_to: date | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "documents": [doc.to_dict() for doc in self.documents],
            "subqueries": list(self.subqueries),
            "date_from": self.date_from.isoformat() if self.date_from else None,
            "date_to": self.date_to.isoformat() if self.date_to else None,
        }


def copy_document(doc: Document, **changes: Any) -> Document:
    return replace(doc, **changes)


def build_search_terms(terms: list[str], context_terms: list[str]) -> SearchTerms:
    """Два списка терминов -> булев запрос. Единственное место сборки (collector.query)."""
    query, violations = build_query(terms, context_terms)
    good_terms, _ = clean_terms(terms, "terms")
    good_context, _ = clean_terms(context_terms, "context_terms")
    return SearchTerms(
        terms=good_terms,
        context_terms=good_context,
        query=query,
        violations=violations,
    )


def _clean_list(values: list[str] | None) -> list[str]:
    """Строки без пустых и без краевых пробелов. Порядок сохраняется."""
    return [str(item).strip() for item in (values or []) if str(item).strip()]


def _optional_str(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None
