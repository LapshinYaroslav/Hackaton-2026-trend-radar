"""Чистые загрузчики справочников: файлы → словари. Без Postgres."""

from __future__ import annotations

import ast
import csv
import json
import re
from pathlib import Path
from typing import Any, Iterable

ROOT = Path(__file__).resolve().parents[1]
SIGNALS_XLSX = ROOT / "data" / "raw" / "dataset.xlsx"
NEGATIVES_XLSX = ROOT / "data" / "raw" / "negatives.xlsx"
NEGATIVES_CSV = ROOT / "labels" / "negatives.csv"
TECHNOLOGIES_CSV = ROOT / "data" / "interim" / "technologies.csv"
GROUPS_CSV = ROOT / "labels" / "signal_groups.csv"
STOPLIST_TXT = ROOT / "labels" / "company_stoplist.txt"
SOURCE_TOTALS_JSON = ROOT / "evidence" / "source_totals_training.json"
ROSPATENT_JSON = ROOT / "evidence" / "rospatent_training_counts.json"
LABELS_DIR = ROOT / "labels"

LINK_RE = re.compile(r"\[([^\]]+)\]\((https?://[^)\s]+)\)")
AREAS = (
    ("Edge", "Edge", 1, "Периферийные устройства, IoT, on-device inference"),
    ("Защита ИИ", "AI security", 2, "Безопасность моделей, агентов и контуров ИИ"),
    ("Индустриальный ИИ", "Industrial AI", 3, "Производство, OT, контроль качества"),
    ("Инфраструктура ИИ", "AI infrastructure", 4, "ЦОД, кремний, сети, охлаждение"),
    ("Роботы", "Robots", 5, "Физический ИИ и робототехника"),
    ("Финтех", "Fintech", 6, "Платежи, идентичность, рынки капитала"),
)

# Веса из docs/methodology.md §8.2 (обучение на 160 строках, s2a2-v1).
MODEL_VERSIONS = (
    {
        "version": "s2a1-v1",
        "is_default": False,
        "features": [
            "volume",
            "growth_research",
            "recency",
            "share_news_wordmatch",
            "share_prev6",
            "age_first_arxiv",
        ],
        "threshold": 0.325,
        "cutoff_date": "2026-09-01",
        "intercept": None,
        "weights": {},
        "metrics": {"cv_auc": 0.805, "loao_auc": 0.811},
        "notes": "Предыдущая боевая модель, сохранена для воспроизводимости отчётов.",
    },
    {
        "version": "s2a2-v1",
        "is_default": True,
        "features": [
            "share_news_wordmatch",
            "recency",
            "share_patent",
            "age_first_arxiv",
            "share_prev6",
            "growth_research",
        ],
        "threshold": 0.400,
        "cutoff_date": "2026-09-01",
        "intercept": 0.381,
        "weights": {
            "share_news_wordmatch": 1.335,
            "recency": 1.049,
            "share_patent": -0.661,
            "age_first_arxiv": -0.418,
            "share_prev6": -0.100,
            "growth_research": 0.003,
        },
        "metrics": {"cv_auc": 0.828, "loao_auc": 0.837, "cv_f1": 0.808},
        "notes": "Боевая модель: volume заменён на share_patent.",
    },
)

PROJECT_SETTINGS = (
    ("cutoff_date", "2026-09-01", "Документы с этой даты не учитываются"),
    ("collection_start", "2020-09-01", "Нижняя граница окна сбора"),
    ("default_model", "s2a2-v1", "Версия модели по умолчанию"),
    ("dataset_as_of", "2026-09", "Срез датасета организаторов"),
    ("training_size", "160", "100 сигналов + 60 негативов"),
)

EVAL_FILES = {
    "g6_eval": "g6_eval_labels.csv",
    "d5_eval": "d5_eval_labels.csv",
    "i2_eval": "i2_eval_labels.csv",
    "g_blind": "g_blind_labels.csv",
    "n_blind": "n_blind_labels.csv",
    "n2_blind": "n2_blind_labels.csv",
    "k_blind": "k_blind_labels.csv",
    "p2_blind": "p2_blind_labels.csv",
}


def parse_links(text: str) -> list[dict[str, str]]:
    return [{"title": title, "url": url} for title, url in LINK_RE.findall(text or "")]


def split_companies(text: str) -> list[str]:
    names: list[str] = []
    for part in re.split(r"\s*[,;]\s*", str(text or "")):
        item = " ".join(part.split()).strip()
        if item:
            names.append(item)
    return names


def _parse_jsonish(value: str) -> Any:
    text = (value or "").strip()
    if not text:
        return []
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        try:
            return ast.literal_eval(text)
        except (ValueError, SyntaxError):
            return text


def _truthy(value: str) -> bool | None:
    text = (value or "").strip().casefold()
    if text in {"true", "1", "yes"}:
        return True
    if text in {"false", "0", "no"}:
        return False
    return None


def _int_or_none(value: Any) -> int | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text or text.lower() == "nan":
        return None
    try:
        return int(float(text))
    except ValueError:
        return None


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return [{k: (v or "").strip() for k, v in row.items()} for row in csv.DictReader(handle)]


def load_technologies_csv(path: Path = TECHNOLOGIES_CSV) -> list[dict[str, Any]]:
    rows = []
    for raw in read_csv(path):
        tech_id = raw["tech_id"]
        label = int(raw["label"])
        rows.append(
            {
                "tech_id": tech_id,
                "name": raw["name"],
                "name_ru": raw.get("name_ru") or raw["name"],
                "name_en": raw.get("name_en") or None,
                "name_en_manual": raw.get("name_en_manual") or None,
                "name_gloss": raw.get("name_gloss") or None,
                "name_inline_gloss": raw.get("name_inline_gloss") or None,
                "tech_key": raw.get("tech_key") or raw.get("name_en") or None,
                "area": raw["area"],
                "label": label,
                "source": "dataset" if label == 1 else "negatives",
                "negative_type": raw.get("negative_type") or None,
                "search_terms_manual": raw.get("search_terms_manual") or None,
                "terms": _parse_jsonish(raw.get("terms") or ""),
                "context_terms": _parse_jsonish(raw.get("context_terms") or ""),
                "anchor_volume": _int_or_none(raw.get("anchor_volume")),
                "n_attested_names": _int_or_none(raw.get("n_attested_names")),
                "blind_choice": _truthy(raw.get("blind_choice") or ""),
                "naming_model": raw.get("model_version") or None,
                "prompt_version_norm": raw.get("prompt_version_norm") or None,
            }
        )
    return rows


def load_negatives(path: Path = NEGATIVES_CSV) -> list[dict[str, Any]]:
    rows = []
    for raw in read_csv(path):
        rows.append(
            {
                "tech_id": raw["id"],
                "name": raw["name"],
                "area": raw["area"],
                "negative_type": raw["negative_type"],
                "pair_with": _int_or_none(raw.get("pair_with")),
                "criterion": raw.get("criterion") or None,
                "evidence_url": raw.get("evidence_url") or None,
                "evidence_url_2": raw.get("evidence_url_2") or None,
                "search_terms": raw.get("search_terms") or None,
            }
        )
    return rows


def load_signal_groups(path: Path = GROUPS_CSV) -> list[dict[str, Any]]:
    rows = []
    for raw in read_csv(path):
        rows.append(
            {
                "signal_id": int(raw["id"]),
                "group_name": raw["group"],
                "reason": raw.get("reason") or "",
            }
        )
    return rows


def load_stoplist(path: Path = STOPLIST_TXT) -> list[str]:
    if not path.exists():
        return []
    return sorted({line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()})


def load_source_totals_training(path: Path = SOURCE_TOTALS_JSON) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    payload = json.loads(path.read_text(encoding="utf-8"))
    rows = []
    for item in payload:
        rows.append(
            {
                "source": item["source"],
                "period": item.get("period") or item.get("window"),
                "n_total": int(item["n_total"]),
                "available": bool(item.get("available", True)),
                "type_filter": item.get("type_filter"),
                "totals_signature": item.get("totals_signature"),
                "collected_at": item.get("collected_at"),
            }
        )
    return rows


def load_rospatent_counts(path: Path = ROSPATENT_JSON) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    payload = json.loads(path.read_text(encoding="utf-8"))
    rows = []
    for item in payload.get("items") or []:
        rows.append(
            {
                "tech_id": item["tech_id"],
                "phrase": item["phrase"],
                "n_pat": int(item["n_pat"]),
                "fetched_at": item.get("fetched_at"),
            }
        )
    return rows


def load_eval_labels(directory: Path = LABELS_DIR) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for pool, filename in EVAL_FILES.items():
        path = directory / filename
        if not path.exists():
            continue
        for raw in read_csv(path):
            item_key = (
                raw.get("n_id")
                or raw.get("id")
                or raw.get("rank")
                or raw.get("name_en")
                or raw.get("canonical_concept")
            )
            if not item_key:
                continue
            rows.append(
                {
                    "pool": pool,
                    "item_key": str(item_key),
                    "name_en": raw.get("name_en") or raw.get("canonical_concept") or None,
                    "label": raw.get("label") or None,
                    "category": raw.get("category") or None,
                    "payload": raw,
                }
            )
    return rows


def load_signals_xlsx(path: Path = SIGNALS_XLSX) -> list[dict[str, Any]]:
    """Полные колонки датасета организаторов. Нет файла — пустой список."""
    if not path.exists():
        return []
    try:
        import pandas as pd
    except ImportError:
        return []

    raw = pd.read_excel(path, header=1)
    if str(raw.columns[0]).startswith("Unnamed"):
        raw = raw.drop(columns=[raw.columns[0]])
    raw = raw.rename(
        columns={
            "№": "id",
            "Технология (слабый сигнал)": "name",
            "Область": "area",
            "Компании": "companies",
            "Почему это слабый сигнал": "rationale",
            "Стадия развития": "stage_raw",
            "Тренд упоминаний": "trend_raw",
            "Балл (стадия+тренд)": "expert_score",
            "Источники": "sources_raw",
        }
    )
    rows = []
    for item in raw.to_dict(orient="records"):
        number = int(item["id"])
        sources_raw = "" if pd.isna(item.get("sources_raw")) else str(item["sources_raw"])
        companies_raw = "" if pd.isna(item.get("companies")) else str(item["companies"])
        rows.append(
            {
                "tech_id": f"s{number}",
                "signal_no": number,
                "name": str(item["name"]).strip(),
                "area": str(item["area"]).strip(),
                "rationale": None if pd.isna(item.get("rationale")) else str(item["rationale"]).strip(),
                "stage_raw": None if pd.isna(item.get("stage_raw")) else str(item["stage_raw"]).strip(),
                "trend_raw": None if pd.isna(item.get("trend_raw")) else str(item["trend_raw"]).strip(),
                "expert_score": _int_or_none(item.get("expert_score")),
                "companies": split_companies(companies_raw),
                "sources": parse_links(sources_raw),
            }
        )
    return rows


def merge_training(
    technologies: Iterable[dict[str, Any]],
    negatives: Iterable[dict[str, Any]],
    groups: Iterable[dict[str, Any]],
    signals: Iterable[dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    """Собирает 160 технологий: CSV как основа, Excel и negatives обогащают метаданные."""
    by_id = {row["tech_id"]: dict(row) for row in technologies}
    group_by_signal = {row["signal_id"]: row for row in groups}
    for row in by_id.values():
        if row["label"] == 1:
            number = int(row["tech_id"][1:])
            group = group_by_signal.get(number)
            row["duplicate_group"] = group["group_name"] if group else row["tech_id"]
            row["pair_with"] = None
            row["criterion"] = None
        else:
            row["duplicate_group"] = row["tech_id"]

    for neg in negatives:
        row = by_id.get(neg["tech_id"])
        if not row:
            continue
        row["pair_with"] = neg["pair_with"]
        row["criterion"] = neg["criterion"]
        row["negative_type"] = neg["negative_type"]
        if neg.get("search_terms") and not row.get("search_terms_manual"):
            row["search_terms_manual"] = neg["search_terms"]
        row["evidence"] = []
        if neg.get("evidence_url"):
            row["evidence"].append({"kind": "evidence", "title": None, "url": neg["evidence_url"]})
        if neg.get("evidence_url_2"):
            row["evidence"].append({"kind": "evidence_2", "title": None, "url": neg["evidence_url_2"]})

    for signal in signals or []:
        row = by_id.get(signal["tech_id"])
        if not row:
            continue
        row["rationale"] = signal.get("rationale")
        row["stage_raw"] = signal.get("stage_raw")
        row["trend_raw"] = signal.get("trend_raw")
        row["expert_score"] = signal.get("expert_score")
        row["companies"] = signal.get("companies") or []
        row["evidence"] = [
            {"kind": "dataset_source", "title": item["title"], "url": item["url"]}
            for item in signal.get("sources") or []
        ]

    return [by_id[key] for key in sorted(by_id, key=_tech_sort)]


def _tech_sort(tech_id: str) -> tuple[int, int]:
    return (0 if tech_id.startswith("s") else 1, int(tech_id[1:]))
