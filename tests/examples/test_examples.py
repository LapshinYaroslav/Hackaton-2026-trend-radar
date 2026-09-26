"""Готовые примеры (И2): каждый examples/*.json — ответ по контракту query_result плюс meta, без секретов и путей."""
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
FILES = sorted((ROOT / "examples").glob("*.json"))
TOP_LEVEL = {"query_id", "topic", "area", "model_version", "threshold", "cutoff_date", "subqueries", "candidate_sources",
             "extract_version", "extract_model", "naming_mode", "candidates_version", "stats", "normalizer_deviations",
             "top", "excluded", "timings", "warnings", "meta"}
DETAIL = {"name_raw", "name_variants", "name_choice_rule", "n_works", "n_institutions", "n_institutions_capped",
          "n_docs", "n_sources", "known_training_label", "quote"}
PATENT = {"n_pat", "share_patent", "rospatent_failed"}
TOP_ITEM = {"rank", "name_ru", "name_en", "score", "explanation_ru", "contributions", "counters", "sources",
            "model_version", "name_ru_source", "name_ru_auto"} | DETAIL | PATENT
EXCLUDED_ITEM = {"name_ru", "name_en", "score", "skipped_reason", "reason_ru", "model_version", "name_ru_source",
                 "name_ru_auto"} | DETAIL
META = {"query", "area", "run_date", "model_version", "candidates_version", "seconds", "counters_cache_share"}
FORBIDDEN = ("api_key", "Bearer", "C:\\\\", "C:/", "/home/", "Users\\\\")


def test_examples_exist() -> None:
    assert [f.stem for f in FILES] == sorted(["ai", "fintech", "cybersec", "robots"])


@pytest.mark.parametrize("path", FILES, ids=[f.stem for f in FILES])
def test_example_matches_contract(path: Path) -> None:
    data = json.loads(path.read_text(encoding="utf-8"))
    assert set(data) == TOP_LEVEL and set(data["meta"]) == META
    assert data["model_version"] == data["meta"]["model_version"] == "s2a2-v1" and data["candidates_version"] == "v3"
    assert 0 < len(data["top"]) <= 15 and [t["rank"] for t in data["top"]] == list(range(1, len(data["top"]) + 1))
    for item in data["top"]:
        assert TOP_ITEM <= set(item) <= TOP_ITEM | {"note_ru"} and item["score"] >= data["threshold"]
        assert item["name_ru_auto"] is True and item["name_ru_source"] in ("translate", "extract", None)
    for item in data["excluded"]:
        assert EXCLUDED_ITEM <= set(item) <= EXCLUDED_ITEM | PATENT | {"note_ru"}
    assert 0.0 <= data["meta"]["counters_cache_share"] <= 1.0


@pytest.mark.parametrize("path", FILES, ids=[f.stem for f in FILES])
def test_example_has_no_secrets_or_machine_paths(path: Path) -> None:
    text = path.read_text(encoding="utf-8")
    assert not [word for word in FORBIDDEN if word in text]


def test_examples_fit_in_2_mb() -> None:
    assert sum(f.stat().st_size for f in (ROOT / "examples").iterdir()) <= 2 * 1024 * 1024
