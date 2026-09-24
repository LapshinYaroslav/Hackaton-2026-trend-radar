"""Ранжирование кандидатов: порядок, объяснения, пропуски и незнакомая область."""
import json
import math

import pandas as pd
import pytest

from model.config import FEATURES_S2A2, OTHER_AREA
from model.predict import load, predict
from model.ranking import (NO_COUNTERS, known_areas, rank_candidates,
                           score_candidate)

KNOWN = "edge model compression"
MAINSTREAM = "humanoid robot"


def test_known_technology_is_scored_with_all_fields() -> None:
    """У посчитанного кандидата заполнены все поля внутреннего формата."""
    got = score_candidate(KNOWN)
    assert 0.0 <= got["score"] <= 1.0
    assert got["is_signal"] == (got["score"] >= got["threshold"])
    assert set(got["contributions"]) == set(FEATURES_S2A2)
    assert set(got["features"]) == set(FEATURES_S2A2)
    assert got["sources"] and got["counters"]
    assert got["skipped_reason"] is None
    assert got["model_version"] == "s2a2-v1" and got["cutoff_date"]


def test_contributions_add_up_to_the_score() -> None:
    """Сумма вкладов плюс свободный член даёт логит возвращённой вероятности."""
    got = score_candidate(KNOWN)
    logit = load()["meta"]["intercept"] + sum(got["contributions"].values())
    assert 1 / (1 + math.exp(-logit)) == pytest.approx(got["score"], abs=1e-12)


def test_signal_outranks_mainstream() -> None:
    """Известный сигнал стоит выше известного мейнстрима."""
    order = [item["name"] for item in rank_candidates([MAINSTREAM, KNOWN])]
    assert order.index(KNOWN) < order.index(MAINSTREAM)


def test_ranking_is_sorted_by_score() -> None:
    """Список отсортирован по убыванию вероятности."""
    scores = [item["score"] for item in
              rank_candidates([MAINSTREAM, KNOWN, "robotic teleoperation data"])]
    assert scores == sorted(scores, reverse=True)


def test_candidate_without_counters_goes_last_with_a_reason() -> None:
    """Не посчитанный кандидат не выбрасывается, а уходит в конец с причиной."""
    got = rank_candidates(["такой технологии в кэше нет", KNOWN])
    assert got[-1]["score"] is None
    assert got[-1]["skipped_reason"] == NO_COUNTERS
    assert got[0]["name"] == KNOWN


def test_limit_does_not_hide_the_skipped() -> None:
    """limit режет посчитанных, но причины пропуска остаются видны."""
    got = rank_candidates([KNOWN, MAINSTREAM, "нет такой технологии"], limit=1)
    assert [item["score"] is None for item in got] == [False, True]


def test_unknown_area_falls_back_and_differs_from_a_known_one() -> None:
    """Незнакомая область считается на общих центре и масштабе, а не падает."""
    unknown = score_candidate(KNOWN, area="область, которой не было")
    known = score_candidate(KNOWN, area="Edge")
    assert unknown["score"] is not None
    assert unknown["score"] != pytest.approx(known["score"], abs=1e-9)
    scaler = load()["pipeline"].named_steps["area_scaler"]
    assert scaler.stats_for("область, которой не было")[0].tolist() == \
        scaler.overall_[0].tolist()


def test_documents_are_passed_through_untouched() -> None:
    """Ссылки на документы модель не добывает, а прикладывает как есть."""
    links = [{"url": "https://example.com/a", "title": "A"}]
    got = rank_candidates([KNOWN], documents={KNOWN: links})
    assert got[0]["documents"] == links


def test_candidate_may_be_a_dict_with_terms() -> None:
    """Кандидат принимается словарём: термины и область идут в расчёт."""
    got = score_candidate({"name": KNOWN, "terms": [KNOWN], "context_terms": [],
                           "area": "Edge"})
    assert got["area"] == "Edge" and got["score"] is not None


def test_top_features_are_json_safe_and_ordered() -> None:
    """top_features сортируются по модулю вклада, NaN отдаётся как null."""
    got = score_candidate("robotic teleoperation data")
    weights = [abs(item["contribution"]) for item in got["top_features"]]
    assert weights == sorted(weights, reverse=True)
    json.dumps(got["top_features"])


def test_whole_answer_serialises_to_json() -> None:
    """Внутренний формат уходит в интерфейс как есть: пропуск — null, а не NaN.

    allow_nan=False: NaN в JSON невалиден, строгий разборщик на нём упадёт.
    """
    got = score_candidate(KNOWN)
    assert "NaN" not in json.dumps(got, ensure_ascii=False, allow_nan=False)


def test_known_areas_are_the_six_from_training() -> None:
    """Интерфейсу предлагаются ровно те области, у которых есть свои центр и масштаб."""
    got = known_areas()
    assert len(got) == 6
    assert "Edge" in got and "Роботы" in got
    assert OTHER_AREA not in got


def test_known_area_is_marked_as_the_main_path() -> None:
    """У знакомой области ответ помечен основным режимом."""
    got = score_candidate(KNOWN, area="Edge")
    assert got["area_known"] is True
    assert got["normalization"] == "по области"


@pytest.mark.parametrize("area", [OTHER_AREA, "", "Квантовые вычисления"])
def test_other_and_unknown_areas_take_the_fallback(area: str) -> None:
    """«Другое», пустая и любая незнакомая область идут одним путём — откатом."""
    got = score_candidate(KNOWN, area=area)
    assert got["area_known"] is False
    assert got["normalization"] == "общая (откат)"
    assert got["score"] is not None


def test_fallback_gives_the_same_answer_for_every_unknown_area() -> None:
    """Откат не зависит от того, как названа незнакомая область."""
    first = score_candidate(KNOWN, area=OTHER_AREA)["score"]
    second = score_candidate(KNOWN, area="что-то ещё")["score"]
    assert first == pytest.approx(second, abs=1e-12)


def test_area_marking_survives_ranking() -> None:
    """Разметка режима доходит до элементов ранжированного списка."""
    got = rank_candidates([KNOWN, MAINSTREAM], area=OTHER_AREA)
    assert all(item["normalization"] == "общая (откат)" for item in got)
