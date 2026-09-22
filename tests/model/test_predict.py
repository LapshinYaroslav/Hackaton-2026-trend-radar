"""Разбор логита по признакам обязан в точности совпадать с пайплайном."""
import math

import numpy as np
import pandas as pd
import pytest

from model.config import ALIASES, FEATURES, NEWS_SHARE, canonical
from model.predict import explain, predict
from model.train import build_pipeline, choose_threshold, describe, score_at


@pytest.fixture(scope="module")
def artifact() -> dict:
    """Маленькая обученная модель на синтетике: тест не зависит от собранных данных."""
    rng = np.random.default_rng(42)
    rows = 80
    frame = pd.DataFrame({name: rng.normal(size=rows) for name in FEATURES})
    frame.loc[:, "area"] = ["A"] * 40 + ["B"] * 40
    frame.loc[:, "label"] = (frame[FEATURES[0]] + frame[FEATURES[3]] > 0).astype(int)
    frame.loc[:, "group"] = [f"g{i}" for i in range(rows)]
    # Пропуски: нужно проверить, что заполнение медианой повторяется в predict.
    frame.loc[0:5, "age_first_arxiv"] = np.nan
    pipeline = build_pipeline().fit(frame[FEATURES + ["area"]], frame["label"])
    meta = describe(pipeline, frame, 0.5)
    return {"pipeline": pipeline, "meta": meta, "frame": frame}


def test_probability_matches_the_pipeline_on_every_row(artifact: dict) -> None:
    """Главная проверка: своя арифметика и пайплайн дают одно и то же число."""
    frame = artifact["frame"]
    expected = artifact["pipeline"].predict_proba(frame[FEATURES + ["area"]])[:, 1]
    for index, row in frame.iterrows():
        got, _ = predict(row.to_dict(), str(row["area"]), artifact)
        assert got == pytest.approx(expected[index], abs=1e-12)


def test_contributions_add_up_to_the_logit(artifact: dict) -> None:
    """Сумма вкладов плюс свободный член — это логит вероятности."""
    row = artifact["frame"].iloc[3]
    probability, contributions = predict(row.to_dict(), str(row["area"]), artifact)
    logit = artifact["meta"]["intercept"] + sum(contributions.values())
    assert 1 / (1 + math.exp(-logit)) == pytest.approx(probability, abs=1e-12)


def test_every_feature_gets_a_contribution(artifact: dict) -> None:
    """Вклад возвращается по каждому признаку модели, без пропусков."""
    _, contributions = predict(artifact["frame"].iloc[0].to_dict(), "A", artifact)
    assert set(contributions) == set(FEATURES)


def test_missing_feature_falls_back_to_the_training_median(artifact: dict) -> None:
    """Пропуск заменяется медианой обучения, а не нулём и не ошибкой."""
    row = artifact["frame"].iloc[0].to_dict()
    row["age_first_arxiv"] = None
    probability, contributions = predict(row, "A", artifact)
    meta = artifact["meta"]
    expected = (meta["coefficients"]["age_first_arxiv"]
                * meta["imputer_median_scaled"]["age_first_arxiv"])
    assert contributions["age_first_arxiv"] == pytest.approx(expected, abs=1e-12)
    assert 0.0 <= probability <= 1.0


def test_unknown_area_uses_overall_centre_and_scale(artifact: dict) -> None:
    """Новая область нормируется общими центром и масштабом, а не падает."""
    row = artifact["frame"].iloc[0].to_dict()
    unknown, _ = predict(row, "область, которой не было", artifact)
    scaler = artifact["pipeline"].named_steps["area_scaler"]
    assert scaler.stats_for("область, которой не было")[0].tolist() == \
        scaler.overall_[0].tolist()
    assert 0.0 <= unknown <= 1.0


def test_area_changes_the_answer(artifact: dict) -> None:
    """Нормировка внутри области значит, что область влияет на ответ."""
    row = artifact["frame"].iloc[0].to_dict()
    first, _ = predict(row, "A", artifact)
    second, _ = predict(row, "B", artifact)
    assert first != pytest.approx(second, abs=1e-9)


def test_old_feature_name_is_accepted(artifact: dict) -> None:
    """Старое имя share_market принимается наравне с новым: кэш не ломается."""
    row = artifact["frame"].iloc[0].to_dict()
    renamed = {ALIASES.get(key, key) if key != NEWS_SHARE else key: value
               for key, value in row.items()}
    renamed["share_market"] = renamed.pop(NEWS_SHARE)
    assert predict(row, "A", artifact)[0] == pytest.approx(
        predict(renamed, "A", artifact)[0], abs=1e-12)


def test_canonical_renames_only_the_old_column() -> None:
    """canonical переименовывает старое имя и не трогает уже канонические."""
    got = canonical(pd.DataFrame({"share_market": [0.1], "volume": [2.0]}))
    assert NEWS_SHARE in got.columns and "share_market" not in got.columns
    already = canonical(pd.DataFrame({NEWS_SHARE: [0.1]}))
    assert list(already.columns) == [NEWS_SHARE]


def test_explain_returns_threshold_decision_and_top_features(artifact: dict) -> None:
    """Готовый ответ выдаче: решение по порогу и самые весомые вклады."""
    row = artifact["frame"].iloc[0].to_dict()
    got = explain(row, "A", artifact, top=2)
    assert got["is_signal"] == (got["score"] >= got["threshold"])
    assert len(got["top_features"]) == 2
    weights = [abs(item["contribution"]) for item in got["top_features"]]
    assert weights == sorted(weights, reverse=True)


def test_threshold_choice_prefers_the_grid_maximum() -> None:
    """Порог берётся из сетки и попадает в точку максимума метрики."""
    labels = np.array([1] * 10 + [0] * 10)
    probability = np.concatenate([np.full(10, 0.8), np.full(10, 0.4)])
    assert 0.4 < choose_threshold(labels, probability) <= 0.8


def test_metrics_counts_accuracy_by_hand() -> None:
    """Матрица ошибок на посчитанном вручную примере: 3 верных из 5."""
    labels = np.array([1, 1, 1, 0, 0])
    probability = np.array([0.9, 0.6, 0.2, 0.7, 0.1])
    got = score_at(labels, probability, 0.5)
    assert (got["TP"], got["FP"], got["FN"], got["TN"]) == (2, 1, 1, 1)
    assert got["accuracy"] == pytest.approx(0.6)
    assert got["precision"] == pytest.approx(2 / 3)
    assert got["recall"] == pytest.approx(2 / 3)
    assert got["f1"] == pytest.approx(2 / 3)
