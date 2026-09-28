"""Запрос → ранжированный список кандидатов с объяснением каждого.

Внутренний формат: контракт выдачи согласуется отдельно, здесь то, что модель
умеет отдать. Описание полей — docs/model_interface.md.

Что модель знает и чего не знает. Знает числа: вероятность, порог, вклад каждого
признака в логит, счётчики по окнам и источники, из которых они взяты. Не знает
документов: их собирает сборщик, поэтому ссылки принимаются параметром и
прикладываются к кандидату как есть.

Запуск: python -m model.ranking "edge model compression" "humanoid robot"
"""
from __future__ import annotations

import math
from typing import Iterable, Mapping

from model.candidate import Fetch, candidate_features
from model.config import EXPLANATIONS, OTHER_AREA
from model.predict import load, predict

# Кандидат без счётчиков не ранжируется: вероятность по пропускам была бы
# вероятностью медианной технологии, а не этой.
NO_COUNTERS = "счётчиков по этим терминам не нашлось"


def known_areas(artifact: dict | None = None) -> list[str]:
    """Области, у которых есть свои центр и масштаб. Интерфейс предлагает их и «другое»."""
    return sorted((artifact or load())["meta"]["by_area"])


def _plain(value: float | None) -> float | None:
    """NaN наружу не отдаём: в JSON он не сериализуется, пропуск — это null."""
    return None if value is None or (isinstance(value, float) and math.isnan(value)) \
        else float(value)


def _normalise(candidate: str | Mapping) -> dict:
    """Кандидат принимается строкой или словарём с терминами и областью."""
    if isinstance(candidate, str):
        return {"name": candidate, "terms": None, "context_terms": None, "area": "",
                "n_pat": None}
    return {"name": candidate["name"], "terms": candidate.get("terms"),
            "context_terms": candidate.get("context_terms"),
            "area": candidate.get("area", ""), "n_pat": candidate.get("n_pat")}


def score_candidate(candidate: str | Mapping, *, area: str = "",
                    fetch: Fetch | None = None, documents: list | None = None,
                    artifact: dict | None = None) -> dict:
    """Один кандидат: признаки, вероятность, вклады, счётчики, источники, ссылки."""
    artifact = artifact or load()
    meta = artifact["meta"]
    spec = _normalise(candidate)
    where = spec["area"] or area
    built = candidate_features(spec["name"], terms=spec["terms"],
                               context_terms=spec["context_terms"], area=where,
                               fetch=fetch, n_pat=spec["n_pat"],
                               version=meta["model_version"])
    area_known = where in known_areas(artifact) and where != OTHER_AREA
    answer = {
        "name": built["name"], "area": where, "area_known": area_known,
        # По какому режиму считался ответ. От этого зависит, какие метрики
        # заявлять: у отката они свои, см. evidence/fallback_mode_*.md.
        "normalization": "по области" if area_known else "общая (откат)",
        "query": built["query"],
        "terms": built["terms"],
        "features": {name: _plain(value)
                     for name, value in built["features"].items()},
        "counters": built["counters"], "sources": built["sources"],
        "documents": list(documents or []),
        "model_version": meta["model_version"], "cutoff_date": meta["cutoff_date"],
        "threshold": meta["threshold"], "score": None, "is_signal": None,
        "contributions": {}, "top_features": [], "skipped_reason": None,
    }
    if not built["complete"]:
        answer["skipped_reason"] = NO_COUNTERS
        return answer
    probability, contributions = predict(built["features"], where, artifact)
    answer["score"] = probability
    answer["is_signal"] = probability >= meta["threshold"]
    answer["contributions"] = contributions
    answer["top_features"] = top_features(answer["features"], contributions)
    return answer


def top_features(features: Mapping[str, float], contributions: Mapping[str, float],
                 top: int = 3) -> list[dict]:
    """Самые весомые вклады по модулю, с готовой строкой для пользователя."""
    ranked = sorted(contributions.items(), key=lambda pair: -abs(pair[1]))[:top]
    return [{"name": name, "value": _plain(features.get(name)), "contribution": value,
             "explanation_ru": EXPLANATIONS.get(name, name)} for name, value in ranked]


def rank_candidates(candidates: Iterable[str | Mapping], *, area: str = "",
                    fetch: Fetch | None = None,
                    documents: Mapping[str, list] | None = None,
                    limit: int | None = None) -> list[dict]:
    """Ранжированный список: сначала посчитанные по убыванию вероятности.

    Кандидаты без счётчиков не выбрасываются, а уходят в конец со списка с
    заполненным skipped_reason: интерфейсу нужно показать, почему их не оценили.
    """
    artifact = load()
    scored = [score_candidate(item, area=area, fetch=fetch, artifact=artifact,
                              documents=(documents or {}).get(_normalise(item)["name"]))
              for item in candidates]
    ready = sorted((item for item in scored if item["score"] is not None),
                   key=lambda item: -item["score"])
    skipped = [item for item in scored if item["score"] is None]
    return (ready[:limit] if limit else ready) + skipped


def main() -> None:
    """Печатает ранжированный список по названиям из аргументов."""
    import sys
    names = sys.argv[1:] or ["edge model compression", "humanoid robot",
                             "robotic teleoperation data"]
    for item in rank_candidates(names):
        if item["score"] is None:
            print(f"  —      {item['name']}: {item['skipped_reason']}")
            continue
        print(f"  {item['score']:.4f} {'сигнал ' if item['is_signal'] else 'отсев '}"
              f"{item['name']}  (порог {item['threshold']})")
        for feature in item["top_features"]:
            print(f"        {feature['name']:22s} вклад {feature['contribution']:+.3f}")


if __name__ == "__main__":
    main()
