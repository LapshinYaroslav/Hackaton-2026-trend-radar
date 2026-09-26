"""Пропуск тестов, которым нужны файлы вне git (data/): в чистой копии они пропускаются с причиной.

В git не лежат артефакты модели, обучающая таблица признаков, кэш счётчиков и сохранённые прогоны пайплайна
(CLAUDE.md, жёсткие правила 1–2). Остальные тесты идут без data/.
"""
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
MODEL = ROOT / "data" / "model" / "s2a2-v1" / "trend_radar_s2a2-v1.joblib"
TRAINING = ROOT / "data" / "experiments"
COUNTERS = ROOT / "data" / "interim" / "collector" / "cache" / "counters"

needs_model = pytest.mark.skipif(not MODEL.exists(), reason=f"нет {MODEL.relative_to(ROOT)}: python -m model.train")
needs_training = pytest.mark.skipif(not any(TRAINING.glob("axis_features_*.csv")),
                                    reason="нет data/experiments/axis_features_*.csv: обучающая таблица не в git")
needs_counters = pytest.mark.skipif(not any(COUNTERS.glob("*.json")) if COUNTERS.exists() else True,
                                    reason="нет кэша счётчиков data/interim/collector/cache/counters")


def require_model() -> None:
    """То же, что needs_model, для помощников и фикстур."""
    if not MODEL.exists():
        pytest.skip(f"нет {MODEL.relative_to(ROOT)}: python -m model.train")
