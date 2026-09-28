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

def _real_model() -> bool:
    """Боевой артефакт, не bootstrap из опубликованных весов."""
    if not MODEL.exists():
        return False
    try:
        import joblib
        return not bool(joblib.load(MODEL).get("meta", {}).get("bootstrap"))
    except Exception:
        return False


needs_model = pytest.mark.skipif(
    not _real_model(),
    reason=f"нет обученного {MODEL.relative_to(ROOT)}: python -m model.train",
)
needs_training = pytest.mark.skipif(not any(TRAINING.glob("axis_features_*.csv")),
                                    reason="нет data/experiments/axis_features_*.csv: обучающая таблица не в git")
needs_counters = pytest.mark.skipif(not any(COUNTERS.glob("*.json")) if COUNTERS.exists() else True,
                                    reason="нет кэша счётчиков data/interim/collector/cache/counters")


def require_model() -> None:
    """Артефакт на диске — боевой или bootstrap. Для оркестратора этого достаточно."""
    if not MODEL.exists():
        pytest.skip(f"нет {MODEL.relative_to(ROOT)}: python -m model.train")
