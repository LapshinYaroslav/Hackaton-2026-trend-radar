"""Корпусные итоги источников, зафиксированные на дату обучения.

Почему файл, а не кэш. Кэш источников изменяемый: у него суточный срок годности,
а OpenAlex индексирует непрерывно, и его корпус растёт между прогонами. Замеренная
разница между двумя снимками сдвигала growth_research на 3.3e-04 — больше допуска
гейта воспроизводимости. Поэтому итоги, на которых обучалась модель, лежат одним
версионируемым файлом и меняются только вместе с моделью.

Отсутствие файла или неполный состав окон — это отказ считать признаки, а не молчаливый
переход на живой кэш: иначе числа поехали бы незаметно.
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from collector.constants import YEAR_WINDOWS

ROOT = Path(__file__).resolve().parents[1]
TRAINING_TOTALS = ROOT / "evidence" / "source_totals_training.json"
# Три источника, по которым считаются признаки модели.
EXPECTED_SOURCES = frozenset({"openalex", "arxiv", "techcrunch"})
EXPECTED_WINDOWS = frozenset(YEAR_WINDOWS)


class CorpusTotalsError(RuntimeError):
    """Корпусные итоги недоступны или неполны: признаки считать нельзя."""


def load_training_totals(path: Path | None = None) -> pd.DataFrame:
    """Итоги корпусов на дату обучения. Колонки source, window, n_total.

    Проверяется состав: три источника на шесть годовых окон, все доступны и
    положительны. Любое отклонение — CorpusTotalsError с указанием, чего не хватает.
    """
    path = path or TRAINING_TOTALS
    if not path.exists():
        raise CorpusTotalsError(
            f"Нет файла корпусных итогов {path}. Он фиксирует числа, на которых "
            f"обучалась модель, и должен лежать в репозитории. Считать по живому "
            f"кэшу нельзя: корпуса растут, и признаки разойдутся с обучением."
        )
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except ValueError as error:
        raise CorpusTotalsError(f"Файл корпусных итогов {path} не читается: {error}") \
            from error
    rows = [item for item in payload if item.get("available")]
    frame = pd.DataFrame(rows, columns=["source", "window", "n_total"])
    _check(frame, path)
    return frame


def _check(frame: pd.DataFrame, path: Path) -> None:
    """Состав окон и источников. Ошибка называет, каких именно пар не хватает."""
    have = {(str(row.source), str(row.window)) for row in frame.itertuples()}
    need = {(source, window) for source in EXPECTED_SOURCES for window in EXPECTED_WINDOWS}
    missing = sorted(need - have)
    if missing:
        raise CorpusTotalsError(
            f"В {path.name} не хватает пар источник-окно: {missing}. Ожидаются все "
            f"{len(need)}: {sorted(EXPECTED_SOURCES)} на {sorted(EXPECTED_WINDOWS)}. "
            f"Неполный набор занизил бы поправку на рост корпуса."
        )
    empty = sorted({(str(row.source), str(row.window))
                    for row in frame.itertuples() if int(row.n_total) <= 0})
    if empty:
        raise CorpusTotalsError(
            f"В {path.name} нулевой корпус у пар {empty}. Ноль в знаменателе поправки "
            f"означает «источник не ответил», а не «документов не было»."
        )


# Патентные счётчики обучения (s2a2-v1): n_pat 160 технологий, зафиксированные на дату
# обучения. Роспатент индексирует непрерывно, поэтому признак share_patent обучающей
# таблицы читается только из этого файла — отсутствие или неполнота означает отказ.
TRAINING_PATENTS = ROOT / "evidence" / "rospatent_training_counts.json"
EXPECTED_PATENT_ROWS = 160


class PatentSnapshotError(RuntimeError):
    """Снимок патентных счётчиков обучения недоступен или неполон: share_patent не считается."""


def load_training_patents(datasets: list[str], path: Path | None = None) -> pd.DataFrame:
    """n_pat обучающих технологий: колонки tech_id, phrase, n_pat.

    datasets — определение признака (model.config.ROSPATENT_DATASETS); снимок, собранный
    по другому набору датасетов, отвергается.
    """
    path = path or TRAINING_PATENTS
    if not path.exists():
        raise PatentSnapshotError(f"Нет снимка патентных счётчиков {path}. Считать share_patent "
                                  f"обучения по живому кэшу нельзя: Роспатент растёт.")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except ValueError as error:
        raise PatentSnapshotError(f"Снимок {path} не читается: {error}") from error
    if payload.get("datasets") != list(datasets):
        raise PatentSnapshotError(f"В {path.name} другой набор датасетов: {payload.get('datasets')}")
    frame = pd.DataFrame(payload.get("items") or [], columns=["tech_id", "phrase", "n_pat"])
    bad = frame["n_pat"].isna() | (pd.to_numeric(frame["n_pat"], errors="coerce") < 0)
    if len(frame) != EXPECTED_PATENT_ROWS or frame["tech_id"].duplicated().any() or bad.any():
        raise PatentSnapshotError(
            f"В {path.name} {len(frame)} строк (ожидается {EXPECTED_PATENT_ROWS}), "
            f"повторов tech_id {int(frame['tech_id'].duplicated().sum())}, "
            f"пустых или отрицательных n_pat {int(bad.sum())}")
    return frame.astype({"tech_id": str, "n_pat": int})
