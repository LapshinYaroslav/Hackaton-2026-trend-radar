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
