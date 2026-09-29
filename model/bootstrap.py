"""Проверка перед стартом: обученный артефакт модели на месте, иначе ошибка.

Заглушки нет: без обученной модели сервис не считает баллы, а падает с понятной причиной.
"""

from __future__ import annotations

from pathlib import Path

from model.config import DEFAULT_VERSION
from model.predict import load
from model.train import existing_artifact


def ensure_artifact(version: str = DEFAULT_VERSION) -> Path:
    """Путь к обученному joblib версии. Нет файла или это заглушка bootstrap — ошибка."""
    path = existing_artifact(version)
    if not path.exists():
        raise FileNotFoundError(
            f"Нет обученной модели {version}: {path}. Положите артефакт из репозитория "
            f"(data/model/{version}/) или обучите: python -m model.train --version {version}")
    if load(path)["meta"].get("bootstrap"):
        raise RuntimeError(
            f"{path} — заглушка из опубликованных весов, а не обученная модель. "
            f"Удалите её и положите артефакт из репозитория (data/model/{version}/)")
    return path
