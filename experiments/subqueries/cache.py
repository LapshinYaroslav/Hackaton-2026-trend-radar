"""Кэш ответов LLM и OpenAlex на диске: повторный прогон не делает сетевых вызовов."""
import hashlib
import json
from datetime import datetime
from pathlib import Path
from typing import Callable

ROOT = Path(__file__).resolve().parents[2]
CACHE_DIR = ROOT / "data" / "interim" / "subquery_eval" / "cache"


def key_of(key_input: dict) -> str:
    """Отпечаток входа: sha256 от JSON с сортировкой ключей, чтобы порядок полей не влиял."""
    text = json.dumps(key_input, ensure_ascii=False, sort_keys=True)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def path_of(kind: str, key: str, cache_dir: Path | None = None) -> Path:
    """Путь к файлу кэша: <кэш>/<вид>/<отпечаток>.json."""
    return (cache_dir or CACHE_DIR) / kind / f"{key}.json"


def read(kind: str, key_input: dict, cache_dir: Path | None = None) -> dict | None:
    """Значение из кэша или None, если его нет или файл битый."""
    path = path_of(kind, key_of(key_input), cache_dir)
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))["value"]
    except (OSError, ValueError, KeyError):
        return None


def write(kind: str, key_input: dict, value: dict, cache_dir: Path | None = None) -> None:
    """Сохраняет значение вместе с входом: по файлу видно, на какой запрос он отвечает."""
    path = path_of(kind, key_of(key_input), cache_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    record = {"ts": datetime.now().astimezone().isoformat(timespec="seconds"),
              "key_input": key_input, "value": value}
    path.write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")


def cached(kind: str, key_input: dict, produce: Callable[[], dict],
           cache_dir: Path | None = None,
           should_cache: Callable[[dict], bool] | None = None) -> tuple[dict, bool]:
    """Значение из кэша или от produce(). Второй элемент — был ли реальный вызов.

    should_cache защищает от записи неудачного ответа: закэшированная ошибка живёт вечно,
    и повторный прогон уже никогда не переспросит.
    """
    hit = read(kind, key_input, cache_dir)
    if hit is not None and (should_cache is None or should_cache(hit)):
        return hit, False
    value = produce()
    if should_cache is None or should_cache(value):
        write(kind, key_input, value, cache_dir)
    return value, True
