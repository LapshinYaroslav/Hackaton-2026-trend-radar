"""Единая точка вызова YandexGPT: запрос к REST API и лог всех обращений."""
from __future__ import annotations

import json
import os
import threading
import time
from datetime import datetime
from pathlib import Path

import requests
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
LOG_FILE = ROOT / "logs" / "llm_calls.jsonl"

COMPLETION_URL = "https://llm.api.cloud.yandex.net/foundationModels/v1/completion"
# Модели не от Yandex в AI Studio доступны только через OpenAI-совместимый API (документация
# AI Studio, 23.09.2026): тот же ключ Api-Key, URI модели gpt://<folder>/<model>/latest.
OPENAI_URL = "https://llm.api.cloud.yandex.net/v1/chat/completions"
OPENAI_MODELS = {"qwen3-235b-a22b-fp8"}
SESSION = requests.Session()  # keep-alive: TLS-рукопожатие не повторяется на каждый вызов
# Явные имена моделей идут в URI без ветки: gpt://<folder>/yandexgpt-5-pro. Старые
# псевдонимы — с веткой /latest, за которой модель может смениться. По документации
# AI Studio (23.09.2026) yandexgpt/latest — это YandexGPT Pro 5, то есть yandexgpt-5-pro.
EXPLICIT_MODELS = {"yandexgpt-5-pro", "yandexgpt-5.1"}
ALLOWED_MODELS = {"yandexgpt", "yandexgpt-lite"} | EXPLICIT_MODELS | OPENAI_MODELS
DEFAULT_MODEL = "yandexgpt-lite"
DEFAULT_TIMEOUT = 60

# Квота Yandex AI Studio: 10 одновременных генераций в синхронном режиме.
# Работаем на квоте целиком, страховка от превышения — повтор при 429 ниже.
MAX_CONCURRENT_CALLS = 10
CALL_LIMIT = threading.Semaphore(MAX_CONCURRENT_CALLS)
RETRY_ATTEMPTS = 3
RETRY_DELAY = 0.5
TIMEOUT_ATTEMPTS = 2  # первая попытка и один повтор при таймауте


def build_model_uri(model: str | None = None) -> str:
    """URI модели: явная модель из аргумента или YANDEX_GPT_MODEL из .env."""
    load_dotenv()
    folder_id = os.getenv("YANDEX_FOLDER_ID", "").strip()
    model = model or os.getenv("YANDEX_GPT_MODEL", "").strip() or DEFAULT_MODEL
    if not folder_id:
        raise ValueError("В .env нет YANDEX_FOLDER_ID")
    if model not in ALLOWED_MODELS:
        raise ValueError(f"Модель {model!r} не разрешена, выберите из {sorted(ALLOWED_MODELS)}")
    if model in EXPLICIT_MODELS:
        return f"gpt://{folder_id}/{model}"
    return f"gpt://{folder_id}/{model}/latest"


def _log_call(record: dict) -> None:
    """Дописывает строку в logs/llm_calls.jsonl. Ключ API в лог не попадает."""
    try:
        LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
        with LOG_FILE.open("a", encoding="utf-8") as file:
            file.write(json.dumps(record, ensure_ascii=False) + "\n")
    except OSError as exc:
        print(f"Предупреждение: не удалось записать лог вызова LLM: {exc}")


def _post_once(body: dict, headers: dict, url: str = COMPLETION_URL):
    """Один POST; при таймауте — ровно один повтор с теми же параметрами, затем ошибка.

    Таймаут — сбой связи, а не содержания ответа, поэтому температура и промпт не меняются.
    Замер Б7 (23.09.2026): один вызов из десяти не уложился в 60 с.
    """
    for attempt in range(TIMEOUT_ATTEMPTS):
        try:
            with CALL_LIMIT:
                return SESSION.post(url, headers=headers, json=body, timeout=DEFAULT_TIMEOUT)
        except requests.Timeout:
            if attempt == TIMEOUT_ATTEMPTS - 1:
                raise


def _post_with_retry(body: dict, headers: dict, url: str = COMPLETION_URL) -> dict:
    """POST под ограничением конкурентности, с повтором при 429 и растущей паузой."""
    delay = RETRY_DELAY
    for attempt in range(RETRY_ATTEMPTS):
        response = _post_once(body, headers, url)
        if getattr(response, "status_code", 200) == 429 and attempt < RETRY_ATTEMPTS - 1:
            time.sleep(delay)
            delay *= 2
            continue
        response.raise_for_status()
        return response.json()
    raise requests.HTTPError("429 после всех повторов")


def _request(model_uri: str, openai: bool, system_prompt: str, user_prompt: str, temperature: float,
             json_object: bool, max_tokens: int) -> tuple[str, dict]:
    """Адрес и тело запроса: completion YandexGPT или OpenAI-совместимый chat/completions."""
    if openai:
        return OPENAI_URL, {"model": model_uri, "temperature": temperature, "max_tokens": max_tokens,
                            "messages": [{"role": "system", "content": system_prompt},
                                         {"role": "user", "content": user_prompt}]}
    return COMPLETION_URL, {
        "modelUri": model_uri,
        "completionOptions": {"stream": False, "temperature": temperature, "maxTokens": str(max_tokens)},
        "messages": [{"role": "system", "text": system_prompt}, {"role": "user", "text": user_prompt}],
        "json_object": json_object,
    }


def _read_answer(payload: dict, openai: bool) -> tuple[str, str | None, dict]:
    """Текст ответа, версия модели и расход токенов из ответа API."""
    if openai:
        return payload["choices"][0]["message"]["content"], payload.get("model"), payload.get("usage", {})
    data = payload.get("result", payload)
    return data["alternatives"][0]["message"]["text"], data.get("modelVersion"), data.get("usage", {})


def ask_llm(system_prompt: str, user_prompt: str, purpose: str, temperature: float = 0.3,
            json_object: bool = True, max_tokens: int = 2000, model: str | None = None) -> dict:
    load_dotenv()
    api_key = os.getenv("YANDEX_API_KEY", "").strip()
    if not api_key:
        raise ValueError("В .env нет YANDEX_API_KEY")
    model_uri = build_model_uri(model)
    openai = (model or os.getenv("YANDEX_GPT_MODEL", "").strip()) in OPENAI_MODELS
    url, body = _request(model_uri, openai, system_prompt, user_prompt, temperature, json_object, max_tokens)
    headers = {"Authorization": f"Api-Key {api_key}", "x-folder-id": os.getenv("YANDEX_FOLDER_ID", "").strip(),
               "Content-Type": "application/json"}
    result = {"text": None, "model_uri": model_uri, "model_version": None,
              "usage": {}, "elapsed_s": 0.0, "error": None}
    started = time.monotonic()
    try:
        payload = _post_with_retry(body, headers, url)
        result["text"], result["model_version"], result["usage"] = _read_answer(payload, openai)
    except (requests.RequestException, ValueError, KeyError, IndexError, TypeError) as exc:
        result["error"] = f"{type(exc).__name__}: {exc}"
    result["elapsed_s"] = round(time.monotonic() - started, 3)
    _log_call({
        "ts": datetime.now().astimezone().isoformat(timespec="seconds"),
        "purpose": purpose,
        "model_uri": model_uri,
        "model_version": result["model_version"],
        "temperature": temperature,
        "system_prompt": system_prompt,
        "user_prompt": user_prompt,
        "response_text": result["text"],
        "elapsed_s": result["elapsed_s"],
        "usage": result["usage"],
        "error": result["error"],
    })
    return result
