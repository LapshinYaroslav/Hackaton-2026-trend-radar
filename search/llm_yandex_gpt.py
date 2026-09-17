"""Единая точка вызова YandexGPT: запрос к REST API и лог всех обращений."""
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
SESSION = requests.Session()  # keep-alive: TLS-рукопожатие не повторяется на каждый вызов
ALLOWED_MODELS = {"yandexgpt", "yandexgpt-lite"}
DEFAULT_MODEL = "yandexgpt-lite"
DEFAULT_TIMEOUT = 60

# Квота Yandex AI Studio: 10 одновременных генераций в синхронном режиме.
# Работаем на квоте целиком, страховка от превышения — повтор при 429 ниже.
MAX_CONCURRENT_CALLS = 10
CALL_LIMIT = threading.Semaphore(MAX_CONCURRENT_CALLS)
RETRY_ATTEMPTS = 3
RETRY_DELAY = 0.5


def build_model_uri() -> str:
    load_dotenv()
    folder_id = os.getenv("YANDEX_FOLDER_ID", "").strip()
    model = os.getenv("YANDEX_GPT_MODEL", "").strip() or DEFAULT_MODEL
    if not folder_id:
        raise ValueError("В .env нет YANDEX_FOLDER_ID")
    if model not in ALLOWED_MODELS:
        raise ValueError(f"Модель {model!r} не разрешена, выберите из {sorted(ALLOWED_MODELS)}")
    return f"gpt://{folder_id}/{model}/latest"


def _log_call(record: dict) -> None:
    """Дописывает строку в logs/llm_calls.jsonl. Ключ API в лог не попадает."""
    try:
        LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
        with LOG_FILE.open("a", encoding="utf-8") as file:
            file.write(json.dumps(record, ensure_ascii=False) + "\n")
    except OSError as exc:
        print(f"Предупреждение: не удалось записать лог вызова LLM: {exc}")


def _post_with_retry(body: dict, headers: dict) -> dict:
    """POST под ограничением конкурентности, с повтором при 429 и растущей паузой."""
    delay = RETRY_DELAY
    for attempt in range(RETRY_ATTEMPTS):
        with CALL_LIMIT:
            response = SESSION.post(COMPLETION_URL, headers=headers, json=body, timeout=DEFAULT_TIMEOUT)
        if getattr(response, "status_code", 200) == 429 and attempt < RETRY_ATTEMPTS - 1:
            time.sleep(delay)
            delay *= 2
            continue
        response.raise_for_status()
        return response.json()
    raise requests.HTTPError("429 после всех повторов")


def ask_llm(system_prompt: str, user_prompt: str, purpose: str, temperature: float = 0.3,
            json_object: bool = True, max_tokens: int = 2000) -> dict:
    load_dotenv()
    api_key = os.getenv("YANDEX_API_KEY", "").strip()
    if not api_key:
        raise ValueError("В .env нет YANDEX_API_KEY")
    model_uri = build_model_uri()
    body = {
        "modelUri": model_uri,
        "completionOptions": {"stream": False, "temperature": temperature, "maxTokens": str(max_tokens)},
        "messages": [
            {"role": "system", "text": system_prompt},
            {"role": "user", "text": user_prompt},
        ],
        "json_object": json_object,
    }
    headers = {"Authorization": f"Api-Key {api_key}", "x-folder-id": os.getenv("YANDEX_FOLDER_ID", "").strip(),
               "Content-Type": "application/json"}
    result = {"text": None, "model_uri": model_uri, "model_version": None,
              "usage": {}, "elapsed_s": 0.0, "error": None}
    started = time.monotonic()
    try:
        payload = _post_with_retry(body, headers)
        data = payload.get("result", payload)
        result["text"] = data["alternatives"][0]["message"]["text"]
        result["model_version"] = data.get("modelVersion")
        result["usage"] = data.get("usage", {})
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
