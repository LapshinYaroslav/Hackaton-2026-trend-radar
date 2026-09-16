"""
Тестовый скрипт для п.4 ТЗ: проверка доступа к YandexGPT (Yandex Cloud Foundation Models).
Отправляет 5 одинаковых запросов, измеряет время ответа и проверяет,
парсится ли ответ моделью как валидный JSON.
"""

import json      # для json.loads() — проверка, что модель вернула корректный JSON
import os        # для чтения переменных окружения (os.getenv)
import re        # для вырезания ```json ... ``` обёртки, если модель её добавляет
import time      # для замера времени ответа (time.perf_counter)

import requests                # для HTTP-запроса к API YandexGPT
from dotenv import load_dotenv  # для загрузки .env в окружение процесса

load_dotenv()  # читает файл .env в корне проекта и кладёт значения в os.environ

# Ключ и ID каталога читаем только через os.getenv — их нельзя хардкодить в коде,
# чтобы они не попали в git (правило хранения ключей из п.3 ТЗ).
API_KEY = os.getenv("YANDEX_API_KEY")
FOLDER_ID = os.getenv("YANDEX_FOLDER_ID")

# Модель по умолчанию — yandexgpt-lite (дешевле и быстрее для теста).
# Если нужна полная YandexGPT, поменяй на "yandexgpt/latest".
MODEL_URI = f"gpt://{FOLDER_ID}/yandexgpt-lite/latest"

# Эндпоинт синхронной генерации Foundation Models.
API_URL = "https://llm.api.cloud.yandex.net/foundationModels/v1/completion"

# Промпт из ТЗ — просим модель вернуть только JSON без пояснений.
PROMPT = (
    "Пользователь ищет: технологии в медицине. "
    "Придумай 6 поисковых подзапросов для научных баз: 3 на русском и 3 на английском. "
    'Верни только JSON вида {"subqueries": ["...", "..."]}, без пояснений.'
)

N_RUNS = 5  # сколько раз прогоняем запрос, как требует п.4 ТЗ


def strip_code_fence(text: str) -> str:
    """Убирает обёртку ```json ... ``` или ``` ... ```, если модель её добавила."""
    match = re.search(r"```(?:json)?\s*(.*?)\s*```", text, re.DOTALL)
    return match.group(1) if match else text


def ask_yandexgpt(prompt: str) -> tuple[str, float]:
    """Отправляет один запрос в YandexGPT и возвращает (текст ответа, время в секундах)."""
    headers = {
        "Authorization": f"Api-Key {API_KEY}",  # авторизация по API-ключу сервисного аккаунта
        "Content-Type": "application/json",
    }
    body = {
        "modelUri": MODEL_URI,
        "completionOptions": {
            "stream": False,       # нам нужен полный ответ целиком, не потоковый
            "temperature": 0.3,    # низкая температура — ответ стабильнее для парсинга JSON
            "maxTokens": "2000",
        },
        "messages": [{"role": "user", "text": prompt}],
    }

    start = time.perf_counter()  # засекаем время перед отправкой запроса
    response = requests.post(API_URL, headers=headers, json=body, timeout=30)
    elapsed = time.perf_counter() - start  # время ответа = после минус до

    response.raise_for_status()  # бросит исключение, если сервер вернул ошибку (401/403/500 и т.д.)
    data = response.json()
    answer_text = data["result"]["alternatives"][0]["message"]["text"]  # текст ответа модели
    return answer_text, elapsed


def main() -> None:
    if not API_KEY or not FOLDER_ID:
        # без ключа/каталога нет смысла даже пытаться — сразу понятная ошибка
        raise SystemExit(
            "Не заданы YANDEX_API_KEY / YANDEX_FOLDER_ID. "
            "Проверь файл .env в корне проекта."
        )

    latencies: list[float] = []  # сюда собираем время каждого запроса
    json_ok_count = 0            # сколько раз ответ распарсился как валидный JSON
    fence_count = 0              # сколько раз модель обернула JSON в ```...```

    for i in range(1, N_RUNS + 1):
        print(f"\n--- Запрос {i}/{N_RUNS} ---")
        try:
            raw_text, elapsed = ask_yandexgpt(PROMPT)
        except requests.RequestException as exc:
            # сетевая ошибка или ошибка API — не роняем весь прогон, просто фиксируем
            print(f"Ошибка запроса: {exc}")
            continue

        latencies.append(elapsed)
        print(f"Время ответа: {elapsed:.2f} сек")
        print(f"Сырой ответ: {raw_text!r}")

        cleaned_text = strip_code_fence(raw_text)
        if cleaned_text != raw_text:
            fence_count += 1  # модель добавила ```json — считаем отдельно

        try:
            parsed = json.loads(cleaned_text)
            json_ok_count += 1
            print(f"JSON распарсен успешно: {parsed}")
        except json.JSONDecodeError as exc:
            print(f"Не удалось распарсить JSON: {exc}")

    print("\n=== Итог ===")
    if latencies:
        avg_latency = sum(latencies) / len(latencies)
        print(f"Среднее время ответа: {avg_latency:.2f} сек ({len(latencies)} успешных запросов)")
    else:
        print("Ни одного успешного запроса — среднее время посчитать нельзя.")
    print(f"Корректный JSON: {json_ok_count}/{N_RUNS}")
    print(f"Ответ был обёрнут в ```: {fence_count}/{N_RUNS}")


if __name__ == "__main__":
    main()
