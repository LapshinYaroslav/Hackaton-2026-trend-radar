"""Minimal HTTP helper with per-source rate limiting. Adapters share one client."""

from __future__ import annotations

import threading
import time
from typing import Any, Protocol

import httpx

from collector.settings import Settings

# Паузы перед повторами после ответа 429, в секундах. Две паузы — три попытки всего.
RETRY_429_DELAYS_S = (10, 30)


class HttpTransport(Protocol):
    def get(
        self,
        url: str,
        *,
        params: dict[str, Any] | None = None,
        headers: dict[str, str] | None = None,
        timeout: float | None = None,
    ) -> httpx.Response: ...

    def close(self) -> None: ...


class RateLimiter:
    def __init__(self, min_interval_s: float) -> None:
        self._min_interval_s = min_interval_s
        self._lock = threading.Lock()
        self._last = 0.0

    def wait(self) -> None:
        with self._lock:
            now = time.perf_counter()
            sleep_for = self._min_interval_s - (now - self._last)
            if sleep_for > 0:
                time.sleep(sleep_for)
            self._last = time.perf_counter()


class HttpxTransport:
    def __init__(self, settings: Settings | None = None, client: httpx.Client | None = None) -> None:
        self._settings = settings or Settings()
        self._owns_client = client is None
        self._client = client or httpx.Client(
            timeout=self._settings.request_timeout_s,
            headers={"User-Agent": self._settings.user_agent},
            follow_redirects=True,
        )

    def get(
        self,
        url: str,
        *,
        params: dict[str, Any] | None = None,
        headers: dict[str, str] | None = None,
        timeout: float | None = None,
    ) -> httpx.Response:
        """GET с повтором при 429: пауза 10 с, повтор, пауза 30 с, повтор; третий 429 — ошибка.

        Одинаково для всех источников. Без повтора 429 в счётчиках давал неполное покрытие
        и no_counters (прогон Г6: 2 кандидата из 19). Пауза повтора больше паузы arXiv в 3 с,
        и запрос остаётся один: ждёт тот же поток, параллельного соединения нет.
        """
        for delay in (*RETRY_429_DELAYS_S, None):
            response = self._client.get(url, params=params, headers=headers, timeout=timeout)
            if response.status_code != 429 or delay is None:
                break
            time.sleep(delay)
        response.raise_for_status()
        return response

    def close(self) -> None:
        if self._owns_client:
            self._client.close()
