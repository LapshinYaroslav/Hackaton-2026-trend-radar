"""Minimal HTTP helper with per-source rate limiting. Adapters share one client."""

from __future__ import annotations

import threading
import time
from typing import Any, Protocol

import httpx

from collector.settings import Settings

# Паузы перед повторами после ответа 429, в секундах. Две паузы — три попытки всего.
RETRY_429_DELAYS_S = (10, 30)
# Задача И3 (разрешено Славой 26.09): arXiv и TechCrunch повторяются при 403, 429, 5xx и таймауте — до трёх
# повторов с паузами 5, 15, 45 с; для 429 — Retry-After, но не больше 60 с. На прогон (жизнь транспорта) —
# не больше 300 с ожидания на источник: дальше повторы источника отключаются, в notes — одна строка.
RETRY_HOSTS = {"export.arxiv.org", "techcrunch.com"}
RETRY_DELAYS_S = (5, 15, 45)
RETRY_AFTER_MAX_S = 60
RETRY_BUDGET_S = 300
BUDGET_NOTE = "повторы {host} отключены до конца прогона: исчерпано {budget} с ожидания"


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
        # Повторы по хостам RETRY_HOSTS: retries, n429, wait_s, disabled — для stats очередей источников.
        self.retry_stats: dict[str, dict[str, Any]] = {}
        self.notes: list[str] = []
        self._retry_lock = threading.Lock()

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
        host = httpx.URL(url).host
        if host in RETRY_HOSTS:
            return self._get_with_retries(host, url, params, headers, timeout)
        for delay in (*RETRY_429_DELAYS_S, None):
            response = self._client.get(url, params=params, headers=headers, timeout=timeout)
            if response.status_code != 429 or delay is None:
                break
            time.sleep(delay)
        response.raise_for_status()
        return response

    def _get_with_retries(self, host: str, url: str, params, headers, timeout) -> httpx.Response:
        """arXiv и TechCrunch: повтор при 403, 429, 5xx и таймауте (см. RETRY_* выше); после исчерпания —
        ошибка, как раньше: HTTPStatusError или исходный таймаут."""
        stats = self.retry_stats.setdefault(host, {"retries": 0, "n429": 0, "wait_s": 0.0, "disabled": False})
        for delay in (*RETRY_DELAYS_S, None):
            try:
                response, failure = self._client.get(url, params=params, headers=headers, timeout=timeout), None
            except httpx.TimeoutException as exc:
                response, failure = None, exc
            code = response.status_code if response is not None else None
            if code == 429:
                stats["n429"] += 1
            if not (failure or code in (403, 429) or (code or 0) >= 500) or delay is None:
                break
            wait = retry_after(response) if code == 429 else None
            if not self._reserve(host, delay if wait is None else wait):
                break
            time.sleep(delay if wait is None else wait)
        if failure is not None:
            raise failure
        response.raise_for_status()
        return response

    def _reserve(self, host: str, seconds: float) -> bool:
        """Берёт seconds из бюджета ожидания хоста; не хватает — повторы хоста отключаются (одна строка в notes)."""
        with self._retry_lock:
            stats = self.retry_stats[host]
            if stats["disabled"] or stats["wait_s"] + seconds > RETRY_BUDGET_S:
                if not stats["disabled"]:
                    stats["disabled"] = True
                    self.notes.append(BUDGET_NOTE.format(host=host, budget=RETRY_BUDGET_S))
                return False
            stats["wait_s"] += seconds
            stats["retries"] += 1
            return True

    def close(self) -> None:
        if self._owns_client:
            self._client.close()


def retry_after(response: httpx.Response | None) -> float | None:
    """Retry-After в секундах (только число), не больше RETRY_AFTER_MAX_S; нет или не число — None."""
    value = (response.headers.get("Retry-After") or "").strip() if response is not None else ""
    return min(float(value), RETRY_AFTER_MAX_S) if value.replace(".", "", 1).isdigit() else None
