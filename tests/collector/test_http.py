"""Повтор при 429 в HttpxTransport: пауза 10 с, пауза 30 с, третий 429 — ошибка. Без сети."""
from unittest.mock import patch

import httpx
import pytest

from collector import http


def transport(statuses: list[int]) -> tuple[http.HttpxTransport, list[int]]:
    """Транспорт, отвечающий статусами по очереди; второй элемент — журнал запросов."""
    calls: list[int] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(1)
        return httpx.Response(statuses[len(calls) - 1], json={})

    client = httpx.Client(transport=httpx.MockTransport(handler))
    return http.HttpxTransport(client=client), calls


def test_429_then_success_waits_ten_seconds() -> None:
    sender, calls = transport([429, 200])
    with patch.object(http.time, "sleep") as sleep:
        assert sender.get("https://api.example/works").status_code == 200
    assert len(calls) == 2
    assert [c.args[0] for c in sleep.call_args_list] == [10]


def test_three_429_fail_after_ten_and_thirty() -> None:
    sender, calls = transport([429, 429, 429])
    with patch.object(http.time, "sleep") as sleep:
        with pytest.raises(httpx.HTTPStatusError):
            sender.get("https://api.example/works")
    assert len(calls) == 3
    assert [c.args[0] for c in sleep.call_args_list] == [10, 30]


def test_other_errors_are_not_retried() -> None:
    sender, calls = transport([400])
    with patch.object(http.time, "sleep") as sleep:
        with pytest.raises(httpx.HTTPStatusError):
            sender.get("https://api.example/works")
    assert len(calls) == 1 and sleep.call_count == 0


def test_rate_limiter_counts_from_start_to_start() -> None:
    """Интервал между началами запросов ≥ min_interval; долгий ответ не добавляет лишней паузы (Л5.1)."""
    import time

    limiter = http.RateLimiter(0.2)
    starts = []
    for duration in (0.05, 0.3, 0.0, 0.1):
        limiter.wait()
        starts.append(time.perf_counter())
        time.sleep(duration)  # «ответ» источника
    gaps = [b - a for a, b in zip(starts, starts[1:])]
    assert all(gap >= 0.199 for gap in gaps)
    # после ответа 0.3 с (дольше интервала) следующий запрос уходит сразу, а не через 0.3 + 0.2
    assert gaps[1] < 0.3 + 0.1
