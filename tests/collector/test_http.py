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


# --- Задача И3: arXiv и TechCrunch — повторы при 403, 429, 5xx и таймауте ---

ARXIV = "https://export.arxiv.org/api/query"
TECHCRUNCH = "https://techcrunch.com/wp-json/wp/v2/posts"


def scripted(answers: list) -> tuple[http.HttpxTransport, list[int]]:
    """Ответы по очереди: код статуса, (код, заголовки) или исключение-таймаут."""
    calls: list[int] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(1)
        answer = answers[min(len(calls), len(answers)) - 1]
        if isinstance(answer, Exception):
            raise answer
        code, headers = answer if isinstance(answer, tuple) else (answer, {})
        return httpx.Response(code, json={}, headers=headers)

    return http.HttpxTransport(client=httpx.Client(transport=httpx.MockTransport(handler))), calls


@pytest.mark.parametrize("first", [403, 429, 500, 503])
def test_retry_hosts_succeed_on_second_try(first) -> None:
    sender, calls = scripted([first, 200])
    with patch.object(http.time, "sleep") as sleep:
        assert sender.get(TECHCRUNCH).status_code == 200
    assert len(calls) == 2 and [c.args[0] for c in sleep.call_args_list] == [5]
    assert sender.retry_stats["techcrunch.com"]["retries"] == 1


def test_timeout_is_retried() -> None:
    sender, calls = scripted([httpx.ReadTimeout("slow"), 200])
    with patch.object(http.time, "sleep"):
        assert sender.get(ARXIV).status_code == 200
    assert len(calls) == 2


def test_exhausted_after_three_retries_5_15_45() -> None:
    sender, calls = scripted([503, 503, 503, 503])
    with patch.object(http.time, "sleep") as sleep:
        with pytest.raises(httpx.HTTPStatusError):
            sender.get(ARXIV)
    assert len(calls) == 4 and [c.args[0] for c in sleep.call_args_list] == [5, 15, 45]
    sender, _ = scripted([httpx.ReadTimeout("slow")] * 4)
    with patch.object(http.time, "sleep"):
        with pytest.raises(httpx.ReadTimeout):
            sender.get(ARXIV)


def test_retry_after_is_used_and_capped_at_60() -> None:
    sender, _ = scripted([(429, {"Retry-After": "7"}), (429, {"Retry-After": "600"}), (429, {"Retry-After": "soon"}), 200])
    with patch.object(http.time, "sleep") as sleep:
        assert sender.get(TECHCRUNCH).status_code == 200
    assert [c.args[0] for c in sleep.call_args_list] == [7, 60, 45]  # не число — обычная пауза
    assert sender.retry_stats["techcrunch.com"]["n429"] == 3


def test_run_budget_300_s_disables_host_with_one_note() -> None:
    sender, calls = scripted([503])
    with patch.object(http.time, "sleep") as sleep:
        for _ in range(6):             # 5 + 15 + 45 = 65 с на запрос: четыре запроса — 260 с
            with pytest.raises(httpx.HTTPStatusError):
                sender.get(ARXIV)
    waited = sum(c.args[0] for c in sleep.call_args_list)
    assert waited <= http.RETRY_BUDGET_S and sender.retry_stats["export.arxiv.org"]["disabled"]
    assert sender.notes == ["повторы export.arxiv.org отключены до конца прогона: исчерпано 300 с ожидания"]
    before = len(calls)
    with patch.object(http.time, "sleep") as sleep:
        with pytest.raises(httpx.HTTPStatusError):
            sender.get(ARXIV)
    assert len(calls) == before + 1 and sleep.call_count == 0  # после отключения — одна попытка


def test_other_hosts_keep_429_only_policy() -> None:
    sender, calls = scripted([503])
    with patch.object(http.time, "sleep") as sleep:
        with pytest.raises(httpx.HTTPStatusError):
            sender.get("https://api.openalex.org/works")
    assert len(calls) == 1 and sleep.call_count == 0 and sender.retry_stats == {}
