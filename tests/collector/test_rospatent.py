"""Счётчик Роспатента: повторы, сбой не равен нулю, кэш задачи П, одна фраза — один запрос."""
import json
import math

import httpx
import pytest

from collector import rospatent as rp

DATASETS = ["ru_since_1994", "us"]


def scripted(*answers):
    """Заглушка POST: по очереди отдаёт ответы; исключение из списка — выбрасывает."""
    calls = []

    def post(url, json=None, headers=None, timeout=None):
        calls.append({"url": url, "body": json, "timeout": timeout})
        answer = answers[len(calls) - 1]
        if isinstance(answer, Exception):
            raise answer
        return answer

    post.calls = calls
    return post


def ok(total):
    return httpx.Response(200, json={"total": total, "available": 0, "hits": []})


def count(post, tmp_path, phrase="robot arm", sleeps=None):
    return rp.count_phrase(phrase, DATASETS, "secret", post=post, cache_dir=tmp_path,
                           sleep=(sleeps.append if sleeps is not None else lambda s: None))


def test_retry_after_connect_error(tmp_path):
    post = scripted(httpx.ConnectError("down"), ok(12))
    sleeps = []
    result = count(post, tmp_path, sleeps=sleeps)
    assert (result["n_pat"], result["failed"], result["retries"]) == (12, False, 1)
    assert sleeps == [1]


def test_timeout_and_5xx_retry_with_growing_pauses(tmp_path):
    post = scripted(httpx.ReadTimeout("slow"), httpx.Response(503), httpx.Response(502), ok(3))
    sleeps = []
    result = count(post, tmp_path, sleeps=sleeps)
    assert result["n_pat"] == 3 and sleeps == [1, 2, 4] and result["requests"] == 4


def test_failure_after_retries_is_none_not_zero(tmp_path):
    post = scripted(*[httpx.ConnectError("down")] * 4)
    result = count(post, tmp_path)
    assert result["n_pat"] is None and result["failed"] is True
    assert result["requests"] == 4
    assert not list(tmp_path.glob("*.json")), "сбой не должен попадать в кэш"


def test_total_zero_is_zero_and_not_failed(tmp_path):
    result = count(scripted(ok(0)), tmp_path)
    assert result["n_pat"] == 0 and result["failed"] is False


def test_429_waits_retry_after_header(tmp_path):
    post = scripted(httpx.Response(429, headers={"Retry-After": "7"}), httpx.Response(429), ok(5))
    sleeps = []
    result = count(post, tmp_path, sleeps=sleeps)
    assert result["n_pat"] == 5 and sleeps == [7.0, 5.0] and result["n429"] == 2


def test_429_gives_up_after_three_waits(tmp_path):
    post = scripted(*[httpx.Response(429)] * 4)
    result = count(post, tmp_path)
    assert result["failed"] is True and result["requests"] == 4


def test_client_error_is_not_retried(tmp_path):
    post = scripted(httpx.Response(401))
    result = count(post, tmp_path)
    assert result["failed"] is True and result["requests"] == 1


def test_request_body_and_timeout(tmp_path):
    post = scripted(ok(1))
    count(post, tmp_path, phrase="edge ai")
    call = post.calls[0]
    assert call["timeout"] == rp.TIMEOUT_S == 15.0
    assert call["body"]["q"] == '"edge ai"' and call["body"]["datasets"] == DATASETS
    assert call["body"]["filter"] == rp.WINDOW and call["body"]["limit"] == rp.ROSPATENT_LIMIT


def test_cache_key_is_the_one_of_task_p():
    """Ключ и путь кэша совпадают с файлами, собранными в задаче П (sha зафиксирован)."""
    from model.config import ROSPATENT_DATASETS
    key = rp.request_key("neuromorphic chip", ROSPATENT_DATASETS)
    assert rp.cache_path(key).name == ("7d4e2410880c13e86802c90f8b6ea8afc36b7964ef8469f"
                                       "3488a5999abd31069.json")


def test_cached_phrase_makes_no_request(tmp_path):
    key = rp.request_key("robot arm", DATASETS)
    rp.cache_path(key, tmp_path).write_text(json.dumps({"key": key, "response": {"total": 42}}),
                                            encoding="utf-8")
    post = scripted()
    result = count(post, tmp_path)
    assert result["n_pat"] == 42 and result["cached"] is True and post.calls == []


def test_datasets_are_part_of_the_key():
    assert rp.cache_path(rp.request_key("x", ["us"])) != rp.cache_path(rp.request_key("x", ["ep"]))


def test_count_all_one_request_per_phrase(tmp_path):
    post = scripted(ok(1), ok(2))
    results, summary = rp.count_all(["a b", "c d", "a b", "c d"], DATASETS, "secret",
                                    post=post, cache_dir=tmp_path, parallel=1,
                                    sleep=lambda s: None)
    assert len(post.calls) == 2 and set(results) == {"a b", "c d"}
    assert (summary["candidates"], summary["requests"], summary["failures"]) == (2, 2, 0)


def test_count_all_without_token_makes_no_request(tmp_path):
    post = scripted()
    results, summary = rp.count_all(["a b"], DATASETS, None, post=post, cache_dir=tmp_path)
    assert post.calls == [] and results["a b"]["failed"] is True and summary["failures"] == 1


@pytest.mark.parametrize("n_pat,n_research,expected", [(0, 0, None), (3, 9, 0.25)])
def test_share_patent_from_counts(n_pat, n_research, expected):
    from model.features import share_patent
    value = share_patent(n_pat, n_research)
    assert (math.isnan(value) if expected is None else value == expected)
